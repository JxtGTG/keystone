#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large file neuron batch processor (mean version)
- Maintains the same directory/flow as the original program: discover model directories → process files one by one → model-level/global summary
- No longer separates think/answer, and does not calculate differences
- For each file: layer by layer, neuron by neuron, sum all token activations and divide by token count, output an activation_means JSON
"""

import os
import json
import argparse
import datetime
from typing import Dict, List, Tuple, Any
import gc
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from tqdm import tqdm

try:
    import ijson
except ImportError:
    print("❌ Missing ijson library, please install: pip install ijson")
    raise SystemExit(1)


class LargeNeuronMeanProcessor:
    """Per-layer, per-neuron mean activation processor (no segments; outputs per-file means)."""

    def __init__(self, chunk_size: int = 500, num_workers: int = None):
        self.chunk_size = chunk_size
        self.num_workers = num_workers or min(cpu_count(), 8)  # Default to CPU core count, max 8

    # ---------------- Directory scanning & model type (keep structure consistent with original) ----------------
    def detect_model_type(self, model_dir_name: str) -> str:
        name = model_dir_name.lower()
        if "mixtral" in name:
            return "mixtral"
        elif "qwen3" in name and ("moe" in name or "a3b" in name):
            return "qwen3moe"
        elif "qwen3" in name:
            return "qwen3"
        elif "qwen" in name:
            return "qwen2"
        elif "llama" in name:
            return "llama"
        elif "gemma" in name:
            return "gemma"
        else:
            return "other"

    def discover_model_directories(self, base_input_dir: str, target_models: List[str] = None) -> Dict[str, Dict]:
        """Discover model directories and token_data files within them (supports new prompt subdirectory structure)
        
        Args:
            base_input_dir: Base input directory
            target_models: List of model directory names to process, if None then process all models
        """
        model_files: Dict[str, Dict[str, Any]] = {}

        if not os.path.exists(base_input_dir):
            print(f"❌ Input directory does not exist: {base_input_dir}")
            return {}

        print(f"🔍 Scanning model directory: {base_input_dir}")
        if target_models:
            print(f"🎯 Target models to process: {', '.join(target_models)}")
        
        for item in os.listdir(base_input_dir):
            item_path = os.path.join(base_input_dir, item)
            if not os.path.isdir(item_path):
                continue

            # If target models are specified, only process those models
            if target_models and item not in target_models:
                continue

            model_name = item
            model_type = self.detect_model_type(model_name)
            print(f"📁 Found model directory: {model_name}  |  🏷️ Model type: {model_type}")

            # New structure: model_name/prompt_name/token_data_file.json
            prompt_files: Dict[str, List[str]] = {}
            for prompt_dir in os.listdir(item_path):
                prompt_path = os.path.join(item_path, prompt_dir)
                if not os.path.isdir(prompt_path):
                    continue
                
                # Find token_data files under the prompt directory
                prompt_token_files = []
                for f in os.listdir(prompt_path):
                    if f.endswith(".json") and "token_data" in f:
                        fp = os.path.join(prompt_path, f)
                        prompt_token_files.append(fp)
                        print(f"    📄 {prompt_dir}/{f}")
                
                if prompt_token_files:
                    prompt_files[prompt_dir] = sorted(prompt_token_files)

            if prompt_files:
                model_files[model_name] = {"type": model_type, "prompt_files": prompt_files}
                total_files = sum(len(files) for files in prompt_files.values())
                print(f"   ✅ Found {len(prompt_files)} prompt directories, total {total_files} token data files")
            else:
                print(f"   ⚠️ No token data files found")

        print(f"\n📊 Total models found: {len(model_files)}")
        for m, info in model_files.items():
            total_files = sum(len(files) for files in info['prompt_files'].values())
            print(f"   {m}: mode={info['type']}, prompts={len(info['prompt_files'])}, files={total_files}")

      
        if target_models:
            not_found = set(target_models) - set(model_files.keys())
            if not_found:
                print(f"⚠️ The following specified model directories were not found: {', '.join(not_found)}")

        return model_files

    # ---------------- Basic utilities: parsing/accumulation/mean ----------------
    def _is_empty_file(self, file_path: str) -> bool:
        """Check if file is empty or contains empty array - optimized version"""
        try:
            # First check file size to avoid reading large files
            file_size = os.path.getsize(file_path)
            if file_size < 10:  # Less than 10 bytes is likely empty
                return True
            if file_size > 1024 * 1024:  # Larger than 1MB, only read first 100 bytes to check
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read(100).strip()
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
            # Check if it's an empty array, empty object, or other empty content
            if not content or content in ["[]", "{}", "null", "null\n"]:
                return True
                
            # Check if it only contains whitespace
            if not content.strip():
                return True
                
        except Exception as e:
            print(f"⚠️ Error checking empty file: {e}")
            return False
            
        return False

    def _iter_token_items_with_fallback(self, file_path: str):
        """
        Compatible with two structures (fast path):
        1) Top-level array: [ {token_text, activations, ...}, ... ]
        2) Top-level object containing token_data array: {"token_data": [ {...}, {...} ]}

        Determines structure by reading the first non-whitespace character at the start of the file,
        avoiding a second read for object structures.
        """
        with open(file_path, "rb") as f:
            # Read file header, determine first non-whitespace character
            head = f.read(128)
            first_non_ws = None
            for b in head:
                if chr(b) not in (" ", "\n", "\r", "\t"):
                    first_non_ws = b
                    break
            # Return to file start for ijson to parse correctly
            f.seek(0)

            if first_non_ws == ord('['):
                it = ijson.items(f, "item")
            else:
               
                it = ijson.items(f, "token_data.item")

            for obj in it:
                yield obj

    def _accumulate(self, activations: Dict[str, Dict[str, list]],
                    sums: Dict[str, Dict[str, np.ndarray]], model_type: str = "other") -> None:
        """Add this token's activations to sums (layer by layer, neuron by neuron)
        
        Args:
            activations: Activation value data
            sums: Accumulator
            model_type: Model type, used to determine if it's a MoE model
        """
        for act_type, layers in activations.items():
            if act_type not in sums:
                sums[act_type] = {}
            
            for layer, values in layers.items():
                # Check if it's MoE model's expert-separated format
                # Only use expert-separated format for MoE models and activation types fwd_up or fwd_down
                if (model_type in ["qwen3moe", "mixtral"] and 
                    act_type in ["fwd_up", "fwd_down"] and 
                    isinstance(values, dict)):
                    # MoE model: values is expert dictionary {expert_0: [...], expert_1: None, ...}
                    self._accumulate_moe_experts(act_type, layer, values, sums)
                else:
                    # Regular model or attention activations: values is direct activation value list
                    self._accumulate_regular(act_type, layer, values, sums)
    
    def _accumulate_moe_experts(self, act_type: str, layer: str, expert_values: Dict[str, list],
                               sums: Dict[str, Dict[str, np.ndarray]]) -> None:
        """Process MoE model's expert-separated activations - directly accumulate all experts - optimized version"""
        for expert_name, expert_activations in expert_values.items():
            # Create independent accumulator key for each expert
            expert_key = f"{layer}_{expert_name}"
            
            # Directly convert to numpy array and accumulate (inactive experts are already 0 values)
            # Use float32 to reduce memory usage
            arr = np.asarray(expert_activations, dtype=np.float32)
            if expert_key not in sums[act_type]:
                sums[act_type][expert_key] = np.zeros_like(arr, dtype=np.float32)
            
            # If lengths don't match, align to shorter length
            if sums[act_type][expert_key].shape[0] != arr.shape[0]:
                n = min(sums[act_type][expert_key].shape[0], arr.shape[0])
                sums[act_type][expert_key][:n] += arr[:n]
            else:
                sums[act_type][expert_key] += arr
    
    def _accumulate_regular(self, act_type: str, layer: str, values: list,
                           sums: Dict[str, Dict[str, np.ndarray]]) -> None:
       
        arr = np.asarray(values, dtype=np.float32)
        if layer not in sums[act_type]:
            sums[act_type][layer] = np.zeros_like(arr, dtype=np.float32)
        
       
        if sums[act_type][layer].shape[0] != arr.shape[0]:
            n = min(sums[act_type][layer].shape[0], arr.shape[0])
            sums[act_type][layer][:n] += arr[:n]
        else:
            sums[act_type][layer] += arr

    def _divide_inplace(self, nested_arrays: Dict[str, Dict[str, np.ndarray]], scalar: float) -> None:
        if scalar == 0:
            return
        for act_type in nested_arrays:
            for layer in nested_arrays[act_type]:
                nested_arrays[act_type][layer] = nested_arrays[act_type][layer] / scalar

    def _to_lists(self, nested_arrays: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, list]]:
        out: Dict[str, Dict[str, list]] = {}
        for act_type, layers in nested_arrays.items():
            out[act_type] = {}
            for layer, arr in layers.items():
                out[act_type][layer] = arr.tolist()
        return out

    def _sum_file(self, file_path: str, model_type: str = "other") -> Tuple[Dict[str, Dict[str, np.ndarray]], int]:
        """Accumulate activations layer by layer, neuron by neuron for all tokens in a single file, return sums and token count"""
        # First check if file is empty
        if self._is_empty_file(file_path):
            return None, 0
            
        sums: Dict[str, Dict[str, np.ndarray]] = {}
        token_count = 0
        pbar = tqdm(desc=f"Accumulating {os.path.basename(file_path)}", unit="tok", leave=False, mininterval=0.3)
        try:
            has_valid_data = False
            for token_item in self._iter_token_items_with_fallback(file_path):
                activations = token_item.get("activations")
                if activations:
                    self._accumulate(activations, sums, model_type)
                    has_valid_data = True
                token_count += 1
                pbar.update(1)
                if token_count % self.chunk_size == 0:
                    gc.collect()
            
            # If file is empty or has no valid activation data, return special identifier
            if token_count == 0 or not has_valid_data:
                return None, 0
                
        finally:
            pbar.close()
        return sums, token_count

    
    def _process_file_worker(self, args_tuple):
     
        file_path, output_dir, model_type, prompt_name, model_name = args_tuple
        return self.process_single_file(file_path, output_dir, model_type, prompt_name, model_name)

    def process_files_parallel(self, file_list: List[Tuple[str, str, str, str, str]], 
                              output_dir: str) -> List[Dict[str, Any]]:
        """Process multiple files in parallel"""
        if len(file_list) == 0:
            return []
            
        print(f"🚀 Using {self.num_workers} processes to process {len(file_list)} files in parallel")
        
        # Create processing function
        worker_func = partial(self._process_file_worker)
        
        results = []
        with Pool(processes=self.num_workers) as pool:
            # Use imap_unordered for better progress display
            for result in tqdm(pool.imap_unordered(worker_func, file_list), 
                             total=len(file_list), 
                             desc="Processing files in parallel", 
                             mininterval=0.5):
                results.append(result)
        
        return results

    # ---------------- Single file processing / saving ----------------
    def _save_means(self, file_path: str, means_nested: Dict[str, Dict[str, list]],
                    output_dir: str, prompt_name: str = None, model_name: str = None) -> str:
        # Extract prompt name and model name from file path (if not provided)
        if prompt_name is None or model_name is None:
            # Extract from path: .../model_name/prompt_name/token_data_file.json
            path_parts = file_path.replace('\\', '/').split('/')
            if len(path_parts) >= 3:
                if prompt_name is None:
                    prompt_name = path_parts[-2]  # Second to last part should be prompt name
                if model_name is None:
                    model_name = path_parts[-3]  # Third to last part should be model name
            else:
                if prompt_name is None:
                    prompt_name = "unknown"
                if model_name is None:
                    model_name = "unknown"
        
        # Create output directory structure: output_dir/model_name/prompt_name/
        model_output_dir = os.path.join(output_dir, model_name)
        prompt_output_dir = os.path.join(model_output_dir, prompt_name)
        os.makedirs(prompt_output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        payload = {
            "meta": {
                "source_file": os.path.abspath(file_path),
                "model_name": model_name,
                "prompt_name": prompt_name,
                "method": "mean_over_all_tokens",
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            },
            "activation_means": means_nested
        }
        out_name = f"{base_name}_activation_means_{ts}.json"
        out_path = os.path.join(prompt_output_dir, out_name)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        return out_path

    def process_single_file(self, file_path: str, output_dir: str, model_type: str = None, prompt_name: str = None, model_name: str = None) -> Dict[str, Any]:
        """Process a single token_data file: output a mean JSON"""
        print(f"\n{'='*60}")
        print(f"🔄 Processing: {os.path.basename(file_path)}")
        if model_name:
            print(f"🤖 Model: {model_name}")
        if prompt_name:
            print(f"📝 Prompt: {prompt_name}")
            
        # Check if target output directory already exists
        if prompt_name and model_name:
            target_prompt_dir = os.path.join(output_dir, model_name, prompt_name)
            if os.path.exists(target_prompt_dir):
                print(f"⏭️ Skipping: target directory already exists {target_prompt_dir}")
                return {
                    "status": "skipped",
                    "file": file_path,
                    "model_type": model_type,
                    "model_name": model_name,
                    "prompt_name": prompt_name,
                    "tokens_count": 0,
                    "reason": "target_directory_exists"
                }
        
        try:
            size_gb = os.path.getsize(file_path) / (1024 ** 3)
            print(f"📊 Size: {size_gb:.2f} GB")
        except Exception:
            pass

        try:
            sums, token_count = self._sum_file(file_path, model_type)
            
            # Check if file is empty or has no valid data
            if sums is None or token_count == 0:
                print(f"⚠️ Skipping: file is empty or has no valid activation data")
                return {
                    "status": "skipped",
                    "file": file_path,
                    "model_type": model_type,
                    "model_name": model_name,
                    "prompt_name": prompt_name,
                    "tokens_count": 0,
                    "reason": "empty_file_or_no_valid_data"
                }
            
            means = {a: {l: arr.copy() for l, arr in layers.items()} for a, layers in sums.items()}
            self._divide_inplace(means, float(token_count))
            means_listed = self._to_lists(means)

            out_path = self._save_means(file_path, means_listed, output_dir, prompt_name, model_name)
            print(f"✅ Complete: mean values saved → {out_path}")

            return {
                "status": "success",
                "file": file_path,
                "model_type": model_type,
                "model_name": model_name,
                "prompt_name": prompt_name,
                "tokens_count": token_count,
                "output_file": out_path
            }
        except Exception as e:
            print(f"❌ Failed: {e}")
            return {
                "status": "failed",
                "file": file_path,
                "model_type": model_type,
                "model_name": model_name,
                "prompt_name": prompt_name,
                "error": str(e)
            }

    # ---------------- Batch processing (by model directory) ----------------
    def process_model_directory(self, model_name: str, model_info: Dict, output_base_dir: str, use_parallel: bool = True) -> Dict[str, Any]:
        print(f"\n{'='*80}")
        print(f"🚀 Starting to process model: {model_name}")
        print(f"🏷️ Model type: {model_info['type']}")
        
        prompt_files = model_info['prompt_files']
        total_files = sum(len(files) for files in prompt_files.values())
        print(f"📁 Number of prompts: {len(prompt_files)}, total files: {total_files}")

        # Prepare task list for all files, also check if prompt directories already exist
        file_tasks = []
        skipped_prompts = []
        results: List[Dict[str, Any]] = []
        
        for prompt_name, files in prompt_files.items():
            # Check if target prompt directory already exists
            target_prompt_dir = os.path.join(output_base_dir, model_name, prompt_name)
            if os.path.exists(target_prompt_dir):
                print(f"⏭️ Skipping prompt: {prompt_name} (target directory already exists: {target_prompt_dir})")
                skipped_prompts.append(prompt_name)
                # Add virtual results for skipped prompts
                for fp in files:
                    results.append({
                        "status": "skipped",
                        "file": fp,
                        "model_type": model_info['type'],
                        "model_name": model_name,
                        "prompt_name": prompt_name,
                        "tokens_count": 0,
                        "reason": "target_directory_exists"
                    })
                continue
            
            for fp in files:
                file_tasks.append((fp, output_base_dir, model_info['type'], prompt_name, model_name))
        success_count = 0
        failed_count = 0
        skipped_count = 0
        total_tokens = 0
        prompt_results = {}

        if use_parallel and len(file_tasks) > 1:
            # Parallel processing
            print(f"🚀 Enabling parallel processing mode (processing {len(file_tasks)} files, processes={self.num_workers})")
            parallel_results = self.process_files_parallel(file_tasks, output_base_dir)
            results.extend(parallel_results)
        else:
            # Serial processing (only process non-skipped prompts)
            print(f"🔄 Using serial processing mode")
            file_counter = 0
            for prompt_name, files in prompt_files.items():
                if prompt_name in skipped_prompts:
                    continue  # Skip already processed prompts
                    
                print(f"\n📝 Processing prompt: {prompt_name} ({len(files)} files)")
                prompt_results[prompt_name] = []
                
                for i, fp in enumerate(files):
                    file_counter += 1
                    print(f"\nProgress: [{file_counter}/{len(file_tasks)}] - {prompt_name}")
                    res = self.process_single_file(fp, output_base_dir, model_info['type'], prompt_name, model_name)
                    results.append(res)
                    prompt_results[prompt_name].append(res)

        # Count results
        for res in results:
            if res.get("status") == "success":
                success_count += 1
                total_tokens += res.get("tokens_count", 0)
            elif res.get("status") == "skipped":
                skipped_count += 1
            else:
                failed_count += 1

        # Reorganize prompt_results (needed for parallel processing)
        if use_parallel and len(file_tasks) > 1:
            prompt_results = {}
            for res in results:
                prompt_name = res.get("prompt_name", "unknown")
                if prompt_name not in prompt_results:
                    prompt_results[prompt_name] = []
                prompt_results[prompt_name].append(res)

        summary = {
            "model_info": {
                "name": model_name,
                "type": model_info['type'],
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "total_prompts": len(prompt_files),
                "total_files": total_files,
                "success_count": success_count,
                "failed_count": failed_count,
                "skipped_count": skipped_count,
                "total_tokens_processed": total_tokens
            },
            "prompt_results": prompt_results,
            "results": results
        }

        # Create model-level output directory to save summary file
        model_output_dir = os.path.join(output_base_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        summary_file = os.path.join(model_output_dir, f"{model_name}_summary_{summary['model_info']['timestamp']}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Model {model_name} processing complete!")
        print(f"   📊 Success: {success_count}/{total_files} files")
        if skipped_count > 0:
            print(f"   ⚠️ Skipped: {skipped_count} files")
        if len(skipped_prompts) > 0:
            print(f"   ⏭️ Skipped prompts: {len(skipped_prompts)} ({', '.join(skipped_prompts)})")
        if failed_count > 0:
            print(f"   ❌ Failed: {failed_count} files")
        print(f"   📝 Processed prompts: {len(prompt_files) - len(skipped_prompts)}/{len(prompt_files)}")
        print(f"   🧮 Total tokens: {total_tokens}")
        print(f"   📄 Summary: {summary_file}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Large file neuron batch processor (mean version: output activation_means per file)")
    parser.add_argument('--input_dir', '-i',
                        default='',
                        help='Base input directory (contains model subdirectories)')
    parser.add_argument('--output_dir', '-o',
                        default='',
                        help='Output directory (organized by model/file subdirectories)')
    parser.add_argument('--chunk_size', '-c', type=int, default=500, help='Number of tokens to process before calling gc.collect()')
    parser.add_argument('--single_file', '-s', type=str, default=None, help='Process only the specified single token_data file (optional)')
    parser.add_argument('--models', '-m', nargs='*', default=[""], 
                        help='Specify model directory names to process (can specify multiple, separated by spaces). If not specified, process all models')
    parser.add_argument('--num_workers', '-w', type=int, default=16, help='Number of processes for parallel processing (default uses CPU core count)')
    parser.add_argument('--no_parallel', action='store_true', help='Disable parallel processing, use serial mode')
    args = parser.parse_args()

    try:
        proc = LargeNeuronMeanProcessor(chunk_size=args.chunk_size, num_workers=args.num_workers)
       

  
        if args.single_file:
            if not os.path.exists(args.single_file):
             
                return

          
            os.makedirs(args.output_dir, exist_ok=True)

            
            model_type = None
            current_path = args.single_file
            for _ in range(5):
                current_path = os.path.dirname(current_path)
                if current_path in ('/', ''):
                    break
                dir_name = os.path.basename(current_path)
                if dir_name and not dir_name.startswith('.'):
                    detected_type = proc.detect_model_type(dir_name)
                    if detected_type != "other": 
                        model_type = detected_type
                        
                        break
            if model_type is None:
                model_type = "other"
              

            _ = proc.process_single_file(args.single_file, args.output_dir, model_type)
            return

        
        model_directories = proc.discover_model_directories(args.input_dir, args.models)
       

        os.makedirs(args.output_dir, exist_ok=True)

        all_results: List[Dict[str, Any]] = []
        total_success = 0
        total_failed = 0
        total_skipped = 0
        total_files = 0
        total_tokens = 0
        total_prompts = 0

        for model_name, model_info in model_directories.items():
            use_parallel = not args.no_parallel
            summary = proc.process_model_directory(model_name, model_info, args.output_dir, use_parallel)
            all_results.append(summary)
            total_success += summary["model_info"]["success_count"]
            total_failed += summary["model_info"]["failed_count"]
            total_skipped += summary["model_info"]["skipped_count"]
            total_files += summary["model_info"]["total_files"]
            total_tokens += summary["model_info"]["total_tokens_processed"]
            total_prompts += summary["model_info"]["total_prompts"]

        global_summary = {
            "global_info": {
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "total_models": len(model_directories),
                "total_prompts": total_prompts,
                "total_files": total_files,
                "total_success": total_success,
                "total_failed": total_failed,
                "total_skipped": total_skipped,
                "total_tokens_processed": total_tokens,
                "input_dir": os.path.abspath(args.input_dir),
                "output_dir": os.path.abspath(args.output_dir)
            },
            "model_results": all_results
        }

        global_summary_file = os.path.join(
            args.output_dir, f"global_summary_{global_summary['global_info']['timestamp']}.json"
        )
        with open(global_summary_file, "w") as f:
            json.dump(global_summary, f, indent=2)

        print(f"\n🎉 All models processing complete!")
        print(f"✅ Total success: {total_success}/{total_files} files")
        if total_skipped > 0:
            print(f"⚠️ Total skipped: {total_skipped} empty files")
        if total_failed > 0:
            print(f"❌ Total failed: {total_failed} files")
        print(f"📝 Total processed: {total_prompts} prompts")
        print(f"🧮 Total tokens: {total_tokens}")
        print(f"📊 Global summary: {global_summary_file}")

    except Exception as e:
        print(f"❌ Program failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
