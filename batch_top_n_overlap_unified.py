#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified neuron overlap analyzer - integrates three extraction methods
- Per-layer analysis (per_layer)
- Global per-type analysis (global_per_type) 
- True global analysis (global)

Supports new file path structure: model_name/prompt_name/activation_means_file.json

Usage examples:
1. List all available prompt names:
   python batch_top_n_overlap_unified.py --list_prompts

2. Process only specific prompts:
   python batch_top_n_overlap_unified.py --prompts "prompt1,prompt2"

3. Use wildcards to filter prompts:
   python batch_top_n_overlap_unified.py --prompts "test_*,eval_*"

4. Combine model and prompt filtering:
   python batch_top_n_overlap_unified.py --models "qwen3_*" --prompts "test_prompt"

5. Multi-process parallel processing (use all CPU cores):
   python batch_top_n_overlap_unified.py --num_processes 0

6. Specify number of processes for parallel processing:
   python batch_top_n_overlap_unified.py --num_processes 4

7. Serial processing (debug mode):
   python batch_top_n_overlap_unified.py --num_processes 1
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import datetime
import numpy as np
import pandas as pd
import fnmatch
from collections import defaultdict
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


ACT_TYPES = ['fwd_up', 'fwd_down', 'attn_q', 'attn_k', 'attn_v', 'attn_o']

def detect_model_type_from_path(file_path: str) -> str:
    path_lower = file_path.lower()
    if "mixtral" in path_lower:
        return "mixtral"
    elif "qwen3" in path_lower and ("moe" in path_lower or "a3b" in path_lower):
        return "qwen3moe"
    elif "qwen3" in path_lower:
        return "qwen3"
    elif "qwen" in path_lower:
        return "qwen2"
    elif "llama" in path_lower:
        return "llama"
    elif "gemma" in path_lower:
        return "gemma"
    else:
        return "other"

def is_moe_model(file_path: str) -> bool:
    """Determine if it's a MoE model"""
    model_type = detect_model_type_from_path(file_path)
    return model_type in ["mixtral", "qwen3moe"]

def _normalize_matrix(data: Dict[str, Any], file_path: str = None) -> Dict[str, Dict[str, List[float]]]:
   
    if isinstance(data, dict):
        if 'activation_means' in data and isinstance(data['activation_means'], dict):
            raw_data = data['activation_means']
        elif 'differences' in data and isinstance(data['differences'], dict):
            raw_data = data['differences']
        else:
            
            has_any = any(k in data for k in ACT_TYPES)
            if has_any:
                raw_data = {k: v for k, v in data.items() if k in ACT_TYPES and isinstance(v, dict)}
            else:
                return {}
    else:
        return {}
    
 
    normalized = {}
    for act_type, layers in raw_data.items():
        if act_type not in ACT_TYPES or not isinstance(layers, dict):
            continue
            
        normalized[act_type] = {}
        
      
        is_moe_from_path = file_path and is_moe_model(file_path)
        is_moe_from_format = any("expert" in str(layer_name) for layer_name in layers.keys())
        
        
        if ((is_moe_from_path or is_moe_from_format) and 
            act_type in ["fwd_up", "fwd_down"]):
            
            for layer_expert, values in layers.items():
                if isinstance(values, list):
                    normalized[act_type][layer_expert] = values
            
           
            if file_path and act_type == 'fwd_up':
                expert_count = sum(1 for k in layers.keys() if "expert" in str(k))
               
        else:
           
            for layer, values in layers.items():
                if isinstance(values, list):
                    normalized[act_type][layer] = values
    
    return normalized

def _is_valid_number(x) -> bool:
    """Check if it's a valid number"""
    try:
        v = float(x)
        return np.isfinite(v)
    except Exception:
        return False

class UnifiedTopNOverlapAnalyzer:
    """Unified Top-N% neuron overlap analyzer (integrates three extraction methods)"""
    
    def __init__(self, file_paths: List[str], 
                 enabled_modules: List[str] = None,
                 per_layer_modules: List[str] = None,
                 global_per_type_modules: List[str] = None,
                 global_modules: List[str] = None):
        self.file_paths = file_paths
        self.file_data: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Any] = {}
        
        # Module configuration
        self.enabled_modules = enabled_modules or ACT_TYPES
        self.per_layer_modules = per_layer_modules or ACT_TYPES
        self.global_per_type_modules = global_per_type_modules or ACT_TYPES
        self.global_modules = global_modules or ACT_TYPES
        
        print(f"🔧 Module configuration:")
        print(f"   📊 Enabled modules: {self.enabled_modules}")
        print(f"   📋 Per-layer analysis modules: {self.per_layer_modules}")
        print(f"   🌐 Global per-type analysis modules: {self.global_per_type_modules}")
        print(f"   🎯 True global analysis modules: {self.global_modules}")
    
    def load_input_files(self) -> bool:
        """Load all input files"""
        try:
            moe_files = 0
            for i, file_path in enumerate(self.file_paths):
                model_type = detect_model_type_from_path(file_path)
                is_moe = is_moe_model(file_path)
                if is_moe:
                    moe_files += 1
                    print(f"📂 Loading file {i+1}: {os.path.basename(file_path)} [MoE model: {model_type}]")
                else:
                    print(f"📂 Loading file {i+1}: {os.path.basename(file_path)} [Regular model: {model_type}]")
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.file_data[file_path] = json.load(f)
            print(f"✅ Successfully loaded {len(self.file_paths)} files (including {moe_files} MoE model files)")
            return True
        except Exception as e:
            print(f"❌ File loading failed: {e}")
            return False
    
    def extract_top_n_neurons_per_layer(self, data: Dict, top_percentage: float, file_path: str = None) -> Dict[str, Dict[str, List[int]]]:
        """Per-layer analysis: take top-N% neurons separately for each layer"""
        top_neurons: Dict[str, Dict[str, List[int]]] = {}
        mat = _normalize_matrix(data, file_path)
        
        if not mat or not any(k in mat for k in ACT_TYPES):
            print(f"⚠️  No activation type fields found in data")
            return top_neurons
        
        # Initialize all modules (including disabled ones)
        for act_type in ACT_TYPES:
            top_neurons[act_type] = {}
            
        # Only process enabled modules
        for act_type, layers in mat.items():
            if act_type not in ACT_TYPES or act_type not in self.per_layer_modules:
                continue
            for layer, values in layers.items():
                if not isinstance(values, list):
                    continue
                indexed = [(i, abs(v)) for i, v in enumerate(values) if _is_valid_number(v)]
                indexed.sort(key=lambda x: x[1], reverse=True)
                select_count = max(1, int(len(indexed) * top_percentage))
                top_indices = [idx for idx, _ in indexed[:select_count]]
                top_neurons[act_type][layer] = top_indices
        return top_neurons
    
    def extract_top_n_neurons_global_per_type(self, data: Dict, top_percentage: float, file_path: str = None) -> Dict[str, Dict[str, List[int]]]:
      
        top_neurons: Dict[str, Dict[str, List[int]]] = {}
        mat = _normalize_matrix(data, file_path)
        
        if not mat or not any(k in mat for k in ACT_TYPES):
            return top_neurons
       
        for act_type in ACT_TYPES:
            top_neurons[act_type] = {}
        
      
        for act_type in ACT_TYPES:
            if act_type not in mat or act_type not in self.global_per_type_modules:
                continue
                
          
            all_global_diffs = []
            for layer, values in mat[act_type].items():
                if not isinstance(values, list):
                    continue
                for local_idx, value in enumerate(values):
                    if _is_valid_number(value):
                        all_global_diffs.append((layer, local_idx, abs(float(value))))
            
            if not all_global_diffs:
                continue
            
            
            all_global_diffs.sort(key=lambda x: x[2], reverse=True)
            select_count = max(1, int(len(all_global_diffs) * top_percentage))
            top_global_neurons = all_global_diffs[:select_count]
            
          
            top_neurons[act_type] = defaultdict(list)
            for layer, local_idx, _ in top_global_neurons:
                top_neurons[act_type][layer].append(local_idx)
            
          
            top_neurons[act_type] = dict(top_neurons[act_type])
        
        return top_neurons
    
    def extract_top_n_neurons_global(self, data: Dict, top_percentage: float, file_path: str = None) -> Dict[str, Dict[str, List[int]]]:
        """True global analysis: unified sorting across all activation types + all layers"""
        mat = _normalize_matrix(data, file_path)
        big_pool: List[Tuple[str, str, int, float]] = []
        
        # Only collect data from enabled modules
        for act_type in ACT_TYPES:
            if act_type not in mat or act_type not in self.global_modules:
                continue
            for layer, values in mat[act_type].items():
                if not isinstance(values, list):
                    continue
                for local_idx, value in enumerate(values):
                    if _is_valid_number(value):
                        big_pool.append((act_type, layer, local_idx, abs(float(value))))
        
        big_pool.sort(key=lambda x: x[3], reverse=True)
        K = max(1, math.ceil(len(big_pool) * top_percentage))
        
        # Initialize all modules (including disabled ones)
        out = {t: defaultdict(list) for t in ACT_TYPES}
        for act_type, layer, local_idx, _ in big_pool[:K]:
            out[act_type][layer].append(local_idx)
        out = {t: dict(d) for t, d in out.items()}
        
        return out
    
    def calculate_overlap_for_method(self, method: str, top_percentage: float) -> Dict[str, Any]:
        """Calculate overlap rate for specified method and top n% value"""
        print(f"🔍 Calculating {method} Top-{top_percentage:.8%} overlap rate...")
        
        # Select extraction function based on method
        if method == "per_layer":
            extract_func = self.extract_top_n_neurons_per_layer
        elif method == "global_per_type":
            extract_func = self.extract_top_n_neurons_global_per_type
        elif method == "global":
            extract_func = self.extract_top_n_neurons_global
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Extract top n% neurons from all files
        all_top = {fp: extract_func(self.file_data[fp], top_percentage, fp) for fp in self.file_paths}
        
        # Check if there's MoE format data
        moe_files = [fp for fp in self.file_paths if is_moe_model(fp)]
        if moe_files:
            print(f"   🔧 Detected {len(moe_files)} MoE files, will analyze in expert-separated format")
        
        total_overlap = 0
        total_union = 0
        total_single_file_neurons = 0
        overlap_details: Dict[str, Dict[str, Any]] = {}
        
        for act_type in ACT_TYPES:
            overlap_details[act_type] = {}
            
            # Only process enabled modules
            if act_type not in self.enabled_modules:
                continue
                
            # Collect all layers/experts (only calculate if all files have this layer)
            all_layers = set()
            for neurons in all_top.values():
                if act_type in neurons:
                    all_layers.update(neurons[act_type].keys())
            
            for layer in sorted(all_layers, key=lambda x: str(x)):
                if not all((act_type in all_top[fp] and layer in all_top[fp][act_type]) for fp in self.file_paths):
                    continue
                
                sets_per_file = [set(all_top[fp][act_type][layer]) for fp in self.file_paths]
                inter = set.intersection(*sets_per_file) if sets_per_file else set()
                uni = set.union(*sets_per_file) if sets_per_file else set()
                
                overlap_count = len(inter)
                union_count = len(uni)
                single_file_count = len(sets_per_file[0]) if sets_per_file else 0
                
                info = {
                    'overlap_count': overlap_count,
                    'total_count': union_count,
                    'overlap_rate': (overlap_count / union_count) if union_count > 0 else 0.0,
                    'intersection': sorted(list(inter)),
                    'file_count': len(self.file_paths),
                    'overlap_vs_single_rate': (overlap_count / single_file_count) if single_file_count > 0 else 0.0,
                    'single_file_neuron_count': single_file_count
                }
                
                # Unique to each file
                for i, _ in enumerate(self.file_paths):
                    mine = sets_per_file[i]
                    others = set.union(*[s for j, s in enumerate(sets_per_file) if j != i]) if len(sets_per_file) > 1 else set()
                    info[f'only_in_file{i+1}'] = sorted(list(mine - others))
                
                overlap_details[act_type][layer] = info
                total_overlap += overlap_count
                total_union += union_count
                total_single_file_neurons += single_file_count
        
        overall_overlap_rate = (total_overlap / total_union) if total_union > 0 else 0.0
        overall_overlap_vs_single_rate = (total_overlap / total_single_file_neurons) if total_single_file_neurons > 0 else 0.0
        
        result = {
            'method': method,
            'top_percentage': top_percentage,
            'overall_stats': {
                'total_overlap': total_overlap,
                'total_neurons': total_union,
                'overall_overlap_rate': overall_overlap_rate,
                'overlap_vs_single_rate': overall_overlap_vs_single_rate,
                'total_single_file_neurons': total_single_file_neurons,
                'file_count': len(self.file_paths),
            },
            'overlap_details': overlap_details,
            'summary': {
                'file_names': [os.path.basename(p) for p in self.file_paths],
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        print(f"   📊 Overall overlap rate: {overall_overlap_rate:.6%} ({total_overlap}/{total_union})")
        print(f"   📊 Overlapping neurons vs single file ratio: {overall_overlap_vs_single_rate:.6%} ({total_overlap}/{total_single_file_neurons})")
        return result
    
    def batch_analyze(self, top_percentages: List[float], methods: List[str] = None) -> Dict[str, Any]:
        """Batch analyze specified methods and top n% values"""
        if methods is None:
            methods = ["per_layer", "global_per_type", "global"]
        
        print(f"🚀 Starting unified batch analysis...")
        print(f"   📊 Analysis methods: {', '.join(methods)}")
        print(f"   📊 Top-N% values: {top_percentages}")
        
        if not self.load_input_files():
            return {}
        
        batch_results = {}
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"🔍 Analysis method: {method}")
            print(f"{'='*60}")
            
            method_results = {}
            for p in top_percentages:
                result = self.calculate_overlap_for_method(method, p)
                method_results[f"top_{p:.8f}"] = result
            
            batch_results[method] = method_results
        
        self.results = batch_results
        return batch_results
    
    def save_results(self, output_dir: str) -> str:
        """Save analysis results, create independent subdirectories for each method (do not save detailed result files)"""
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create independent subdirectories for each method (but do not save detailed result files)
        for method, method_results in self.results.items():
            method_dir = os.path.join(output_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            
            # Comment out saving detailed result files, only save overlapping neuron files
            # method_results_file = os.path.join(method_dir, f"{method}_top_n_overlap_{ts}.json")
            # with open(method_results_file, 'w', encoding='utf-8') as f:
            #     json.dump(method_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Result directory created: {output_dir}")
        print(f"   📁 Method subdirectories: {', '.join(self.results.keys())}")
        print(f"   ⚠️  Skipped saving detailed result files (only save overlapping neuron files)")
        return output_dir
    
    def save_overlap_neurons(self, output_dir: str, method: str, top_percentage: float) -> str:
        """Save overlapping neurons as standard format file to corresponding method's subdirectory"""
        if not self.results or method not in self.results:
            print(f"⚠️  No result data found for method {method}")
            return ""
        
        target = None
        for top_n_key, result in self.results[method].items():
            if abs(result['top_percentage'] - top_percentage) < 1e-12:
                target = result
                break
        
        if not target:
            print(f"⚠️  No result found for method {method} top {top_percentage:.8f}%")
            return ""
        
        overlap_details = target['overlap_details']
        overlap_neurons = {t: {} for t in ACT_TYPES}
        
        for act_type in ACT_TYPES:
            if act_type in overlap_details:
                for layer, d in overlap_details[act_type].items():
                    # Only save layers/experts with overlapping neurons, skip empty arrays
                    if d['overlap_count'] > 0:
                        overlap_neurons[act_type][layer] = d['intersection']
        
        # Create method subdirectory
        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(method_dir, f"overlap_neurons_{top_percentage:.8f}_{ts}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(overlap_neurons, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Method {method} overlapping neuron file saved: {method}/{os.path.basename(out_file)}")
        return out_file
    
    def print_summary_report(self):
        """Print summary report"""
        if not self.results:
            print("⚠️  No result data")
            return
        
        print("\n" + "="*80)
        print("📊 Unified Top-N% Neuron Overlap Analysis Summary Report")
        print("="*80)
        print(f"\n📁 Analysis files: {', '.join([os.path.basename(p) for p in self.file_paths])}")
        
        for method, method_results in self.results.items():
            print(f"\n🔍 Analysis method: {method}")
            print(f"{'Top-N%':<12} {'Total Overlap':<12} {'Total Neurons':<12} {'Overlap Rate':<10} {'Overlap vs Single':<15}")
            print("-" * 70)
            
            for top_n_key, result in method_results.items():
                p = result['top_percentage']
                st = result['overall_stats']
                print(f"{p:<12.8f} {st['total_overlap']:<12} {st['total_neurons']:<12} "
                      f"{st['overall_overlap_rate']:.6%} {st['overlap_vs_single_rate']:.6%}")
        
        print("\n💡 Notes:")
        print("   • per_layer: Take top-N% neurons separately for each layer")
        print("   • global_per_type: Global sort within each activation type, take top-N%")
        print("   • global: Unified sort across all activation types + all layers, take top-N%")
        print("   • Overlap rate = Intersection of all files / Union of all files")
        print("   • Overlap vs single file ratio = Intersection / Single file count")

# ========= Multi-process processing functions =========
def process_single_model(model_task: Dict[str, Any]) -> Dict[str, Any]:
    """Task function for processing a single model, used for multi-processing"""
    try:
        model_dir = Path(model_task['model_dir'])
        model_name = model_task['model_name']
        output_root = Path(model_task['output_root'])
        top_ps = model_task['top_ps']
        methods = model_task['methods']
        file_patterns = model_task['file_patterns']
        prompt_patterns = model_task['prompt_patterns']
        min_files = model_task['min_files']
        enabled_modules = model_task['enabled_modules']
        per_layer_modules = model_task['per_layer_modules']
        global_per_type_modules = model_task['global_per_type_modules']
        global_modules = model_task['global_modules']
        force_reprocess = model_task['force_reprocess']
        skip_existing = model_task['skip_existing']
        
        print(f"[Process {os.getpid()}] 🔄 Starting to process model: {model_name}")
        start_time = time.time()
        
        # Check output directory
        out_dir = output_root / model_name
        should_skip_existing = skip_existing and not force_reprocess
        
        if should_skip_existing and out_dir.exists() and out_dir.is_dir():
            print(f"[Process {os.getpid()}] ⏭️  Model {model_name} result directory already exists, skipping")
            return {
                "model": model_name, 
                "files": 0, 
                "prompts": 0, 
                "status": "skipped_existing",
                "duration": 0,
                "process_id": os.getpid()
            }
        elif force_reprocess and out_dir.exists() and out_dir.is_dir():
            print(f"[Process {os.getpid()}] 🔄 Force reprocessing model {model_name} (result directory exists but will be overwritten)")
        
        # Find files
        target_files, found_prompts = find_activation_means_files(model_dir, file_patterns, prompt_patterns)
        
        if found_prompts:
            print(f"[Process {os.getpid()}] 🎯 {model_name} found prompts: {', '.join(sorted(found_prompts))}")
        
        if len(target_files) < min_files:
            print(f"[Process {os.getpid()}] ⚠️  {model_name} only found {len(target_files)} matching files (threshold {min_files}), skipping.")
            return {
                "model": model_name, 
                "files": len(target_files), 
                "prompts": len(found_prompts), 
                "status": "skipped",
                "duration": 0,
                "process_id": os.getpid()
            }
        
        # Create output directory
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Execute analysis
        analyzer = UnifiedTopNOverlapAnalyzer(
            target_files,
            enabled_modules=enabled_modules,
            per_layer_modules=per_layer_modules,
            global_per_type_modules=global_per_type_modules,
            global_modules=global_modules
        )
        
        results = analyzer.batch_analyze(top_ps, methods)
        if not results:
            print(f"[Process {os.getpid()}] ❌ {model_name} analysis failed")
            return {
                "model": model_name, 
                "files": len(target_files), 
                "prompts": len(found_prompts), 
                "status": "failed",
                "duration": time.time() - start_time,
                "process_id": os.getpid()
            }
        
        # Save results
        analyzer.save_results(str(out_dir))
        
        # Export intersection files
        for method in methods:
            for p in top_ps:
                analyzer.save_overlap_neurons(str(out_dir), method, p)
        
        duration = time.time() - start_time
        
        
        return {
            "model": model_name, 
            "files": len(target_files), 
            "prompts": len(found_prompts), 
            "status": "done",
            "duration": duration,
            "process_id": os.getpid()
        }
        
    except Exception as e:
       
        return {
            "model": model_task.get('model_name', 'unknown'), 
            "files": 0, 
            "prompts": 0, 
            "status": "error",
            "error": str(e),
            "duration": 0,
            "process_id": os.getpid()
        }

# ========= Directory batch processing entry =========
def collect_models(input_root: Path) -> List[Path]:
    """Return list of model subdirectories under input_root"""
    return [p for p in sorted(input_root.iterdir()) if p.is_dir()]

def list_available_prompts(input_root: Path, file_patterns: List[str]) -> Dict[str, List[str]]:
    """List all available prompt names, grouped by model"""
    models = collect_models(input_root)
    prompt_info = {}
    
    for model_dir in models:
        model_name = model_dir.name
        prompts = set()
        
        for pat in file_patterns:
            for p in model_dir.rglob(pat):
                if p.is_file():
                    # Extract prompt name from file path
                    relative_path = p.relative_to(model_dir)
                    path_parts = relative_path.parts
                    
                    # Find prompt name (usually in second level directory)
                    for part in path_parts:
                        if part not in ['', '.'] and not part.endswith('.json'):
                            prompts.add(part)
                            break
        
        if prompts:
            prompt_info[model_name] = sorted(list(prompts))
    
    return prompt_info

def find_activation_means_files(model_dir: Path, patterns: List[str], prompt_patterns: List[str] = None) -> Tuple[List[str], List[str]]:
    """Recursively find matching files under model directory, supports prompt name filtering
    
    Returns:
        Tuple[List[str], List[str]]: (found file list, found prompt name list)
    """
    found: List[str] = []
    found_prompts: List[str] = []
    seen = set()
    seen_prompts = set()
    
    for pat in patterns:
        for p in model_dir.rglob(pat):
            if p.is_file() and str(p) not in seen:
                # Extract prompt name from file path (assumes structure: model_name/prompt_name/activation_means_file.json)
                relative_path = p.relative_to(model_dir)
                path_parts = relative_path.parts
                
                # Find prompt name (usually in second level directory)
                prompt_name = None
                for part in path_parts:
                    if part not in ['', '.'] and not part.endswith('.json'):
                        prompt_name = part
                        break
                
                # If prompt filtering is specified, check if it matches
                if prompt_patterns:
                    if not prompt_name or not any(fnmatch.fnmatch(prompt_name, prompt_pattern) for prompt_pattern in prompt_patterns):
                        continue
                
                seen.add(str(p))
                found.append(str(p))
                
                # Record found prompt name
                if prompt_name and prompt_name not in seen_prompts:
                    seen_prompts.add(prompt_name)
                    found_prompts.append(prompt_name)
    
    return found, found_prompts

def main():
    parser = argparse.ArgumentParser(
        description="Unified neuron overlap analyzer (integrates three extraction methods)"
    )
    parser.add_argument("--input_root", "-i",
                        default="",
                        help="Input root directory (contains model subdirectories)")
    parser.add_argument("--output_root", "-o",
                        default="",
                        help="Output root directory (each model's results saved in subdirectories under this directory)")
    parser.add_argument("--top_percentages", "-p",
                        default="",
                        help="Comma-separated Top-N percentage list")
    parser.add_argument("--file_globs",
                        default="*activation_means*.json",
                        help="Comma-separated file matching patterns")
    parser.add_argument("--min_files", type=int, default=1,
                        help="Minimum number of matching files required per model to execute analysis")
    parser.add_argument("--models", "-m", default="",
                        help="Only process these subdirectories (comma-separated; supports wildcards, e.g.: llama_8b,qwen3_*; empty=all)")
    parser.add_argument("--prompts", default="",
                        help="Only process these prompt names (comma-separated; supports wildcards, e.g.: prompt1,prompt2_*; empty=all)")
    parser.add_argument("--list_prompts", action="store_true",
                        help="List all available prompt names and exit")
    parser.add_argument("--methods", default="per_layer",
                        help="Specify analysis methods to run (comma-separated, options: per_layer,global_per_type,global; default=all)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip models with existing result directories (enabled by default)")
    parser.add_argument("--force_reprocess", action="store_true",
                        help="Force reprocess all models, even if result directories exist")
    
    # Multi-process configuration parameters
    parser.add_argument("--num_processes", "-j", type=int, default=0,
                        help="Number of processes for parallel processing (0=auto-detect CPU cores, 1=serial processing)")
    parser.add_argument("--max_workers", type=int, default=0,
                        help="Maximum worker processes (0=auto-detect, recommended not to exceed CPU core count)")
    
    # Module configuration parameters
    parser.add_argument("--enabled_modules", default="fwd_up,fwd_down,attn_q,attn_k,attn_v,attn_o",
                        help="List of enabled modules (comma-separated, default=all enabled)")
    parser.add_argument("--per_layer_modules", default="fwd_up,fwd_down,attn_q,attn_k,attn_v,attn_o",
                        help="List of enabled modules for per-layer analysis (comma-separated, default=all enabled)")
    parser.add_argument("--global_per_type_modules", default="fwd_up,fwd_down,attn_q,attn_k,attn_v,attn_o",
                        help="List of enabled modules for global per-type analysis (comma-separated, default=all enabled)")
    parser.add_argument("--global_modules", default="fwd_up,fwd_down,attn_q,attn_k,attn_v,attn_o",
                        help="List of enabled modules for true global analysis (comma-separated, default=all enabled)")
    
    args = parser.parse_args()
    
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.is_dir():
        raise SystemExit(f"❌ Input root directory does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Parse top percentages
    try:
        top_ps = sorted(float(x.strip()) for x in args.top_percentages.split(",") if x.strip())
    except Exception:
        raise SystemExit("❌ --top_percentages format error, should be comma-separated floating point numbers")
    
    # Parse module configuration
    try:
        enabled_modules = [x.strip() for x in args.enabled_modules.split(",") if x.strip()]
        per_layer_modules = [x.strip() for x in args.per_layer_modules.split(",") if x.strip()]
        global_per_type_modules = [x.strip() for x in args.global_per_type_modules.split(",") if x.strip()]
        global_modules = [x.strip() for x in args.global_modules.split(",") if x.strip()]
        
        # Validate module names
        valid_modules = set(ACT_TYPES)
        for modules, name in [(enabled_modules, "enabled_modules"), 
                             (per_layer_modules, "per_layer_modules"),
                             (global_per_type_modules, "global_per_type_modules"),
                             (global_modules, "global_modules")]:
            invalid = set(modules) - valid_modules
            if invalid:
                raise SystemExit(f"❌ {name} contains invalid modules: {invalid}. Valid modules: {ACT_TYPES}")
                
    except Exception as e:
        raise SystemExit(f"❌ Module configuration parsing error: {e}")
    
    # Parse analysis methods
    try:
        methods = [x.strip() for x in args.methods.split(",") if x.strip()]
        valid_methods = {"per_layer", "global_per_type", "global"}
        invalid_methods = set(methods) - valid_methods
        if invalid_methods:
            raise SystemExit(f"❌ Invalid analysis methods: {invalid_methods}. Valid methods: {valid_methods}")
    except Exception as e:
        raise SystemExit(f"❌ Analysis method parsing error: {e}")
    
    # Collect all first-level subdirectories
    models = collect_models(input_root)
    if not models:
        raise SystemExit(f"❌ No model subdirectories found under {input_root}")
    
    # Filter: --models supports wildcards
    if args.models and args.models.strip():
        patterns = [s.strip() for s in args.models.split(",") if s.strip()]
        selected = [mdir for mdir in models if any(fnmatch.fnmatch(mdir.name, pat) for pat in patterns)]
        if not selected:
            pats = ", ".join(patterns)
            raise SystemExit(f"❌ No subdirectories found matching patterns ({pats}). Available subdirectories: {', '.join(p.name for p in models)}")
        models = selected
    
    # Parse file matching patterns
    file_patterns = [p.strip() for p in args.file_globs.split(",") if p.strip()]
    
    # If just listing prompts, execute and exit
    if args.list_prompts:
        print("🔍 Scanning all available prompt names...")
        prompt_info = list_available_prompts(input_root, file_patterns)
        
        if not prompt_info:
            print("❌ No prompts found")
            return
        
        print("\n📋 All available prompt names:")
        print("=" * 60)
        for model_name, prompts in prompt_info.items():
            print(f"\n🏷️  Model: {model_name}")
            print(f"   Prompt count: {len(prompts)}")
            print(f"   Prompt list: {', '.join(prompts)}")
        
        # Count all unique prompts
        all_prompts = set()
        for prompts in prompt_info.values():
            all_prompts.update(prompts)
        
        print(f"\n📊 Statistics:")
        print(f"   Total models: {len(prompt_info)}")
        print(f"   Unique prompt count: {len(all_prompts)}")
        print(f"   All unique prompts: {', '.join(sorted(all_prompts))}")
        return
    
    # Parse prompt filtering patterns
    prompt_patterns = None
    if args.prompts and args.prompts.strip():
        prompt_patterns = [p.strip() for p in args.prompts.split(",") if p.strip()]
    
    print(f"📁 Input root directory: {input_root}")
    print(f"📦 Output root directory: {output_root}")
    print(f"🔢 Top-N% list: {top_ps}")
    print(f"🔎 File matching: **/{{{', '.join(file_patterns)}}}")
    if prompt_patterns:
        print(f"🎯 Prompt filtering: {', '.join(prompt_patterns)}")
    else:
        print(f"🎯 Prompt filtering: None (process all prompts)")
    print(f"🔧 Analysis methods: {', '.join(methods)}")
    print(f"⏭️  Skip existing directories: {'Yes' if args.skip_existing and not args.force_reprocess else 'No'}")
    if args.force_reprocess:
        print(f"🔄 Force reprocess: Yes")
    print(f"📚 Planned model directories to process: {len(models)}  -> {', '.join(m.name for m in models)}\n")
    
    # Determine parallel processing configuration
    cpu_count = mp.cpu_count()
    if args.num_processes == 0:
        num_processes = cpu_count
    elif args.num_processes == 1:
        num_processes = 1  # Serial processing
    else:
        num_processes = min(args.num_processes, cpu_count)
    
    if args.max_workers == 0:
        max_workers = num_processes
    else:
        max_workers = min(args.max_workers, num_processes)
    
    print(f"🔄 Parallel configuration:")
    print(f"   💻 CPU core count: {cpu_count}")
    print(f"   🚀 Parallel processes: {num_processes}")
    print(f"   👥 Maximum worker processes: {max_workers}")
    
    summary_rows = []
    
    # Prepare task list
    model_tasks = []
    for mdir in models:
        task = {
            'model_dir': str(mdir),
            'model_name': mdir.name,
            'output_root': str(output_root),
            'top_ps': top_ps,
            'methods': methods,
            'file_patterns': file_patterns,
            'prompt_patterns': prompt_patterns,
            'min_files': args.min_files,
            'enabled_modules': enabled_modules,
            'per_layer_modules': per_layer_modules,
            'global_per_type_modules': global_per_type_modules,
            'global_modules': global_modules,
            'force_reprocess': args.force_reprocess,
            'skip_existing': args.skip_existing
        }
        model_tasks.append(task)
    
    start_time = time.time()
    
    if num_processes == 1:
        # Serial processing (maintain original logic)
        print(f"\n🔄 Using serial processing mode...")
        for task in model_tasks:
            result = process_single_model(task)
            summary_rows.append(result)
    else:
        # Multi-process processing
        print(f"\n🚀 Using multi-process processing mode...")
        completed_count = 0
        total_tasks = len(model_tasks)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_single_model, task): task for task in model_tasks}
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    summary_rows.append(result)
                    completed_count += 1
                    
                    # Display progress
                    progress = (completed_count / total_tasks) * 100
                    elapsed_time = time.time() - start_time
                    avg_time_per_task = elapsed_time / completed_count
                    remaining_tasks = total_tasks - completed_count
                    estimated_remaining_time = avg_time_per_task * remaining_tasks
                    
                    print(f"📊 Progress: {completed_count}/{total_tasks} ({progress:.1f}%) | "
                          f"Elapsed: {elapsed_time:.1f}s | "
                          f"Estimated remaining: {estimated_remaining_time:.1f}s | "
                          f"Model: {result['model']} [{result['status']}]")
                    
                except Exception as e:
                    task = future_to_task[future]
                    print(f"❌ Error occurred while processing task: {task['model_name']} - {e}")
                    summary_rows.append({
                        "model": task['model_name'], 
                        "files": 0, 
                        "prompts": 0, 
                        "status": "error",
                        "error": str(e),
                        "duration": 0,
                        "process_id": -1
                    })
    
    total_time = time.time() - start_time
    print(f"\n✅ All complete! Total time: {total_time:.2f} seconds")
    
    # Print summary statistics
    print(f"\n📊 Processing results summary:")
    print(f"{'Model':<20} {'Files':<8} {'Prompts':<10} {'Status':<15} {'Time(s)':<10} {'Process ID':<8}")
    print("-" * 80)
    
    status_counts = {}
    total_files = 0
    total_prompts = 0
    
    for result in summary_rows:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1
        total_files += result.get('files', 0)
        total_prompts += result.get('prompts', 0)
        
        print(f"{result['model']:<20} {result.get('files', 0):<8} {result.get('prompts', 0):<10} "
              f"{status:<15} {result.get('duration', 0):<10.2f} {result.get('process_id', -1):<8}")
    
    print("-" * 80)
    print(f"Total: {len(summary_rows)} models")
    print(f"Total files: {total_files}")
    print(f"Total prompts: {total_prompts}")
    print(f"Status statistics: {status_counts}")
    
    if num_processes > 1:
        avg_time_per_task = total_time / len(summary_rows)
        speedup_ratio = (total_time * num_processes) / total_time if total_time > 0 else 1
        print(f"Average time per task: {avg_time_per_task:.2f} seconds")
        print(f"Theoretical speedup ratio: {speedup_ratio:.2f}x")

if __name__ == "__main__":
    main()
