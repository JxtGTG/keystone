import torch
import numpy as np
from transformers import (
    LlamaForCausalLM, LlamaConfig, Gemma2ForCausalLM, 
    AutoConfig, AutoModelForCausalLM, AutoTokenizer
)

try:
    from transformers import Gemma3ForConditionalGeneration
    GEMMA3_AVAILABLE = True
except Exception:
    GEMMA3_AVAILABLE = False


try:
    from transformers import Gemma3ForCausalLM
    GEMMA3_TEXT_AVAILABLE = True
except Exception:
    GEMMA3_TEXT_AVAILABLE = False
from typing import Dict, List
import json
from collections import defaultdict
import argparse
import random
import math
import os

import shutil
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Try importing MoE models
try:
    from transformers import Qwen2MoeForCausalLM, MixtralForCausalLM
    # Try importing Qwen3Moe model (if available)
    try:
        from transformers import Qwen3MoeForCausalLM
        QWEN3MOE_AVAILABLE = True
    except ImportError:
        QWEN3MOE_AVAILABLE = False
    MOE_AVAILABLE = True
except ImportError:
    MOE_AVAILABLE = False
    QWEN3MOE_AVAILABLE = False
    print("⚠️ MoE model import failed, will use AutoModelForCausalLM as fallback")

def detect_model_type(model_path: str) -> str:
    """Detect model type"""
    path_lower = model_path.lower()
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

def is_moe_model(model_path: str) -> bool:
   
    model_type = detect_model_type(model_path)
    return model_type in ["mixtral", "qwen3moe"]

def zero_out_indices(weight_tensor, dim, indices_to_zero, factor, head_dim=128):
    """
    Zero out neurons at specified indices
    
    For K/V projections (factor > 1), use correct GQA mapping logic:
        q_head_id = q_idx // head_dim
        kv_head_id = q_head_id // kv_repeat
        offset_in_head = q_idx % head_dim
        k_idx = kv_head_id * head_dim + offset_in_head
    """
    weight_tensor = weight_tensor.clone()
    
    # For K/V (factor > 1), need correct mapping
    if factor > 1:
        mapped_indices = set()
        kv_repeat = int(factor)
        for idx in indices_to_zero:
            q_head_id = idx // head_dim
            kv_head_id = q_head_id // kv_repeat
            offset_in_head = idx % head_dim
            k_idx = kv_head_id * head_dim + offset_in_head
            mapped_indices.add(k_idx)
        indices_to_zero = sorted(mapped_indices)
    
    if dim == 0:
        weight_tensor[indices_to_zero, :] = 0
    elif dim == 1:
        weight_tensor[:, indices_to_zero] = 0
    return weight_tensor

def apply_zero_mask_to_model(model, deactivate_indices_dict, head_dim=128):
    state_dict = model.state_dict()
    new_state_dict = {}
    for name, param in state_dict.items():
        if name in deactivate_indices_dict:
            info = deactivate_indices_dict[name]
            
            param = zero_out_indices(param, info["dim"], info["indices"], info["factor"], head_dim)
        new_state_dict[name] = param
    model.load_state_dict(new_state_dict)

def build_deactivate_indices_dict(
    model_path: str,
    activate_keys_q_set: Dict[int, List[int]],
    activate_keys_k_set: Dict[int, List[int]],
    activate_keys_v_set: Dict[int, List[int]],
    activate_keys_o_set: Dict[int, List[int]],
    activate_keys_fwd_up_set: Dict[int, List[int]],
    activate_keys_fwd_down_set: Dict[int, List[int]],
    total_layers: int,
    intermediate_size: int,
    hidden_size: int,
):
    deactivate_dict = {}
    model_config = AutoConfig.from_pretrained(model_path)
    if hasattr(model_config, "text_config"):
        model_config = model_config.text_config
    kv_factor = model_config.num_attention_heads / model_config.num_key_value_heads
    print(f"📊 KV factor: {kv_factor}")
    
    is_moe = is_moe_model(model_path)
    print(f"🔧 Model type: {'MoE' if is_moe else 'Regular'}")

    for idx in range(total_layers):
        layer_idx = str(idx)

        # Process attention layers
        if layer_idx in activate_keys_q_set:
            deactivate_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = {
                "dim": 0,
                "indices": list(set(activate_keys_q_set[layer_idx])),
                "factor": 1
            }
        if layer_idx in activate_keys_k_set:
            deactivate_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = {
                "dim": 0,
                "indices": list(set(activate_keys_k_set[layer_idx])),
                "factor": kv_factor
            }
        if layer_idx in activate_keys_v_set:
            deactivate_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = {
                "dim": 0,
                "indices": list(set(activate_keys_v_set[layer_idx])),
                "factor": kv_factor
            }
        if layer_idx in activate_keys_o_set:
            deactivate_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = {
                "dim": 1,
                "indices": list(set(activate_keys_o_set[layer_idx])),
                "factor": 1
            }

        # Process FFN layers
        if is_moe:
            # MoE model: process expert-separated format
            _process_moe_ffn_layer(
                deactivate_dict, layer_idx, 
                activate_keys_fwd_up_set, activate_keys_fwd_down_set,
                model_config
            )
        else:
            # Regular model: process standard FFN format
            if layer_idx in activate_keys_fwd_up_set:
                deactivate_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = {
                    "dim": 0,
                    "indices": list(set(activate_keys_fwd_up_set[layer_idx])),
                    "factor": 1
                }
            if layer_idx in activate_keys_fwd_down_set:
                deactivate_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = {
                    "dim": 1,
                    "indices": list(set(activate_keys_fwd_down_set[layer_idx])),
                    "factor": 1
                }

    return deactivate_dict

def _process_moe_ffn_layer(
    deactivate_dict: dict, 
    layer_idx: str,
    activate_keys_fwd_up_set: Dict[str, List[int]],
    activate_keys_fwd_down_set: Dict[str, List[int]],
    model_config
):

    for key, indices in activate_keys_fwd_up_set.items():
        if key.startswith(f"{layer_idx}_expert_"):
            expert_id = key.split("_expert_")[1]
       
            weight_name = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.w1.weight"
            deactivate_dict[weight_name] = {
                "dim": 0,
                "indices": list(set(indices)),
                "factor": 1
            }
    

    for key, indices in activate_keys_fwd_down_set.items():
        if key.startswith(f"{layer_idx}_expert_"):
            expert_id = key.split("_expert_")[1]
         
            weight_name = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.w2.weight"
            deactivate_dict[weight_name] = {
                "dim": 1,
                "indices": list(set(indices)),
                "factor": 1
            }


def deactivation_params_saving(model_path, detected_neuron, save_path, processor_source: str = None):
    """
    Perform model deactivation based on detected neurons, and ensure tokenizer and processor are saved together.
    - Priority: save AutoProcessor (when online and have permissions)
    - If fails, copy preprocessor_config.json / processor_config.json from local source directory
    - If fails again, try to download only processor config from HF snapshot_download
    - If still fails, write minimal usable Gemma3 processor config (to ensure vLLM doesn't crash due to missing files)
    """
    import json, os, shutil, tempfile
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    # Optional: MoE etc. imports omitted (consistent with existing code)

    def _is_dir(p): 
        return isinstance(p, str) and os.path.isdir(p)

    def _write_minimal_gemma3_processor(dst_dir: str):
        preproc = {
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_processor_type": "Gemma3ImageProcessor",
            "image_seq_length": 256,
            "image_std": [0.5, 0.5, 0.5],
            "resample": 2,
            "rescale_factor": 0.00392156862745098,
            "size": {"height": 896, "width": 896},
            "processor_class": "Gemma3Processor"
        }
        proc = {
            "image_seq_length": 256,
            "processor_class": "Gemma3Processor"
        }
        with open(os.path.join(dst_dir, "preprocessor_config.json"), "w", encoding="utf-8") as f:
            json.dump(preproc, f, indent=2, ensure_ascii=False)
        with open(os.path.join(dst_dir, "processor_config.json"), "w", encoding="utf-8") as f:
            json.dump(proc, f, indent=2, ensure_ascii=False)
        print("✅ Written minimal usable Gemma3 processor config (preprocessor_config.json / processor_config.json)")

    def _copy_processor_files(src_dir: str, dst_dir: str) -> bool:
        ok = False
        for fn in ("preprocessor_config.json", "processor_config.json"):
            src = os.path.join(src_dir, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_dir, fn))
                print(f"✅ Copied {fn} from local")
                ok = True or ok
        return ok

    def _extract_repo_id_from_path(path: str) -> Optional[str]:
        """Extract repo_id from local snapshot path (e.g., models--google--gemma-3-1b-it -> google/gemma-3-1b-it)"""
        if not _is_dir(path):
            return None
        # Try to extract from path: models--namespace--repo_name
        parts = path.split(os.sep)
        for i, part in enumerate(parts):
            if part.startswith("models--") and "--" in part:
                # Extract models--namespace--repo_name part
                model_part = part
                # Convert to namespace/repo_name format
                repo_id = model_part.replace("models--", "").replace("--", "/")
                # Validate format: should be namespace/repo_name
                if "/" in repo_id and not repo_id.startswith("/") and not repo_id.endswith("/"):
                    return repo_id
        return None

    def _is_valid_repo_id(repo_id: str) -> bool:
        """Check if it's a valid HF repo_id format (namespace/repo_name or repo_name)"""
        if not repo_id or "/" in repo_id and (repo_id.startswith("/") or repo_id.endswith("/")):
            return False
        # Cannot be absolute path
        if os.path.isabs(repo_id):
            return False
        # Cannot contain special path characters
        if os.sep in repo_id or "\\" in repo_id:
            return False
        return True

    def _save_or_copy_processor(model_id_or_dir: str, dst_dir: str) -> None:
        # If already exists in directory, don't repeat
        has_pre = os.path.exists(os.path.join(dst_dir, "preprocessor_config.json"))
        has_proc = os.path.exists(os.path.join(dst_dir, "processor_config.json"))
        if has_pre and has_proc:
            print("ℹ️ Processor config already exists, skipping save.")
            return

        # 1) Priority: AutoProcessor
        try:
            from transformers import AutoProcessor
            print("🧩 Saving AutoProcessor...")
            processor = AutoProcessor.from_pretrained(model_id_or_dir, trust_remote_code=True)
            processor.save_pretrained(dst_dir)
            print("✅ Processor saved")
            return
        except Exception as e:
            print(f"⚠️ AutoProcessor save failed: {e}")

        # 2) Next: Copy from local source directory
        #    - If processor_source not explicitly provided and model_path itself is a directory, use model_path as source
        src_dir = processor_source if _is_dir(processor_source) else (model_id_or_dir if _is_dir(model_id_or_dir) else None)
        if src_dir and _copy_processor_files(src_dir, dst_dir):
            return

        # 3) Next: Download only processor config from HF (requires huggingface_hub installed and online+permissions)
        #    Only try if model_id_or_dir is a valid repo_id format
        repo_id_to_try = None
        if _is_valid_repo_id(model_id_or_dir):
            repo_id_to_try = model_id_or_dir
        elif _is_dir(model_id_or_dir):
            # Try to extract repo_id from local path
            repo_id_to_try = _extract_repo_id_from_path(model_id_or_dir)
        
        if repo_id_to_try:
            try:
                from huggingface_hub import snapshot_download
                print(f"🌐 Attempting to download only processor config from Hugging Face (repo_id: {repo_id_to_try})...")
                tmpd = tempfile.mkdtemp()
                snapshot_download(
                    repo_id=repo_id_to_try,
                    allow_patterns=["preprocessor_config.json", "processor_config.json"],
                    local_dir=tmpd,
                    local_dir_use_symlinks=False
                )
                if _copy_processor_files(tmpd, dst_dir):
                    print("✅ Processor config downloaded from HF")
                    shutil.rmtree(tmpd, ignore_errors=True)
                    return
                shutil.rmtree(tmpd, ignore_errors=True)
            except Exception as e:
                print(f"⚠️ Failed to download processor config from HF: {e}")
        else:
            print("ℹ️ Cannot extract valid repo_id from path, skipping HF download step")

        # 4) Fallback: Gemma3 minimal usable config
        print("🛟 Using Gemma3 minimal usable processor config as fallback")
        _write_minimal_gemma3_processor(dst_dir)

    print(f"🚀 Starting to process model: {model_path}")
    print("📥 Loading original model...")
    model_type = detect_model_type(model_path)  # Your existing function
    is_moe = is_moe_model(model_path)          # Your existing function

    # === Precisely select model class based on config to avoid weight reinitialization ===
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg_model_type = getattr(cfg, "model_type", None)

    print(f"🔍 Detected config.model_type: {cfg_model_type}")

    if cfg_model_type == "gemma3_text":
        # Text version Gemma3 (e.g., gemma-3-1b-pt) should use Gemma3ForCausalLM or AutoModelForCausalLM
        if GEMMA3_TEXT_AVAILABLE:
            model = Gemma3ForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    elif cfg_model_type == "gemma3" and GEMMA3_AVAILABLE:
        # Multimodal Gemma3 uses ConditionalGeneration class
        model = Gemma3ForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True)
    elif is_moe and MOE_AVAILABLE:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    config = model.config
    print(f"✅ Model loaded successfully: {getattr(config, 'model_type', 'unknown')} {'[MoE]' if is_moe else ''}")

    # === Parse neurons & build deactivate dictionary (maintain your existing implementation) ===
    print("📊 Parsing neuron data...")
    activate_keys_q_set = detected_neuron.get('attn_q', {})
    activate_keys_k_set = detected_neuron.get('attn_k', {})
    activate_keys_v_set = detected_neuron.get('attn_v', {})
    activate_keys_o_set = detected_neuron.get('attn_o', {})
    activate_keys_fwd_up_set = detected_neuron.get('fwd_up', {})
    activate_keys_fwd_down_set = detected_neuron.get('fwd_down', {})

    total_neurons = sum(len(v) for d in [activate_keys_q_set, activate_keys_k_set, activate_keys_v_set,
                                         activate_keys_o_set, activate_keys_fwd_up_set, activate_keys_fwd_down_set]
                        for v in d.values())
    print(f"🧠 Total neurons detected: {total_neurons}")

    if hasattr(config, 'num_hidden_layers'):
        total_layers = config.num_hidden_layers
        inter_size = config.intermediate_size
        hidden_size = config.hidden_size
    elif hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
        total_layers = config.text_config.num_hidden_layers
        inter_size = config.text_config.intermediate_size
        hidden_size = config.text_config.hidden_size
    else:
        raise ValueError(f"Cannot get layer count information from config: {type(config)}")

    # Get head_dim for correct K/V mapping
    model_config_for_head = config.text_config if hasattr(config, 'text_config') else config
    num_attention_heads = getattr(model_config_for_head, 'num_attention_heads', 32)
    head_dim = hidden_size // num_attention_heads
    print(f"📐 Head dimension: {head_dim}")

    print("🔧 Building deactivation configuration...")
    deact_dict = build_deactivate_indices_dict(
        model_path,
        activate_keys_q_set, activate_keys_k_set, activate_keys_v_set, activate_keys_o_set,
        activate_keys_fwd_up_set, activate_keys_fwd_down_set,
        total_layers=total_layers, intermediate_size=inter_size, hidden_size=hidden_size
    )
    print(f"📋 Number of weight entries to process: {len(deact_dict)}")

    print("⚡ Applying neuron deactivation...")
    apply_zero_mask_to_model(model, deact_dict, head_dim)

    print(f"💾 Saving modified model to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print("✅ Model saved")

    # Ensure exported config.json maintains consistent sliding_window with source model (if source has this key)
    try:
        dst_cfg_path = os.path.join(save_path, "config.json")
        src_cfg_path = os.path.join(model_path, "config.json")
        if os.path.exists(dst_cfg_path):
            with open(dst_cfg_path, "r", encoding="utf-8") as f:
                dst_cfg = json.load(f)
            src_sw = None
            # Read sliding_window from source model config (supports flat or in text_config)
            if os.path.exists(src_cfg_path):
                with open(src_cfg_path, "r", encoding="utf-8") as f:
                    src_cfg = json.load(f)
                if isinstance(src_cfg, dict):
                    if "sliding_window" in src_cfg:
                        src_sw = src_cfg.get("sliding_window")
                    elif isinstance(src_cfg.get("text_config"), dict) and "sliding_window" in src_cfg["text_config"]:
                        src_sw = src_cfg["text_config"].get("sliding_window")
            # Only write to export directory if source has this key
            if src_sw is not None and "sliding_window" not in dst_cfg:
                dst_cfg["sliding_window"] = src_sw
                with open(dst_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(dst_cfg, f, indent=2, ensure_ascii=False)
                print(f"🛠️ Synced sliding_window from source config: {src_sw}")
    except Exception as e:
        print(f"⚠️ Failed to sync sliding_window: {e}")

    print("📝 Saving tokenizer...")
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tok.save_pretrained(save_path)
        print("✅ Tokenizer saved")
    except Exception as e:
        print(f"⚠️ Failed to save tokenizer: {e}")

    # Key: Whether online or not, ensure processor config is finally written to directory
    print("🧩 Preparing to save/copy Processor config...")
    _save_or_copy_processor(model_path, save_path)

    # Validation
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json",
               "preprocessor_config.json", "processor_config.json"):
        path = os.path.join(save_path, fn)
        print(("✅ Verified exists: " if os.path.exists(path) else "⚠️  Not found: ") + fn)

    print(f"🎉 Deactivation complete! Model saved at: {save_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Neuron Deactivation Tool')
    parser.add_argument("--model_path", "-m", type=str, default="",
                       help="Original model path")
    parser.add_argument("--neurons_file", "-n", type=str, default="",
                       help="Neuron JSON file path")
    parser.add_argument("--output_dir", "-o", type=str, default="",
                       help="Output directory path")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Custom output model name (optional)")

    args = parser.parse_args()

 
    
    if not os.path.exists(args.neurons_file):
        print(f"❌ Neuron file does not exist: {args.neurons_file}")
        return

    # Load neuron data
    print(f"📂 Loading neuron file: {args.neurons_file}")
    try:
        # Try multiple encoding methods to read file
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        detected_neurons = None
        
        for encoding in encodings:
            try:
                with open(args.neurons_file, "r", encoding=encoding) as f:
                    detected_neurons = json.load(f)
                print(f"✅ Neuron file loaded successfully (encoding: {encoding})")
                break
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed (encoding: {encoding}): {e}")
                continue
        
        if detected_neurons is None:
            print(f"❌ Unable to read neuron file with any encoding: {args.neurons_file}")
            return
            
    except Exception as e:
        print(f"❌ Neuron file loading failed: {e}")
        return

    # Determine output path
    if args.model_name:
        output_path = os.path.join(args.output_dir, args.model_name)
    else:
        # Use original model name + _deactivated
        model_name = os.path.basename(args.model_path.rstrip('/'))
        output_path = os.path.join(args.output_dir, f"{model_name}_deactivated")

    print(f"🎯 Starting deactivation...")
    print(f"   Original model: {args.model_path}")
    print(f"   Neuron file: {args.neurons_file}")
    print(f"   Output path: {output_path}")

    try:
        deactivation_params_saving(
            model_path=args.model_path,
            detected_neuron=detected_neurons,
            save_path=output_path
        )
        print(f"\n🎉 Deactivation completed successfully!")
        print(f"📁 Modified model saved at: {output_path}")
        
    except Exception as e:
        print(f"❌ Deactivation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

    

    

    
    
