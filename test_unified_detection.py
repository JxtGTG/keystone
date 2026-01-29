#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified neuron detection test program - supports Qwen2, Qwen3, and Llama models
Automatically identifies model type and selects corresponding implementation
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import datetime
import time
import gc
from pathlib import Path


try:
    import orjson
    HAS_ORJSON = True
    print("✅ Using orjson for high-performance JSON processing")
except ImportError:
    HAS_ORJSON = False
    print("⚠️ orjson not installed, using standard json module")
    print("💡 Installing orjson can significantly improve performance: pip install orjson")


sys.path.append('./transformers_standard_with_detection')

# Dynamically import implementations for different models
from transformers import AutoTokenizer, AutoConfig


class UnifiedDetectionTest:
    """Unified neuron detection tester, supports multiple models"""
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """Initialize detector"""
        print(f"🔄 Initializing unified neuron detector...")
        print(f"📍 Model path: {model_path}")
        print(f"🖥️ Device: {device}")
        
        self.model_path = model_path
        self.device = device
        
        # Automatically identify model type
        self.model_type = self._identify_model_type(model_path)
        print(f"🔍 Identified model type: {self.model_type}")
        
        # Dynamically import corresponding model class
        self.model_class = self._get_model_class()
        print(f"🔍 Resolved model class: {self.model_class}")
        if self.model_class is None:
            raise ValueError("Model class is None, cannot continue initialization")
        
        # Load config
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Handle config structures for different models
        if self.model_type == "Gemma3":
            # Gemma3 config is nested; layer count is in text_config
            if hasattr(self.config, 'text_config'):
                self.text_config = self.config.text_config
                print(f"📊 Model layers: {self.text_config.num_hidden_layers}")
            else:
                self.text_config = self.config
                print(f"📊 Model layers: {self.text_config.num_hidden_layers}")
        else:
            # Other models use config directly
            self.text_config = self.config
            print(f"📊 Model layers: {self.config.num_hidden_layers}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        print(f"✅ Tokenizer loaded")
        
        # Load the corresponding model
        print(f"🔄 Loading {self.model_type} model (this may take some time)...")
        
        # Handle device mapping
        if isinstance(device, str) and "," in device:
            # Multi-GPU setup, e.g. "cuda:0,1" or "cuda:0,1,5"
            # Remove "cuda:" prefix and extract GPU indices
            device_clean = device.replace("cuda:", "").strip()
            gpu_ids = [gpu_id.strip() for gpu_id in device_clean.split(",") if gpu_id.strip()]
            
            if gpu_ids:
                # Set CUDA_VISIBLE_DEVICES, e.g. "0,1"
                visible_devices = ",".join(gpu_ids)
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
                print(f"🔧 Detected multi-GPU setting: {device}, set CUDA_VISIBLE_DEVICES={visible_devices}")
                print(f"   GPUs in use: {', '.join(f'cuda:{gid}' for gid in gpu_ids)}")
                device_map_setting = "auto"  # Auto-assign within restricted GPUs
            else:
                device_map_setting = "auto"
        elif isinstance(device, str) and "cuda:" in device:
            # Single GPU setup, e.g., "cuda:5"
            device_index = int(device.split(":")[1])
            device_map_setting = device_index
            print(f"🔧 Using single GPU: {device} (index: {device_index})")
        else:
            # Other cases (auto, cpu, etc.)
            device_map_setting = device if isinstance(device, str) else "auto"
            print(f"🔧 Using device setting: {device_map_setting}")
            
        # Use correct config depending on model type
        if self.model_type == "Gemma3":
            # Gemma3 needs special config handling
            model_config = self.text_config
            self.model = self.model_class.from_pretrained(
                model_path,
                config=model_config,
                device_map=device_map_setting,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                 attn_implementation="eager" 
            )
        elif self.model_type in ["Llama", "Llama32"]:
            # Llama loads custom model directly
            print("🔄 Loading custom Llama model...")
            self.model = self.model_class.from_pretrained(
                model_path,
                device_map=device_map_setting,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation="eager" 
            )
            print("✅ Llama model loaded")
        else:
            # Other models (including Qwen3Moe) don't pass config parameter; let model handle it
            self.model = self.model_class.from_pretrained(
                model_path,
                device_map=device_map_setting,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation="eager" 

            )
        self.model.eval()
        print(f"✅ {self.model_type} model loaded")
        
        # Clean up GPU memory
        self._cleanup_gpu_memory()
        
        # Monitor GPU memory after model load
        self._monitor_gpu_memory("after model load")
    
    def _identify_model_type(self, model_path: str) -> str:
        """Automatically identify model type"""
        model_name = model_path.lower()
        
        if "mixtral" in model_name:
            return "Mixtral"
        elif "gpt-oss" in model_name or "gpt_oss" in model_name:
            return "GptOss"
        elif "qwen3" in model_name and ("moe" in model_name or "a3b" in model_name):
            return "Qwen3Moe"
        elif "gemma3" in model_name or "gemma-3" in model_name:
            return "Gemma3"
       
        elif "llama" in model_name:
            return "Llama"
        elif "qwen3" in model_name:
            return "Qwen3"
        elif "qwen" in model_name:
            return "Qwen2"
        else:
            # Default to Qwen2, as most Qwen models are Qwen2
            return "Llama"
    
    def _get_model_class(self):
        """Get corresponding model class based on model type"""
        try:
            if self.model_type == "Mixtral":
                from transformers_standard_with_detection.modeling_mixtral import MixtralForCausalLM
                print(f"✅ Successfully imported MixtralForCausalLM")
                return MixtralForCausalLM
            elif self.model_type == "GptOss":
                from transformers_standard_with_detection.modeling_gpt_oss import GptOssForCausalLM
                print(f"✅ Successfully imported GptOssForCausalLM")
                return GptOssForCausalLM
            elif self.model_type == "Qwen3Moe":
                from transformers_standard_with_detection.modeling_qwen3_moe import Qwen3MoeForCausalLM
                print(f"✅ Successfully imported Qwen3MoeForCausalLM")
                return Qwen3MoeForCausalLM
            elif self.model_type == "Gemma3":
                from transformers_standard_with_detection.modeling_gemma3 import Gemma3ForCausalLM
                print(f"✅ Successfully imported Gemma3ForCausalLM")
                return Gemma3ForCausalLM
            elif self.model_type == "Llama32":
                from transformers_standard_with_detection.modeling_llama import LlamaForCausalLM
                print(f"✅ Successfully imported LlamaForCausalLM (Llama-3.2)")
                return LlamaForCausalLM
            elif self.model_type == "Llama":
                from transformers_standard_with_detection.modeling_llama import LlamaForCausalLM
                print(f"✅ Successfully imported LlamaForCausalLM")
                return LlamaForCausalLM
            elif self.model_type == "Qwen3":
                from transformers_standard_with_detection.modeling_qwen3 import Qwen3ForCausalLM
                print(f"✅ Successfully imported Qwen3ForCausalLM")
                return Qwen3ForCausalLM
            elif self.model_type == "Qwen2":
                from transformers_standard_with_detection.modeling_qwen2 import Qwen2ForCausalLM
                print(f"✅ Successfully imported Qwen2ForCausalLM")
                return Qwen2ForCausalLM
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except ImportError as e:
            print(f"❌ Failed to import {self.model_type} model: {e}")
            raise
        except Exception as e:
            print(f"❌ Unknown error occurred while getting {self.model_type} model class: {e}")
            raise
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            # Force garbage collection
            gc.collect()
            # Get current GPU memory usage
            print(f"🧹 GPU memory cleanup complete")
            print(f"📊 Current GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"📊 Current GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    def _monitor_gpu_memory(self, stage: str = ""):
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 {stage} GPU memory status:")
            print(f"   - Allocated: {allocated:.2f} GB")
            print(f"   - Cached: {reserved:.2f} GB")
            print(f"   - Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved:.2f} GB")
    
    def test_detection_mode(
        self, 
        prompt: str,
        max_new_tokens: int = 10,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.9,
        output_dir: str = "./test_data",
        prompt_name: str = "default"
    ) -> Dict[str, Any]:
        """Test activation value recording mode"""
        print(f"\n🎯 Testing activation value recording mode")
        print(f"📝 Prompt: {prompt}")
        
        # Create output directory (create subdirectory by prompt name)
        output_path = Path(output_dir) / prompt_name
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare input
        chat = [{"role": "user", "content": prompt}]
        
        try:
            # Try using thinking template (specific to Qwen3 and Qwen3Moe)
            if self.model_type in ["Qwen3", "Qwen3Moe"]:
                templated_prompt = self.tokenizer.apply_chat_template(
                    chat, 
                    tokenize=False, 
                    add_generation_prompt=True, 
                    enable_thinking=True  # Thinking mode for Qwen3 and Qwen3Moe
                )
                print(f"✅ Using thinking template ({self.model_type})")
            else:
                # Check if chat template exists
                if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                    templated_prompt = self.tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True
                    )
                    print("✅ Using standard template")
                else:
                    # If no chat template, use original prompt directly
                    templated_prompt = prompt
                    print("✅ Using original prompt (no chat template)")
        except (TypeError, ValueError) as e:
            # If chat template is unavailable, use original prompt directly
            templated_prompt = prompt
            print(f"✅ Using original prompt (chat template error: {e})")
        
        inputs = self.tokenizer(templated_prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        print(f"🔤 Input token length: {input_length}")
        
        # Get candidate layers (all layers)
        candidate_layers = list(range(self.text_config.num_hidden_layers))
        print(f"🔍 Detection layers: {len(candidate_layers)}")
        
        # Store data
        all_token_data = []
        generated_tokens = []  # Store token text
        generated_token_ids = []  # Store token IDs for repetition penalty
        
        print(f"\n🚀 Starting token-by-token generation and activation value recording...")
        
        # Display sampling parameters
        if temperature < 1e-6:
            print(f"🎲 Sampling mode: Greedy sampling (temperature={temperature}, will use argmax)")
        else:
            print(f"🎲 Sampling parameters: temperature={temperature}, top_k={top_k}, top_p={top_p}")
        
        print(f"📝 Model response: ", end="", flush=True)  # Start real-time output
        
        # Monitor GPU memory state before starting
        self._monitor_gpu_memory("Before generation starts")
        
        with torch.no_grad():
            
            current_input_ids = inputs.input_ids.clone()
            attention_mask = inputs.attention_mask.clone()
            
            for step in range(max_new_tokens):
                # Safety mechanism: prevent infinite loop
                if step > max_new_tokens * 2:  # If exceeds 2x expected steps, force exit
                    print(f"\n⚠️ Reached maximum step limit ({max_new_tokens * 2}), forcing exit from generation")
                    break
                # Forward pass and record activations - directly copy logic from test_qwen3_moe_with_detection.py
                try:
                    outputs, layer_activations = self.model(
                        input_ids=current_input_ids,
                        attention_mask=attention_mask,
                        record_activations=True,
                        target_layers=candidate_layers
                    )
                except Exception as e:
                    print(f"\n❌ Step {step} activation retrieval failed: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                
                # Get next token's logits - directly copy logic from test_qwen3_moe_with_detection.py
                next_token_logits = outputs.logits[0, -1, :]
                
                # Apply repetition penalty (before other sampling strategies)
                repetition_penalty = 1.1
                for token_id in set(generated_token_ids[-64:]):  # Only apply penalty to last 64 tokens
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= repetition_penalty
                    else:
                        next_token_logits[token_id] /= repetition_penalty
                
                # Special handling: if temperature is 0 or very small, use greedy sampling
                if temperature < 1e-6:
                    # Greedy sampling: directly select token with maximum logits
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    # Apply temperature sampling
                    next_token_logits = next_token_logits / temperature
                    
                    # Apply top_k sampling
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        logits_mask = torch.full_like(next_token_logits, float('-inf'))
                        logits_mask.scatter_(-1, top_k_indices, top_k_logits)
                        next_token_logits = logits_mask
                    
                    # Apply top_p sampling - directly copy logic from test_qwen3_moe_with_detection.py
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Comment out debug info, maintain original output mode
                # if step < 5 or step % 10 == 0:  # Show first 5 steps and every 10 steps
                #     token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
                #     print(f"\n🔍 Step {step}: Generated token ID={next_token.item()}, text={repr(token_text)}")
                
                # Add token ID to list
                generated_token_ids.append(next_token.item())
                
                # Decode token and display in real time
                if self.model_type == "Gemma3":
                    token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
                    generated_tokens.append(token_text)
                elif self.model_type in ["Qwen3Moe", "Qwen3"]:
                    token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
                    generated_tokens.append(token_text)
                elif self.model_type == "Llama" or self.model_type == "Llama32":
                    token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
                    generated_tokens.append(token_text)
                
                elif self.model_type in ["Mixtral"]:
                    token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                    generated_tokens.append(token_text)
                elif self.model_type == "Qwen2":
                    token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
                    generated_tokens.append(token_text)
                
                
                
                
                
                # Real-time output token text (like a conversation)
                print(token_text, end="", flush=True)
                
              
                # Format activations to be compatible with detection code
                formatted_activations = self._format_activations_for_storage(layer_activations)
                
                # Check if it's a terminator, add special marker
                is_eos_token = False
                eos_reason = None
                
                if self.model_type == "Gemma3":
                   
                    end_of_turn_id = self.tokenizer.convert_tokens_to_ids('<end_of_turn>')
                    if next_token.item() == end_of_turn_id:
                        is_eos_token = True
                        eos_reason = "end_of_turn"
                    elif next_token.item() == self.tokenizer.eos_token_id:
                        is_eos_token = True
                        eos_reason = "eos_token"
                elif self.model_type in ["Qwen3Moe", "Qwen3", "Llama", "Llama32", "Mixtral"]:
                    if next_token.item() == self.tokenizer.eos_token_id:
                        is_eos_token = True
                        eos_reason = "eos_token"
                elif next_token.item() == self.tokenizer.eos_token_id:
                    is_eos_token = True
                    eos_reason = "eos_token"
                
                # Record activation data (including terminator marker)
                token_data = {
                    'step': step,
                    'token_id': next_token.item(),
                    'token_text': token_text,
                    'is_eos_token': is_eos_token,
                    'eos_reason': eos_reason,
                    'activations': formatted_activations
                }
                
                all_token_data.append(token_data)
                
                # Check if generation should end (after saving data)
                if is_eos_token:
                    if self.model_type == "Gemma3":
                        print(f"\n🏁 eos_reason: {eos_reason}")
                    else:
                        print(f"\n🛑 eos_reason: {eos_reason}")
                    break
                
                # Update input - directly copy logic from test_qwen3_moe_with_detection.py
                current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(0)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.model.device)], dim=1)
        
        # Generate complete text
        full_generated_text = ''.join(generated_tokens)
        print(f"\n\n📄 completion:")
        print(f"  - Token count: {len(generated_tokens)}")
        print(f"  - Text length: {len(full_generated_text)} characters")
        
        # Monitor GPU memory state after generation completes
        self._monitor_gpu_memory("completed")
        
        # Comment out repeated display of generated content, as it's already displayed in real-time
        # print(f"  - Generated content: {full_generated_text}")
        
        # Save original token data
        token_file = output_path / f"{self.model_type.lower()}_detection_token_data_{timestamp}.json"
        self._save_json_optimized(all_token_data, token_file)
        print(f"💾 Token data saved: {token_file}")
        
        # Data validation (commented out)
        # validation_result = self._validate_data(all_token_data)
        
        # Save validation results
        summary = {
            'timestamp': timestamp,
            'model_type': f'{self.model_type.lower()}_with_detection',
            'model_path': self.model_path,
            'prompt': prompt,
            'generated_text': full_generated_text,
            'total_tokens': len(generated_tokens),
            'model_layers': self.text_config.num_hidden_layers,
            # 'validation': validation_result,  # Commented out
            'files': {
                'token_data': str(token_file)
            }
        }
        
        summary_file = output_path / f"{self.model_type.lower()}_detection_summary_{timestamp}.json"
        self._save_json(summary, summary_file)
        print(f"📋 Test summary saved: {summary_file}")
        
        # Clean up GPU memory
        self._cleanup_gpu_memory()
        
        return summary
    
    def _format_activations_for_storage(self, layer_activations: dict) -> dict:
        """Format activations to storage format - unified format for all models"""
        formatted = {
            'fwd_up': {},
            'fwd_down': {},
            'attn_q': {},
            'attn_k': {},
            'attn_v': {},
            'attn_o': {}
        }
        
        for layer_idx, activations in layer_activations.items():
            layer_str = str(layer_idx)
            
            # Handle activation value formats for different models
            if self.model_type == "Gemma3":
                # Gemma3 model uses direct key names
                if activations.get('fwd_up') is not None:
                    formatted['fwd_up'][layer_str] = self._convert_tensor_to_list(activations['fwd_up'])
                if activations.get('fwd_down') is not None:
                    formatted['fwd_down'][layer_str] = self._convert_tensor_to_list(activations['fwd_down'])
                
                # Attention activations
                if activations.get('attn_q') is not None:
                    formatted['attn_q'][layer_str] = self._convert_tensor_to_list(activations['attn_q'])
                if activations.get('attn_k') is not None:
                    formatted['attn_k'][layer_str] = self._convert_tensor_to_list(activations['attn_k'])
                if activations.get('attn_v') is not None:
                    formatted['attn_v'][layer_str] = self._convert_tensor_to_list(activations['attn_v'])
                if activations.get('attn_o') is not None:
                    formatted['attn_o'][layer_str] = self._convert_tensor_to_list(activations['attn_o'])
            elif self.model_type in ["Qwen3Moe", "Mixtral"]:
                # MoE models use new expert-separated format - do not merge activations
                # MLP activations - save by expert separately
                if activations.get('moe_activations') is not None:
                    # Create separate activation records for each expert
                    expert_up_activations = {}
                    expert_down_activations = {}
                    
                    for expert_name, expert_data in activations['moe_activations'].items():
                        if expert_data.get('is_activated', False):
                            # Save activations for each expert
                            expert_up_activations[expert_name] = self._convert_tensor_to_list(expert_data.get('up_activations', []))
                            expert_down_activations[expert_name] = self._convert_tensor_to_list(expert_data.get('down_activations', []))
                        else:
                            # Inactive experts saved as zero-value lists
                            expert_up_activations[expert_name] = self._convert_tensor_to_list(expert_data.get('up_activations', []))
                            expert_down_activations[expert_name] = self._convert_tensor_to_list(expert_data.get('down_activations', []))
                    
                    # Save expert-separated activations
                    formatted['fwd_up'][layer_str] = expert_up_activations
                    formatted['fwd_down'][layer_str] = expert_down_activations
                else:
                    # Backward compatible: if no moe_activations, use old format
                    if activations.get('up_activations') is not None:
                        formatted['fwd_up'][layer_str] = self._convert_tensor_to_list(activations['up_activations'])
                    if activations.get('down_activations') is not None:
                        formatted['fwd_down'][layer_str] = self._convert_tensor_to_list(activations['down_activations'])
                
                # Attention activations
                if activations.get('attn_activations') is not None:
                    attn_acts = activations['attn_activations']
                    formatted['attn_q'][layer_str] = self._convert_tensor_to_list(attn_acts.get('q'))
                    formatted['attn_k'][layer_str] = self._convert_tensor_to_list(attn_acts.get('k'))
                    formatted['attn_v'][layer_str] = self._convert_tensor_to_list(attn_acts.get('v'))
                    formatted['attn_o'][layer_str] = self._convert_tensor_to_list(attn_acts.get('o'))
            else:
                # Other models (Qwen2, Llama, Qwen3, etc.) use nested structure
                # MLP activations
                if activations.get('up_activations') is not None:
                    formatted['fwd_up'][layer_str] = self._convert_tensor_to_list(activations['up_activations'])
                if activations.get('down_activations') is not None:
                    formatted['fwd_down'][layer_str] = self._convert_tensor_to_list(activations['down_activations'])
                
                # Attention activations
                if activations.get('attn_activations') is not None:
                    attn_acts = activations['attn_activations']
                    formatted['attn_q'][layer_str] = self._convert_tensor_to_list(attn_acts.get('q'))
                    formatted['attn_k'][layer_str] = self._convert_tensor_to_list(attn_acts.get('k'))
                    formatted['attn_v'][layer_str] = self._convert_tensor_to_list(attn_acts.get('v'))
                    formatted['attn_o'][layer_str] = self._convert_tensor_to_list(attn_acts.get('o'))
        
        return formatted
    
    def _convert_tensor_to_list(self, tensor_data):
        """
        Convert tensor data to serializable list format
        
    
        """
        if tensor_data is None:
            return None
        if isinstance(tensor_data, torch.Tensor):
            return tensor_data.detach().cpu().numpy().tolist()
        elif isinstance(tensor_data, list):
            return [self._convert_tensor_to_list(item) for item in tensor_data]
        else:
            return tensor_data
    
   
    
    def _save_json(self, data: Any, file_path: Path):
       
        if HAS_ORJSON:
           
            with open(file_path, 'wb') as f:
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        else:
          
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_json_optimized(self, data: Any, file_path: Path):
        
        print(f"🔄 save json to {file_path}")
        
        if HAS_ORJSON:
          
            print(f"🚀 use orjson to save json")
            
            if isinstance(data, list) and len(data) > 1000:
                
                
                
                with open(file_path, 'wb') as f:
                    f.write(b'[\n')
                    
                    for i, item in enumerate(data):
                        if i > 0:
                            f.write(b',\n')
                        
                      
                        item_json = orjson.dumps(item, option=orjson.OPT_INDENT_2)
                        f.write(item_json)
                      
                        if i % 100 == 0:
                            f.flush()
                            if i % 1000 == 0:
                                print(f"  📝 Saved {i}/{len(data)} items")
                    
                    f.write(b'\n]')
            else:
                
                with open(file_path, 'wb') as f:
                    f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        else:
           
            print(f"⚠️ use standard json module")
            
            with open(file_path, 'w', encoding='utf-8', buffering=8192*4) as f:
                if isinstance(data, list) and len(data) > 1000:
                   
                    
                    f.write('[\n')
                    for i, item in enumerate(data):
                        if i > 0:
                            f.write(',\n')
                        
                        json.dump(item, f, ensure_ascii=False, indent=2)
                        
                        if i % 100 == 0:
                            f.flush()
                            if i % 1000 == 0:
                                print(f"  📝 saved {i}/{len(data)} items")
                    
                    f.write('\n]')
                else:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ save json completed")


def batch_process_models(models_config, base_output_dir: str, device: str = "cuda:0,1,5"):
    """Batch process multiple models and prompts"""
    print(f"🚀 Starting batch processing for {len(models_config)} models...")
    
    all_results = {}
    
    for i, config in enumerate(models_config):
        model_path = config['model_path']
        prompts = config['prompts']
        model_name = config.get('model_name', model_path.split('/')[-1])
        
        print(f"\n{'='*60}")
        print(f"📋 Processing model {i+1}/{len(models_config)}: {model_name}")
        print(f"📍 Model path: {model_path}")
        print(f"📝 Processing {len(prompts)} prompts")
        print(f"{'='*60}")
        
        try:
            # Create independent output directory for each model
            model_output_dir = Path(base_output_dir) / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created model-specific directory: {model_output_dir}")
            
            # Create unified detector
            detector = UnifiedDetectionTest(
                model_path=model_path,
                device=device
            )
            
            model_results = []
            
            # Process each prompt
            for j, prompt_config in enumerate(prompts):
                prompt = prompt_config['prompt']
                prompt_name = prompt_config.get('name', f'prompt_{j}')
                max_new_tokens = prompt_config.get('max_new_tokens', 1000)
                temperature = prompt_config.get('temperature', 0.6)
                
                print(f"\n🎯 Processing prompt {j+1}/{len(prompts)}: {prompt_name}")
                print(f"📝 Content: {prompt}")
                
                try:
                  
                    detection_result = detector.test_detection_mode(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        output_dir=str(model_output_dir),
                        prompt_name=prompt_name
                    )
                    
                 
                    detection_result['prompt_name'] = prompt_name
                    detection_result['prompt_content'] = prompt
                    detection_result['prompt_config'] = prompt_config
                    
                    model_results.append(detection_result)
                    
                    print(f"✅ Prompt '{prompt_name}' completed")
                    
                except Exception as e:
                    print(f"❌ Prompt '{prompt_name}' failed: {e}")
                    error_result = {
                        'prompt_name': prompt_name,
                        'prompt_content': prompt,
                        'error': str(e),
                        'status': 'failed'
                    }
                    model_results.append(error_result)
                    continue
            
            # Save model-level summary
            model_summary = {
                'model_name': model_name,
                'model_path': model_path,
                'model_type': detector.model_type,
                'total_prompts': len(prompts),
                'successful_prompts': len([r for r in model_results if r.get('status') != 'failed']),
                'failed_prompts': len([r for r in model_results if r.get('status') == 'failed']),
                'results': model_results,
                'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            
            summary_file = model_output_dir / f"{model_name}_batch_summary.json"
            if HAS_ORJSON:
                with open(summary_file, 'wb') as f:
                    f.write(orjson.dumps(model_summary, option=orjson.OPT_INDENT_2))
            else:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(model_summary, f, ensure_ascii=False, indent=2)
            
            print(f"📋 Model summary saved: {summary_file}")
            all_results[model_name] = model_summary
            
            # Cleanup GPU memory for current model
            detector._cleanup_gpu_memory()
            
        except Exception as e:
            print(f"❌ Model {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save global summary
    global_summary = {
        'total_models': len(models_config),
        'successful_models': len([r for r in all_results.values() if r.get('status') != 'failed']),
        'failed_models': len([r for r in all_results.values() if r.get('status') == 'failed']),
        'models': all_results,
        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    global_summary_file = Path(base_output_dir) / "global_batch_summary.json"
    if HAS_ORJSON:
        with open(global_summary_file, 'wb') as f:
            f.write(orjson.dumps(global_summary, option=orjson.OPT_INDENT_2))
    else:
        with open(global_summary_file, 'w', encoding='utf-8') as f:
            json.dump(global_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Batch processing complete!")
    print(f"📊 Global summary saved: {global_summary_file}")
    
    return all_results


def batch_process_models_cross_combination(models_list, prompts_list, base_output_dir: str, device: str = "cuda:0,1,5", 
                                         default_max_new_tokens: int = 1000, temperature: float = 0.6, 
                                         top_k: int = 50, top_p: float = 0.9):
    """Cross-combination mode: each model runs all prompts; supports per-prompt max_new_tokens"""
    print(f"🚀 Starting cross-combination batch processing...")
    print(f"📊 Model count: {len(models_list)}")
    print(f"📝 Prompt count: {len(prompts_list)}")
    print(f"🔧 Default params: max_new_tokens={default_max_new_tokens}, temperature={temperature}, top_k={top_k}, top_p={top_p}")
    print(f"💡 Each prompt can set its own max_new_tokens")
    
    all_results = {}
    
    for i, model_config in enumerate(models_list):
        model_path = model_config['model_path']
        model_name = model_config.get('model_name', model_path.split('/')[-1])
        
        print(f"\n{'='*60}")
        print(f"📋 Processing model {i+1}/{len(models_list)}: {model_name}")
        print(f"📍 Model path: {model_path}")
        print(f"📝 Will process {len(prompts_list)} prompts")
        print(f"{'='*60}")
        
        try:
            # Create independent output directory for each model
            model_output_dir = Path(base_output_dir) / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created model-specific directory: {model_output_dir}")
            
            # Create unified detector
            detector = UnifiedDetectionTest(
                model_path=model_path,
                device=device
            )
            
            model_results = []
            
            # Process each prompt (supports per-prompt max_new_tokens)
            for j, prompt_config in enumerate(prompts_list):
                prompt = prompt_config['prompt']
                prompt_name = prompt_config.get('name', f'prompt_{j}')
                # Get per-prompt max_new_tokens (fallback to default)
                prompt_max_tokens = prompt_config.get('max_new_tokens', default_max_new_tokens)
                
                print(f"\n🎯 Processing prompt {j+1}/{len(prompts_list)}: {prompt_name}")
                print(f"📝 Content: {prompt}")
                print(f"⚙️ Params: max_new_tokens={prompt_max_tokens}, temperature={temperature}, top_k={top_k}, top_p={top_p}")
                
                try:
                    # Test activation recording mode (use per-prompt max_new_tokens)
                    detection_result = detector.test_detection_mode(
                        prompt=prompt,
                        max_new_tokens=prompt_max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        output_dir=str(model_output_dir),
                        prompt_name=prompt_name
                    )
                    
                    # Attach prompt info to results
                    detection_result['prompt_name'] = prompt_name
                    detection_result['prompt_content'] = prompt
                    detection_result['prompt_config'] = {
                        'name': prompt_name,
                        'prompt': prompt,
                        'max_new_tokens': prompt_max_tokens,
                        'temperature': temperature,
                        'top_k': top_k,
                        'top_p': top_p
                    }
                    
                    model_results.append(detection_result)
                    
                    print(f"✅ Prompt '{prompt_name}' completed")
                    
                except Exception as e:
                    print(f"❌ Prompt '{prompt_name}' failed: {e}")
                    error_result = {
                        'prompt_name': prompt_name,
                        'prompt_content': prompt,
                        'error': str(e),
                        'status': 'failed'
                    }
                    model_results.append(error_result)
                    continue
            
            # Save model-level summary
            model_summary = {
                'model_name': model_name,
                'model_path': model_path,
                'model_type': detector.model_type,
                'total_prompts': len(prompts_list),
                'successful_prompts': len([r for r in model_results if r.get('status') != 'failed']),
                'failed_prompts': len([r for r in model_results if r.get('status') == 'failed']),
                'unified_params': {
                    'default_max_new_tokens': default_max_new_tokens,
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p
                },
                'results': model_results,
                'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            
            summary_file = model_output_dir / f"{model_name}_cross_combination_summary.json"
            if HAS_ORJSON:
                with open(summary_file, 'wb') as f:
                    f.write(orjson.dumps(model_summary, option=orjson.OPT_INDENT_2))
            else:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(model_summary, f, ensure_ascii=False, indent=2)
            
            print(f"📋 Model summary saved: {summary_file}")
            all_results[model_name] = model_summary
            
            # Cleanup GPU memory for current model
            detector._cleanup_gpu_memory()
            
        except Exception as e:
            print(f"❌ Model {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save global summary
    global_summary = {
        'processing_mode': 'cross_combination',
        'total_models': len(models_list),
        'total_prompts': len(prompts_list),
        'unified_params': {
            'default_max_new_tokens': default_max_new_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p
        },
        'successful_models': len([r for r in all_results.values() if r.get('status') != 'failed']),
        'failed_models': len([r for r in all_results.values() if r.get('status') == 'failed']),
        'models': all_results,
        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    global_summary_file = Path(base_output_dir) / "global_cross_combination_summary.json"
    if HAS_ORJSON:
        with open(global_summary_file, 'wb') as f:
            f.write(orjson.dumps(global_summary, option=orjson.OPT_INDENT_2))
    else:
        with open(global_summary_file, 'w', encoding='utf-8') as f:
            json.dump(global_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Cross-combination batch processing complete!")
    print(f"📊 Global summary saved: {global_summary_file}")
    
    return all_results


def load_config_from_file(config_file: str = "batch_config.json"):
    """Load batch processing config from file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Successfully loaded config from file: {config_file}")
        return config
    except FileNotFoundError:
        print(f"⚠️ Config file {config_file} not found, using default config")
        return None
    except Exception as e:
        print(f"❌ Failed to load config file: {e}")
        return None


def load_cross_combination_config(config_file: str = "cross_combination_config.json"):
    """Load cross-combination mode config from file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Successfully loaded cross-combination config from {config_file}")
        return config
    except Exception as e:
        print(f"❌ Failed to load cross-combination config file: {e}")
        return None



def main():
    """Main test function"""
    print("🧪 Unified neuron detection test program - batch processing mode")
    print("=" * 60)
    print("Supported model types:")
    print("  - Mixtral family (contains 'mixtral')")
    print("  - GPT-OSS family (contains 'gpt-oss' or 'gpt_oss')")
    print("  - Qwen3 MoE family (contains 'qwen3' and 'moe')")
    print("  - Gemma3 family (contains 'gemma3' or 'gemma-3')")
    print("  - Llama family (contains 'llama')")
    print("  - Qwen3 family (contains 'qwen3')")
    print("  - Qwen2 family (contains 'qwen')")
    print("=" * 60)
    print("Supported processing modes:")
    print("  1. Standard mode: each model has its own prompts and parameters")
    print("  2. Cross-combination mode: each model runs all prompts with unified parameters")
    print("=" * 60)
    
    # Select processing mode
    mode = input("Select processing mode (1: standard, 2: cross-combination): ").strip()
    
    if mode == "1":
        # Standard mode: load from config file
        print("\n🎯 Selected standard mode")
        config = load_config_from_file("batch_config.json")
     
        # Extract config parameters
        base_output_dir = config["base_output_dir"]
        device = config["device"]
        models_config = config["models"]
        
        print(f"📁 Base output dir: {base_output_dir}")
        print(f"🖥️ Device: {device}")
        print(f"📊 Configured models: {len(models_config)}")
        
        try:
            # Start batch processing
            all_results = batch_process_models(
                models_config=models_config,
                base_output_dir=base_output_dir,
                device=device
            )
            
            # Print final summary
            print(f"\n{'='*60}")
            print("🎯 Final summary (standard mode)")
            print(f"{'='*60}")
            
            for model_name, result in all_results.items():
                print(f"\n📋 Model: {model_name}")
                print(f"  - Total prompts: {result['total_prompts']}")
                print(f"  - Success: {result['successful_prompts']}")
                print(f"  - Failed: {result['failed_prompts']}")
                print(f"  - Model type: {result['model_type']}")
                
        except BaseException as e:
            print(f"❌ Standard-mode batch processing failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif mode == "2":
        # Cross-combination mode
        print("\n🎯 Selected cross-combination mode")
        
        # Try loading cross-combination config from file
        cross_config = load_cross_combination_config("cross_single.json")
    
        print("📋 Using cross-combination config from file")
        models_list = cross_config["models"]
        prompts_list = cross_config["prompts"]
        default_max_new_tokens = cross_config["unified_params"]["max_new_tokens"]
        temperature = cross_config["unified_params"]["temperature"]
        top_k = cross_config["unified_params"].get("top_k", 50)
        top_p = cross_config["unified_params"].get("top_p", 0.9)
        base_output_dir = cross_config["base_output_dir"]
        device = cross_config["device"]
        
        print(f"📊 Model count: {len(models_list)}")
        print(f"📝 Prompt count: {len(prompts_list)}")
        print(f"🔧 Default params: max_new_tokens={default_max_new_tokens}, temperature={temperature}, top_k={top_k}, top_p={top_p}")
        print(f"💡 Each prompt can set its own max_new_tokens")
        print(f"📁 Output dir: {base_output_dir}")
        print(f"🖥️ Device: {device}")
        
        try:
            # Start cross-combination batch processing
            all_results = batch_process_models_cross_combination(
                models_list=models_list,
                prompts_list=prompts_list,
                base_output_dir=base_output_dir,
                device=device,
                default_max_new_tokens=default_max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Print final summary
            print(f"\n{'='*60}")
            print("🎯 Final summary (cross-combination mode)")
            print(f"{'='*60}")
            
            for model_name, result in all_results.items():
                print(f"\n📋 Model: {model_name}")
                print(f"  - Total prompts: {result['total_prompts']}")
                print(f"  - Success: {result['successful_prompts']}")
                print(f"  - Failed: {result['failed_prompts']}")
                print(f"  - Model type: {result['model_type']}")
                
        except BaseException as e:
            print(f"❌ Cross-combination batch processing failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("❌ Invalid selection, exiting")
        return
    
    # Clean up GPU memory at program end
    print("\n🧹 Program finished, cleaning up GPU memory...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU memory cleanup complete")


if __name__ == "__main__":
    main()
