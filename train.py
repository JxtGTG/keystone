#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from trl import DataCollatorForCompletionOnlyLM
from transformers import TrainerCallback
from processors import get_prompt_builder
from prepare_benchmark_data import BENCHMARK_CONFIGS, format_example, _is_valid_example

# Optional: your custom Trainer (if your project has it)
# If not, you can directly use HF Trainer: from transformers import Trainer
try:
    from transformers_standard_with_detection.trainer import Trainer  # Your custom Trainer
except Exception:
    from transformers import Trainer  # Fallback to official Trainer


cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
print(f"🚀 CUDA_VISIBLE_DEVICES={cuda_devices}")

def train(
    # model/data params
    base_model: str = "",
    data_dir: str = "",
    benchmark_name: str = "",

    # training hyperparams
    batch_size: int = 32,          # Expected global batch_size (≈ world_size * micro_batch_size * grad_accum)
    micro_batch_size: int = 4,     # per_device_batch_size per GPU
    num_epochs: int = 2,
    learning_rate: float = 5e-5,
    cutoff_len: int = 2048,
    val_set_size: int = 2000,
    eval_steps: int = 200,
    save_steps: int = 200,
    max_train_samples: int = 200,
    max_eval_samples: int = 0,
    top_k: int = None,
    min_length_filter: int = 100,  # Only keep samples with length >= 100

    # output
    output_base_dir: str = "",
    run_name: str = "",
    data_loading_mode: str = "original",  # "original", "mixed", "auto"
    neuron_update_strategy: str = "freeze",     # "", "freeze", "restore"
    neuron_path: str = "",

    # llm hyperparams
    group_by_length: bool = False,

    # other
    cache_dir: str = "",
    local_rank: int = 0,  # torchrun injects LOCAL_RANK; this is just a placeholder
    resume_from_checkpoint: str = None,
):
    """
    Supervised Fine-Tuning script with FSDP (no DeepSpeed required).
    Can run on single/multi-GPU: single GPU uses plain python; multi-GPU uses torchrun.
    """
    # --- Basic checks and output directory ---
    assert base_model, "Please specify --base_model"
    final_output_dir = os.path.join(output_base_dir, run_name)
    os.makedirs(final_output_dir, exist_ok=True)

    # --- Distributed environment detection ---
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1
    if is_dist:
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                pass
        except Exception:
            pass
    print(f"🌎 WORLD_SIZE={world_size} | Distributed={is_dist} | FSDP=ON")

    # Calculate effective gradient accumulation steps (try to make global_batch ≈ batch_size)
    # global_batch = micro_batch_size * world_size * grad_accum
    grad_accum = max(1, math.ceil(batch_size / max(1, micro_batch_size * world_size)))
    effective_global = micro_batch_size * max(1, world_size) * grad_accum
    print(f"🧮 per_device_batch={micro_batch_size}, grad_accum={grad_accum}, "
          f"≈global_batch={effective_global} (target={batch_size})")

    # --- Model and tokenizer ---
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side ="right"
    model.config.use_cache = False  # Disable cache during training
    model.gradient_checkpointing_enable()

    # --- Save tokenizer to run directory for inference ---
    try:
        rank_str = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
        is_main_process_save = str(rank_str) == "0"
    except Exception:
        is_main_process_save = True
    if is_main_process_save:
        try:
            tokenizer.save_pretrained(final_output_dir)
            print(f"💾 Saved tokenizer to: {final_output_dir}")
        except Exception as e:
            print(f"⚠️ Failed to save tokenizer: {e}")

    # === Assistant header marker (for collator response_template) ===
    assistant_hdr = "<|start_header_id|>assistant<|end_header_id|>\n\n"


    def analyze_length_distribution(dataset, name="Dataset"):
        lengths = []
       
        for i, data_point in enumerate(dataset):
            if i >= 1:
                break
            user_prompt = data_point["prompt"]
            ground_truth = data_point["formatted_ground_truth"]
            user_messages = [{"role": "user", "content": user_prompt}]
            full_messages = user_messages + [{"role": "assistant", "content": ground_truth}]
            full_text = tokenizer.apply_chat_template(
                full_messages, add_generation_prompt=False, tokenize=False
            )
            enc = tokenizer(
                full_text,
                truncation=False,
                add_special_tokens=False,  # Keep consistent with tokenize_function
            )
            lengths.append(len(enc["input_ids"]))
        if lengths:
            lengths.sort()
            q1 = lengths[len(lengths)//4]
            med = lengths[len(lengths)//2]
            q3 = lengths[len(lengths)*3//4]
            avg = sum(lengths)/len(lengths)
            print(f"📊 {name} length distribution (samples<=16000): min={lengths[0]}, p25={q1}, p50={med}, p75={q3}, max={lengths[-1]}, avg={avg:.1f}, cutoff_len={cutoff_len}")

    def filter_by_length(data_point):
        user_prompt = data_point["prompt"]
        ground_truth = data_point["formatted_ground_truth"]
        user_messages = [{"role": "user", "content": user_prompt}]
        full_messages = user_messages + [{"role": "assistant", "content": ground_truth}]
        full_text = tokenizer.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=False
        )
        enc = tokenizer(
            full_text,
            truncation=False,
            add_special_tokens=False,  # Keep consistent with tokenize_function
        )
        token_length = len(enc["input_ids"])
        return (0 <= token_length <= cutoff_len)

    def tokenize_function(data_point):
        user_prompt = data_point["prompt"]
        ground_truth = data_point["formatted_ground_truth"]
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": ground_truth},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=cutoff_len,
            padding=False,           # Leave to collator for dynamic padding
            return_attention_mask=True,
        )
        return enc

    # --- Data loading (mixed / original / auto) ---
    benchmark_dir = os.path.join(data_dir, benchmark_name)
    train_file = os.path.join(benchmark_dir, "train.jsonl")
    test_file = os.path.join(benchmark_dir, "test.jsonl")
    # raw directory is no longer used; training loads from standard paths
    incorrect_file = os.path.join(benchmark_dir, "incorrect_samples.jsonl")
    correct_file = os.path.join(benchmark_dir, "correct_samples.jsonl")

    use_mixed_loading = False
    if data_loading_mode == "mixed":
        if os.path.exists(incorrect_file) and os.path.exists(correct_file):
            use_mixed_loading = True
        else:
            raise FileNotFoundError("Mixed mode requires incorrect_samples.jsonl and correct_samples.jsonl")
    elif data_loading_mode == "original":
        use_mixed_loading = False
    elif data_loading_mode == "auto":
        if os.path.exists(incorrect_file) and os.path.exists(correct_file):
            use_mixed_loading = True

    if use_mixed_loading:
        print("🔍 Using mixed data loading")
        incorrect_dataset = load_dataset("json", data_files=incorrect_file, cache_dir=cache_dir)
        correct_dataset = load_dataset("json", data_files=correct_file, cache_dir=cache_dir)
        incorrect_data = incorrect_dataset["train"]
        correct_data = correct_dataset["train"]
        print(f"📊 Original: incorrect={len(incorrect_data)}, correct={len(correct_data)}")
        analyze_length_distribution(incorrect_data, "Incorrect samples")
        analyze_length_distribution(correct_data, "Correct samples")
        print("🔍 Length filtering...")
        incorrect_filtered = incorrect_data.filter(filter_by_length)
        correct_filtered = correct_data.filter(filter_by_length)
        print(f"📊 After filtering: incorrect={len(incorrect_filtered)}, correct={len(correct_filtered)}")
        target = max_train_samples // 2 if max_train_samples is not None else min(len(incorrect_filtered), len(correct_filtered))
        incorrect_selected = incorrect_filtered.select(range(min(target, len(incorrect_filtered))))
        correct_selected = correct_filtered.select(range(min(target, len(correct_filtered))))
        print(f"📊 Selected: incorrect={len(incorrect_selected)}, correct={len(correct_selected)}, total={len(incorrect_selected)+len(correct_selected)}")
        train_data = concatenate_datasets([incorrect_selected, correct_selected]).shuffle(seed=42)
        # Cleanup
        del incorrect_selected, correct_selected, incorrect_filtered, correct_filtered
        del incorrect_data, correct_data, incorrect_dataset, correct_dataset
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Mixed data loading complete")
    else:
        print("🔍 Loading data from standard paths and building prompts in the training pipeline")
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        # Read core prompt + ground_truth (+formatted)
        base_train_dataset = load_dataset("json", data_files=train_file, cache_dir=cache_dir)
        base_train = base_train_dataset["train"]
        # Apply prompt builder in the training pipeline
        prompt_builder_fn = get_prompt_builder(benchmark_name)
        def _apply_prompt_builder_only(dp):
            core_prompt = dp.get("prompt", "")
            dp_out = dict(dp)
            dp_out["prompt"] = prompt_builder_fn(core_prompt)
            return dp_out
        train_data = base_train.map(
            _apply_prompt_builder_only,
            load_from_cache_file=False,
            num_proc=max(1, (os.cpu_count() or 1)//2),
        )
        # Select a subset first (avoid loading all data)
        train_data = train_data.select(range(min(25000, len(train_data))))
        analyze_length_distribution(train_data, "Original training data (built in training)")
        print("🔍 Length filtering...")
        train_data = train_data.filter(filter_by_length)
        print(f"📊 Training samples after filtering: {len(train_data)}")
        if max_train_samples is not None:
            train_data = train_data.select(range(min(max_train_samples, len(train_data))))

    # Eval data
    # Eval data: load from standard paths and build prompts in the pipeline
    eval_dataset = load_dataset("json", data_files=test_file, cache_dir=cache_dir)
    base_eval = eval_dataset["train"]
    prompt_builder_fn_eval = get_prompt_builder(benchmark_name)
    def _apply_prompt_builder_only_eval(dp):
        core_prompt = dp.get("prompt", "")
        dp_out = dict(dp)
        dp_out["prompt"] = prompt_builder_fn_eval(core_prompt)
        return dp_out
    eval_data = base_eval.map(
        _apply_prompt_builder_only_eval,
        load_from_cache_file=False,
        num_proc=max(1, (os.cpu_count() or 1)//2),
    )
    if max_eval_samples is not None:
        eval_data = eval_data.select(range(min(max_eval_samples, len(eval_data))))

    # --- Reasonable map concurrency in distributed scenarios ---
    cpu_cnt = max(1, os.cpu_count() or 1)
    map_workers = max(1, cpu_cnt // max(1, world_size))
    print(f"🧵 tokenize map workers = {map_workers}")

    # ✅ No longer generate labels in map; labels handled by DataCollatorForCompletionOnlyLM
    # --- Debug: print a few raw concatenated texts before tokenization (incl. special markers & invisible chars) ---
    try:
        rank_str = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
        is_main_process = str(rank_str) == "0"
    except Exception:
        is_main_process = True
    if is_main_process:
        debug_count = min(3, len(train_data)) if hasattr(train_data, "__len__") else 0
        if debug_count > 0:
            print(f"🔎 Debug samples (raw concatenated text incl. special markers), first {debug_count}:")
            for i in range(debug_count):
                dp = train_data[i]
                user_prompt = dp.get("prompt", "")
                ground_truth = dp.get("formatted_ground_truth", "")
                messages_dbg = [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": ground_truth},
                ]
                full_text_dbg = tokenizer.apply_chat_template(
                    messages_dbg, add_generation_prompt=False, tokenize=False
                )
                print(f"—— [sample {i}] full_text:")
                print(full_text_dbg)
                print("repr:", repr(full_text_dbg))

    train_data = train_data.map(tokenize_function, load_from_cache_file=False, num_proc=map_workers, remove_columns=None)
    eval_data = eval_data.map(tokenize_function, load_from_cache_file=False, num_proc=map_workers, remove_columns=None)
    print(f"LOAD DATA FINISHED - Train: {len(train_data)}, Eval: {len(eval_data)}")

    # --- Neuron configuration (keep your interface) ---
    def read_neuron(path, top_k=-1):
        with open(path, 'r') as f:
            raw = json.load(f)
        out = {}
        for layer, proj_map in raw.items():
            out[layer] = {}
            for proj, idx_list in proj_map.items():
                if not isinstance(idx_list, (list, tuple)):
                    idx_list = [idx_list]
                if top_k is not None and top_k > 0:
                    idx_list = list(idx_list)[:top_k]
                out[layer][proj] = set(idx_list)
        return out

    activate_neuron = read_neuron(neuron_path, top_k=-1) if neuron_path else None

    # --- FSDP config: choose wrap classes by model type; fallback to size-based ---
    model_type = getattr(getattr(model, "config", None), "model_type", None)

    fsdp_layer_map = {
        "llama": ["LlamaDecoderLayer"],
        "mistral": ["MistralDecoderLayer"],
        "mixtral": ["MixtralDecoderLayer"],
        "qwen2": ["Qwen2DecoderLayer"],
        "qwen2_moe": ["Qwen2MoeDecoderLayer"],
        "gemma": ["GemmaDecoderLayer"],
        "gemma2": ["Gemma2DecoderLayer"],
    }

    if model_type in fsdp_layer_map:
        # ✅ Class-name strategy: do not set min_num_params (conflicts with transformer_layer_cls_to_wrap)

        fsdp_cfg = {
            "fsdp_transformer_layer_cls_to_wrap": fsdp_layer_map[model_type],
            "xla": False,
            "use_orig_params": True,
            "forward_prefetch": True,
            "limit_all_gathers": True,
        }
        print(f"🧩 Using class-based FSDP wrap for model_type={model_type}: {fsdp_cfg['fsdp_transformer_layer_cls_to_wrap']}")
        fsdp_arg = "full_shard auto_wrap"
    else:
        # ✅ Fallback: size-based strategy; set min_num_params only, no transformer_layer_cls_to_wrap
        fsdp_cfg = {
            "xla": False,
            "min_num_params": int(5e6),  # Tune up/down based on memory
            "use_orig_params": True,
            "forward_prefetch": True,
            "limit_all_gathers": True,
        }
        print(f"🧩 Fallback to size-based FSDP auto-wrap (model_type={model_type}), min_num_params={fsdp_cfg['min_num_params']}")
        fsdp_arg = "full_shard auto_wrap"

    # === Set response_template based on model type (Llama vs Qwen2.5) ===
    try:
        if isinstance(model_type, str) and model_type.startswith("qwen2"):
            # Qwen2/2.5 ChatML template: generation starts with assistant header
            assistant_hdr = "<|im_start|>assistant\n"
        else:
            # Default to Llama3 template
            assistant_hdr = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    except Exception:
        # Fallback: keep Llama3 template
        assistant_hdr = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
#     # === Bootstrap "assistant segment start" template from tokenizer (strict consistency) ===
#     assistant_hdr = tokenizer.apply_chat_template(
#     [{"role": "assistant", "content": ""}],
#     add_generation_prompt=False,
#     tokenize=False,
# )


    # --- TrainingArguments (FSDP enabled, DeepSpeed disabled) ---
    training_args = transformers.TrainingArguments(
        run_name=run_name,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=final_output_dir,
        save_total_limit=1,               # ✅ Keep at most 1 checkpoint
        load_best_model_at_end=True,      # ✅ Auto-load best model at training end
        metric_for_best_model="eval_loss",# ✅ Choose best by eval loss (or your custom metric)
        greater_is_better=False,   
      
        ddp_find_unused_parameters=False if is_dist else False,
        group_by_length=group_by_length,
        report_to=['wandb'],           # To disable: ['none']
        save_only_model=False,             # ✅ Compatible with load_best_model_at_end
        fsdp=fsdp_arg,                 # ✅ Enable FSDP
        fsdp_config=fsdp_cfg,          # ✅ FSDP config
    )
    if not hasattr(training_args, "use_ipex"):
        training_args.use_ipex = False
    # Compatible with custom Trainer extended fields
    training_args.tp_size = 1
    training_args.activate_neuron = activate_neuron
    training_args.neuron_update_strategy = neuron_update_strategy

    # --- Collator: build labels mask automatically based on response_template ---
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=assistant_hdr,
        instruction_template=None,
    )

    # --- Optional: include terminator tokens (<|eot_id|>/eos) in loss computation ---
    try:
        terminator_ids = []
        # eot_id (e.g., Llama3 <|eot_id|>)
        try:
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        except Exception:
            eot_id = None
        if isinstance(eot_id, int) and eot_id >= 0:
            terminator_ids.append(eot_id)
        # eos_token_id
        if hasattr(tokenizer, "eos_token_id") and isinstance(tokenizer.eos_token_id, int) and tokenizer.eos_token_id >= 0:
            terminator_ids.append(tokenizer.eos_token_id)
        # ChatML terminator (commonly used by Qwen2/2.5)
        try:
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            im_end_id = None
        if isinstance(im_end_id, int) and im_end_id >= 0:
            terminator_ids.append(im_end_id)

        class IncludeTerminatorInLossCollator:
            def __init__(self, base_collator, term_ids, ignore_index=-100):
                self.base_collator = base_collator
                self.term_ids = set([tid for tid in term_ids if isinstance(tid, int)])
                self.ignore_index = ignore_index

            def __call__(self, features):
                batch = self.base_collator(features)
                try:
                    labels = batch["labels"]
                    input_ids = batch["input_ids"]
                    attention_mask = batch.get("attention_mask", None)
                    if isinstance(labels, torch.Tensor) and len(self.term_ids) > 0:
                        batch_size, seq_len = labels.shape
                        for b in range(batch_size):
                            # Candidate: is terminator, original label is ignore, and not padding position
                            cand = torch.zeros(seq_len, dtype=torch.bool, device=labels.device)
                            for tid in self.term_ids:
                                cand |= (input_ids[b] == tid)
                            cand &= (labels[b] == self.ignore_index)
                            if isinstance(attention_mask, torch.Tensor):
                                cand &= (attention_mask[b] == 1)
                            if not cand.any():
                                continue
                            # Prefer terminator right after supervised span (previous token participates in loss)
                            prev_supervised = torch.zeros(seq_len, dtype=torch.bool, device=labels.device)
                            prev_supervised[1:] = (labels[b][:-1] != self.ignore_index)
                            cand_pref = cand & prev_supervised
                            idxs = torch.nonzero(cand_pref, as_tuple=False).view(-1)
                            if idxs.numel() == 0:
                                # Fallback: choose the last candidate terminator
                                idxs = torch.nonzero(cand, as_tuple=False).view(-1)
                            last_idx = int(idxs[-1].item())
                            labels[b, last_idx] = input_ids[b, last_idx]
                        batch["labels"] = labels
                except Exception as e:
                    print(f"⚠️ Failed to include terminator in loss: {e}")
                return batch

        if len(terminator_ids) > 0:
            data_collator = IncludeTerminatorInLossCollator(data_collator, terminator_ids)
            if is_main_process:
                print(f"✅ Terminators will be included in loss: terminator_ids={terminator_ids}")
    except Exception as e:
        print(f"⚠️ Unable to enable terminators in loss: {e}")

    # --- Debug: build a small batch via collator and print input_ids/labels/decoded text and loss-mask text ---
    if is_main_process:
        try:
            sample_count = min(2, len(train_data)) if hasattr(train_data, "__len__") else 0
            if sample_count > 0:
                # Keep only fields needed by collator to avoid tensorization failure on string fields
                features = []
                for i in range(sample_count):
                    item = train_data[i]
                    feat = {k: item[k] for k in ("input_ids", "attention_mask") if k in item}
                    features.append(feat)
                batch_dbg = data_collator(features)
                input_ids_0_full = batch_dbg["input_ids"][0].tolist()
                labels_0_full = batch_dbg["labels"][0].tolist()
                attn_0 = batch_dbg.get("attention_mask")[0].tolist() if "attention_mask" in batch_dbg else [1] * len(input_ids_0_full)
                # Remove left padding before printing to avoid lots of pad(eos/eot) noise
                try:
                    first_token_idx = next(i for i, a in enumerate(attn_0) if a == 1)
                except StopIteration:
                    first_token_idx = 0
                left_pad = first_token_idx
                input_ids_0 = input_ids_0_full[first_token_idx:]
                labels_0 = labels_0_full[first_token_idx:]
                decoded_0 = tokenizer.decode(input_ids_0, skip_special_tokens=False)
                print("—— collator output example (item #1):")
                print(f"left_pad_removed={left_pad}")
                print("input_ids:", input_ids_0)
                print("labels:", labels_0)
                print("decoded:")
                print(decoded_0)
                # Tokens participating in loss and their text
                loss_token_ids_0 = [tid for tid, lab in zip(input_ids_0, labels_0) if lab != -100]
                if len(loss_token_ids_0) > 0:
                    loss_text_0 = tokenizer.decode(loss_token_ids_0, skip_special_tokens=False)
                else:
                    loss_text_0 = ""
                print("loss_token_ids:", loss_token_ids_0)
                print("loss_text:")
                print(loss_text_0)
        except Exception as e:
            print(f"⚠️ Failed to debug-print collator batch: {e}")

    # --- Register training loss logging callback ---
    class LossPrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            try:
                rank_str = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
                is_main = str(rank_str) == "0"
            except Exception:
                is_main = True
            if not is_main:
                return
            loss_val = logs.get("loss")
            lr_val = logs.get("learning_rate")
            if loss_val is not None:
                try:
                    print(f"🧾 step={state.global_step} | loss={float(loss_val):.6f} | lr={lr_val}")
                except Exception:
                    print(f"🧾 step={state.global_step} | loss={loss_val} | lr={lr_val}")

    class TokenizerSaverCallback(TrainerCallback):
        def __init__(self, tok):
            self.tokenizer = tok

        def on_save(self, args, state, control, **kwargs):
            try:
                rank_str = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
                is_main = str(rank_str) == "0"
            except Exception:
                is_main = True
            if not is_main:
                return
            try:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.tokenizer.save_pretrained(ckpt_dir)
                print(f"💾 Wrote tokenizer to: {ckpt_dir}")
            except Exception as e:
                print(f"⚠️ Failed to save tokenizer to checkpoint: {e}")

    # Callback to keep only the best checkpoint (delete others)
    class BestOnlyCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            try:
                best_ckpt = getattr(state, "best_model_checkpoint", None)
                if not best_ckpt or not isinstance(best_ckpt, str):
                    return
                out_dir = args.output_dir
                if not os.path.isdir(out_dir):
                    return
                # Iterate and delete checkpoint-* directories except the best one
                for name in os.listdir(out_dir):
                    if not name.startswith("checkpoint-"):
                        continue
                    path = os.path.join(out_dir, name)
                    # Skip best checkpoint
                    if os.path.abspath(path) == os.path.abspath(best_ckpt):
                        continue
                    try:
                        import shutil
                        shutil.rmtree(path, ignore_errors=True)
                    except Exception:
                        pass
            except Exception:
                # Fail silently to avoid impacting training
                pass

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=training_args,
        data_collator=data_collator,
    )

    try:
        trainer.add_callback(LossPrinterCallback())
        trainer.add_callback(TokenizerSaverCallback(tokenizer))
        # Enable "best-only checkpoint" strategy with save_total_limit=1 and load_best_model_at_end=True
        trainer.add_callback(BestOnlyCheckpointCallback())
    except Exception as e:
        print(f"⚠️ Unable to register loss printing callback: {e}")

    # --- Training ---
    print("Starting training with FSDP...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Training completed successfully!")

    # --- Cleanup training artifacts to save disk space ---
    if is_main_process_save:
        try:
            import glob
            import shutil
            
            print("🧹 Starting cleanup of training artifacts...")
            cleanup_patterns = [
                "optimizer.bin",
                "optimizer.pt",
                "pytorch_model_fsdp.bin",
                "scheduler.pt",
                "rng_state_*.pth",
                "trainer_state.json",
                "training_args.bin",
            ]
            
            # Iterate over all checkpoint directories under output dir
            for ckpt_dir in glob.glob(os.path.join(final_output_dir, "checkpoint-*")):
                if not os.path.isdir(ckpt_dir):
                    continue
                    
                print(f"  Cleaning directory: {ckpt_dir}")
                total_freed = 0
                
                for pattern in cleanup_patterns:
                    if "*" in pattern:
                        # Handle glob patterns (e.g., rng_state_*.pth)
                        files_to_delete = glob.glob(os.path.join(ckpt_dir, pattern))
                    else:
                        # Handle specific filenames
                        files_to_delete = [os.path.join(ckpt_dir, pattern)]
                    
                    for file_path in files_to_delete:
                        if os.path.isfile(file_path):
                            try:
                                file_size = os.path.getsize(file_path) / (1024**3)  # Convert to GB
                                os.remove(file_path)
                                total_freed += file_size
                                print(f"    ✓ Deleted: {os.path.basename(file_path)} ({file_size:.2f}GB)")
                            except Exception as e:
                                print(f"    ✗ Failed to delete {os.path.basename(file_path)}: {e}")
                
                print(f"  📦 {os.path.basename(ckpt_dir)} freed space: {total_freed:.2f}GB")
            
            print("✅ Training artifacts cleanup complete!")
        except Exception as e:
            print(f"⚠️ Error cleaning training files (does not affect training results): {e}")


if __name__ == "__main__":
    import fire
    fire.Fire(train)
