# Modified standard library, supports neuron activation value recording
# Only import models that work properly, avoid import errors

# Try importing each model, skip if fails
try:
    from .modeling_llama import LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel
    _has_llama = True
except ImportError:
    _has_llama = False

try:
    from .modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model, Qwen2PreTrainedModel
    _has_qwen2 = True
except ImportError:
    _has_qwen2 = False

try:
    from .modeling_gemma3 import Gemma3ForCausalLM, Gemma3Model, Gemma3PreTrainedModel
    _has_gemma3 = True
except ImportError:
    _has_gemma3 = False

try:
    from .modeling_mixtral import MixtralForCausalLM, MixtralModel, MixtralPreTrainedModel
    _has_mixtral = True
except ImportError:
    _has_mixtral = False

try:
    from .modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel, Qwen3MoePreTrainedModel
    _has_qwen3_moe = True
except ImportError:
    _has_qwen3_moe = False

try:
    from .modeling_gpt_oss import GptOssForCausalLM, GptOssModel, GptOssPreTrainedModel
    _has_gpt_oss = True
except ImportError:
    _has_gpt_oss = False

# Build list of available models
__all__ = []

if _has_llama:
    __all__.extend(["LlamaForCausalLM", "LlamaModel", "LlamaPreTrainedModel"])

if _has_qwen2:
    __all__.extend(["Qwen2ForCausalLM", "Qwen2Model", "Qwen2PreTrainedModel"])

if _has_gemma3:
    __all__.extend(["Gemma3ForCausalLM", "Gemma3Model", "Gemma3PreTrainedModel"])

if _has_mixtral:
    __all__.extend(["MixtralForCausalLM", "MixtralModel", "MixtralPreTrainedModel"])

if _has_qwen3_moe:
    __all__.extend(["Qwen3MoeForCausalLM", "Qwen3MoeModel", "Qwen3MoePreTrainedModel"])

if _has_gpt_oss:
    __all__.extend(["GptOssForCausalLM", "GptOssModel", "GptOssPreTrainedModel"])
