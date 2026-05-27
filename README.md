## Workflow for “Tiny Brains, Giant Impact: Uncovering the Keystone Neurons of LLM with Just a Few Prompts”

### Step 1: Record Neuron Activations
Run models with prompts and record neuron activations.

**Script**: `test_unified_detection.py`

**Output**: `token_data_*.json` files containing raw activation data per token.

### Step 2: Calculate Mean Activations
Process raw activation data to calculate mean activations per neuron per layer.

**Script**: `process_neuron_mean_activation.py`

**Input**: `token_data_*.json` files  
**Output**: `activation_means_*.json` files

### Step 3: Analyze Neuron Overlaps
Identify top-N activated neurons and analyze overlaps across different prompts.

**Script**: `batch_top_n_overlap_unified.py`

**Input**: `activation_means_*.json` files  
**Output**: JSON files containing top-N neurons and overlap statistics

### Step 4: Modify Model
Modify the model by amplifying or deactivating identified neurons.

**Amplify neurons**: `scaling.py`  
**Deactivate neurons**: `deactivation.py`

**Input**: Original model + neuron JSON file  
**Output**: Modified model

### Step 5: Supervised Fine-Tuning(targeted fine-tuning)
Fine-tune the modified model using supervised fine-tuning.

**Script**: `train.py`

**Input**: Modified model from Step 4  
**Output**: Fine-tuned model

## Supported Models

- Qwen2, Qwen3, Qwen3-MoE
- Llama
- Mixtral
- Gemma3

