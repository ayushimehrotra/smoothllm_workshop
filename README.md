# Towards Realistic Guarantees: A Probabilistic Certificate for SmoothLLM

Code accompanying the paper **"Towards Realistic Guarantees: A Probabilistic Certificate for SmoothLLM"**. The repository provides scripts for reproducing adversarial attack generation and evaluating the SmoothLLM defense against jailbreaking prompts.

## Repository structure

- `smoothllm/` – core library code
  - `language_models.py` – thin wrapper around Hugging Face models with FastChat conversation templates.
  - `model_configs.py` – local paths and templates for supported models (Llama-2-7b-chat, Vicuna-13B). Update these paths to match your environment.
  - `perturbations.py` – random perturbation primitives (swap, patch, insert) used by SmoothLLM.
  - `defenses.py` / `experiment_defenses.py` – SmoothLLM defense implementations for primary and experimental setups.
  - `attacks.py` / `experiment_attacks.py` – loaders for logged attack prompts and perturbation-aware variants.
  - `make_attacks.py` – runs new GCG attacks with NanoGCG to extend the control strings in the provided logs.
- `run_attacks.py` – example script to regenerate GCG attack controls for Vicuna.
- `experiment_k.py` – example experiment measuring SmoothLLM robustness across perturbation settings.
- `llm-attacks/` – submodule containing upstream attack assets (see submodule docs for details).

## Requirements

- Python 3.10+
- CUDA-capable GPU if running generation locally.
- Installed Python packages: `torch`, `transformers`, `fastchat`, `pandas`, `numpy`, `tqdm`, `nanogcg`.

Install dependencies into your environment:

```bash
pip install torch transformers fastchat pandas numpy tqdm nanogcg
```

> Note: Model downloads are expected to be available locally. Set `local_files_only=True` is used in `language_models.LLM`, so ensure weights and tokenizers are already present at the configured paths.

## Model setup

The default model locations are configured in `smoothllm/model_configs.py`. Update the `model_path` and `tokenizer_path` entries to point to your local checkpoints before running any scripts. Conversation templates should align with the chosen model (e.g., `llama-2`, `vicuna`).

## Running the included scripts

### 1. Generate or extend GCG attack logs

`run_attacks.py` launches NanoGCG to produce control strings for the GCG attack and appends them to the provided log file (`smoothllm/lib/llmattacks_vicuna.json` by default).

```bash
python run_attacks.py
```

Key options inside the script:
- `target_model`: selects the model configuration key from `model_configs.MODELS`.
- `attack_logfile`: path to the JSON log that will be updated with new controls.
- `start_index` / `end_index`: slice of goal/target pairs to attack.

### 2. Evaluate SmoothLLM robustness across k

`experiment_k.py` instantiates a target model, wraps it with the SmoothLLM defense, and measures jailbreak success under varying perturbation budgets for one or more attack types.

Typical invocation mirroring the paper sweeps:

```bash
python experiment_k.py \
  --target_model vicuna \
  --attack_names GCG PAIR \
  --attack_logfiles smoothllm/lib/llmattacks_vicuna.json smoothllm/lib/pair_vicuna.pkl \
  --perturbation_types RandomPatchPerturbation RandomSwapPerturbation \
  --k_values 0 1 2 3 4 5 6 7 8 9 10 \
  --trials 3 \
  --max_prompts 500 \
  --output data/vicuna_attack_success.csv
```

Notable CLI options:
- `--attack_names` / `--attack_logfiles`: aligned lists of attack loaders (e.g., `GCG`, `PAIR`) and their logs.
- `--perturbation_types`: perturbation strategies to sweep (e.g., `RandomPatchPerturbation`, `RandomSwapPerturbation`).
- `--k_values`: perturbation budgets to evaluate.
- `--trials`: number of repeated runs for averaging and confidence intervals.
- `--max_prompts`: optional cap on prompts taken from each attack log.
- `--confidence_z`: z-score used for Agresti–Coull confidence intervals (default 1.96 for 95% CI).

Each run saves a CSV with per-attack, per-perturbation metrics including the mean jailbreak rate and Agresti–Coull bounds.

### 3. Reproduce paper figures

`generate_k_data.sh` orchestrates the Vicuna and Llama-2 sweeps used for Figures 5–10. Update the logfile paths in the script to match your environment, then run:

```bash
bash generate_k_data.sh
```

The script writes CSV outputs under `data/` for the Vicuna (RandomPatch + RandomSwap) and Llama-2 (RandomPatch) experiments across both GCG and PAIR attacks.

## Using the library components

You can also import the components directly to build custom experiments:

```python
from smoothllm.language_models import LLM
from smoothllm.defenses import SmoothLLM
from smoothllm.attacks import GCG, PAIR
from smoothllm.model_configs import MODELS

config = MODELS['vicuna']
model = LLM(
    model_path=config['model_path'],
    tokenizer_path=config['tokenizer_path'],
    conv_template_name=config['conversation_template'],
    device='cuda:0'
)
defense = SmoothLLM(target_model=model, pert_type='RandomSwapPerturbation', pert_pct=3, num_copies=64)
attack = GCG(logfile='smoothllm/lib/llmattacks_vicuna.json', target_model=model)
```

The `SmoothLLM` defense perturbs the attack prompt multiple times, generates responses in batches, and returns a majority-vote response that aims to reject jailbreaks.

## Dataset and logs

Example attack logs live under `smoothllm/lib/` (not tracked here). Place the appropriate JSON (for GCG) or pickle (for PAIR) logs in that directory when running the scripts. The code expects fields like `goal`, `target`, and `control` (for GCG) or `jailbreak_prompt` (for PAIR).

## Citation

If you use this codebase, please cite the accompanying paper:

> Kumarappan, A., Mehrotra, A. (2024). *Towards Realistic Guarantees: A Probabilistic Certificate for SmoothLLM*.
