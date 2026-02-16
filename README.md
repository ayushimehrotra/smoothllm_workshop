# Towards Realistic Guarantees: A Probabilistic Certificate for SmoothLLM

Code accompanying the paper **"Towards Realistic Guarantees: A Probabilistic Certificate for SmoothLLM"** ([arXiv:2511.18721](https://arxiv.org/abs/2511.18721)). The repository provides scripts for reproducing adversarial attack generation, evaluating the SmoothLLM defense against jailbreaking prompts, and computing the probabilistic certificate (DSP) from the paper.

## Repository structure

- `smoothllm/` – core library code
  - `__init__.py` – package init with public API exports.
  - `prompt.py` – shared `Prompt` class used across attack/defense modules.
  - `language_models.py` – thin wrapper around Hugging Face models with FastChat conversation templates. Supports CUDA, MPS, and CPU.
  - `model_configs.py` – model paths and templates for supported models (Llama-2-7b-chat, Vicuna-13B). Paths are configurable via environment variables.
  - `perturbations.py` – random perturbation primitives (swap, patch, insert) used by SmoothLLM.
  - `defenses.py` / `experiment_defenses.py` – SmoothLLM defense implementations for primary and experimental setups.
  - `attacks.py` / `experiment_attacks.py` – loaders for logged attack prompts and perturbation-aware variants.
  - `make_attacks.py` – runs new GCG attacks with NanoGCG to extend the control strings in the provided logs.
  - `certificate.py` – DSP computation, hypergeometric PMF, binomial CDF, alpha bounds (Propositions 1 & 2).
  - `curve_fitting.py` – exponential ASR model fitting: `ASR(k) = a * exp(-b*k) + c`.
  - `plotting.py` – figure generation (ASR vs k with CI bands, DSP vs N).
  - `sensitivity.py` – sensitivity analysis and Section 3.7 parameter selection pipeline.
  - `lib/` – directory for attack log files (see `lib/README.md` for details).
- `experiment_k.py` – evaluate SmoothLLM robustness across perturbation budgets k.
- `run_attacks.py` – generate GCG attack control strings.
- `generate_k_data.sh` – orchestrate full experiment sweeps for paper figures.

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended for model inference (MPS and CPU also supported).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model setup

Model paths are configured in `smoothllm/model_configs.py`. Set them via environment variables:

```bash
export SMOOTHLLM_LLAMA2_PATH="/path/to/Llama-2-7b-chat-hf"
export SMOOTHLLM_VICUNA_PATH="/path/to/vicuna-13b-v1.5"
```

If environment variables are not set, the code falls back to looking for model directories inside `smoothllm/`.

> Note: `local_files_only=True` is used in `language_models.LLM`, so ensure weights and tokenizers are already present at the configured paths.

## Running the included scripts

### 1. Generate or extend GCG attack logs

`run_attacks.py` launches NanoGCG to produce control strings for the GCG attack and appends them to the provided log file.

```bash
python run_attacks.py
```

### 2. Evaluate SmoothLLM robustness across k

`experiment_k.py` instantiates a target model, wraps it with the SmoothLLM defense, and measures jailbreak success under varying perturbation budgets.

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

### 3. Reproduce paper figures

`generate_k_data.sh` orchestrates the Vicuna and Llama-2 sweeps used for Figures 5-10:

```bash
bash generate_k_data.sh
```

## Using the library components

### Attack and defense experiments

```python
from smoothllm.language_models import LLM
from smoothllm.model_configs import MODELS

config = MODELS['vicuna']
model = LLM(
    model_path=config['model_path'],
    tokenizer_path=config['tokenizer_path'],
    conv_template_name=config['conversation_template'],
)
```

### Probabilistic certificate (no GPU needed)

```python
from smoothllm.certificate import compute_dsp, alpha_lower_bound
from smoothllm.curve_fitting import fit_asr_curve, make_asr_func, find_k_for_epsilon

# Fit ASR curve from experiment CSV
a, b, c, _ = fit_asr_curve(k_values, asr_values)

# Find minimum k for target epsilon
k = find_k_for_epsilon(a, b, c, epsilon=0.05)

# Compute alpha and DSP
M = round(0.10 * 240)  # 10% perturbation rate on 240-char region
alpha = alpha_lower_bound(k, m=240, m_S=100, M=M, epsilon=0.05)
dsp = compute_dsp(alpha, N=11)
```

### End-to-end parameter selection (Section 3.7)

```python
from smoothllm.sensitivity import parameter_selection_pipeline

result = parameter_selection_pipeline(
    a=0.2921, b=0.3756, c=0.0133,
    m=240, m_S=100, q=0.10,
    epsilon=0.05, target_dsp=0.95,
    perturbation_type='RandomSwapPerturbation'
)
print(f"k={result['k']}, N={result['N']}, DSP={result['dsp']:.4f}")
```

### Plotting

```python
from smoothllm.plotting import plot_asr_vs_k, plot_dsp_vs_N

# ASR vs k with confidence intervals and fitted curve
plot_asr_vs_k("data/vicuna_attack_success.csv",
              attack_name="GCG",
              perturbation_type="RandomSwapPerturbation",
              fit_params=(0.29, 0.38, 0.01),
              save_path="figures/asr_vs_k.png")

# DSP vs N
plot_dsp_vs_N(alpha=0.91, target_dsp=0.95,
              save_path="figures/dsp_vs_N.png")
```

## Dataset and logs

Example attack logs should be placed under `smoothllm/lib/`. See `smoothllm/lib/README.md` for the expected formats and how to obtain them.

## Citation

If you use this codebase, please cite the accompanying paper ([arXiv:2511.18721](https://arxiv.org/abs/2511.18721)):

> Kumarappan, A., Mehrotra, A. (2024). *Towards Realistic Guarantees: A Probabilistic Certificate for SmoothLLM*.
