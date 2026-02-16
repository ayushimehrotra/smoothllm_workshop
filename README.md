# Towards Realistic Guarantees: A Probabilistic Certificate for SmoothLLM

Code accompanying the paper **"Towards Realistic Guarantees: A Probabilistic Certificate for SmoothLLM"** ([arXiv:2511.18721](https://arxiv.org/abs/2511.18721)).

This repository provides:
- The probabilistic certificate (DSP computation, alpha bounds from Propositions 1 & 2)
- Exponential ASR curve fitting and sensitivity analysis (Section 3.7)
- Scripts for reproducing all paper figures and experimental results
- Attack generation and SmoothLLM defense evaluation

## Quick start (no GPU needed)

The theoretical components — certificate computation, curve fitting, sensitivity analysis, and figure generation — run on CPU with no model weights required.

```bash
pip install -r requirements.txt

# Reproduce Figures 1, 2, and 4
python generate_figures.py

# Run the full verification suite (178 tests)
python tests/test_paper_verification.py
```

### Compute the probabilistic certificate

```python
from smoothllm.certificate import compute_dsp, alpha_lower_bound
from smoothllm.curve_fitting import fit_asr_curve, make_asr_func, find_k_for_epsilon

# Fit ASR curve from experiment data
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

## Repository structure

```
smoothllm/                    Core library
  __init__.py                 Package init with public API exports
  certificate.py              DSP, hypergeometric PMF, alpha bounds (Propositions 1 & 2)
  curve_fitting.py            Exponential ASR model: ASR(k) = a * exp(-b*k) + c
  sensitivity.py              Sensitivity analysis + Section 3.7 parameter selection
  plotting.py                 Figure generation (ASR vs k, DSP vs N)
  prompt.py                   Shared Prompt class
  perturbations.py            Random perturbation primitives (swap, patch, insert)
  language_models.py           Hugging Face model wrapper (CUDA/MPS/CPU)
  model_configs.py            Model paths and templates (env var configurable)
  defenses.py                 SmoothLLM defense (N copies + majority vote)
  attacks.py                  Attack loaders from logged prompts
  experiment_attacks.py       Perturbation-aware attack variants for k experiments
  experiment_defenses.py      Simplified defense for single-pass k evaluation
  make_attacks.py             NanoGCG attack generation
  lib/                        Attack log files (see lib/README.md)

experiment_k.py               Evaluate SmoothLLM robustness across k values
run_attacks.py                Generate GCG attack control strings
generate_k_data.sh            Orchestrate full experiment sweeps
generate_figures.py           Reproduce all paper figures (1-2, 4-10)
tests/
  test_paper_verification.py  178-test suite verifying all paper formulas
```

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended for model inference (MPS and CPU also supported)
- No GPU needed for certificate computation, curve fitting, or figure generation

```bash
pip install -r requirements.txt
```

## Reproducing paper figures

```bash
# Figures 1, 2, 4 — from paper's fitted parameters (no GPU needed)
python generate_figures.py

# Figures 5-10 — require experimental data from GPU experiments
bash generate_k_data.sh
python generate_figures.py --data-dir data/

# Generate a specific figure
python generate_figures.py --figures 4
```

| Figure | Content | Requirements |
|--------|---------|-------------|
| 1 | Llama2 + RandomPatch + GCG: ASR vs k | None (uses paper params) |
| 2 | Llama2 + RandomSwap + GCG: ASR vs k | None (uses paper params) |
| 3 | Conceptual pipeline diagram | Not code-generated |
| 4 | Certified DSP vs N | None (pure math) |
| 5-10 | Additional model/attack/perturbation combos | GPU + experimental CSVs |

## Running experiments (GPU required)

### Model setup

Set model paths via environment variables:

```bash
export SMOOTHLLM_LLAMA2_PATH="/path/to/Llama-2-7b-chat-hf"
export SMOOTHLLM_VICUNA_PATH="/path/to/vicuna-13b-v1.5"
```

If not set, the code falls back to looking for model directories inside `smoothllm/`.

> Note: `local_files_only=True` is used in `language_models.LLM`, so ensure weights and tokenizers are already present at the configured paths.

### 1. Generate or extend GCG attack logs

```bash
python run_attacks.py
```

### 2. Evaluate SmoothLLM robustness across k

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

### 3. Run all experiment sweeps

```bash
bash generate_k_data.sh
```

## Using the library

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

## Testing

The verification suite checks every formula, numerical constant, and claim from the paper:

```bash
python tests/test_paper_verification.py
```

This runs 178 tests covering:
- Hypergeometric PMF (Eq. 6)
- Alpha bounds (Propositions 1 & 2, Eqs. 2, 10, 11)
- DSP computation (Eqs. 1, 5)
- Exponential ASR model fitting
- Section 3.6/3.7 case study values
- Figure 4 parameters and threshold crossings
- Patch overlap PMF (Appendix D)
- Sensitivity analysis (Appendix C, Eqs. 17-18)
- Figure generation and parameter correctness
- Full independent cross-check of the DSP pipeline

## Dataset and logs

Attack logs should be placed under `smoothllm/lib/`. See `smoothllm/lib/README.md` for expected formats.

## Citation

If you use this codebase, please cite the accompanying paper ([arXiv:2511.18721](https://arxiv.org/abs/2511.18721)):

> Kumarappan, A., Mehrotra, A. (2024). *Towards Realistic Guarantees: A Probabilistic Certificate for SmoothLLM*.
