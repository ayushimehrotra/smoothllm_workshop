#!/usr/bin/env bash
set -euo pipefail

# Generate attack success data across perturbation budgets (k) for the
# experiments shown in Figures 5–10 of the SmoothLLM paper. Adjust the
# paths for the attack logfiles to match your local environment.

# Shared settings
K_VALUES=(0 1 2 3 4 5 6 7 8 9 10)
TRIALS=3
MAX_PROMPTS=500
OUTPUT_DIR="data"
mkdir -p "$OUTPUT_DIR"

# Attack log files (update these to point to your downloaded artifacts)
VICUNA_GCG_LOG="smoothllm/lib/llmattacks_vicuna.json"
VICUNA_PAIR_LOG="smoothllm/lib/pair_vicuna.pkl"
LLAMA2_GCG_LOG="smoothllm/lib/llmattacks_llama2.json"
LLAMA2_PAIR_LOG="smoothllm/lib/pair_llama2.pkl"

echo "Starting Vicuna sweeps (RandomPatch + RandomSwap across GCG and PAIR)"
python experiment_k.py \
  --target_model vicuna \
  --attack_names GCG PAIR \
  --attack_logfiles "$VICUNA_GCG_LOG" "$VICUNA_PAIR_LOG" \
  --perturbation_types RandomPatchPerturbation RandomSwapPerturbation \
  --k_values "${K_VALUES[@]}" \
  --trials "$TRIALS" \
  --max_prompts "$MAX_PROMPTS" \
  --output "$OUTPUT_DIR/vicuna_attack_success.csv"

echo "Starting Llama2 sweeps (RandomPatch across GCG and PAIR)"
python experiment_k.py \
  --target_model llama2 \
  --attack_names GCG PAIR \
  --attack_logfiles "$LLAMA2_GCG_LOG" "$LLAMA2_PAIR_LOG" \
  --perturbation_types RandomPatchPerturbation \
  --k_values "${K_VALUES[@]}" \
  --trials "$TRIALS" \
  --max_prompts "$MAX_PROMPTS" \
  --output "$OUTPUT_DIR/llama2_patch_attack_success.csv"

echo "Starting Llama2 sweeps (RandomSwap across GCG and PAIR)"
python experiment_k.py \
  --target_model llama2 \
  --attack_names GCG PAIR \
  --attack_logfiles "$LLAMA2_GCG_LOG" "$LLAMA2_PAIR_LOG" \
  --perturbation_types RandomSwapPerturbation \
  --k_values "${K_VALUES[@]}" \
  --trials "$TRIALS" \
  --max_prompts "$MAX_PROMPTS" \
  --output "$OUTPUT_DIR/llama2_swap_attack_success.csv"

echo "Attack success data saved under $OUTPUT_DIR/"
