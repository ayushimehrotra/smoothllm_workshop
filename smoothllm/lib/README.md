# Attack Log Files

This directory should contain the attack log files used by the experiment scripts.

## Required Files

| File | Format | Attack | Model | Description |
|------|--------|--------|-------|-------------|
| `llmattacks_vicuna.json` | JSON | GCG | Vicuna-13B | GCG control strings for Vicuna |
| `llmattacks_llama2.json` | JSON | GCG | Llama-2-7B | GCG control strings for Llama-2 |
| `pair_vicuna.pkl` | Pickle | PAIR | Vicuna-13B | PAIR jailbreak prompts for Vicuna |
| `pair_llama2.pkl` | Pickle | PAIR | Llama-2-7B | PAIR jailbreak prompts for Llama-2 |

## JSON Format (GCG)

```json
{
    "goal": ["goal_1", "goal_2", ...],
    "target": ["target_1", "target_2", ...],
    "control": ["control_string_1", "control_string_2", ...]
}
```

- `goal`: The adversarial objective strings.
- `target`: The desired model outputs.
- `control`: The optimized adversarial suffixes (produced by GCG / NanoGCG).

## Pickle Format (PAIR)

A pandas DataFrame with at least a `jailbreak_prompt` column containing the
jailbreak prompt strings.

## How to Obtain

1. **GCG logs**: Run `run_attacks.py` or `make_attacks.py` with NanoGCG to
   generate control strings, or use logs from the original
   [llm-attacks](https://github.com/llm-attacks/llm-attacks) repository.
2. **PAIR logs**: Generate using the
   [PAIR](https://github.com/patrickrchao/JailbreakingLLMs) codebase and
   save the resulting DataFrame as a pickle file.
