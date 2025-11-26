import argparse
import inspect
import json
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import smoothllm.experiment_attacks as attacks
import smoothllm.experiment_defenses as defenses
import smoothllm.language_models as language_models
import smoothllm.model_configs as model_configs


torch.cuda.empty_cache()


def build_attack(attack_name, logfile, target_model, pert_type, pert_pct):
    attack_cls = getattr(attacks, attack_name)
    attack_signature = inspect.signature(attack_cls.__init__)

    attack_kwargs = {
        "logfile": logfile,
        "target_model": target_model,
    }

    if "pert_type" in attack_signature.parameters:
        attack_kwargs["pert_type"] = pert_type
    if "pert_pct" in attack_signature.parameters:
        attack_kwargs["pert_pct"] = pert_pct

    return attack_cls(**attack_kwargs)


def agresti_coull_interval(successes: int, total: int, z_value: float) -> tuple:
    """Compute Agresti-Coull interval for a binomial proportion."""

    if total == 0:
        return 0.0, 0.0, 0.0

    adjusted_total = total + z_value**2
    adjusted_rate = (successes + 0.5 * z_value**2) / adjusted_total
    margin = z_value * np.sqrt(adjusted_rate * (1 - adjusted_rate) / adjusted_total)
    return (
        adjusted_rate,
        max(0.0, adjusted_rate - margin),
        min(1.0, adjusted_rate + margin),
    )


def evaluate_attack_success(
    target_model,
    attack_name,
    attack_logfile,
    pert_type,
    k,
    trials,
    max_prompts,
    z_value,
):
    trial_means: List[float] = []
    successes = 0
    total_samples = 0
    samples_per_trial: List[int] = []

    for trial in range(1, trials + 1):
        attack = build_attack(
            attack_name=attack_name,
            logfile=attack_logfile,
            target_model=target_model,
            pert_type=pert_type,
            pert_pct=k,
        )

        defense = defenses.SmoothLLM(
            target_model=target_model,
            pert_type=pert_type,
            pert_pct=k,
        )

        prompts = attack.prompts
        if max_prompts is not None:
            prompts = prompts[:max_prompts]

        jailbreak_results = []
        for prompt in tqdm(
            prompts,
            desc=f"{attack_name} | {pert_type} | k={k} | trial {trial}/{trials}",
        ):
            output = defense(prompt)
            jailbreak_results.append(defense.is_jailbroken(output))

        samples_per_trial.append(len(jailbreak_results))
        successes += int(np.sum(jailbreak_results))
        total_samples += len(jailbreak_results)
        trial_means.append(np.mean(jailbreak_results) if jailbreak_results else 0.0)

    attack_success_mean = successes / total_samples if total_samples else 0.0
    adjusted_rate, lower, upper = agresti_coull_interval(
        successes, total_samples, z_value
    )

    return {
        "attack_name": attack_name,
        "perturbation_type": pert_type,
        "k": k,
        "trials": trials,
        "avg_samples_per_trial": float(np.mean(samples_per_trial))
        if samples_per_trial
        else 0.0,
        "total_samples": total_samples,
        "successful_attacks": successes,
        "attack_success_mean": float(attack_success_mean),
        "attack_success_std": float(np.std(trial_means)),
        "agresti_coull_center": float(adjusted_rate),
        "agresti_coull_low": float(lower),
        "agresti_coull_high": float(upper),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate attack success across perturbation budgets k for different "
            "perturbation strategies and attack types."
        )
    )
    parser.add_argument(
        "--target_model",
        default="vicuna",
        help="Model key defined in smoothllm/model_configs.py",
    )
    parser.add_argument(
        "--attack_names",
        nargs="+",
        help="Attack class names from smoothllm.experiment_attacks",
    )
    parser.add_argument(
        "--attack_logfiles",
        nargs="+",
        help="Log files aligned to --attack_names (broadcast if length is 1)",
    )
    parser.add_argument(
        "--attack_name",
        help="Deprecated: single attack name; prefer --attack_names",
    )
    parser.add_argument(
        "--attack_logfile",
        help="Deprecated: single logfile; prefer --attack_logfiles",
    )
    parser.add_argument(
        "--perturbation_types",
        nargs="+",
        default=["RandomPatchPerturbation"],
        help="Perturbation strategies to evaluate",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=list(range(0, 11)),
        help="Perturbation budgets k to sweep",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of times to rerun each k for averaging",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=500,
        help="Maximum number of prompts to evaluate per attack (None uses all)",
    )
    parser.add_argument(
        "--confidence_z",
        type=float,
        default=1.96,
        help="Z-score for Agresti-Coull confidence interval (default 95% CI)",
    )
    parser.add_argument(
        "--output",
        default="k_curve_results.csv",
        help="CSV file to store aggregated results",
    )

    args = parser.parse_args()

    attacks_to_run = args.attack_names or (
        [] if args.attack_name is None else [args.attack_name]
    )
    logfiles_to_use = args.attack_logfiles or (
        [] if args.attack_logfile is None else [args.attack_logfile]
    )

    if not attacks_to_run:
        parser.error(
            "Please supply at least one attack via --attack_names or --attack_name."
        )

    if not logfiles_to_use:
        parser.error(
            "Please supply attack logfiles via --attack_logfiles or --attack_logfile."
        )

    if len(logfiles_to_use) == 1 and len(attacks_to_run) > 1:
        logfiles_to_use = logfiles_to_use * len(attacks_to_run)

    if len(attacks_to_run) != len(logfiles_to_use):
        parser.error("Number of attack logfiles must match the number of attacks.")

    config = model_configs.MODELS[args.target_model]
    target_model = language_models.LLM(
        model_path=config["model_path"],
        tokenizer_path=config["tokenizer_path"],
        conv_template_name=config["conversation_template"],
        device="cuda:0",
    )

    results = []
    for attack_name, attack_logfile in zip(attacks_to_run, logfiles_to_use):
        for pert_type in args.perturbation_types:
            for k in args.k_values:
                result = evaluate_attack_success(
                    target_model=target_model,
                    attack_name=attack_name,
                    attack_logfile=attack_logfile,
                    pert_type=pert_type,
                    k=k,
                    trials=args.trials,
                    max_prompts=args.max_prompts,
                    z_value=args.confidence_z,
                )
                results.append(result)
                print(
                    json.dumps(
                        {
                            "attack": attack_name,
                            "perturbation": pert_type,
                            "k": k,
                            "attack_success_mean": result["attack_success_mean"],
                            "agresti_coull_low": result["agresti_coull_low"],
                            "agresti_coull_high": result["agresti_coull_high"],
                        }
                    )
                )

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
