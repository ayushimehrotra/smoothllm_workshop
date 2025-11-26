import json
import pandas as pd
import nanogcg

from nanogcg import GCGConfig

import smoothllm.lib.perturbations as perturbations


class Prompt:
    def __init__(self, full_prompt, perturbable_prompt, max_new_tokens):
        self.full_prompt = full_prompt
        self.perturbable_prompt = perturbable_prompt
        self.max_new_tokens = max_new_tokens

    def perturb(self, perturbation_fn):
        perturbed_prompt = perturbation_fn(self.perturbable_prompt)
        self.full_prompt = self.full_prompt.replace(
            self.perturbable_prompt, perturbed_prompt
        )
        self.perturbable_prompt = perturbed_prompt


class Attack:
    def __init__(self, logfile, target_model):
        self.logfile = logfile
        self.target_model = target_model


class GCG(Attack):
    """Greedy Coordinate Gradient attack.

    Title: Universal and Transferable Adversarial Attacks on
                Aligned Language Models
    Authors: Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson
    Paper: https://arxiv.org/abs/2307.15043
    """

    def __init__(self, logfile, target_model, start_index, end_index):
        super(GCG, self).__init__(logfile, target_model)

        # Load the log file
        with open(self.logfile, "r") as f:
            self.log = json.load(f)

        self.goals = self.log["goal"][start_index:end_index]
        self.targets = self.log["target"][start_index:end_index]

        config = GCGConfig(
            num_steps=500, search_width=64, topk=64, seed=42, verbosity="WARNING"
        )

        for g, t in zip(self.goals, self.targets):
            control_attack_result = self.create_prompt(g, t, config)
            self.log["control"].append(control_attack_result)

            with open(self.logfile, "w") as f:
                json.dump(self.log, f, indent=4)

    def create_prompt(
        self, goal, target, config, max_new_len=100
    ):  # Add config as an argument
        """Create GCG prompt and run the attack."""

        max_new_tokens = max(
            len(self.target_model.tokenizer(target).input_ids) + 2, max_new_len
        )

        control = nanogcg.run(
            self.target_model.model, self.target_model.tokenizer, goal, target, config
        )

        return control.best_string
