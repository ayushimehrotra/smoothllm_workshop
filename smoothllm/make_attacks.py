import json

import nanogcg
from nanogcg import GCGConfig

from smoothllm.prompt import Prompt


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

    def create_prompt(self, goal, target, config):
        """Create GCG prompt and run the attack."""

        control = nanogcg.run(
            self.target_model.model, self.target_model.tokenizer, goal, target, config
        )

        return control.best_string
