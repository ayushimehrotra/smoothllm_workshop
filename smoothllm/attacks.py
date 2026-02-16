import json

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

    def __init__(self, logfile, target_model):
        super(GCG, self).__init__(logfile, target_model)

        with open(self.logfile, "r") as f:
            log = json.load(f)

        self.goals = log["goal"]
        self.targets = log["target"]
        self.controls = log["control"]

        self.prompts = [
            self.create_prompt(g, c, t)
            for (g, c, t) in zip(self.goals, self.controls, self.targets)
        ]

    def create_prompt(self, goal, control, target):
        """Create GCG prompt."""

        max_new_tokens = max(
            len(self.target_model.tokenizer(target).input_ids) + 2, 100
        )

        # Create full prompt for LLM
        conv_template = self.target_model.conv_template
        conv_template.append_message(conv_template.roles[0], f"{goal} {control}")
        conv_template.append_message(conv_template.roles[1], "")
        prompt = conv_template.get_prompt()

        # As per the GCG source code, we encode then decode the full prompt
        encoding = self.target_model.tokenizer(prompt)
        full_prompt = (
            self.target_model.tokenizer.decode(encoding.input_ids)
            .replace("<s>", "")
            .replace("</s>", "")
        )

        # Clear the conv template
        conv_template.messages = []

        start_index = full_prompt.find(goal)
        end_index = full_prompt.find(control) + len(control)
        perturbable_prompt = full_prompt[start_index:end_index]

        return Prompt(full_prompt, perturbable_prompt, max_new_tokens)


class PAIR(Attack):
    """Prompt Automatic Iterative Refinement (PAIR) attack.

    Title: Jailbreaking Black Box Large Language Models in Twenty Queries
    Authors: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani,
                George J. Pappas, Eric Wong
    Paper: https://arxiv.org/abs/2310.08419
    """

    def __init__(self, logfile, target_model):
        super(PAIR, self).__init__(logfile, target_model)

        import pandas as pd

        df = pd.read_pickle(logfile)
        jailbreak_prompts = df["jailbreak_prompt"].to_list()

        self.prompts = [self.create_prompt(prompt) for prompt in jailbreak_prompts]

    def create_prompt(self, prompt):
        conv_template = self.target_model.conv_template
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        full_prompt = conv_template.get_prompt()

        # Clear the conv template
        conv_template.messages = []

        return Prompt(full_prompt, prompt, max_new_tokens=100)
