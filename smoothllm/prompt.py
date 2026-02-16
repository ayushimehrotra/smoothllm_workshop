class Prompt:
    """A prompt with a perturbable region for SmoothLLM experiments.

    Attributes:
        full_prompt: The complete prompt string sent to the model.
        perturbable_prompt: The substring of full_prompt that may be perturbed.
        max_new_tokens: Maximum number of new tokens for generation.
    """

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
