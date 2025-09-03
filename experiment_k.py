import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import fastchat

import smoothllm.perturbations as perturbations
import smoothllm.experiment_defenses as defenses
import smoothllm.experiment_attacks as attacks
import smoothllm.language_models as language_models
import smoothllm.model_configs as model_configs


torch.cuda.empty_cache()

# Targeted LLM
target_model= 'vicuna'

# Attacking LLM
attack_name='GCG'
attack_logfile='smoothllm/lib/llmattacks_vicuna.json'

# SmoothLLM
smoothllm_pert_pct=3
smoothllm_pert_type='RandomSwapPerturbation'

if __name__ == "__main__":
    # Instantiate the targeted LLM
    config = model_configs.MODELS[target_model]
    target_model = language_models.LLM(
        model_path=config['model_path'],
        tokenizer_path=config['tokenizer_path'],
        conv_template_name=config['conversation_template'],
        device='cuda:0'
    )

    # Create SmoothLLM instance
    defense = defenses.SmoothLLM(
        target_model=target_model,
        pert_type=smoothllm_pert_type,
        pert_pct=smoothllm_pert_pct
    )

    # Checking defense success rate with different positions
    jb_percentage = []
    for _ in range(2):
        attack = vars(attacks)[attack_name](
            logfile=attack_logfile,
            target_model=target_model,
            pert_type=smoothllm_pert_type,
            pert_pct=smoothllm_pert_pct
        )
        for i, prompt in tqdm(enumerate(attack.prompts)):
            output = defense(prompt)
            jb = defense.is_jailbroken(output)
            jb_percentage.append(jb)
        print(f"For k={smoothllm_pert_pct}, Attack Accuracy: {np.mean(jb_percentage)}, Standard Dev.: {np.std(jb_percentage)}")

