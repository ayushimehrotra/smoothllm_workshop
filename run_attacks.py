import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import fastchat

import smoothllm.perturbations as perturbations
import smoothllm.defenses as defenses
import smoothllm.make_attacks as attacks
import smoothllm.language_models as language_models
import smoothllm.model_configs as model_configs


torch.cuda.empty_cache()

# Targeted LLM
target_model= 'vicuna'

# Attacking LLM
attack='GCG'
attack_logfile='smoothllm/lib/llmattacks_vicuna.json'

# Instantiate the targeted LLM
config = model_configs.MODELS[target_model]
target_model = language_models.LLM(
    model_path=config['model_path'],
    tokenizer_path=config['tokenizer_path'],
    conv_template_name=config['conversation_template'],
    device='cuda:0'
)

# Create attack instance, used to create prompts
attack = vars(attacks)[attack](
    logfile=attack_logfile,
    target_model=target_model,
    start_index=0,
    end_index=500
)