import sys
import warnings

warnings.filterwarnings("ignore")


user_site_packages = '/home/am5715/.local/lib/python3.11/site-packages'

if user_site_packages in sys.path:
    sys.path.remove(user_site_packages)

sys.path.insert(0, user_site_packages)

import os

private_libs_path = '/home/am5715/smoothllm_workshop/my_private_libs'

if private_libs_path not in sys.path:
    sys.path.insert(0, private_libs_path)


os.environ["ACCELERATE_DISABLE_BNB"] = "1"
os.environ["TRANSFORMERS_SKIP_AWS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("--- Final sys.path ---")
for i, path in enumerate(sys.path[:5]):
    print(f"{i}: {path}")
print("----------------------")



import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import fastchat

import smoothllm.lib.perturbations as perturbations
import smoothllm.lib.defenses as defenses
import smoothllm.lib.attacks as attacks
import smoothllm.lib.language_models as language_models
import smoothllm.lib.model_configs as model_configs

os.getcwd()


torch.cuda.empty_cache()

# Targeted LLM
target_model= 'llama2'

# Attacking LLM
attack='GCG'
attack_logfile='smoothllm/data/GCG/llama2_behaviors.json'

# SmoothLLM
smoothllm_num_copies=1
smoothllm_pert_pct=1
smoothllm_pert_type='RandomSwapPerturbation'



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
    pert_pct=smoothllm_pert_pct,
    num_copies=smoothllm_num_copies
)

# Create attack instance, used to create prompts
attack = vars(attacks)[attack](
    logfile=attack_logfile,
    target_model=target_model,
    start_index=450,
    end_index=500
)