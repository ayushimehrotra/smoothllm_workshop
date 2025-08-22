import os, sys, site, sysconfig, importlib.abc

os.environ["ACCELERATE_DISABLE_BNB"] = "1"
os.environ["TRANSFORMERS_SKIP_AWS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


paths = sysconfig.get_paths()
stdlib = paths["stdlib"]
platstdlib = paths.get("platstdlib", stdlib)
lib_dyn = os.path.join(stdlib, "lib-dynload")

def _ins(p):
    if p and p not in sys.path:
        sys.path.insert(0, p)

_ins(stdlib)
_ins(platstdlib)
_ins(lib_dyn)

# Annoying Spack stuff
user_site = site.getusersitepackages()
_ins(user_site)

def _is_spack_sitepkgs(p: str) -> bool:
    n = p.replace("\\", "/")
    return ("/spack/" in n or "/.spack-env/" in n) and ("site-packages" in n or "dist-packages" in n)
sys.path = [p for p in sys.path if not _is_spack_sitepkgs(p)]

# More annoying Spack stuff
class _BlockAWS(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("boto3", "botocore")):
            return None  # say "not found" (safe for importlib.util.find_spec)
        return None
sys.meta_path.insert(0, _BlockAWS())

for m in ("boto3", "botocore"):
    if m in sys.modules:
        del sys.modules[m]
        
        
        
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
        pert_pct=smoothllm_pert_pct,
        num_copies=smoothllm_num_copies
    )

    # Create attack instance, used to create prompts
    attack = vars(attacks)[attack](
        logfile=attack_logfile,
        target_model=target_model
    )

    # Checking defense success rate with different positions
    jb_percentage = []
    for _ in range(30):
        for i, prompt in tqdm(enumerate(attack.prompts)):
            output = defense(prompt)
            jb = defense.is_jailbroken(output)
            jb_percentage.append(jb)
    print(f"For k=1, Attack Accuracy: {np.mean(jb_percentage)}, Standard Dev.: {np.std(jb_percentage)}")

