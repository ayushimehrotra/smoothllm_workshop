"""SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks.

Public API
----------
Prompt : class
    Shared prompt representation with perturbable region.
compute_dsp : function
    Compute Defense Success Probability via majority vote.
alpha_lower_bound : function
    Conservative alpha bound (Proposition 1).
alpha_tighter_bound : function
    Data-informed alpha bound (Proposition 2).
fit_asr_curve : function
    Fit exponential ASR decay model to empirical data.
make_asr_func : function
    Build a callable ASR(k) from fitted parameters.
parameter_selection_pipeline : function
    End-to-end parameter selection (Section 3.7).
"""

from smoothllm.prompt import Prompt
from smoothllm.certificate import compute_dsp, alpha_lower_bound, alpha_tighter_bound
from smoothllm.curve_fitting import fit_asr_curve, make_asr_func
from smoothllm.sensitivity import parameter_selection_pipeline

__all__ = [
    "Prompt",
    "compute_dsp",
    "alpha_lower_bound",
    "alpha_tighter_bound",
    "fit_asr_curve",
    "make_asr_func",
    "parameter_selection_pipeline",
]
