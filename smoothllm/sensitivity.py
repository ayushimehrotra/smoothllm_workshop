"""Sensitivity analysis and parameter selection (Section 3.7).

Implements derivatives of alpha and DSP with respect to epsilon,
and the end-to-end parameter selection pipeline from the paper.
"""

import math

from scipy.stats import hypergeom, binom

from smoothllm.certificate import (
    alpha_lower_bound,
    alpha_tighter_bound,
    alpha_patch_tighter_bound,
    compute_dsp,
    find_minimum_N,
)
from smoothllm.curve_fitting import find_k_for_epsilon, make_asr_func


def d_alpha_d_epsilon(k, m, m_S, M):
    """Derivative of alpha (lower bound) with respect to epsilon.

    d(alpha)/d(epsilon) = -P_{k+}

    where P_{k+} = P(X >= k) under the hypergeometric model.

    Parameters
    ----------
    k : int
        Suffix perturbation threshold.
    m, m_S, M : int
        Hypergeometric parameters.

    Returns
    -------
    float
        The derivative (always non-positive).
    """
    p_k_plus = float(hypergeom.sf(k - 1, m, m_S, M))
    return -p_k_plus


def d_dsp_d_alpha(alpha, N):
    """Derivative of DSP with respect to alpha.

    DSP = 1 - Binom.cdf(t, N, alpha) where t = ceil(N/2) - 1.
    d(DSP)/d(alpha) = sum_{j=0}^{t} C(N,j) * [j*alpha^{j-1}*(1-alpha)^{N-j}
                        - alpha^j*(N-j)*(1-alpha)^{N-j-1}]
    which simplifies to N * binom.pmf(t, N-1, alpha) via the identity
    d/dp CDF(t, N, p) = -N * binom.pmf(t, N-1, p).

    Parameters
    ----------
    alpha : float
        Per-copy correct-rejection probability.
    N : int
        Number of copies.

    Returns
    -------
    float
        d(DSP)/d(alpha) (always non-negative).
    """
    t = math.ceil(N / 2) - 1
    # d/dp [1 - Binom.cdf(t, N, p)] = N * binom.pmf(t, N-1, p)
    if N <= 1:
        return 1.0
    return float(N * binom.pmf(t, N - 1, alpha))


def d_dsp_d_epsilon(k, m, m_S, M, alpha, N):
    """Derivative of DSP with respect to epsilon via chain rule.

    d(DSP)/d(epsilon) = d(alpha)/d(epsilon) * d(DSP)/d(alpha)

    Parameters
    ----------
    k : int
        Suffix perturbation threshold.
    m, m_S, M : int
        Hypergeometric parameters.
    alpha : float
        Current per-copy correct-rejection probability.
    N : int
        Number of copies.

    Returns
    -------
    float
        Sensitivity of DSP to epsilon (typically negative: increasing epsilon
        decreases DSP).
    """
    return d_alpha_d_epsilon(k, m, m_S, M) * d_dsp_d_alpha(alpha, N)


def parameter_selection_pipeline(
    a, b, c, m, m_S, q, epsilon, target_dsp, perturbation_type="RandomSwapPerturbation"
):
    """End-to-end parameter selection (Section 3.7 case study).

    Steps:
    1. Find minimum k such that ASR(k) <= epsilon.
    2. Compute M (number of perturbed characters) from q and m.
    3. Compute alpha using the tighter bound with the fitted ASR curve.
    4. Find minimum N such that DSP >= target_dsp.
    5. Compute sensitivity of DSP to epsilon.

    Parameters
    ----------
    a, b, c : float
        Fitted exponential ASR curve parameters.
    m : int
        Total perturbable region length.
    m_S : int
        Adversarial suffix length.
    q : float
        Perturbation rate as a fraction (e.g. 0.10 for 10%).
    epsilon : float
        Target maximum ASR.
    target_dsp : float
        Desired Defense Success Probability (e.g. 0.95).
    perturbation_type : str
        ``"RandomSwapPerturbation"`` or ``"RandomPatchPerturbation"``.

    Returns
    -------
    dict
        Keys: k, M, alpha_lower, alpha_tighter, N, dsp, sensitivity.
    """
    # Step 1: find k
    k = find_k_for_epsilon(a, b, c, epsilon)

    # Step 2: compute M
    M = round(q * m)

    # Step 3: compute alpha bounds
    asr_func = make_asr_func(a, b, c)
    alpha_lo = alpha_lower_bound(k, m, m_S, M, epsilon)

    if perturbation_type == "RandomPatchPerturbation":
        alpha_tight = alpha_patch_tighter_bound(k, m, m_S, M, epsilon, asr_func)
    else:
        alpha_tight = alpha_tighter_bound(k, m, m_S, M, epsilon, asr_func)

    # Step 4: find minimum N (use tighter bound)
    N = find_minimum_N(alpha_tight, target_dsp)

    # Step 5: sensitivity
    if N is not None:
        dsp = compute_dsp(alpha_tight, N)
        sens = d_dsp_d_epsilon(k, m, m_S, M, alpha_tight, N)
    else:
        dsp = None
        sens = None

    return {
        "k": k,
        "M": M,
        "alpha_lower": alpha_lo,
        "alpha_tighter": alpha_tight,
        "N": N,
        "dsp": dsp,
        "sensitivity": sens,
    }
