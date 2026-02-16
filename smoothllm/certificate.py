"""Probabilistic certificate computations for SmoothLLM (Propositions 1 & 2).

Implements the DSP (Defense Success Probability) framework from
"Towards Realistic Guarantees: A Probabilistic Certificate for SmoothLLM"
(arXiv:2511.18721).
"""

import math

from scipy.stats import hypergeom, binom


def hypergeometric_pmf(i, m, m_S, M):
    """Probability that exactly *i* of *M* swapped characters fall in the suffix.

    Models RandomSwapPerturbation: drawing *M* positions uniformly without
    replacement from a perturbable region of length *m*, where *m_S* positions
    belong to the adversarial suffix.

    Parameters
    ----------
    i : int
        Number of suffix characters hit.
    m : int
        Total length of the perturbable region.
    m_S : int
        Length of the adversarial suffix within the perturbable region.
    M : int
        Number of characters swapped (drawn).

    Returns
    -------
    float
        Pr[X = i] where X ~ Hypergeometric(N=m, K=m_S, n=M).
    """
    return float(hypergeom.pmf(i, m, m_S, M))


def alpha_lower_bound(k, m, m_S, M, epsilon):
    """Conservative lower bound on correct-rejection probability (Proposition 1).

    Assumes that whenever fewer than *k* suffix characters are perturbed the
    attack always succeeds (worst case).

    alpha >= (1 - epsilon) * P(X >= k)

    Parameters
    ----------
    k : int
        Minimum number of suffix perturbations for the attack to fail.
    m, m_S, M : int
        Hypergeometric parameters (see :func:`hypergeometric_pmf`).
    epsilon : float
        Residual attack success rate at perturbation level *k*.

    Returns
    -------
    float
        Lower bound on alpha (per-copy correct-rejection probability).
    """
    p_k_plus = float(hypergeom.sf(k - 1, m, m_S, M))  # P(X >= k)
    return (1.0 - epsilon) * p_k_plus


def alpha_tighter_bound(k, m, m_S, M, epsilon, asr_func):
    """Data-informed lower bound on correct-rejection probability (Proposition 2).

    Uses the empirical ASR curve for perturbation levels below *k* instead of
    assuming worst-case (ASR = 1).

    alpha >= (1 - epsilon) * P(X >= k) + sum_{i<k} (1 - ASR(i)) * P(X = i)

    Parameters
    ----------
    k : int
        Minimum suffix perturbations threshold.
    m, m_S, M : int
        Hypergeometric parameters.
    epsilon : float
        Residual ASR at level *k*.
    asr_func : callable
        Function mapping perturbation count *i* -> empirical ASR(i).

    Returns
    -------
    float
        Tighter lower bound on alpha.
    """
    p_k_plus = float(hypergeom.sf(k - 1, m, m_S, M))
    alpha = (1.0 - epsilon) * p_k_plus

    for i in range(k):
        p_i = float(hypergeom.pmf(i, m, m_S, M))
        alpha += (1.0 - asr_func(i)) * p_i

    return alpha


def patch_overlap_pmf(i, m, m_S, M):
    """Probability that a contiguous patch of width *M* overlaps *i* suffix chars.

    Models RandomPatchPerturbation where the patch is a contiguous block.
    The suffix occupies the last *m_S* characters of the perturbable region.

    Parameters
    ----------
    i : int
        Number of suffix characters overlapped.
    m : int
        Total perturbable region length.
    m_S : int
        Suffix length (assumed at the end of the region).
    M : int
        Patch width.

    Returns
    -------
    float
        Pr[overlap = i].
    """
    if M > m or m_S > m:
        return 0.0

    num_positions = m - M + 1
    if num_positions <= 0:
        return 1.0 if i == min(m_S, M) else 0.0

    count = 0
    suffix_start = m - m_S

    for start in range(num_positions):
        end = start + M
        overlap = max(0, min(end, m) - max(start, suffix_start))
        if overlap == i:
            count += 1

    return count / num_positions


def alpha_patch_tighter_bound(k, m, m_S, M, epsilon, asr_func):
    """Tighter alpha bound for RandomPatchPerturbation.

    Same logic as :func:`alpha_tighter_bound` but uses patch overlap geometry.

    Parameters
    ----------
    k : int
        Minimum overlap threshold.
    m, m_S, M : int
        Patch geometry parameters.
    epsilon : float
        Residual ASR at level *k*.
    asr_func : callable
        Empirical ASR function.

    Returns
    -------
    float
        Tighter lower bound on alpha for patch perturbations.
    """
    max_overlap = min(m_S, M)

    p_k_plus = sum(patch_overlap_pmf(i, m, m_S, M) for i in range(k, max_overlap + 1))
    alpha = (1.0 - epsilon) * p_k_plus

    for i in range(k):
        p_i = patch_overlap_pmf(i, m, m_S, M)
        alpha += (1.0 - asr_func(i)) * p_i

    return alpha


def compute_dsp(alpha, N):
    """Compute Defense Success Probability via majority vote.

    DSP = P(at least ceil(N/2) of N copies correctly reject)
        = 1 - Binom.cdf(ceil(N/2) - 1, N, alpha)

    Parameters
    ----------
    alpha : float
        Per-copy correct-rejection probability.
    N : int
        Number of perturbed copies (must be odd for a clean majority).

    Returns
    -------
    float
        Defense Success Probability.
    """
    threshold = math.ceil(N / 2) - 1
    return 1.0 - float(binom.cdf(threshold, N, alpha))


def compute_dsp_sweep(alpha, N_values):
    """Compute DSP across a range of N values.

    Parameters
    ----------
    alpha : float
        Per-copy correct-rejection probability.
    N_values : iterable of int
        Values of N to evaluate.

    Returns
    -------
    list of dict
        Each dict contains ``{"N": int, "dsp": float}``.
    """
    return [{"N": N, "dsp": compute_dsp(alpha, N)} for N in N_values]


def find_minimum_N(alpha, target_dsp, max_N=1001):
    """Find the smallest odd N such that DSP >= target_dsp.

    Uses linear search over odd values of N.

    Parameters
    ----------
    alpha : float
        Per-copy correct-rejection probability.
    target_dsp : float
        Desired minimum DSP (e.g. 0.95).
    max_N : int
        Upper search bound.

    Returns
    -------
    int or None
        Smallest N achieving the target, or None if not found within max_N.
    """
    for N in range(1, max_N + 1, 2):
        if compute_dsp(alpha, N) >= target_dsp:
            return N
    return None
