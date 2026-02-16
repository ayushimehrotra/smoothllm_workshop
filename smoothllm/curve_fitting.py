"""Exponential ASR curve fitting for SmoothLLM experiments.

Fits the model  ASR(k) = a * exp(-b * k) + c  to empirical attack success
rate data collected across perturbation budgets k.
"""

import math

import numpy as np
from scipy.optimize import curve_fit


def exponential_model(k, a, b, c):
    """Exponential decay model: ``a * exp(-b * k) + c``."""
    return a * np.exp(-b * k) + c


def fit_asr_curve(k_values, asr_values):
    """Fit the exponential ASR model to empirical data.

    Parameters
    ----------
    k_values : array-like
        Perturbation budget values.
    asr_values : array-like
        Observed attack success rates at each k.

    Returns
    -------
    tuple (a, b, c, pcov)
        Fitted parameters and the covariance matrix from curve_fit.
    """
    k_values = np.asarray(k_values, dtype=float)
    asr_values = np.asarray(asr_values, dtype=float)

    p0 = [asr_values[0], 0.3, asr_values[-1]]
    bounds = ([0.0, 0.0, 0.0], [1.0, np.inf, 1.0])

    popt, pcov = curve_fit(
        exponential_model, k_values, asr_values, p0=p0, bounds=bounds, maxfev=10000
    )

    return popt[0], popt[1], popt[2], pcov


def make_asr_func(a, b, c):
    """Return a callable ``ASR(k) = a * exp(-b * k) + c``.

    Suitable for passing to :func:`smoothllm.certificate.alpha_tighter_bound`.
    """

    def asr_func(k):
        return a * math.exp(-b * k) + c

    return asr_func


def find_k_for_epsilon(a, b, c, epsilon):
    """Find the smallest integer k such that ASR(k) <= epsilon.

    Analytically solves  a * exp(-b * k) + c = epsilon  and rounds up.

    Parameters
    ----------
    a, b, c : float
        Fitted exponential model parameters.
    epsilon : float
        Target maximum ASR.

    Returns
    -------
    int
        Minimum perturbation budget achieving ASR <= epsilon.

    Raises
    ------
    ValueError
        If epsilon <= c (the asymptote) or a <= 0, making the target
        unreachable.
    """
    if epsilon <= c:
        raise ValueError(
            f"Target epsilon={epsilon} is at or below the asymptote c={c}; "
            f"ASR can never reach this level."
        )
    if a <= 0:
        raise ValueError(f"Parameter a={a} must be positive.")
    if b <= 0:
        raise ValueError(f"Parameter b={b} must be positive.")

    k_exact = -math.log((epsilon - c) / a) / b
    return math.ceil(k_exact)


def fit_from_csv(csv_path, attack_name=None, perturbation_type=None):
    """Load experiment CSV and fit the ASR curve.

    Parameters
    ----------
    csv_path : str
        Path to a CSV produced by ``experiment_k.py``.
    attack_name : str, optional
        Filter rows to this attack (e.g. ``"GCG"``).
    perturbation_type : str, optional
        Filter rows to this perturbation type.

    Returns
    -------
    tuple (a, b, c, pcov)
        Fitted parameters and covariance matrix.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    if attack_name is not None:
        df = df[df["attack_name"] == attack_name]
    if perturbation_type is not None:
        df = df[df["perturbation_type"] == perturbation_type]

    df = df.sort_values("k")
    return fit_asr_curve(df["k"].values, df["attack_success_mean"].values)
