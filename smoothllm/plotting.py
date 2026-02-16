"""Plotting utilities for SmoothLLM experiments.

Generates figures reproducing the paper's ASR-vs-k and DSP-vs-N plots.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # allow import without matplotlib for headless envs
    plt = None

from smoothllm.certificate import compute_dsp


def _require_matplotlib():
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")


def agresti_coull_interval(successes, total, z=1.96):
    """Compute Agresti-Coull confidence interval."""
    if total == 0:
        return 0.0, 0.0, 0.0
    n_tilde = total + z**2
    p_tilde = (successes + 0.5 * z**2) / n_tilde
    margin = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return p_tilde, max(0.0, p_tilde - margin), min(1.0, p_tilde + margin)


def plot_asr_vs_k(
    csv_path,
    attack_name=None,
    perturbation_type=None,
    fit_params=None,
    z_value=1.96,
    title=None,
    ax=None,
    save_path=None,
):
    """Plot ASR vs perturbation budget k with Agresti-Coull CI bands.

    Parameters
    ----------
    csv_path : str
        Path to CSV from ``experiment_k.py``.
    attack_name : str, optional
        Filter to this attack.
    perturbation_type : str, optional
        Filter to this perturbation type.
    fit_params : tuple (a, b, c), optional
        If provided, overlay the fitted exponential curve.
    z_value : float
        Z-score for confidence intervals (default 1.96 for 95% CI).
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; creates a new figure if None.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()
    import pandas as pd

    df = pd.read_csv(csv_path)
    if attack_name is not None:
        df = df[df["attack_name"] == attack_name]
    if perturbation_type is not None:
        df = df[df["perturbation_type"] == perturbation_type]
    df = df.sort_values("k")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    k_vals = df["k"].values
    means = df["attack_success_mean"].values
    lows = df["agresti_coull_low"].values
    highs = df["agresti_coull_high"].values

    ax.plot(k_vals, means, "o-", label="Empirical ASR")
    ax.fill_between(k_vals, lows, highs, alpha=0.2, label=f"{z_value:.2f}-z CI")

    if fit_params is not None:
        a, b, c = fit_params
        k_smooth = np.linspace(k_vals.min(), k_vals.max(), 200)
        asr_fit = a * np.exp(-b * k_smooth) + c
        ax.plot(k_smooth, asr_fit, "--", label=f"Fit: {a:.3f}e^(-{b:.3f}k)+{c:.3f}")

    ax.set_xlabel("Perturbation budget k")
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    if title:
        ax.set_title(title)

    if save_path:
        ax.get_figure().savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_dsp_vs_N(
    alpha,
    N_range=None,
    target_dsp=None,
    title=None,
    ax=None,
    save_path=None,
):
    """Plot DSP vs number of copies N.

    Parameters
    ----------
    alpha : float
        Per-copy correct-rejection probability.
    N_range : iterable of int, optional
        N values to plot (default: odd values 1..51).
    target_dsp : float, optional
        If provided, draw a horizontal threshold line.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()

    if N_range is None:
        N_range = range(1, 52, 2)

    N_vals = list(N_range)
    dsp_vals = [compute_dsp(alpha, N) for N in N_vals]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(N_vals, dsp_vals, "o-", label=f"DSP (alpha={alpha:.4f})")

    if target_dsp is not None:
        ax.axhline(y=target_dsp, color="r", linestyle="--", label=f"Target DSP={target_dsp}")

    ax.set_xlabel("Number of copies N")
    ax.set_ylabel("Defense Success Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    if title:
        ax.set_title(title)

    if save_path:
        ax.get_figure().savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
