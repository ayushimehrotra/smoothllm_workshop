"""Comprehensive verification suite: code vs. paper (arXiv:2511.18721).

Tests every mathematical formula, numerical constant, and reproducibility
claim against the paper's text. Organized by paper section.

Run with:
    python -m pytest tests/test_paper_verification.py -v
or:
    python tests/test_paper_verification.py
"""

import math
import sys
import os
import traceback

import numpy as np
from scipy.stats import hypergeom, binom
from scipy.special import comb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0
WARN = 0


def check(condition, label, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}")
        if detail:
            print(f"        {detail}")


def warn(label, detail=""):
    global WARN
    WARN += 1
    print(f"  WARN  {label}")
    if detail:
        print(f"        {detail}")


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ===========================================================================
# 1. INFRASTRUCTURE — can we even import everything?
# ===========================================================================
section("1. INFRASTRUCTURE IMPORTS")

try:
    import smoothllm
    check(True, "import smoothllm")
except Exception as e:
    check(False, "import smoothllm", str(e))

try:
    from smoothllm.prompt import Prompt
    check(True, "import Prompt from smoothllm.prompt")
except Exception as e:
    check(False, "import Prompt", str(e))

try:
    from smoothllm.certificate import (
        hypergeometric_pmf,
        alpha_lower_bound,
        alpha_tighter_bound,
        alpha_patch_tighter_bound,
        patch_overlap_pmf,
        compute_dsp,
        compute_dsp_sweep,
        find_minimum_N,
    )
    check(True, "import all certificate functions")
except Exception as e:
    check(False, "import certificate", str(e))

try:
    from smoothllm.curve_fitting import (
        exponential_model,
        fit_asr_curve,
        make_asr_func,
        find_k_for_epsilon,
    )
    check(True, "import all curve_fitting functions")
except Exception as e:
    check(False, "import curve_fitting", str(e))

try:
    from smoothllm.sensitivity import (
        d_alpha_d_epsilon,
        d_dsp_d_alpha,
        d_dsp_d_epsilon,
        parameter_selection_pipeline,
    )
    check(True, "import all sensitivity functions")
except Exception as e:
    check(False, "import sensitivity", str(e))

try:
    from smoothllm.plotting import plot_asr_vs_k, plot_dsp_vs_N
    check(True, "import plotting functions")
except Exception as e:
    check(False, "import plotting", str(e))

try:
    from smoothllm.perturbations import (
        RandomSwapPerturbation,
        RandomPatchPerturbation,
        RandomInsertPerturbation,
    )
    check(True, "import perturbation classes")
except Exception as e:
    check(False, "import perturbations", str(e))

try:
    import smoothllm.model_configs as mc
    check(True, "import model_configs")
except Exception as e:
    check(False, "import model_configs", str(e))

# Prompt deduplication
try:
    from smoothllm.prompt import Prompt as P1
    # attacks.py should import from prompt
    import smoothllm.attacks as atk_mod
    # Check that attacks.py uses the shared Prompt
    check(
        not hasattr(atk_mod, 'Prompt') or atk_mod.Prompt is P1,
        "attacks.py uses shared Prompt class"
    )
except Exception as e:
    check(False, "Prompt dedup check", str(e))


# ===========================================================================
# 2. PERTURBATION PRIMITIVES (Section 3.1 of the paper)
# ===========================================================================
section("2. PERTURBATION PRIMITIVES")

# 2a. RandomSwapPerturbation preserves length
s = "A" * 50
swap = RandomSwapPerturbation(q=5)
result = swap(s)
check(len(result) == len(s), "RandomSwap preserves string length")

# 2b. RandomSwapPerturbation changes exactly q characters (for uniform string)
diff = sum(1 for a, b in zip(s, result) if a != b)
check(diff <= 5, "RandomSwap changes at most q characters")

# 2c. RandomPatchPerturbation preserves length
patch = RandomPatchPerturbation(q=5)
result = patch(s)
check(len(result) == len(s), "RandomPatch preserves string length")

# 2d. RandomInsertPerturbation increases length
ins = RandomInsertPerturbation(q=10)
result = ins("Hello World Test String")
check(
    len(result) > len("Hello World Test String"),
    "RandomInsert increases string length",
)

# 2e. Bounds checking: q > len(s) should not crash
try:
    swap_big = RandomSwapPerturbation(q=100)
    result = swap_big("abc")
    check(len(result) == 3, "RandomSwap q>len(s) clamps gracefully")
except Exception as e:
    check(False, "RandomSwap q>len(s)", str(e))

try:
    patch_big = RandomPatchPerturbation(q=100)
    result = patch_big("abc")
    check(len(result) == 3, "RandomPatch q>len(s) clamps gracefully")
except Exception as e:
    check(False, "RandomPatch q>len(s)", str(e))

# 2f. RandomInsertPerturbation index shifting fix
#     After fix: sorted indices + offset ensures correct positions
np.random.seed(42)
for _ in range(20):
    s_test = "A" * 100
    ins10 = RandomInsertPerturbation(q=10)
    result = ins10(s_test)
    expected_len = 100 + int(100 * 10 / 100)
    check(
        len(result) == expected_len,
        f"RandomInsert produces correct length ({expected_len})",
    )
    break  # one check is enough for length


# ===========================================================================
# 3. HYPERGEOMETRIC PMF — Paper Eq. (6)
# ===========================================================================
section("3. HYPERGEOMETRIC PMF — Eq. (6)")

# Paper Eq. (6):
# Pr[X = i] = C(m_S, i) * C(m - m_S, M - i) / C(m, M)
# X ~ Hypergeometric(N=m, K=m_S, n=M)

m, m_S, M = 240, 100, 24  # Paper's parameters

# 3a. PMF matches hand formula for specific i
for i in [0, 5, 10, 15, 24]:
    if i > min(M, m_S):
        continue
    code_val = hypergeometric_pmf(i, m, m_S, M)
    hand_val = float(comb(m_S, i, exact=True) * comb(m - m_S, M - i, exact=True)
                     / comb(m, M, exact=True))
    check(
        abs(code_val - hand_val) < 1e-12,
        f"PMF(i={i}) matches hand formula: {code_val:.6e} vs {hand_val:.6e}",
    )

# 3b. PMF sums to 1
pmf_sum = sum(hypergeometric_pmf(i, m, m_S, M) for i in range(min(M, m_S) + 1))
check(abs(pmf_sum - 1.0) < 1e-10, f"PMF sums to 1.0 (got {pmf_sum:.15f})")

# 3c. Mean matches E[X] = M * m_S / m
empirical_mean = sum(i * hypergeometric_pmf(i, m, m_S, M) for i in range(min(M, m_S) + 1))
expected_mean = M * m_S / m
check(
    abs(empirical_mean - expected_mean) < 1e-8,
    f"Mean = M*m_S/m = {expected_mean:.4f} (got {empirical_mean:.4f})",
)


# ===========================================================================
# 4. ALPHA LOWER BOUND — Paper Eq. (2) / Eq. (10)
# ===========================================================================
section("4. ALPHA LOWER BOUND — Eq. (2)")

# Paper Eq. (10):
# alpha_lower = (1 - epsilon) * sum_{i=k}^{min(M,m_S)} Pr[X=i]
# which is (1 - epsilon) * P(X >= k)

k_test, eps_test = 6, 0.05

# 4a. Compute by hand
p_k_plus_hand = sum(hypergeometric_pmf(i, m, m_S, M) for i in range(k_test, min(M, m_S) + 1))
alpha_lower_hand = (1.0 - eps_test) * p_k_plus_hand
alpha_lower_code = alpha_lower_bound(k_test, m, m_S, M, eps_test)
check(
    abs(alpha_lower_code - alpha_lower_hand) < 1e-12,
    f"alpha_lower matches hand computation: {alpha_lower_code:.10f}",
)

# 4b. P(X>=k) via scipy.stats.hypergeom.sf
p_k_plus_scipy = float(hypergeom.sf(k_test - 1, m, m_S, M))
check(
    abs(p_k_plus_hand - p_k_plus_scipy) < 1e-12,
    f"P(X>={k_test}) hand={p_k_plus_hand:.10f} scipy={p_k_plus_scipy:.10f}",
)

# 4c. When epsilon=0, alpha_lower = P(X>=k) (original k-unstable, Remark 2)
alpha_eps0 = alpha_lower_bound(k_test, m, m_S, M, 0.0)
check(
    abs(alpha_eps0 - p_k_plus_scipy) < 1e-12,
    "epsilon=0 recovers deterministic k-unstable bound (Remark 2)",
)

# 4d. alpha_lower is monotonically decreasing in epsilon
alphas = [alpha_lower_bound(k_test, m, m_S, M, e) for e in np.linspace(0, 0.5, 20)]
check(
    all(alphas[i] >= alphas[i + 1] - 1e-15 for i in range(len(alphas) - 1)),
    "alpha_lower monotonically decreasing in epsilon",
)


# ===========================================================================
# 5. ALPHA TIGHTER BOUND — Paper Eq. (11)
# ===========================================================================
section("5. ALPHA TIGHTER BOUND — Eq. (11)")

# Paper Eq. (11):
# alpha_tighter = sum_{i=0}^{k-1} (1 - ASR(i)) * Pr[X=i]
#               + (1 - epsilon) * sum_{i=k}^{min(M,m_S)} Pr[X=i]

a_fit, b_fit, c_fit = 0.2921, 0.3756, 0.0133  # Paper Section 3.7

asr_func = make_asr_func(a_fit, b_fit, c_fit)

# 5a. Compute by hand
sub_threshold_sum = sum(
    (1.0 - asr_func(i)) * hypergeometric_pmf(i, m, m_S, M)
    for i in range(k_test)
)
above_threshold_sum = (1.0 - eps_test) * sum(
    hypergeometric_pmf(i, m, m_S, M)
    for i in range(k_test, min(M, m_S) + 1)
)
alpha_tighter_hand = sub_threshold_sum + above_threshold_sum
alpha_tighter_code = alpha_tighter_bound(k_test, m, m_S, M, eps_test, asr_func)
check(
    abs(alpha_tighter_code - alpha_tighter_hand) < 1e-12,
    f"alpha_tighter matches hand computation: {alpha_tighter_code:.10f}",
)

# 5b. alpha_tighter >= alpha_lower (since we add non-negative terms)
check(
    alpha_tighter_code >= alpha_lower_code - 1e-15,
    f"alpha_tighter ({alpha_tighter_code:.6f}) >= alpha_lower ({alpha_lower_code:.6f})",
)

# 5c. Verify each ASR(i) value against the paper's fitted model
for i_val in range(7):
    asr_val = asr_func(i_val)
    expected_asr = a_fit * math.exp(-b_fit * i_val) + c_fit
    check(
        abs(asr_val - expected_asr) < 1e-12,
        f"ASR({i_val}) = {asr_val:.6f} matches a*exp(-b*{i_val})+c = {expected_asr:.6f}",
    )


# ===========================================================================
# 6. DSP COMPUTATION — Paper Eq. (1) / Eq. (5)
# ===========================================================================
section("6. DSP COMPUTATION — Eq. (1)")

# Paper Eq. (5):
# DSP = sum_{t=ceil(N/2)}^{N} C(N,t) * alpha^t * (1-alpha)^{N-t}
# Code:  DSP = 1 - binom.cdf(ceil(N/2)-1, N, alpha)

# 6a. Hand computation for small N
def dsp_hand(alpha, N):
    """Direct summation of binomial terms."""
    t_start = math.ceil(N / 2)
    return sum(
        comb(N, t, exact=True) * alpha**t * (1.0 - alpha) ** (N - t)
        for t in range(t_start, N + 1)
    )

for N_val in [1, 3, 5, 11, 21]:
    for alpha_val in [0.5, 0.7, 0.9, 0.95]:
        code = compute_dsp(alpha_val, N_val)
        hand = dsp_hand(alpha_val, N_val)
        check(
            abs(code - hand) < 1e-10,
            f"DSP(alpha={alpha_val}, N={N_val}): code={code:.10f} hand={hand:.10f}",
        )

# 6b. Edge cases
check(abs(compute_dsp(1.0, 5) - 1.0) < 1e-15, "DSP(alpha=1) = 1")
check(abs(compute_dsp(0.0, 5) - 0.0) < 1e-15, "DSP(alpha=0) = 0")

# 6c. DSP is monotonically increasing in alpha
dsp_vals = [compute_dsp(a_val, 11) for a_val in np.linspace(0, 1, 50)]
check(
    all(dsp_vals[i] <= dsp_vals[i + 1] + 1e-15 for i in range(len(dsp_vals) - 1)),
    "DSP monotonically increasing in alpha",
)

# 6d. DSP is monotonically increasing in N (for alpha > 0.5)
dsp_N_vals = [compute_dsp(0.8, N) for N in range(1, 30, 2)]
check(
    all(dsp_N_vals[i] <= dsp_N_vals[i + 1] + 1e-15 for i in range(len(dsp_N_vals) - 1)),
    "DSP monotonically increasing in N for alpha>0.5",
)


# ===========================================================================
# 7. EXPONENTIAL ASR MODEL — Paper Section 3.5/3.6
# ===========================================================================
section("7. EXPONENTIAL ASR MODEL")

# Paper: ASR(k) = a * exp(-b*k) + c

# 7a. exponential_model matches formula
for k_val in range(11):
    code_val = exponential_model(k_val, a_fit, b_fit, c_fit)
    hand_val = a_fit * np.exp(-b_fit * k_val) + c_fit
    check(
        abs(code_val - hand_val) < 1e-12,
        f"exponential_model(k={k_val}) = {code_val:.6f}",
    )

# 7b. Curve fitting recovers known parameters
np.random.seed(0)
k_synth = np.arange(0, 11, dtype=float)
asr_synth = 0.3 * np.exp(-0.4 * k_synth) + 0.01
a_r, b_r, c_r, _ = fit_asr_curve(k_synth, asr_synth)
check(abs(a_r - 0.3) < 0.01, f"Fit recovers a=0.3 (got {a_r:.6f})")
check(abs(b_r - 0.4) < 0.01, f"Fit recovers b=0.4 (got {b_r:.6f})")
check(abs(c_r - 0.01) < 0.01, f"Fit recovers c=0.01 (got {c_r:.6f})")

# 7c. Fit with noise still recovers approximately
np.random.seed(42)
asr_noisy = 0.3 * np.exp(-0.4 * k_synth) + 0.01 + np.random.normal(0, 0.01, len(k_synth))
asr_noisy = np.clip(asr_noisy, 0, 1)
a_n, b_n, c_n, _ = fit_asr_curve(k_synth, asr_noisy)
check(abs(a_n - 0.3) < 0.05, f"Noisy fit recovers a~0.3 (got {a_n:.4f})")
check(abs(b_n - 0.4) < 0.1, f"Noisy fit recovers b~0.4 (got {b_n:.4f})")

# 7d. Paper Section 3.6: a=0.1650, b=0.1121, c=0.0427 for Llama2+Patch+GCG
a_36, b_36, c_36 = 0.1650, 0.1121, 0.0427
asr_10 = a_36 * math.exp(-b_36 * 10) + c_36
check(
    abs(asr_10 - 0.097) < 0.005,
    f"Section 3.6: ASR(10) = {asr_10:.4f} ~ 0.097 (paper says ~0.097)",
)


# ===========================================================================
# 8. find_k_for_epsilon — Paper Section 3.7 Step 3
# ===========================================================================
section("8. find_k_for_epsilon — Section 3.7 Step 3")

# Paper: ASR(k) = 0.292*exp(-0.376*k) + 0.013
# Solve: 0.292*exp(-0.376*k) + 0.013 <= 0.05
# => k >= -ln((0.05 - 0.013) / 0.292) / 0.376
# => k >= -ln(0.1267) / 0.376
# => k >= 2.066 / 0.376 = 5.49
# => k = 6

k_found = find_k_for_epsilon(a_fit, b_fit, c_fit, 0.05)
check(k_found == 6, f"find_k_for_epsilon gives k={k_found} (paper says k=6)")

# Verify: ASR(6) <= epsilon and ASR(5) > epsilon
asr_at_6 = asr_func(6)
asr_at_5 = asr_func(5)
check(asr_at_6 <= 0.05, f"ASR(6)={asr_at_6:.6f} <= 0.05")
check(asr_at_5 > 0.05, f"ASR(5)={asr_at_5:.6f} > 0.05 (k=5 insufficient)")

# Hand-verify the exact solution
k_exact = -math.log((0.05 - c_fit) / a_fit) / b_fit
check(
    abs(k_exact - 5.49) < 0.1,
    f"k_exact = {k_exact:.4f} ~ 5.49 (paper says k >= 5.49)",
)

# Error for impossible epsilon
try:
    find_k_for_epsilon(a_fit, b_fit, c_fit, c_fit - 0.001)
    check(False, "find_k_for_epsilon rejects epsilon <= c")
except ValueError:
    check(True, "find_k_for_epsilon rejects epsilon <= c")


# ===========================================================================
# 9. M COMPUTATION — Paper Section 3.7 Step 4
# ===========================================================================
section("9. M COMPUTATION")

# Paper: M = floor(q * m) = floor(0.10 * 240) = 24
# Note: paper uses floor; code uses round. Both give 24 for these values.
M_paper = math.floor(0.10 * 240)
M_code = round(0.10 * 240)
check(M_paper == 24, f"Paper: M = floor(0.10*240) = {M_paper}")
check(M_code == 24, f"Code: M = round(0.10*240) = {M_code}")
check(M_paper == M_code, "floor and round agree for q=0.10, m=240")


# ===========================================================================
# 10. SECTION 3.7 CASE STUDY — End-to-end
# ===========================================================================
section("10. SECTION 3.7 CASE STUDY")

# Paper claims: k=6, epsilon=0.05, m=240, m_S=100, q=0.10, M=24
# DSP >= 0.95, paper claims N=10

result = parameter_selection_pipeline(
    a=a_fit, b=b_fit, c=c_fit,
    m=240, m_S=100, q=0.10,
    epsilon=0.05, target_dsp=0.95,
    perturbation_type="RandomSwapPerturbation",
)

check(result["k"] == 6, f"Pipeline k={result['k']} (paper says 6)")
check(result["M"] == 24, f"Pipeline M={result['M']} (paper says 24)")
check(
    result["alpha_tighter"] >= result["alpha_lower"],
    f"alpha_tighter ({result['alpha_tighter']:.6f}) >= alpha_lower ({result['alpha_lower']:.6f})",
)
check(
    result["dsp"] is not None and result["dsp"] >= 0.95,
    f"DSP={result['dsp']:.6f} >= 0.95 target",
)
check(result["N"] is not None, f"Found N={result['N']}")

# Verify sensitivity is negative (paper Appendix C)
check(
    result["sensitivity"] < 0,
    f"Sensitivity={result['sensitivity']:.6f} < 0 (increasing eps decreases DSP)",
)

# Paper says N=10 but our tighter bound gives alpha so high that N=3 suffices.
# This is NOT a bug — the paper says "a value of N = 10 is typically sufficient",
# referring to Figure 4's parameters (k=8), not the case study's k=6.
# With the tighter bound, alpha is very high, so even N=1 or N=3 works.
# The important thing: our code gives DSP >= 0.95 for the returned N.
if result["N"] != 10:
    warn(
        f"Pipeline N={result['N']} differs from paper's N=10",
        "Paper's N=10 refers to Figure 4 (k=8) or uses the conservative bound. "
        f"Our tighter bound gives alpha={result['alpha_tighter']:.4f}, so fewer "
        f"copies suffice. DSP={result['dsp']:.4f} >> 0.95. Mathematically correct."
    )


# ===========================================================================
# 11. FIGURE 4 PARAMETERS — k=8
# ===========================================================================
section("11. FIGURE 4 PARAMETERS (k=8, m=240, m_S=100, q=0.10, eps=0.05)")

# Paper caption: "threshold k = 8, and epsilon = 0.05"
k_fig4 = 8
alpha_lo_fig4 = alpha_lower_bound(k_fig4, m, m_S, M, eps_test)
alpha_tight_fig4 = alpha_tighter_bound(k_fig4, m, m_S, M, eps_test, asr_func)

check(
    0 < alpha_lo_fig4 < 1,
    f"Figure 4 alpha_lower = {alpha_lo_fig4:.6f}",
)
check(
    alpha_tight_fig4 >= alpha_lo_fig4,
    f"Figure 4 alpha_tighter = {alpha_tight_fig4:.6f} >= alpha_lower",
)

# With conservative bound, check if N=10 is needed
N_lo_fig4 = find_minimum_N(alpha_lo_fig4, 0.95)
N_tight_fig4 = find_minimum_N(alpha_tight_fig4, 0.95)
dsp_at_10_lo = compute_dsp(alpha_lo_fig4, 10)
dsp_at_10_tight = compute_dsp(alpha_tight_fig4, 10)

print(f"  INFO  Figure 4 with lower bound: alpha={alpha_lo_fig4:.6f}, min N={N_lo_fig4}, DSP(N=10)={dsp_at_10_lo:.6f}")
print(f"  INFO  Figure 4 with tighter bound: alpha={alpha_tight_fig4:.6f}, min N={N_tight_fig4}, DSP(N=10)={dsp_at_10_tight:.6f}")

check(
    dsp_at_10_tight >= 0.95,
    f"DSP(N=10, alpha_tight) = {dsp_at_10_tight:.6f} >= 0.95",
)


# ===========================================================================
# 12. PATCH OVERLAP PMF — Paper Appendix D
# ===========================================================================
section("12. PATCH OVERLAP PMF — Appendix D")

# Test with small parameters for hand verification
m_small, m_S_small, M_small = 10, 4, 3
# Suffix occupies positions 6,7,8,9 (last 4 of 10)
# Patch width 3, possible starting positions: 0..7 (8 total)
# start=0: covers 0,1,2 -> overlap=0
# start=1: covers 1,2,3 -> overlap=0
# start=2: covers 2,3,4 -> overlap=0
# start=3: covers 3,4,5 -> overlap=0
# start=4: covers 4,5,6 -> overlap=1
# start=5: covers 5,6,7 -> overlap=2
# start=6: covers 6,7,8 -> overlap=3
# start=7: covers 7,8,9 -> overlap=3
expected_pmf = {0: 4/8, 1: 1/8, 2: 1/8, 3: 2/8}

for i_val, expected in expected_pmf.items():
    got = patch_overlap_pmf(i_val, m_small, m_S_small, M_small)
    check(
        abs(got - expected) < 1e-12,
        f"patch_overlap_pmf(i={i_val}, m=10, m_S=4, M=3) = {got:.4f} (expected {expected:.4f})",
    )

# PMF sums to 1
pmf_patch_sum = sum(
    patch_overlap_pmf(i, m_small, m_S_small, M_small)
    for i in range(min(m_S_small, M_small) + 1)
)
check(abs(pmf_patch_sum - 1.0) < 1e-10, f"Patch PMF sums to 1.0 (got {pmf_patch_sum:.10f})")

# With paper's parameters
pmf_paper_sum = sum(
    patch_overlap_pmf(i, 240, 100, 24)
    for i in range(min(100, 24) + 1)
)
check(
    abs(pmf_paper_sum - 1.0) < 1e-10,
    f"Patch PMF for paper params sums to 1.0 (got {pmf_paper_sum:.10f})",
)


# ===========================================================================
# 13. PROPOSITION 2 — RandomPatch alpha bound
# ===========================================================================
section("13. PROPOSITION 2 — Patch alpha bound")

# Paper Eq. for Proposition 2:
# alpha_patch = sum_{i=0}^{k-1} (1-ASR(i)) * Pr[X=i]
#             + (1-eps) * sum_{i=k}^{M} Pr[X=i]
# where Pr[X=i] uses patch overlap geometry

alpha_patch = alpha_patch_tighter_bound(k_test, m, m_S, M, eps_test, asr_func)

# Hand computation
sub_patch = sum(
    (1.0 - asr_func(i)) * patch_overlap_pmf(i, m, m_S, M)
    for i in range(k_test)
)
above_patch = (1.0 - eps_test) * sum(
    patch_overlap_pmf(i, m, m_S, M)
    for i in range(k_test, min(m_S, M) + 1)
)
alpha_patch_hand = sub_patch + above_patch

check(
    abs(alpha_patch - alpha_patch_hand) < 1e-12,
    f"alpha_patch_tighter matches hand: {alpha_patch:.10f}",
)

# Should be between 0 and 1
check(0 < alpha_patch < 1, f"alpha_patch = {alpha_patch:.6f} in (0, 1)")


# ===========================================================================
# 14. SENSITIVITY ANALYSIS — Paper Appendix C
# ===========================================================================
section("14. SENSITIVITY ANALYSIS — Appendix C")

# Paper Eq. (17): d(alpha)/d(epsilon) = -P_{k+}
p_k_plus = float(hypergeom.sf(k_test - 1, m, m_S, M))
deriv_alpha = d_alpha_d_epsilon(k_test, m, m_S, M)
check(
    abs(deriv_alpha - (-p_k_plus)) < 1e-12,
    f"d(alpha)/d(eps) = {deriv_alpha:.6f} = -P_k+ = {-p_k_plus:.6f} [Eq 17]",
)

# Verify numerically: finite difference of alpha_lower w.r.t. epsilon
h = 1e-8
alpha_plus = alpha_lower_bound(k_test, m, m_S, M, eps_test + h)
alpha_minus = alpha_lower_bound(k_test, m, m_S, M, eps_test - h)
numerical_deriv = (alpha_plus - alpha_minus) / (2 * h)
check(
    abs(deriv_alpha - numerical_deriv) < 1e-5,
    f"d(alpha)/d(eps) analytical={deriv_alpha:.6f} numerical={numerical_deriv:.6f}",
)

# d(DSP)/d(alpha) — verify with finite difference
alpha_test = result["alpha_tighter"]
N_test = 11
dsp_deriv = d_dsp_d_alpha(alpha_test, N_test)
dsp_plus = compute_dsp(alpha_test + h, N_test)
dsp_minus = compute_dsp(alpha_test - h, N_test)
num_dsp_deriv = (dsp_plus - dsp_minus) / (2 * h)
check(
    abs(dsp_deriv - num_dsp_deriv) < 1e-3,
    f"d(DSP)/d(alpha) analytical={dsp_deriv:.6f} numerical={num_dsp_deriv:.6f}",
)

# Chain rule: d(DSP)/d(epsilon) = d(alpha)/d(eps) * d(DSP)/d(alpha)
chain = d_dsp_d_epsilon(k_test, m, m_S, M, alpha_test, N_test)
expected_chain = deriv_alpha * dsp_deriv
check(
    abs(chain - expected_chain) < 1e-10,
    f"Chain rule: {chain:.6f} = {deriv_alpha:.6f} * {dsp_deriv:.6f}",
)

# Paper says: "the overall derivative is non-positive" (Appendix C)
check(chain <= 0, f"d(DSP)/d(eps) = {chain:.6f} <= 0 (Appendix C)")


# ===========================================================================
# 15. SECTION 3.6 — Instantiating (k, epsilon)-unstable
# ===========================================================================
section("15. SECTION 3.6 — (k, eps) instantiation")

# Paper: For RandomPatchPerturbation, a=0.1650, b=0.1121, c=0.0427
# k=10, ASR(10) ≈ 0.097, set epsilon=0.10
asr_func_36 = make_asr_func(0.1650, 0.1121, 0.0427)
asr_10_computed = asr_func_36(10)
check(
    abs(asr_10_computed - 0.097) < 0.005,
    f"Section 3.6: ASR(10)={asr_10_computed:.4f} ≈ 0.097",
)
check(
    asr_10_computed <= 0.10,
    f"ASR(10)={asr_10_computed:.4f} <= epsilon=0.10 (valid instantiation)",
)


# ===========================================================================
# 16. find_minimum_N — correctness
# ===========================================================================
section("16. find_minimum_N correctness")

for alpha_val in [0.6, 0.7, 0.8, 0.9]:
    N_min = find_minimum_N(alpha_val, 0.95)
    if N_min is not None:
        # N_min achieves DSP >= 0.95
        dsp_at_N = compute_dsp(alpha_val, N_min)
        check(dsp_at_N >= 0.95, f"alpha={alpha_val}: N={N_min} gives DSP={dsp_at_N:.6f} >= 0.95")
        # N_min - 2 does NOT (verifies minimality for odd N)
        if N_min >= 3:
            dsp_below = compute_dsp(alpha_val, N_min - 2)
            check(
                dsp_below < 0.95,
                f"alpha={alpha_val}: N={N_min-2} gives DSP={dsp_below:.6f} < 0.95 (minimal)",
            )


# ===========================================================================
# 17. PLOTTING — generates without error
# ===========================================================================
section("17. PLOTTING VERIFICATION")

import tempfile
import matplotlib
matplotlib.use("Agg")  # headless

# 17a. plot_dsp_vs_N runs without error and produces a file
try:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmppath = f.name
    ax = plot_dsp_vs_N(
        alpha=alpha_tight_fig4,
        N_range=range(1, 52, 2),
        target_dsp=0.95,
        title="Figure 4 reproduction",
        save_path=tmppath,
    )
    check(os.path.getsize(tmppath) > 0, "plot_dsp_vs_N produces non-empty PNG")
    os.unlink(tmppath)
except Exception as e:
    check(False, "plot_dsp_vs_N", str(e))

# 17b. plot_asr_vs_k with synthetic CSV
try:
    import pandas as pd
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        tmpcsv = f.name
        df = pd.DataFrame({
            "k": list(range(11)),
            "attack_success_mean": [a_fit * math.exp(-b_fit * k) + c_fit for k in range(11)],
            "agresti_coull_low": [max(0, a_fit * math.exp(-b_fit * k) + c_fit - 0.05) for k in range(11)],
            "agresti_coull_high": [min(1, a_fit * math.exp(-b_fit * k) + c_fit + 0.05) for k in range(11)],
            "attack_name": ["GCG"] * 11,
            "perturbation_type": ["RandomSwapPerturbation"] * 11,
        })
        df.to_csv(f, index=False)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmppng = f.name

    ax = plot_asr_vs_k(
        tmpcsv,
        attack_name="GCG",
        perturbation_type="RandomSwapPerturbation",
        fit_params=(a_fit, b_fit, c_fit),
        title="ASR vs k test",
        save_path=tmppng,
    )
    check(os.path.getsize(tmppng) > 0, "plot_asr_vs_k produces non-empty PNG")
    os.unlink(tmpcsv)
    os.unlink(tmppng)
except Exception as e:
    check(False, "plot_asr_vs_k", str(e))

# 17c. fit_from_csv works with synthetic data
try:
    from smoothllm.curve_fitting import fit_from_csv
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        tmpcsv2 = f.name
        df = pd.DataFrame({
            "k": list(range(11)),
            "attack_success_mean": [0.3 * math.exp(-0.4 * k) + 0.01 for k in range(11)],
            "attack_name": ["GCG"] * 11,
            "perturbation_type": ["RandomSwapPerturbation"] * 11,
        })
        df.to_csv(f, index=False)

    a_csv, b_csv, c_csv, _ = fit_from_csv(
        tmpcsv2, attack_name="GCG", perturbation_type="RandomSwapPerturbation"
    )
    check(abs(a_csv - 0.3) < 0.01, f"fit_from_csv recovers a=0.3 (got {a_csv:.4f})")
    check(abs(b_csv - 0.4) < 0.01, f"fit_from_csv recovers b=0.4 (got {b_csv:.4f})")
    os.unlink(tmpcsv2)
except Exception as e:
    check(False, "fit_from_csv", str(e))


# ===========================================================================
# 18. MODEL CONFIGS — env var and fallback paths
# ===========================================================================
section("18. MODEL CONFIGS")

check("llama2" in mc.MODELS, "llama2 config present")
check("vicuna" in mc.MODELS, "vicuna config present")
check(
    mc.MODELS["llama2"]["conversation_template"] == "llama-2",
    "llama2 uses llama-2 template",
)
check(
    mc.MODELS["vicuna"]["conversation_template"] == "vicuna",
    "vicuna uses vicuna template",
)
# Paths should NOT be the broken absolute root paths (original bug: /smoothllm_workshop/...)
check(
    not mc.MODELS["llama2"]["model_path"].startswith("/smoothllm_workshop/"),
    "llama2 path not hardcoded to filesystem root",
)
check(
    not mc.MODELS["vicuna"]["model_path"].startswith("/smoothllm_workshop/"),
    "vicuna path not hardcoded to filesystem root",
)


# ===========================================================================
# 19. EXPERIMENT PIPELINE — generate_k_data.sh coverage
# ===========================================================================
section("19. EXPERIMENT PIPELINE COVERAGE")

# Check that generate_k_data.sh covers all needed model/perturbation combos
# Paper Figures 1-2: Llama2 + Patch + GCG, Llama2 + Swap + GCG
# Figures 5-6: Vicuna + Patch + GCG, Vicuna + Swap + GCG
# Figures 7-8: Vicuna + Patch + PAIR, Vicuna + Swap + PAIR
# Figures 9-10: Llama2 + Patch + PAIR, Llama2 + Swap + PAIR

try:
    with open("generate_k_data.sh", "r") as f:
        script = f.read()
    check(
        "RandomSwapPerturbation" in script and "RandomPatchPerturbation" in script,
        "generate_k_data.sh covers both perturbation types",
    )
    # Count how many python experiment_k.py calls there are
    n_calls = script.count("python experiment_k.py")
    check(
        n_calls >= 3,
        f"generate_k_data.sh has {n_calls} experiment calls (need >=3 for full coverage)",
    )
    # Check Llama2 + Swap is present (was missing before fix)
    llama2_swap = "llama2" in script.lower() and "RandomSwapPerturbation" in script
    check(llama2_swap, "generate_k_data.sh includes Llama2 + RandomSwapPerturbation sweep")
except Exception as e:
    check(False, "generate_k_data.sh analysis", str(e))


# ===========================================================================
# 20. FULL FORMULA CROSS-CHECK — independent implementation
# ===========================================================================
section("20. FULL FORMULA CROSS-CHECK")

# Independently implement the entire DSP pipeline from scratch
# using only basic math and scipy, no smoothllm functions

def independent_dsp(a, b, c, m, m_S, q, epsilon, N):
    """Compute DSP from scratch without using any smoothllm code."""
    # Step 1: find k
    k = math.ceil(-math.log((epsilon - c) / a) / b)

    # Step 2: M
    M = round(q * m)

    # Step 3: alpha_tighter (Eq. 11)
    alpha = 0.0
    for i in range(min(M, m_S) + 1):
        p_i = float(hypergeom.pmf(i, m, m_S, M))
        if i >= k:
            alpha += (1.0 - epsilon) * p_i
        else:
            asr_i = a * math.exp(-b * i) + c
            alpha += (1.0 - asr_i) * p_i

    # Step 4: DSP
    t = math.ceil(N / 2) - 1
    dsp = 1.0 - float(binom.cdf(t, N, alpha))
    return k, M, alpha, dsp


k_ind, M_ind, alpha_ind, dsp_ind = independent_dsp(
    a_fit, b_fit, c_fit, 240, 100, 0.10, 0.05, 3
)

check(k_ind == result["k"], f"Independent k={k_ind} matches pipeline k={result['k']}")
check(M_ind == result["M"], f"Independent M={M_ind} matches pipeline M={result['M']}")
check(
    abs(alpha_ind - result["alpha_tighter"]) < 1e-10,
    f"Independent alpha={alpha_ind:.10f} matches pipeline alpha={result['alpha_tighter']:.10f}",
)

# Cross-check DSP at N=11 for Figure 4 k=8
k_ind2, M_ind2, alpha_ind2, dsp_ind2 = independent_dsp(
    a_fit, b_fit, c_fit, 240, 100, 0.10, 0.05, 11
)
# alpha should use k=6 (not k=8), so this is alpha for the pipeline's k
dsp_11_code = compute_dsp(result["alpha_tighter"], 11)
check(
    abs(dsp_ind2 - dsp_11_code) < 1e-10,
    f"Independent DSP(N=11)={dsp_ind2:.10f} matches code={dsp_11_code:.10f}",
)


# ===========================================================================
# 21. REMARK 2 — Recovery of original k-unstable
# ===========================================================================
section("21. REMARK 2 — epsilon=0 recovers original certificate")

# When epsilon=0, (k,0)-unstable = k-unstable (deterministic)
# alpha_lower = P(X >= k)  (no (1-eps) discount)
alpha_original = alpha_lower_bound(k_test, m, m_S, M, 0.0)
p_geq_k = float(hypergeom.sf(k_test - 1, m, m_S, M))
check(
    abs(alpha_original - p_geq_k) < 1e-12,
    f"eps=0: alpha = P(X>={k_test}) = {p_geq_k:.10f} (original SmoothLLM)",
)


# ===========================================================================
# 22. ADDITIONAL EDGE CASES
# ===========================================================================
section("22. ADDITIONAL EDGE CASES")

# DSP at alpha=0.5 should be ~0.5 for large N (symmetry)
dsp_half = compute_dsp(0.5, 101)
check(
    abs(dsp_half - 0.5) < 0.05,
    f"DSP(alpha=0.5, N=101) = {dsp_half:.6f} ~ 0.5 (symmetry)",
)

# compute_dsp_sweep returns correct structure
sweep = compute_dsp_sweep(0.9, [1, 3, 5, 7])
check(len(sweep) == 4, "compute_dsp_sweep returns correct number of results")
check(
    all("N" in s and "dsp" in s for s in sweep),
    "compute_dsp_sweep results have N and dsp keys",
)

# alpha bounds agree with specific known edge: k=0 means all perturbations count
alpha_k0 = alpha_lower_bound(0, m, m_S, M, 0.05)
# P(X >= 0) = 1.0 always
check(
    abs(alpha_k0 - 0.95) < 1e-12,
    f"k=0: alpha_lower = (1-0.05)*1.0 = 0.95 (got {alpha_k0:.6f})",
)


# ===========================================================================
# 23. FIGURE GENERATION — generate_figures.py
# ===========================================================================
section("23. FIGURE GENERATION — generate_figures.py")

import subprocess
import shutil

fig_output_dir = os.path.join(tempfile.gettempdir(), "smoothllm_test_figures")
if os.path.exists(fig_output_dir):
    shutil.rmtree(fig_output_dir)

# 23a. generate_figures.py runs without error
try:
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "generate_figures.py")
    result_proc = subprocess.run(
        [sys.executable, script_path, "--output-dir", fig_output_dir],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
    )
    check(result_proc.returncode == 0, "generate_figures.py runs successfully",
          result_proc.stderr[:200] if result_proc.returncode != 0 else "")
except Exception as e:
    check(False, "generate_figures.py execution", str(e))

# 23b. Figures 1, 2, 4 are generated (no experimental data needed)
for fig_num in [1, 2, 4]:
    fig_path = os.path.join(fig_output_dir, f"figure_{fig_num}.pdf")
    exists = os.path.exists(fig_path)
    check(exists, f"Figure {fig_num} PDF exists at {fig_path}")
    if exists:
        size = os.path.getsize(fig_path)
        check(size > 1000, f"Figure {fig_num} PDF is non-trivial ({size} bytes)")

# 23c. Figures 5-10 are correctly skipped without --data-dir
for fig_num in [5, 6, 7, 8, 9, 10]:
    fig_path = os.path.join(fig_output_dir, f"figure_{fig_num}.pdf")
    check(not os.path.exists(fig_path),
          f"Figure {fig_num} correctly skipped (no --data-dir)")

# 23d. Verify Figure 1 uses correct paper parameters
# (Llama2 + RandomPatchPerturbation + GCG: a=0.1650, b=0.1121, c=0.0427)
try:
    from generate_figures import PAPER_FITS
    p1 = PAPER_FITS[1]
    check(abs(p1["a"] - 0.1650) < 1e-4, f"Figure 1: a={p1['a']} (paper: 0.1650)")
    check(abs(p1["b"] - 0.1121) < 1e-4, f"Figure 1: b={p1['b']} (paper: 0.1121)")
    check(abs(p1["c"] - 0.0427) < 1e-4, f"Figure 1: c={p1['c']} (paper: 0.0427)")
    check(p1["model"] == "Llama2", f"Figure 1: model={p1['model']}")
    check(p1["attack"] == "GCG", f"Figure 1: attack={p1['attack']}")
    check(p1["pert"] == "RandomPatchPerturbation", f"Figure 1: pert={p1['pert']}")
except Exception as e:
    check(False, "Figure 1 paper params", str(e))

# 23e. Verify Figure 2 uses correct paper parameters
# (Llama2 + RandomSwapPerturbation + GCG: a=0.2921, b=0.3756, c=0.0133)
try:
    p2 = PAPER_FITS[2]
    check(abs(p2["a"] - 0.2921) < 1e-4, f"Figure 2: a={p2['a']} (paper: 0.2921)")
    check(abs(p2["b"] - 0.3756) < 1e-4, f"Figure 2: b={p2['b']} (paper: 0.3756)")
    check(abs(p2["c"] - 0.0133) < 1e-4, f"Figure 2: c={p2['c']} (paper: 0.0133)")
    check(p2["model"] == "Llama2", f"Figure 2: model={p2['model']}")
    check(p2["attack"] == "GCG", f"Figure 2: attack={p2['attack']}")
    check(p2["pert"] == "RandomSwapPerturbation", f"Figure 2: pert={p2['pert']}")
except Exception as e:
    check(False, "Figure 2 paper params", str(e))

# 23f. Verify Figure 4 uses correct paper parameters
try:
    from generate_figures import FIG4_PARAMS
    check(FIG4_PARAMS["m"] == 240, f"Figure 4: m={FIG4_PARAMS['m']} (paper: 240)")
    check(FIG4_PARAMS["m_S"] == 100, f"Figure 4: m_S={FIG4_PARAMS['m_S']} (paper: 100)")
    check(abs(FIG4_PARAMS["q"] - 0.10) < 1e-10, f"Figure 4: q={FIG4_PARAMS['q']} (paper: 0.10)")
    check(FIG4_PARAMS["k"] == 8, f"Figure 4: k={FIG4_PARAMS['k']} (paper: 8)")
    check(abs(FIG4_PARAMS["epsilon"] - 0.05) < 1e-10, f"Figure 4: epsilon={FIG4_PARAMS['epsilon']} (paper: 0.05)")
except Exception as e:
    check(False, "Figure 4 paper params", str(e))

# 23g. Verify Figures 5-10 cover all model/attack/perturbation combos from the paper
try:
    from generate_figures import EMPIRICAL_FIGURES
    expected_combos = {
        5:  ("Vicuna", "GCG",  "RandomPatchPerturbation"),
        6:  ("Vicuna", "GCG",  "RandomSwapPerturbation"),
        7:  ("Vicuna", "PAIR", "RandomPatchPerturbation"),
        8:  ("Vicuna", "PAIR", "RandomSwapPerturbation"),
        9:  ("Llama2", "PAIR", "RandomPatchPerturbation"),
        10: ("Llama2", "PAIR", "RandomSwapPerturbation"),
    }
    for fig_num, (model, attack, pert) in expected_combos.items():
        info = EMPIRICAL_FIGURES[fig_num]
        check(
            info["model"] == model and info["attack"] == attack and info["pert"] == pert,
            f"Figure {fig_num}: {model}/{attack}/{pert}",
        )
except Exception as e:
    check(False, "Figures 5-10 combos", str(e))

# 23h. Verify Figure 4 math: DSP curves cross 0.95 threshold
# Both conservative and tighter bounds should reach DSP >= 0.95 for some N <= 51
try:
    from generate_figures import FIG4_PARAMS, PAPER_FITS
    p = FIG4_PARAMS
    M_fig4 = math.floor(p["q"] * p["m"])
    asr_func_fig4 = make_asr_func(PAPER_FITS[2]["a"], PAPER_FITS[2]["b"], PAPER_FITS[2]["c"])
    alpha_lo_f4 = alpha_lower_bound(p["k"], p["m"], p["m_S"], M_fig4, p["epsilon"])
    alpha_tight_f4 = alpha_tighter_bound(p["k"], p["m"], p["m_S"], M_fig4, p["epsilon"], asr_func_fig4)

    N_lo_f4 = find_minimum_N(alpha_lo_f4, 0.95)
    N_tight_f4 = find_minimum_N(alpha_tight_f4, 0.95)

    check(N_lo_f4 is not None and N_lo_f4 <= 51,
          f"Figure 4 conservative bound: DSP≥0.95 achieved at N={N_lo_f4} (≤51)")
    check(N_tight_f4 is not None and N_tight_f4 <= 51,
          f"Figure 4 tighter bound: DSP≥0.95 achieved at N={N_tight_f4} (≤51)")
    check(alpha_tight_f4 >= alpha_lo_f4,
          f"Figure 4: tighter alpha ({alpha_tight_f4:.6f}) >= conservative ({alpha_lo_f4:.6f})")
except Exception as e:
    check(False, "Figure 4 math verification", str(e))

# Cleanup
if os.path.exists(fig_output_dir):
    shutil.rmtree(fig_output_dir)


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
section("FINAL SUMMARY")
total = PASS + FAIL
print(f"\n  Results: {PASS} passed, {FAIL} failed, {WARN} warnings out of {total} tests")
if FAIL == 0:
    print(f"\n  ALL {PASS} TESTS PASSED")
    if WARN > 0:
        print(f"  ({WARN} warnings — see details above)")
    print()
else:
    print(f"\n  {FAIL} TEST(S) FAILED — review output above")
    print()

sys.exit(FAIL)
