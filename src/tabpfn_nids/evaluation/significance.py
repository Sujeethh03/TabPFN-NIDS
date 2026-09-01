"""Statistical comparison of two experiment arms (Build Plan step 5.2).

A caution that governs how this module should be used. The Wilcoxon
signed-rank test needs enough paired observations before a two-sided p-value
below 0.05 is even *reachable*: at n=5 the smallest attainable p is 0.0625,
and at n=3 it is 0.25. Running three seeds and reporting "p = 0.25, not
significant" says nothing about the data -- the design made significance
impossible before any number was computed.

``minimum_reachable_p`` exposes that bound, and ``wilcoxon_test`` refuses to
return a p-value it could never have rejected, returning ``underpowered=True``
instead. For small seed counts, prefer ``summarize_paired`` and report
mean +/- std with the observed differences.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Smallest two-sided p attainable by the Wilcoxon signed-rank test, by number
# of non-zero paired differences. Below n=6 the test cannot reach 0.05.
MINIMUM_REACHABLE_P: dict[int, float] = {
    1: 1.0,
    2: 0.5,
    3: 0.25,
    4: 0.125,
    5: 0.0625,
    6: 0.03125,
}

ALPHA: float = 0.05


def minimum_reachable_p(n_pairs: int) -> float:
    """Return the smallest two-sided p-value Wilcoxon can produce.

    Args:
        n_pairs: Number of non-zero paired differences.

    Returns:
        The floor on the two-sided p-value. For n >= 6 this is below 0.05, so
        the test can in principle reject.
    """
    if n_pairs <= 0:
        return 1.0
    return MINIMUM_REACHABLE_P.get(n_pairs, 2.0 ** (-(n_pairs - 1)))


def wilcoxon_test(
    baseline_scores: list[float] | np.ndarray,
    enhanced_scores: list[float] | np.ndarray,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Compare two arms with a paired Wilcoxon signed-rank test.

    Scores must be paired: element i of each sequence is the same seed or
    fold, evaluated under the two conditions.

    Args:
        baseline_scores: Per-seed metric values for the baseline arm.
        enhanced_scores: Per-seed metric values for the enhanced arm.
        alpha: Significance threshold.

    Returns:
        A dict containing ``n_pairs``, ``mean_difference``,
        ``median_difference``, ``min_reachable_p``, ``underpowered``,
        ``statistic``, ``p_value`` and ``significant``. When the design is
        underpowered, ``p_value`` and ``significant`` are None and a warning
        is logged: a p-value that could never have crossed alpha is not
        evidence and should not be reported as though it were.

    Raises:
        ValueError: If the two sequences differ in length or are empty.

    Example:
        >>> wilcoxon_test([0.73, 0.75, 0.77], [0.74, 0.76, 0.79])["underpowered"]
        True
    """
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    enhanced = np.asarray(enhanced_scores, dtype=np.float64)

    if baseline.shape != enhanced.shape:
        raise ValueError(
            f"paired comparison needs equal lengths; got {baseline.shape} "
            f"and {enhanced.shape}. Scores must be paired by seed."
        )
    if baseline.size == 0:
        raise ValueError("cannot compare empty score sequences")

    differences = enhanced - baseline
    n_nonzero = int(np.count_nonzero(differences))
    floor = minimum_reachable_p(n_nonzero)
    underpowered = floor > alpha

    result: dict[str, Any] = {
        "n_pairs": int(baseline.size),
        "n_nonzero_differences": n_nonzero,
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "min_reachable_p": floor,
        "underpowered": underpowered,
        "statistic": None,
        "p_value": None,
        "significant": None,
    }

    if underpowered:
        logger.warning(
            "Wilcoxon is underpowered with %d non-zero differences: the "
            "smallest reachable two-sided p is %.4f, above alpha=%.2f. "
            "Reporting mean +/- std instead of a p-value.",
            n_nonzero,
            floor,
            alpha,
        )
        return result

    statistic, p_value = stats.wilcoxon(baseline, enhanced)
    result["statistic"] = float(statistic)
    result["p_value"] = float(p_value)
    result["significant"] = bool(p_value < alpha)
    return result


def summarize_paired(
    baseline_scores: list[float] | np.ndarray,
    enhanced_scores: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Describe two arms with means, spreads and their difference.

    This is the honest report at small seed counts, where a hypothesis test
    cannot reach significance.

    Args:
        baseline_scores: Per-seed values for the baseline arm.
        enhanced_scores: Per-seed values for the enhanced arm.

    Returns:
        A dict of per-arm mean and standard deviation, the mean difference,
        and whether that difference exceeds the baseline's own seed-to-seed
        spread -- a far more meaningful check at n=3 than a p-value.

    Raises:
        ValueError: If the sequences differ in length or are empty.
    """
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    enhanced = np.asarray(enhanced_scores, dtype=np.float64)

    if baseline.shape != enhanced.shape:
        raise ValueError("paired comparison needs equal lengths")
    if baseline.size == 0:
        raise ValueError("cannot summarise empty score sequences")

    baseline_std = (
        statistics.stdev(baseline.tolist()) if baseline.size > 1 else 0.0
    )
    enhanced_std = (
        statistics.stdev(enhanced.tolist()) if enhanced.size > 1 else 0.0
    )
    difference = float(np.mean(enhanced) - np.mean(baseline))

    return {
        "n": int(baseline.size),
        "baseline_mean": float(np.mean(baseline)),
        "baseline_std": baseline_std,
        "enhanced_mean": float(np.mean(enhanced)),
        "enhanced_std": enhanced_std,
        "mean_difference": difference,
        "difference_exceeds_baseline_spread": abs(difference) > baseline_std,
    }
