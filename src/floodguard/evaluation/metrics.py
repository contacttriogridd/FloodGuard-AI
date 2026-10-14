"""Regression + classification metrics, bootstrap CIs and statistical tests."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score


# ------------------------------------------------------------- regression
# NumPy implementations (identical to sklearn's, without input-validation overhead —
# they are called thousands of times by the bootstrap).
def r2(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else 0.0


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - np.asarray(p, dtype=float)) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))))


REGRESSION: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {"r2": r2, "rmse": rmse, "mae": mae}


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {k: f(y, p) for k, f in REGRESSION.items()}


# --------------------------------------------------------- classification
def to_risk_class(values: np.ndarray, cutoffs: Sequence[float]) -> np.ndarray:
    return np.digitize(np.asarray(values), list(cutoffs))


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    """ECE with equal-width bins: Σ_b (n_b / n) |acc_b − conf_b|."""
    y, prob = np.asarray(y), np.asarray(prob)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - prob[m].mean())
    return float(ece)


def reliability_curve(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> dict[str, list[float]]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    conf, acc, count = [], [], []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf.append(float(prob[m].mean()))
            acc.append(float(np.asarray(y)[m].mean()))
            count.append(int(m.sum()))
    return {"confidence": conf, "accuracy": acc, "count": count}


def classification_metrics(classes_true: np.ndarray, classes_pred: np.ndarray, high_true: np.ndarray,
                           high_prob: np.ndarray, bins: int = 10) -> dict[str, float]:
    high_prob = np.clip(high_prob, 1e-7, 1 - 1e-7)
    out = {"macro_f1": float(f1_score(classes_true, classes_pred, average="macro"))}
    if len(np.unique(high_true)) == 2:
        out.update({"roc_auc": float(roc_auc_score(high_true, high_prob)),
                    "pr_auc": float(average_precision_score(high_true, high_prob)),
                    "log_loss": float(log_loss(high_true, high_prob, labels=[0, 1])),
                    "brier": float(brier_score_loss(high_true, high_prob)),
                    "ece": expected_calibration_error(high_true, high_prob, bins)})
    return out


# ---------------------------------------------------------------- bootstrap
def bootstrap_ci(metric: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray, p: np.ndarray,
                 n: int = 1000, ci: float = 0.95, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    samples = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        samples[i] = metric(y[idx], p[idx])
    alpha = (1 - ci) / 2
    return {"value": metric(y, p), "lo": float(np.quantile(samples, alpha)), "hi": float(np.quantile(samples, 1 - alpha)),
            "std": float(samples.std())}


def paired_bootstrap(metric: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray, p_a: np.ndarray,
                     p_b: np.ndarray, n: int = 1000, ci: float = 0.95, seed: int = 42) -> dict[str, float]:
    """Bootstrap of metric(a) − metric(b) on the same resamples; one-sided p = P(diff ≤ 0)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx])
    alpha = (1 - ci) / 2
    return {"diff": metric(y, p_a) - metric(y, p_b), "lo": float(np.quantile(diffs, alpha)),
            "hi": float(np.quantile(diffs, 1 - alpha)), "p_value": float((diffs <= 0).mean())}


# ---------------------------------------------------------- hypothesis tests
def wilcoxon_folds(scores_a: Sequence[float], scores_b: Sequence[float]) -> dict[str, float | str]:
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    if len(a) < 2 or np.allclose(a, b):
        return {"statistic": None, "p_value": None, "n": int(len(a)), "note": "too few folds or identical scores"}
    res = stats.wilcoxon(a, b, alternative="greater")
    note = "with 5 folds the smallest attainable one-sided p is 0.031" if len(a) <= 5 else ""
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "n": int(len(a)), "note": note}


def friedman_nemenyi(score_matrix: np.ndarray, names: Sequence[str], alpha: float = 0.05) -> dict:
    """Friedman test over blocks (rows) × models (columns) + Nemenyi critical difference."""
    n_blocks, k = score_matrix.shape
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, score_matrix)  # rank 1 = best
    avg = ranks.mean(axis=0)
    stat, p = stats.friedmanchisquare(*score_matrix.T) if k >= 3 else (np.nan, np.nan)
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6 * n_blocks)))
    return {"statistic": float(stat), "p_value": float(p), "n_blocks": int(n_blocks), "k": int(k), "alpha": alpha,
            "critical_difference": cd, "average_ranks": dict(zip(names, avg.round(4).tolist()))}
