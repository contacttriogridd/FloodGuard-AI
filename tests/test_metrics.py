"""Metric implementations, calibration error, bootstrap CIs and hypothesis tests."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from floodguard.evaluation import metrics as M


def test_regression_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    p = y + rng.normal(scale=0.3, size=500)
    assert np.isclose(M.r2(y, p), r2_score(y, p))
    assert np.isclose(M.rmse(y, p), np.sqrt(mean_squared_error(y, p)))
    assert np.isclose(M.mae(y, p), mean_absolute_error(y, p))


def test_ece_is_small_for_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 50000)
    y = (rng.uniform(0, 1, 50000) < p).astype(int)
    assert M.expected_calibration_error(y, p) < 0.02
    overconfident = np.where(p > 0.5, 0.99, 0.01)
    assert M.expected_calibration_error(y, overconfident) > 0.15


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.normal(size=400)
    p = y + rng.normal(scale=0.5, size=400)
    ci = M.bootstrap_ci(M.r2, y, p, n=300)
    assert ci["lo"] <= ci["value"] <= ci["hi"]
    assert ci["hi"] - ci["lo"] > 0


def test_paired_bootstrap_detects_better_model():
    rng = np.random.default_rng(3)
    y = rng.normal(size=600)
    good, bad = y + rng.normal(scale=0.2, size=600), y + rng.normal(scale=0.8, size=600)
    res = M.paired_bootstrap(M.r2, y, good, bad, n=300)
    assert res["diff"] > 0 and res["lo"] > 0 and res["p_value"] < 0.05


def test_friedman_nemenyi_ranks_and_cd():
    rng = np.random.default_rng(4)
    scores = np.column_stack([rng.normal(0.9, 0.01, 10), rng.normal(0.8, 0.01, 10), rng.normal(0.7, 0.01, 10)])
    res = M.friedman_nemenyi(scores, ["a", "b", "c"])
    assert res["average_ranks"]["a"] < res["average_ranks"]["c"]
    assert res["p_value"] < 0.01 and res["critical_difference"] > 0


def test_classification_metrics_and_risk_classes():
    assert M.to_risk_class(np.array([0.1, 0.5, 0.9]), [0.4, 0.6]).tolist() == [0, 1, 2]
    y_cls = np.array([0, 1, 2, 2, 1, 0])
    prob = np.array([0.1, 0.3, 0.9, 0.8, 0.4, 0.05])
    out = M.classification_metrics(y_cls, y_cls, (y_cls == 2).astype(int), prob)
    assert out["macro_f1"] == 1.0 and out["roc_auc"] == 1.0
    assert set(out) == {"macro_f1", "roc_auc", "pr_auc", "log_loss", "brier", "ece"}
