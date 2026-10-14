"""Ensemble layer: blend weights are non-negative and sum to one; cut-off tuning; deployable model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from floodguard.models.ensemble import FloodGuardRegressor, fast_macro_f1, hill_climb_weights, nnls_weights, tune_cutoffs


def _members(seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=2000)
    P = np.column_stack([y + rng.normal(scale=s, size=2000) for s in (0.3, 0.5, 1.5)])
    return y, P


def test_nnls_weights_non_negative_and_sum_to_one():
    y, P = _members()
    w = nnls_weights(P, y)
    assert (w >= 0).all()
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[2]  # the most accurate member receives the largest weight


def test_hill_climbing_weights_non_negative_and_sum_to_one():
    y, P = _members(1)
    w = hill_climb_weights(P, y, iters=50)
    assert (w >= 0).all()
    assert np.isclose(w.sum(), 1.0)


def test_blend_is_no_worse_than_best_member():
    y, P = _members(2)
    w = nnls_weights(P, y)
    mse = lambda p: np.mean((p - y) ** 2)  # noqa: E731
    assert mse(P @ w) <= min(mse(P[:, i]) for i in range(P.shape[1])) + 1e-9


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 3, 1000), rng.integers(0, 3, 1000)
    assert np.isclose(fast_macro_f1(a, b), f1_score(a, b, average="macro"))


def test_tuned_cutoffs_are_ordered():
    rng = np.random.default_rng(4)
    pred = rng.uniform(0, 1, 3000)
    classes = np.digitize(pred + rng.normal(0, 0.05, 3000), [0.33, 0.66])
    lo, hi = tune_cutoffs(pred, classes)
    assert lo < hi and abs(lo - 0.33) < 0.1 and abs(hi - 0.66) < 0.1


class _Const:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v)


def test_flood_guard_regressor_blends_members():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    model = FloodGuardRegressor([("m1", ["a"], _Const(0.4)), ("m2", ["b"], _Const(0.6))], "blend", [0.25, 0.75])
    np.testing.assert_allclose(model.predict(X), [0.55, 0.55])
    assert model.__sklearn_is_fitted__()
