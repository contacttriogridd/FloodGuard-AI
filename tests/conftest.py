"""Shared fixtures: synthetic frames and a tiny model bundle in the real save format."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from floodguard.utils.common import save_json  # noqa: E402
from floodguard.utils.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def factors(cfg):
    return list(cfg.features.factors)


def make_frame(factors: list[str], n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(np.clip(rng.poisson(5, size=(n, len(factors))), 0, 16), columns=factors).astype(float)
    y = 0.5 + 0.0056 * (X.sum(axis=1).to_numpy() - 100) + rng.normal(0, 0.02, n)
    X["is_original"] = 0.0
    return X, np.clip(y, 0.28, 0.73)


@pytest.fixture(scope="session")
def frame(factors):
    return make_frame(factors)


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory, cfg, factors):
    """A small ensemble (Ridge + LightGBM blend) saved exactly like the ensemble stage does."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from floodguard.features.engineering import ColumnSelector, build_feature_pipeline
    from floodguard.models.ensemble import FloodGuardRegressor
    from floodguard.models.zoo import fit_model, make_model

    X, y = make_frame(factors, n=800, seed=1)
    feat = build_feature_pipeline(cfg, "default", "median")
    F = feat.fit_transform(X, y)
    cols = list(F.columns)
    ridge, _ = fit_model("ridge", make_model("ridge", {}, cfg), F, y, None, None, cfg, early_stopping=False)
    lgbm, _ = fit_model("lgbm", make_model("lgbm", {"n_estimators": 60}, cfg), F, y, None, None, cfg, early_stopping=False)
    model = FloodGuardRegressor([("ridge", cols, ridge), ("lgbm", cols, lgbm)], "blend", [0.6, 0.4])
    pipeline = Pipeline(feat.steps[:-1] + [("select", ColumnSelector(cols)), ("model", model)])
    pred = pipeline.predict(X)
    cut = [float(np.quantile(y, 0.333)), float(np.quantile(y, 0.667))]
    bounds = {f: [0, 16] for f in factors}
    bundle = {
        "pipeline": pipeline, "strategy": "blend_nnls", "members": ["ridge", "lgbm"], "factors": factors, "bounds": bounds,
        "risk_classes": ["Low", "Medium", "High"], "true_cutoffs": cut, "prediction_cutoffs": cut,
        "high_calibrator": LogisticRegression().fit(pred.reshape(-1, 1), (y >= cut[1]).astype(int)),
        "residual_quantiles": {0.05: -0.03, 0.95: 0.03}, "background": X.iloc[:20], "themes": dict(cfg.features.themes),
        "version": "test", "data_source": "synthetic", "trained_at": "2026-01-01T00:00:00",
    }
    out = tmp_path_factory.mktemp("artifacts")
    joblib.dump(bundle, out / "floodguard_pipeline.joblib")
    save_json({"feature_bounds": bounds, "oof_r2": None}, out / "model_card.json")
    return out
