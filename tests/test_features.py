"""Feature transformers: correct shapes, deterministic outputs and no target / test leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodguard.features.engineering import (
    FactorImputer,
    FloodFeatureBuilder,
    FrequencyEncoder,
    OOFTargetEncoder,
    RowStatistics,
    SortedRowValues,
    ThemeAggregates,
    build_feature_pipeline,
)


def test_row_statistics_values(frame, factors):
    X, _ = frame
    out = RowStatistics(factors, (5, 10)).fit_transform(X)
    assert out.shape == (len(X), 9 + 2)
    np.testing.assert_allclose(out["row_sum"], X[factors].sum(axis=1))
    np.testing.assert_array_equal(out["row_ge_5"], (X[factors] >= 5).sum(axis=1))
    assert not out.isna().any().any()


def test_theme_aggregates_and_interactions(frame, factors, cfg):
    X, _ = frame
    t = ThemeAggregates(dict(cfg.features.themes), [("Urbanization", "DrainageSystems")])
    out = t.fit_transform(X)
    assert len(out.columns) == len(cfg.features.themes) + 1
    np.testing.assert_allclose(out["ix_Urbanization_x_DrainageSystems"], X["Urbanization"] * X["DrainageSystems"])


def test_sorted_values_are_sorted(frame, factors):
    X, _ = frame
    out = SortedRowValues(factors).fit_transform(X).to_numpy()
    assert out.shape == (len(X), len(factors))
    assert (np.diff(out, axis=1) >= 0).all()


def test_frequency_encoder_learns_from_train_only(factors):
    train = pd.DataFrame({f: [1, 1, 2, 3] for f in factors})
    test = pd.DataFrame({f: [1, 9, 9, 9] for f in factors})  # 9 never appears in train
    enc = FrequencyEncoder(factors).fit(train)
    maps_before = {k: dict(v) for k, v in enc.maps_.items()}
    out = enc.transform(test)
    assert out[f"freq_{factors[0]}"].tolist() == [0.5, 0.0, 0.0, 0.0]
    assert enc.maps_ == maps_before  # transforming test data must not update the encoder


def test_target_encoder_is_cross_fitted(frame, factors):
    X, y = frame
    enc = OOFTargetEncoder(["Urbanization"], cv=5)
    oof = enc.fit_transform(X, y)
    full = enc.transform(X)
    assert oof.shape == full.shape == (len(X), 1)
    # In-sample encodings use out-of-fold statistics, so they differ from the full-data mapping.
    assert not np.allclose(oof.to_numpy(), full.to_numpy())


def test_imputer_fills_with_valid_integers(frame, factors):
    X, _ = frame
    Xm = X.copy()
    Xm.iloc[::7, 3] = np.nan
    for strategy in ("median", "knn", "iterative"):
        out = FactorImputer(factors, strategy, value_range=(0, 25)).fit(Xm).transform(Xm)
        vals = out[factors].to_numpy()
        assert not np.isnan(vals).any()
        assert (vals % 1 == 0).all() and vals.min() >= 0


def test_builder_shapes_and_column_order(frame, factors, cfg):
    X, y = frame
    builder = FloodFeatureBuilder(factors=factors, themes=dict(cfg.features.themes),
                                  interactions=[tuple(p) for p in cfg.features.interactions],
                                  te_features=["Urbanization", "DrainageSystems"], te_cv=3)
    tr = builder.fit_transform(X.iloc[:400], y[:400])
    te = builder.transform(X.iloc[400:])
    assert list(tr.columns) == list(te.columns)
    assert te.shape == (200, tr.shape[1])
    assert not te.isna().any().any()


def test_pair_combination_encoder_is_high_cardinality_and_cross_fitted(frame, factors):
    from floodguard.features.engineering import PairCombinationEncoder

    X, y = frame
    enc = PairCombinationEncoder([("Urbanization", "DrainageSystems")], cv=5)
    tr = enc.fit_transform(X.iloc[:500], y[:500])
    key = "pair_Urbanization__DrainageSystems"
    assert enc.cardinality_[key] > X["Urbanization"].nunique() * 2  # far more levels than one factor
    assert list(tr.columns) == [f"freq_{key}", f"te_{key}"]
    # target encoding is out-of-fold in-sample, full-fit on new data
    assert not np.allclose(tr[f"te_{key}"], enc.transform(X.iloc[:500])[f"te_{key}"])
    unseen = X.iloc[:1].copy()
    unseen[["Urbanization", "DrainageSystems"]] = 25
    assert enc.transform(unseen)[f"freq_{key}"].iloc[0] == 0.0


def test_lgbm_gpu_switch_is_ignored_without_gpu(cfg):
    from floodguard.models.zoo import make_model
    from floodguard.utils.config import Config

    local = Config({**cfg.to_dict(), "models": {**cfg.to_dict()["models"], "use_gpu": False, "lgbm_device": "gpu"}})
    assert "device_type" not in make_model("lgbm", {}, local).get_params()


@pytest.mark.parametrize("variant", ["default", "raw", "sum_only", "no_row_features", "no_target_encoding",
                                     "no_combination_encoding"])
def test_feature_pipeline_variants(frame, cfg, variant):
    X, y = frame
    pipe = build_feature_pipeline(cfg, variant, "median")
    out = pipe.fit_transform(X.iloc[:400], y[:400])
    test = pipe.transform(X.iloc[400:])
    assert out.shape[1] == test.shape[1]
    if variant == "sum_only":
        assert list(out.columns) == ["row_sum", "is_original"]
    if variant == "no_row_features":
        assert not any(c.startswith(("row_", "sorted_")) for c in out.columns)
    if variant == "default":
        assert any(c.startswith("te_pair_") for c in out.columns)
    if variant in ("no_combination_encoding", "raw"):
        assert not any("pair_" in c for c in out.columns)


def test_autoencoder_embedding_shape(frame, factors):
    pytest.importorskip("torch")
    from floodguard.features.reduction import AutoencoderEmbedder

    X, _ = frame
    ae = AutoencoderEmbedder(factors, hidden=(8,), bottleneck=3, epochs=2, batch_size=128).fit(X)
    emb = ae.transform(X)
    assert emb.shape == (len(X), 3) and list(emb.columns) == ["ae_0", "ae_1", "ae_2"]
