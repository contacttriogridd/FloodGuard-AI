"""Leakage-free feature engineering as sklearn transformers.

Every transformer learns only from the rows passed to ``fit`` (the training
fold). Target encoding is cross-fitted inside ``fit_transform`` (out-of-fold),
so training rows never see their own target.

    python -m floodguard.features.engineering --override configs/smoke.yaml   # runs feature selection
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

PASSTHROUGH = ("is_original",)


class _Stateless:
    """Marks transformers that learn nothing as fitted (sklearn's check_is_fitted)."""

    def __sklearn_is_fitted__(self) -> bool:
        return True


# ----------------------------------------------------------------- imputation
class FactorImputer(TransformerMixin, BaseEstimator):
    """Median / KNN / Iterative imputation of the factor columns.

    KNN and iterative imputers are fitted on at most ``max_fit_rows`` rows so
    the serialised pipeline stays small. Imputed values are rounded to the
    nearest valid integer because the factors are integer scores.
    """

    def __init__(self, factors: Sequence[str] = (), strategy: str = "median", max_fit_rows: int = 30000,
                 round_to_int: bool = True, value_range: tuple[float, float] = (0, 25), random_state: int = 42):
        self.factors = factors
        self.strategy = strategy
        self.max_fit_rows = max_fit_rows
        self.round_to_int = round_to_int
        self.value_range = value_range
        self.random_state = random_state

    def _make(self):
        if self.strategy == "median":
            return SimpleImputer(strategy="median")
        if self.strategy == "knn":
            return KNNImputer(n_neighbors=5)
        if self.strategy == "iterative":
            return IterativeImputer(max_iter=10, random_state=self.random_state)
        raise ValueError(f"Unknown imputer strategy '{self.strategy}'")

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FactorImputer":
        Xf = X[list(self.factors)]
        if self.strategy != "median" and len(Xf) > self.max_fit_rows:
            Xf = Xf.sample(n=self.max_fit_rows, random_state=self.random_state)
        self.imputer_ = self._make().fit(Xf.to_numpy(dtype=float))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        cols = list(self.factors)
        values = out[cols].to_numpy(dtype=float, copy=True)  # pandas copy-on-write returns read-only views
        rows = np.isnan(values).any(axis=1)
        if rows.any():
            filled = self.imputer_.transform(values[rows])
            if self.round_to_int:
                filled = np.clip(np.rint(filled), *self.value_range)
            values[rows] = filled
            out[cols] = values
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(input_features if input_features is not None else self.factors, dtype=object)


# --------------------------------------------------------------- row features
class RowStatistics(_Stateless, TransformerMixin, BaseEstimator):
    """Row statistics across the factors: sum, mean, std, … and counts of values ≥ k."""

    def __init__(self, factors: Sequence[str] = (), count_thresholds: Sequence[int] = (5, 6, 7, 8, 9, 10)):
        self.factors = factors
        self.count_thresholds = count_thresholds

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RowStatistics":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        import warnings

        V = X[list(self.factors)].to_numpy(dtype=float)
        # Constant rows have undefined skew/kurtosis (set to 0 below); silence scipy's precision warning.
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = {
                "row_sum": V.sum(1), "row_mean": V.mean(1), "row_std": V.std(1), "row_min": V.min(1),
                "row_max": V.max(1), "row_median": np.median(V, 1),
                "row_skew": np.nan_to_num(stats.skew(V, axis=1)), "row_kurt": np.nan_to_num(stats.kurtosis(V, axis=1)),
                "row_range": V.max(1) - V.min(1),
            }
        for k in self.count_thresholds:
            out[f"row_ge_{k}"] = (V >= k).sum(1)
        return pd.DataFrame(out, index=X.index)


class ThemeAggregates(_Stateless, TransformerMixin, BaseEstimator):
    """Theme means (hydro-climatic, urban, environmental, governance) and interaction products."""

    def __init__(self, themes: dict[str, Sequence[str]] | None = None, interactions: Sequence[Sequence[str]] = ()):
        self.themes = themes
        self.interactions = interactions

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ThemeAggregates":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {f"theme_{name}": X[list(cols)].mean(axis=1) for name, cols in (self.themes or {}).items()}
        for a, b in self.interactions:
            out[f"ix_{a}_x_{b}"] = X[a] * X[b]
        return pd.DataFrame(out, index=X.index)


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Value-count encoding: each integer factor treated as a high-cardinality categorical.

    Counts come from the training rows plus optional ``reference_counts`` taken
    from *unlabeled* data (the competition test set) — no target is involved.
    """

    def __init__(self, factors: Sequence[str] = (), reference_counts: dict[str, dict] | None = None):
        self.factors = factors
        self.reference_counts = reference_counts

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FrequencyEncoder":
        self.maps_: dict[str, dict[float, float]] = {}
        for c in self.factors:
            counts = X[c].value_counts()
            ref = (self.reference_counts or {}).get(c)
            if ref:
                counts = counts.add(pd.Series(ref, dtype=float), fill_value=0)
            total = float(counts.sum())
            self.maps_[c] = {float(k): float(v) / total for k, v in counts.items()}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f"freq_{c}": X[c].astype(float).map(self.maps_[c]).fillna(0.0).to_numpy()
                             for c in self.factors}, index=X.index)


class SortedRowValues(_Stateless, TransformerMixin, BaseEstimator):
    """The factor values of each row sorted ascending (permutation-invariant view)."""

    def __init__(self, factors: Sequence[str] = ()):
        self.factors = factors

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SortedRowValues":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        V = np.sort(X[list(self.factors)].to_numpy(dtype=float), axis=1)
        return pd.DataFrame(V, columns=[f"sorted_{i:02d}" for i in range(V.shape[1])], index=X.index)


class OOFTargetEncoder(TransformerMixin, BaseEstimator):
    """Smoothed target encoding, cross-fitted (out-of-fold) during ``fit_transform``."""

    def __init__(self, features: Sequence[str] = (), cv: int = 5, smooth: str | float = "auto", random_state: int = 42):
        self.features = features
        self.cv = cv
        self.smooth = smooth
        self.random_state = random_state

    def _encoder(self) -> TargetEncoder:
        # scikit-learn >= 1.3: cv must be an int (TargetEncoder handles shuffling internally).
        return TargetEncoder(target_type="continuous", smooth=self.smooth, cv=int(self.cv))

    def _frame(self, values: np.ndarray, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(values, columns=[f"te_{c}" for c in self.features], index=index)

    def fit(self, X: pd.DataFrame, y: Any) -> "OOFTargetEncoder":
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **_: Any) -> pd.DataFrame:
        if y is None:
            raise ValueError("OOFTargetEncoder needs the target during fit")
        self.encoder_ = self._encoder()
        return self._frame(self.encoder_.fit_transform(X[list(self.features)].astype(int), np.asarray(y)), X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._frame(self.encoder_.transform(X[list(self.features)].astype(int)), X.index)


class PairCombinationEncoder(TransformerMixin, BaseEstimator):
    """High-cardinality encoding of factor *pairs*.

    Each pair (a, b) is turned into one categorical key with up to
    ``(range_a + 1) × (range_b + 1)`` levels (e.g. ~400 for Urbanization ×
    DrainageSystems), then encoded by frequency and by smoothed target encoding
    that is cross-fitted (out-of-fold) during ``fit_transform``.
    """

    def __init__(self, pairs: Sequence[Sequence[str]] = (), target_encoding: bool = True, cv: int = 5,
                 random_state: int = 42):
        self.pairs = pairs
        self.target_encoding = target_encoding
        self.cv = cv
        self.random_state = random_state

    @staticmethod
    def _name(a: str, b: str) -> str:
        return f"pair_{a}__{b}"

    def _keys(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({self._name(a, b): np.rint(X[a]).astype(int) * 1000 + np.rint(X[b]).astype(int)
                             for a, b in self.pairs}, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **_: Any) -> pd.DataFrame:
        keys = self._keys(X)
        self.cardinality_ = {c: int(keys[c].nunique()) for c in keys.columns}
        self.freq_maps_ = {c: (keys[c].value_counts() / len(keys)).to_dict() for c in keys.columns}
        frames = [self._freq(keys)]
        if self.target_encoding:
            if y is None:
                raise ValueError("PairCombinationEncoder needs the target for target encoding")
            from sklearn.model_selection import KFold

            self.te_ = TargetEncoder(target_type="continuous", smooth="auto", cv=int(self.cv))
            frames.append(pd.DataFrame(self.te_.fit_transform(keys, np.asarray(y)),
                                       columns=[f"te_{c}" for c in keys.columns], index=X.index))
        return pd.concat(frames, axis=1)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PairCombinationEncoder":
        self.fit_transform(X, y)
        return self

    def _freq(self, keys: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f"freq_{c}": keys[c].map(self.freq_maps_[c]).fillna(0.0).to_numpy() for c in keys.columns},
                            index=keys.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keys = self._keys(X)
        frames = [self._freq(keys)]
        if self.target_encoding:
            frames.append(pd.DataFrame(self.te_.transform(keys), columns=[f"te_{c}" for c in keys.columns], index=X.index))
        return pd.concat(frames, axis=1)


class ColumnSelector(_Stateless, TransformerMixin, BaseEstimator):
    """Keep a fixed list of columns (selected features)."""

    def __init__(self, columns: Sequence[str] | None = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ColumnSelector":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            return X
        return X[[c for c in self.columns if c in X.columns]]


# --------------------------------------------------------------- composition
class FloodFeatureBuilder(TransformerMixin, BaseEstimator):
    """Raw factors + passthrough flags + every enabled feature group."""

    def __init__(self, factors: Sequence[str] = (), themes: dict | None = None, interactions: Sequence = (),
                 row_stats: bool = True, count_thresholds: Sequence[int] = (5, 6, 7, 8, 9, 10),
                 theme_aggregates: bool = True, frequency_encoding: bool = True,
                 frequency_reference: dict | None = None, sorted_values: bool = True, target_encoding: bool = True,
                 te_features: Sequence[str] = (), te_cv: int = 5, autoencoder: Any = None,
                 passthrough: Sequence[str] = PASSTHROUGH, random_state: int = 42,
                 combination_encoding: bool = False, combination_pairs: Sequence[Sequence[str]] = ()):
        self.factors = factors
        self.themes = themes
        self.interactions = interactions
        self.row_stats = row_stats
        self.count_thresholds = count_thresholds
        self.theme_aggregates = theme_aggregates
        self.frequency_encoding = frequency_encoding
        self.frequency_reference = frequency_reference
        self.sorted_values = sorted_values
        self.target_encoding = target_encoding
        self.te_features = te_features
        self.te_cv = te_cv
        self.autoencoder = autoencoder
        self.passthrough = passthrough
        self.random_state = random_state
        self.combination_encoding = combination_encoding
        self.combination_pairs = combination_pairs

    def _uses_pairs(self) -> bool:
        # getattr keeps builders pickled before this option existed loadable.
        return bool(getattr(self, "combination_encoding", False) and getattr(self, "combination_pairs", ()))

    def _stateless(self) -> list[TransformerMixin]:
        parts: list[TransformerMixin] = []
        if self.row_stats:
            parts.append(RowStatistics(self.factors, self.count_thresholds))
        if self.theme_aggregates:
            parts.append(ThemeAggregates(self.themes, self.interactions))
        if self.sorted_values:
            parts.append(SortedRowValues(self.factors))
        return parts

    def _base(self, X: pd.DataFrame) -> list[pd.DataFrame]:
        keep = list(self.factors) + [c for c in self.passthrough if c in X.columns]
        frames = [X[keep].astype(float)] + [p.transform(X) for p in self._stateless()]
        if self.autoencoder is not None:
            frames.append(self.autoencoder.transform(X))
        return frames

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **_: Any) -> pd.DataFrame:
        frames = self._base(X)
        if self.frequency_encoding:
            self.freq_ = FrequencyEncoder(self.factors, self.frequency_reference).fit(X)
            frames.append(self.freq_.transform(X))
        if self.target_encoding and self.te_features:
            self.te_ = OOFTargetEncoder(self.te_features, cv=self.te_cv, random_state=self.random_state)
            frames.append(self.te_.fit_transform(X, y))
        if self._uses_pairs():
            self.pairs_ = PairCombinationEncoder([tuple(p) for p in self.combination_pairs], self.target_encoding,
                                                 cv=self.te_cv, random_state=self.random_state)
            frames.append(self.pairs_.fit_transform(X, y))
        out = pd.concat(frames, axis=1)
        self.feature_names_out_ = list(out.columns)
        return out

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FloodFeatureBuilder":
        self.fit_transform(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frames = self._base(X)
        if self.frequency_encoding:
            frames.append(self.freq_.transform(X))
        if self.target_encoding and self.te_features:
            frames.append(self.te_.transform(X))
        if self._uses_pairs():
            frames.append(self.pairs_.transform(X))
        return pd.concat(frames, axis=1)[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)


# ------------------------------------------------------------------- variants
VARIANTS = ("default", "raw", "sum_only", "ae", "no_row_features", "no_target_encoding", "no_combination_encoding")


def variant_flags(cfg, variant: str = "default") -> dict[str, Any]:
    f = cfg.features
    flags = dict(row_stats=f.row_stats, theme_aggregates=f.theme_aggregates, frequency_encoding=f.frequency_encoding,
                 sorted_values=f.sorted_values, target_encoding=f.target_encoding, use_autoencoder=False,
                 combination_encoding=bool(f.get("combination_encoding", False)))
    if variant in ("raw", "sum_only"):
        flags.update(theme_aggregates=False, frequency_encoding=False, sorted_values=False, target_encoding=False,
                     combination_encoding=False, row_stats=variant == "sum_only")
    elif variant == "no_combination_encoding":
        flags["combination_encoding"] = False
    elif variant == "ae":
        flags["use_autoencoder"] = True
    elif variant == "no_row_features":
        flags.update(row_stats=False, sorted_values=False)
    elif variant == "no_target_encoding":
        flags["target_encoding"] = False
    elif variant != "default":
        raise ValueError(f"Unknown feature variant '{variant}'")
    return flags


def build_feature_pipeline(cfg, variant: str = "default", imputer: str = "median", selected: Sequence[str] | None = None,
                           autoencoder: Any = None, frequency_reference: dict | None = None) -> Pipeline:
    """imputer -> feature engineering -> selection (sklearn Pipeline, fitted per fold)."""
    flags = variant_flags(cfg, variant)
    builder = FloodFeatureBuilder(
        factors=list(cfg.features.factors), themes=dict(cfg.features.themes),
        interactions=[tuple(p) for p in cfg.features.interactions], row_stats=flags["row_stats"],
        count_thresholds=list(cfg.features.count_thresholds), theme_aggregates=flags["theme_aggregates"],
        frequency_encoding=flags["frequency_encoding"],
        frequency_reference=frequency_reference if cfg.features.get("frequency_use_unlabeled_test") else None,
        sorted_values=flags["sorted_values"], target_encoding=flags["target_encoding"],
        te_features=list(cfg.features.target_encoding_features), te_cv=int(cfg.features.target_encoding_cv),
        autoencoder=autoencoder if flags["use_autoencoder"] else None, random_state=cfg.project.seed,
        combination_encoding=flags["combination_encoding"],
        combination_pairs=[tuple(p) for p in cfg.features.get("combination_pairs", cfg.features.interactions)],
    )
    if variant == "sum_only":
        columns: list[str] | None = ["row_sum", "is_original"]
    elif variant == "raw":
        columns = None
    elif selected is not None:
        ae_cols = [f"ae_{i}" for i in range(int(cfg.reduction.autoencoder.bottleneck))] if flags["use_autoencoder"] else []
        columns = list(dict.fromkeys(list(selected) + ae_cols))
    else:
        columns = None
    imp = FactorImputer(factors=list(cfg.features.factors), strategy=imputer,
                        max_fit_rows=int(cfg.data.missing.max_fit_rows), value_range=tuple(cfg.data.factor_range),
                        random_state=cfg.project.seed)
    return Pipeline([("impute", imp), ("features", builder), ("select", ColumnSelector(columns))])


def frequency_reference_counts(frame: pd.DataFrame | None, factors: Sequence[str]) -> dict[str, dict] | None:
    if frame is None:
        return None
    return {c: {float(k): int(v) for k, v in frame[c].dropna().value_counts().items()} for c in factors}


# ------------------------------------------------------------------- selection
def run(cfg) -> dict:
    """Feature stage: Spearman analysis, MI + permutation ranking, top-k chosen by CV."""
    from lightgbm import LGBMRegressor
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold, cross_val_score, train_test_split

    from floodguard.data.prepare import load_prepared
    from floodguard.utils.common import get_logger, load_json, run_paths, save_json, seed_everything

    log = get_logger("floodguard.features")
    seed_everything(cfg.project.seed)
    paths, data = run_paths(cfg), load_prepared(cfg)
    factors, scfg, seed = list(cfg.features.factors), cfg.features.selection, cfg.project.seed
    sample = data.train.sample(n=min(int(scfg.sample_rows), len(data.train)), random_state=seed)
    y = sample[data.target].to_numpy()
    imputer = load_json(paths.audit / "data_audit.json")["imputation"]["selected"]

    pipe = build_feature_pipeline(cfg, "default", imputer,
                                  frequency_reference=frequency_reference_counts(data.comp_test, factors))
    F = pipe.fit_transform(data.X(sample), y)

    # Spearman correlation among factors, the factor sum and the target (figure 03).
    corr_frame = sample[factors].copy()
    corr_frame["row_sum"] = sample[factors].sum(axis=1)
    corr_frame[data.target] = y
    spearman = corr_frame.corr(method="spearman")
    pearson_sum = float(np.corrcoef(corr_frame["row_sum"].fillna(corr_frame["row_sum"].median()), y)[0, 1])
    save_json({"columns": list(spearman.columns), "matrix": spearman.round(4).to_numpy(),
               "target_vs_row_sum": {"spearman": float(spearman.loc["row_sum", data.target]), "pearson": pearson_sum},
               "target_vs_factor_max_abs": float(spearman.loc[factors, data.target].abs().max())},
              paths.features / "correlation.json")

    fast = dict(n_estimators=200, learning_rate=0.08, num_leaves=31, verbose=-1, random_state=seed, n_jobs=-1)
    mi = mutual_info_regression(F.to_numpy(), y, random_state=seed)
    Xa, Xb, ya, yb = train_test_split(F, y, test_size=0.25, random_state=seed)
    model = LGBMRegressor(**fast).fit(Xa, ya)
    perm = permutation_importance(model, Xb, yb, n_repeats=int(scfg.permutation_repeats), random_state=seed,
                                  scoring="r2", n_jobs=1)
    scores = pd.DataFrame({"feature": F.columns, "mutual_info": mi, "permutation": perm.importances_mean})
    scores["rank"] = (scores["mutual_info"].rank(ascending=False) + scores["permutation"].rank(ascending=False)) / 2
    scores = scores.sort_values("rank").reset_index(drop=True)

    cv = KFold(n_splits=int(scfg.cv_folds), shuffle=True, random_state=seed)
    search = []
    for k in scfg.candidate_k:
        cols = list(scores["feature"]) if k == "all" else list(scores["feature"][: int(k)])
        r2 = cross_val_score(LGBMRegressor(**fast), F[cols], y, cv=cv, scoring="r2")
        search.append({"k": len(cols), "label": str(k), "cv_r2_mean": float(r2.mean()), "cv_r2_std": float(r2.std())})
        log.info("Feature selection: top-%s -> CV R² %.4f", k, r2.mean())
    best = max(search, key=lambda s: (round(s["cv_r2_mean"], 4), -s["k"])) if scfg.enabled else search[-1]
    selected = list(scores["feature"][: best["k"]])

    save_json(scores.round(6).to_dict(orient="records"), paths.features / "feature_scores.json")
    builder = pipe.named_steps["features"]
    high_card = getattr(builder, "pairs_", None)
    result = {"selected_k": best["k"], "selected": selected, "search": search, "n_engineered": int(F.shape[1]),
              "imputer": imputer,
              "factor_cardinality": {c: int(sample[c].nunique()) for c in factors},
              "pair_cardinality": high_card.cardinality_ if high_card is not None else {}}
    save_json(result, paths.features / "selected_features.json")
    log.info("Selected %d of %d features (Spearman target~row_sum = %.3f). Pair-key cardinality: %s", best["k"],
             F.shape[1], float(spearman.loc["row_sum", data.target]), result["pair_cardinality"])
    return result


if __name__ == "__main__":
    from floodguard.utils.config import cli

    config, _ = cli(__doc__)
    run(config)
