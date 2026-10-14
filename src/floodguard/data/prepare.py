"""Load, merge, audit, clean and split the flood data.

    python -m floodguard.data.prepare --config configs/default.yaml --override configs/smoke.yaml

Outputs (``<run_dir>/``):
    data/train.parquet      training split (S4E5 + optional original rows) with a ``fold`` column
    data/holdout.parquet    15 % S4E5 test split, stratified by target deciles (touched once)
    data/external.parquet   original dataset (external validation)
    data/comp_test.parquet  unlabeled competition test set (frequency encoding + submission)
    audit/data_audit.json   schema, duplicates, missingness, outliers, splits, risk thresholds
    audit/imputer_comparison.json
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from floodguard.features.engineering import FactorImputer
from floodguard.utils.common import RunPaths, get_logger, load_json, run_paths, save_json, seed_everything
from floodguard.utils.config import Config, cli, resolve_path

log = get_logger("floodguard.prepare")
FLAG = "is_original"


# --------------------------------------------------------------------------- loading
def load_raw(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    train_path = resolve_path(cfg.data.train_path)
    if not train_path.exists():
        hint = ("python scripts/make_synthetic_flood.py" if cfg.data.source == "synthetic"
                else "python scripts/run_pipeline.py --stages download")
        raise FileNotFoundError(f"{train_path} not found. Run: {hint}")
    train = pd.read_csv(train_path)
    if cfg.data.get("sample_rows"):
        train = train.sample(n=min(int(cfg.data.sample_rows), len(train)), random_state=cfg.project.seed)
    test_path, orig_path = resolve_path(cfg.data.test_path), resolve_path(cfg.data.original_path)
    comp_test = pd.read_csv(test_path) if test_path.exists() else None
    original = pd.read_csv(orig_path) if orig_path.exists() else None
    return train, comp_test, original


# --------------------------------------------------------------------------- audit
def audit_schema(df: pd.DataFrame, cfg: Config, name: str, labeled: bool = True) -> dict:
    factors = list(cfg.features.factors)
    required = factors + ([cfg.data.target] if labeled else [])
    lo, hi = cfg.data.factor_range
    report: dict = {
        "name": name, "rows": int(len(df)), "columns": int(df.shape[1]),
        "missing_columns": [c for c in required if c not in df.columns],
        "extra_columns": [c for c in df.columns if c not in required + [cfg.data.id_col]],
        "dtypes": {c: str(df[c].dtype) for c in required if c in df.columns},
        "exact_duplicates": int(df.drop(columns=[cfg.data.id_col], errors="ignore").duplicated().sum()),
        "ranges": {}, "out_of_range": {}, "non_integer": {},
    }
    for c in factors:
        if c not in df.columns:
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        report["ranges"][c] = [float(col.min()), float(col.max())]
        report["out_of_range"][c] = int(((col < lo) | (col > hi)).sum())
        report["non_integer"][c] = int((col.dropna() % 1 != 0).sum())
    if labeled and cfg.data.target in df.columns:
        t = pd.to_numeric(df[cfg.data.target], errors="coerce")
        report["target_range"] = [float(t.min()), float(t.max())]
        report["target_out_of_range"] = int(((t < 0) | (t > 1)).sum())
    if report["missing_columns"]:
        raise ValueError(f"{name}: missing required columns {report['missing_columns']}")
    return report


def clean(df: pd.DataFrame, cfg: Config, labeled: bool = True) -> tuple[pd.DataFrame, dict]:
    factors = list(cfg.features.factors)
    lo, hi = cfg.data.factor_range
    out = df.copy()
    summary: dict = {"rows_in": int(len(out))}
    for c in factors:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        bad = out[c].notna() & ((out[c] < lo) | (out[c] > hi) | (out[c] % 1 != 0))
        summary.setdefault("invalid_set_to_missing", {})[c] = int(bad.sum())
        out.loc[bad, c] = np.nan
    if labeled:
        t = cfg.data.target
        out[t] = pd.to_numeric(out[t], errors="coerce")
        bad_t = out[t].isna() | (out[t] < 0) | (out[t] > 1)
        summary["rows_dropped_bad_target"] = int(bad_t.sum())
        out = out[~bad_t]
    cols = factors + ([cfg.data.target] if labeled else [])
    before = len(out)
    out = out.drop_duplicates(subset=cols).reset_index(drop=True)
    summary["duplicates_removed"] = before - len(out)
    summary["rows_out"] = int(len(out))
    return out, summary


def missingness(df: pd.DataFrame, cols: list[str]) -> dict:
    counts = df[cols].isna().sum()
    return {"per_column": {c: int(v) for c, v in counts.items()},
            "rate": {c: round(float(v) / max(len(df), 1), 6) for c, v in counts.items()},
            "rows_with_missing": int(df[cols].isna().any(axis=1).sum())}


# ------------------------------------------------------------------ imputer study
def imputer_study(train: pd.DataFrame, cfg: Config, paths: RunPaths) -> dict:
    """Inject MCAR missingness into a copy and compare imputers by downstream CV R²."""
    mcfg, seed = cfg.data.missing, cfg.project.seed
    factors = list(cfg.features.factors)
    rng = np.random.default_rng(seed)
    complete = train.dropna(subset=factors)
    sample = complete.sample(n=min(int(mcfg.study_rows), len(complete)), random_state=seed)
    X_true = sample[factors].to_numpy(dtype=float)
    mask = rng.random(X_true.shape) < float(mcfg.inject_rate)
    X_missing = pd.DataFrame(np.where(mask, np.nan, X_true), columns=factors, index=sample.index)
    y = sample[cfg.data.target].to_numpy()
    np.save(paths.audit / "missing_mask_sample.npy", mask[:200])

    cv = KFold(n_splits=int(mcfg.cv_folds), shuffle=True, random_state=seed)
    results = {}
    for strategy in mcfg.imputers:
        imputer = FactorImputer(factors=factors, strategy=strategy, max_fit_rows=int(mcfg.max_fit_rows),
                                value_range=tuple(cfg.data.factor_range), random_state=seed)
        model = Pipeline([("impute", imputer), ("ridge", Ridge(alpha=1.0))])
        start = time.perf_counter()
        scores = cross_val_score(model, X_missing, y, cv=cv, scoring="r2")
        imputed = imputer.fit(X_missing).transform(X_missing).to_numpy(dtype=float)
        results[strategy] = {
            "cv_r2_mean": float(scores.mean()), "cv_r2_std": float(scores.std()),
            "imputation_rmse": float(np.sqrt(np.mean((imputed[mask] - X_true[mask]) ** 2))),
            "seconds": round(time.perf_counter() - start, 2),
        }
        log.info("Imputer %-9s CV R² %.4f ± %.4f (imputation RMSE %.3f)", strategy,
                 results[strategy]["cv_r2_mean"], results[strategy]["cv_r2_std"], results[strategy]["imputation_rmse"])
    best = max(results, key=lambda k: results[k]["cv_r2_mean"])
    out = {"inject_rate": float(mcfg.inject_rate), "rows": int(len(sample)), "downstream_model": "Ridge on imputed factors",
           "results": results, "selected": best}
    save_json(out, paths.audit / "imputer_comparison.json")
    return out


# -------------------------------------------------------------------- outliers
def outlier_flags(train: pd.DataFrame, frames: dict[str, pd.DataFrame], cfg: Config) -> dict:
    """IQR (per factor) and IsolationForest flags. Rows are flagged, never dropped."""
    factors = list(cfg.features.factors)
    ocfg, seed = cfg.data.outliers, cfg.project.seed
    q1, q3 = train[factors].quantile(0.25), train[factors].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - ocfg.iqr_k * iqr, q3 + ocfg.iqr_k * iqr
    fit_rows = train[factors].fillna(train[factors].median())
    fit_rows = fit_rows.sample(n=min(int(ocfg.iforest_max_rows), len(fit_rows)), random_state=seed)
    forest = IsolationForest(contamination=float(ocfg.iforest_contamination), random_state=seed, n_jobs=-1).fit(fit_rows)

    report: dict = {"iqr_bounds": {c: [float(lower[c]), float(upper[c])] for c in factors}}
    for name, frame in frames.items():
        X = frame[factors]
        cell_flags = (X.lt(lower) | X.gt(upper))
        frame["outlier_iqr"] = cell_flags.any(axis=1).astype(int)
        frame["outlier_iforest"] = (forest.predict(X.fillna(train[factors].median())) == -1).astype(int)
        report[name] = {
            "iqr_rows": int(frame["outlier_iqr"].sum()),
            "iqr_cells_per_column": {c: int(v) for c, v in cell_flags.sum().items()},
            "iforest_rows": int(frame["outlier_iforest"].sum()),
            "both": int((frame["outlier_iqr"] & frame["outlier_iforest"]).sum()),
        }
    return report


# ---------------------------------------------------------------------- splits
def decile_bins(y: pd.Series | np.ndarray, bins: int) -> np.ndarray:
    return pd.qcut(pd.Series(y).rank(method="first"), q=bins, labels=False).to_numpy()


def assign_folds(df: pd.DataFrame, target: str, n_folds: int, bins: int, seed: int) -> np.ndarray:
    folds = np.full(len(df), -1)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(skf.split(df, decile_bins(df[target], bins))):
        folds[idx] = k
    return folds


def risk_cutoffs(y: np.ndarray, quantiles: list[float]) -> list[float]:
    return [float(np.quantile(y, q)) for q in quantiles]


# ------------------------------------------------------------------- prepared
@dataclass
class PreparedData:
    train: pd.DataFrame
    holdout: pd.DataFrame
    external: pd.DataFrame | None
    comp_test: pd.DataFrame | None
    audit: dict
    cfg: Config

    @property
    def feature_cols(self) -> list[str]:
        return list(self.cfg.features.factors) + [FLAG]

    @property
    def target(self) -> str:
        return self.cfg.data.target

    def X(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.feature_cols]

    @property
    def cutoffs(self) -> list[float]:
        return self.audit["risk"]["cutoffs"]


def load_prepared(cfg: Config) -> PreparedData:
    paths = run_paths(cfg)
    if not (paths.data / "train.parquet").exists():
        raise FileNotFoundError("Prepared data not found — run the 'prepare' stage first.")
    read = lambda name: pd.read_parquet(paths.data / name) if (paths.data / name).exists() else None  # noqa: E731
    return PreparedData(read("train.parquet"), read("holdout.parquet"), read("external.parquet"),
                        read("comp_test.parquet"), load_json(paths.audit / "data_audit.json"), cfg)


# ------------------------------------------------------------------------- run
def run(cfg: Config) -> PreparedData:
    seed_everything(cfg.project.seed)
    paths = run_paths(cfg)
    factors, target, seed = list(cfg.features.factors), cfg.data.target, cfg.project.seed
    train_raw, comp_raw, orig_raw = load_raw(cfg)

    audit: dict = {"source": cfg.data.source, "schema": {}, "cleaning": {}}
    audit["schema"]["s4e5_train"] = audit_schema(train_raw, cfg, "s4e5_train")
    s4e5, audit["cleaning"]["s4e5_train"] = clean(train_raw, cfg)

    original = None
    if orig_raw is not None:
        audit["schema"]["original"] = audit_schema(orig_raw, cfg, "original")
        original, audit["cleaning"]["original"] = clean(orig_raw, cfg)
        original[FLAG] = 1
    comp_test = None
    if comp_raw is not None:
        audit["schema"]["comp_test"] = audit_schema(comp_raw, cfg, "comp_test", labeled=False)
        comp_test, audit["cleaning"]["comp_test"] = clean(comp_raw, cfg, labeled=False)
        comp_test[FLAG] = 0
    s4e5[FLAG] = 0

    # Hold out 15 % of S4E5 (competition distribution), stratified by target deciles.
    train, holdout = train_test_split(
        s4e5, test_size=float(cfg.data.holdout_size), random_state=seed,
        stratify=decile_bins(s4e5[target], int(cfg.cv.stratify_bins)),
    )
    train, holdout = train.reset_index(drop=True), holdout.reset_index(drop=True)
    train["fold"] = assign_folds(train, target, int(cfg.cv.n_folds), int(cfg.cv.stratify_bins), seed)
    if original is not None and cfg.data.merge_original:
        extra = original.copy()
        extra["fold"] = assign_folds(extra, target, int(cfg.cv.n_folds), int(cfg.cv.stratify_bins), seed)
        train = pd.concat([train, extra], ignore_index=True)

    frames = {"train": train, "holdout": holdout}
    if original is not None:
        frames["external"] = original
    audit["missingness"] = {k: missingness(v, factors) for k, v in frames.items()}
    audit["imputation"] = imputer_study(train, cfg, paths)
    audit["outliers"] = outlier_flags(train, frames, cfg)

    cutoffs = risk_cutoffs(train.loc[train[FLAG] == 0, target].to_numpy(), list(cfg.risk.class_quantiles))
    audit["risk"] = {"class_names": list(cfg.risk.class_names), "quantiles": list(cfg.risk.class_quantiles),
                     "cutoffs": cutoffs, "definition": "Ground-truth classes from tertiles of the training target"}
    audit["splits"] = {
        "train_rows": int(len(train)), "train_original_rows": int(train[FLAG].sum()),
        "holdout_rows": int(len(holdout)), "external_rows": int(len(original)) if original is not None else 0,
        "comp_test_rows": int(len(comp_test)) if comp_test is not None else 0,
        "n_folds": int(cfg.cv.n_folds), "holdout_size": float(cfg.data.holdout_size),
        "fold_sizes": {int(k): int(v) for k, v in train["fold"].value_counts().sort_index().items()},
    }
    audit["target_stats"] = {k: {"mean": float(v[target].mean()), "std": float(v[target].std()),
                                 "min": float(v[target].min()), "max": float(v[target].max())}
                             for k, v in frames.items()}

    train.to_parquet(paths.data / "train.parquet", index=False)
    holdout.to_parquet(paths.data / "holdout.parquet", index=False)
    if original is not None:
        original.to_parquet(paths.data / "external.parquet", index=False)
    if comp_test is not None:
        comp_test.to_parquet(paths.data / "comp_test.parquet", index=False)
    save_json(audit, paths.audit / "data_audit.json")
    log.info("Prepared: train=%d (original rows %d), holdout=%d, external=%d | imputer=%s | risk cut-offs=%s",
             len(train), audit["splits"]["train_original_rows"], len(holdout), audit["splits"]["external_rows"],
             audit["imputation"]["selected"], [round(c, 4) for c in cutoffs])
    return load_prepared(cfg)


if __name__ == "__main__":
    config, _ = cli(__doc__)
    run(config)
