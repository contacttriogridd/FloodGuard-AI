"""K-fold out-of-fold (OOF) training engine shared by every model.

For each fold the feature pipeline (imputer -> engineering -> selection) is
fitted on the training folds only, the model is trained with early stopping on
the validation fold, OOF predictions are collected and the held-out test set
is predicted (averaged over folds). Transformed folds are cached per feature
variant so several models / Optuna trials reuse them.

    python -m floodguard.models.cv --override configs/smoke.yaml --model lgbm
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from floodguard.data.prepare import FLAG, PreparedData, assign_folds, load_prepared
from floodguard.features.engineering import build_feature_pipeline, frequency_reference_counts
from floodguard.features.reduction import load_autoencoder
from floodguard.models.zoo import FEATURE_VARIANT, display_name, fit_model, make_model, predict, resolve_params
from floodguard.utils.common import get_logger, load_json, run_paths, save_json
from floodguard.utils.config import Config

log = get_logger("floodguard.cv")


@dataclass
class Fold:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    Xtr: pd.DataFrame
    ytr: np.ndarray
    Xva: pd.DataFrame
    yva: np.ndarray
    Xte: pd.DataFrame | None = None
    Xext: pd.DataFrame | None = None


@dataclass
class CVResult:
    name: str
    oof: np.ndarray
    test: np.ndarray | None
    fold_scores: list[float]
    best_iterations: list[int | None]
    curves: list[Any] = field(default_factory=list)
    seconds: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def oof_r2(self) -> float:
        return float(np.mean(self.fold_scores))


def selected_features(cfg: Config) -> list[str] | None:
    sel = load_json(run_paths(cfg).features / "selected_features.json")
    return sel["selected"] if sel else None


def chosen_imputer(cfg: Config) -> str:
    audit = load_json(run_paths(cfg).audit / "data_audit.json", {})
    return audit.get("imputation", {}).get("selected", "median")


def make_feature_pipeline(cfg: Config, data: PreparedData, variant: str = "default"):
    ae = load_autoencoder(cfg) if variant == "ae" else None
    if variant == "ae" and ae is None:
        raise RuntimeError("Autoencoder features requested but no trained autoencoder — run the 'reduce' stage.")
    return build_feature_pipeline(cfg, variant, chosen_imputer(cfg), selected_features(cfg), ae,
                                  frequency_reference_counts(data.comp_test, list(cfg.features.factors)))


class FoldCache:
    """Transformed folds keyed by (feature variant, data subset)."""

    def __init__(self, cfg: Config, data: PreparedData):
        self.cfg, self.data = cfg, data
        self._store: dict[tuple, list[Fold]] = {}

    def folds(self, variant: str = "default", frame: pd.DataFrame | None = None, n_folds: int | None = None,
              key: str = "full", with_test: bool = True) -> list[Fold]:
        cache_key = (variant, key, n_folds)
        if cache_key in self._store:
            return self._store[cache_key]
        data, cfg = self.data, self.cfg
        frame = data.train if frame is None else frame
        if n_folds is None:
            fold_ids = frame["fold"].to_numpy()
        else:
            fold_ids = assign_folds(frame, data.target, n_folds, int(cfg.cv.stratify_bins), cfg.project.seed)
        X, y = data.X(frame), frame[data.target].to_numpy()
        out: list[Fold] = []
        for k in sorted(np.unique(fold_ids)):
            tr, va = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
            pipe = make_feature_pipeline(cfg, data, variant)
            Xtr = pipe.fit_transform(X.iloc[tr], y[tr]).astype(np.float32)
            fold = Fold(tr, va, Xtr, y[tr], pipe.transform(X.iloc[va]).astype(np.float32), y[va])
            if with_test:
                fold.Xte = pipe.transform(data.X(data.holdout)).astype(np.float32)
            out.append(fold)
        self._store[cache_key] = out
        log.info("Built %d folds for variant '%s' (%s rows, %d features).", len(out), variant, len(frame), out[0].Xtr.shape[1])
        return out

    def clear(self) -> None:
        self._store.clear()


def run_cv(name: str, cfg: Config, cache: FoldCache, params: dict[str, Any] | None = None, variant: str | None = None,
           frame: pd.DataFrame | None = None, key: str = "full", save: bool = True, tracker=None) -> CVResult:
    """Train ``name`` on every fold; returns OOF + averaged test predictions."""
    variant = variant or FEATURE_VARIANT.get(name, "default")
    folds = cache.folds(variant, frame=frame, key=key)
    n_rows = sum(len(f.valid_idx) for f in folds)
    oof = np.zeros(n_rows)
    test = np.zeros(len(folds[0].Xte)) if folds[0].Xte is not None else None
    scores, iters, curves = [], [], []
    start = time.perf_counter()
    for k, fold in enumerate(folds):
        model = make_model(name, params, cfg)
        model, info = fit_model(name, model, fold.Xtr, fold.ytr, fold.Xva, fold.yva, cfg)
        pred = predict(model, fold.Xva)
        oof[fold.valid_idx] = pred
        scores.append(float(r2_score(fold.yva, pred)))
        iters.append(info["best_iteration"])
        curves.append(info["curve"])
        if test is not None:
            test += predict(model, fold.Xte) / len(folds)
    result = CVResult(name, oof, test, scores, iters, curves, round(time.perf_counter() - start, 2),
                      resolve_params(cfg, name, params))
    log.info("%-32s OOF R² %.5f (folds: %s) in %.1fs", display_name(name), result.oof_r2,
             ", ".join(f"{s:.4f}" for s in scores), result.seconds)
    if save:
        save_cv_result(cfg, result, variant)
    if tracker is not None:
        with tracker.run(f"train/{name}", nested=True, tags={"stage": "train", "model": name, "variant": variant}):
            tracker.log_params({k: v for k, v in result.params.items()})
            for k, s in enumerate(scores):
                tracker.log_metrics({"fold_r2": s}, step=k)
            tracker.log_metrics({"oof_r2": result.oof_r2, "train_seconds": result.seconds})
    return result


def save_cv_result(cfg: Config, result: CVResult, variant: str) -> None:
    paths = run_paths(cfg)
    np.save(paths.oof / f"{result.name}_oof.npy", result.oof)
    if result.test is not None:
        np.save(paths.oof / f"{result.name}_test.npy", result.test)
    curve = next((c for c in result.curves if c), None)
    save_json({"name": result.name, "display_name": display_name(result.name), "variant": variant,
               "fold_scores": result.fold_scores, "oof_r2": result.oof_r2, "best_iterations": result.best_iterations,
               "seconds": result.seconds, "params": result.params, "curve_fold0": curve},
              paths.metrics / f"cv_{result.name}.json")


def load_predictions(cfg: Config, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    paths = run_paths(cfg)
    oof, test = paths.oof / f"{name}_oof.npy", paths.oof / f"{name}_test.npy"
    if not oof.exists() or not test.exists():
        return None
    return np.load(oof), np.load(test)


def fit_full(name: str, cfg: Config, data: PreparedData, params: dict[str, Any] | None = None,
             variant: str | None = None, frame: pd.DataFrame | None = None, n_iter_from: CVResult | None = None):
    """Refit feature pipeline + model on all training rows (used for deployment and SHAP)."""
    from floodguard.models.zoo import refit_params

    variant = variant or FEATURE_VARIANT.get(name, "default")
    frame = data.train if frame is None else frame
    pipe = make_feature_pipeline(cfg, data, variant)
    X = pipe.fit_transform(data.X(frame), frame[data.target].to_numpy()).astype(np.float32)
    p = resolve_params(cfg, name, params)
    if n_iter_from is not None:
        p.update(refit_params(name, n_iter_from.best_iterations))
    model, _ = fit_model(name, make_model(name, p, cfg), X, frame[data.target].to_numpy(), None, None, cfg,
                         early_stopping=False)
    return pipe, model


def run(cfg: Config, models: list[str] | None = None) -> dict[str, float]:
    """Train stage: OOF-train the baselines and every enabled model."""
    from floodguard.models.zoo import BASELINES, available
    from floodguard.utils.common import seed_everything
    from floodguard.utils.tracking import Tracker

    seed_everything(cfg.project.seed)
    data = load_prepared(cfg)
    cache, tracker = FoldCache(cfg, data), Tracker(cfg)
    names = models or list(BASELINES) + list(cfg.models.enabled)
    scores: dict[str, float] = {}
    with tracker.run("train", tags={"stage": "train"}):
        for name in names:
            if not available(name):
                log.warning("Skipping %s — optional dependency not installed.", name)
                continue
            if FEATURE_VARIANT.get(name) == "ae" and load_autoencoder(cfg) is None:
                log.warning("Skipping %s — no trained autoencoder (run the 'reduce' stage).", name)
                continue
            scores[name] = run_cv(name, cfg, cache, tracker=tracker).oof_r2
    save_json(scores, run_paths(cfg).metrics / "oof_scores.json")
    return scores


def main() -> None:
    from floodguard.utils.config import cli

    cfg, args = cli(__doc__, lambda p: p.add_argument("--model", action="append", default=None,
                                                      help="model(s) to train; default = all enabled"))
    run(cfg, args.model)


if __name__ == "__main__":
    main()

__all__ = ["Fold", "CVResult", "FoldCache", "run_cv", "fit_full", "load_predictions", "FLAG"]
