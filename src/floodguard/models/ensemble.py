"""The FloodGuard ensemble: blending, stacking and soft voting on OOF predictions.

Candidates (all scored on out-of-fold predictions only):
    best_single   the best tuned base model
    blend_nnls    non-negative least squares weights, constrained to sum to 1
    blend_hill    Caruana hill-climbing ensemble selection (with replacement)
    stack_ridge   Ridge meta-learner on OOF predictions + selected row features
    stack_lgbm    LightGBM meta-learner on the same inputs
plus a soft-voting classifier for the risk classes (averaged calibrated class
probabilities of every base model). The final ensemble is chosen by OOF R²,
refitted on the full training split and saved as ONE sklearn Pipeline.

    python -m floodguard.models.ensemble --override configs/smoke.yaml
"""
from __future__ import annotations

import platform
import time
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize, nnls
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score, log_loss, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from floodguard.data.prepare import load_prepared
from floodguard.features.engineering import ColumnSelector, RowStatistics
from floodguard.models.cv import load_predictions, make_feature_pipeline
from floodguard.models.zoo import BASELINES, FEATURE_VARIANT, display_name, fit_model, make_model, refit_params, resolve_params
from floodguard.utils.common import get_logger, load_json, run_paths, save_json, seed_everything
from floodguard.utils.config import Config
from floodguard.utils.tracking import Tracker

log = get_logger("floodguard.ensemble")


# ------------------------------------------------------------------- weights
def nnls_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 that minimise squared error (NNLS start + SLSQP)."""
    w0, _ = nnls(P, y)
    w0 = w0 / w0.sum() if w0.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])
    res = minimize(lambda w: np.mean((P @ w - y) ** 2), w0, method="SLSQP", bounds=[(0, 1)] * P.shape[1],
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x if res.success else w0, 0, None)
    return w / w.sum()


def hill_climb_weights(P: np.ndarray, y: np.ndarray, iters: int = 100) -> np.ndarray:
    """Caruana et al. (2004) greedy ensemble selection with replacement."""
    counts = np.zeros(P.shape[1])
    current = np.zeros(len(y))
    for step in range(1, iters + 1):
        trial = (current[:, None] * (step - 1) + P) / step
        best = int(np.argmin(np.mean((trial - y[:, None]) ** 2, axis=0)))
        counts[best] += 1
        current = trial[:, best]
    return counts / counts.sum()


def meta_matrix(P: np.ndarray, meta: np.ndarray | None) -> np.ndarray:
    return P if meta is None else np.hstack([P, meta])


def stack_oof(meta_model: Any, Z: np.ndarray, y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    return cross_val_predict(clone(meta_model), Z, y, cv=KFold(folds, shuffle=True, random_state=seed))


def lgbm_meta(seed: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, verbose=-1,
                         random_state=seed, n_jobs=-1)


# ----------------------------------------------------------- deployable model
class FloodGuardRegressor(RegressorMixin, BaseEstimator):
    """Fitted base models combined by blend weights or a stacking meta-learner.

    ``members`` is a list of (name, columns, fitted model). Each member reads its
    own column subset of the engineered feature frame.
    """

    def __init__(self, members: Sequence[tuple[str, list[str], Any]] = (), strategy: str = "blend",
                 weights: Sequence[float] | None = None, meta_model: Any = None, meta_features: Sequence[str] = ()):
        self.members = members
        self.strategy = strategy
        self.weights = weights
        self.meta_model = meta_model
        self.meta_features = meta_features

    def fit(self, X: Any, y: Any = None) -> "FloodGuardRegressor":
        return self  # members are fitted before assembly

    def __sklearn_is_fitted__(self) -> bool:
        return len(self.members) > 0

    def member_predictions(self, X: pd.DataFrame) -> np.ndarray:
        from floodguard.models.zoo import predict

        return np.column_stack([predict(model, X[cols]) for _, cols, model in self.members])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        P = self.member_predictions(X)
        if self.strategy == "stack":
            meta = X[list(self.meta_features)].to_numpy(dtype=float) if self.meta_features else None
            return self.meta_model.predict(meta_matrix(P, meta))
        return P @ np.asarray(self.weights)


# ---------------------------------------------------------- soft voting
def class_probabilities(pred: np.ndarray, classes: np.ndarray, oof: bool, seed: int, fitted: LogisticRegression | None = None):
    """Calibrate a model's regression output into risk-class probabilities."""
    Z = pred.reshape(-1, 1)
    model = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    if oof:
        proba = cross_val_predict(model, Z, classes, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                  method="predict_proba")
        return proba, model.fit(Z, classes)
    return fitted.predict_proba(Z), fitted


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)))


def tune_cutoffs(pred: np.ndarray, classes: np.ndarray, grid: int = 60, max_rows: int = 300000, seed: int = 0) -> list[float]:
    """Two prediction cut-offs (Low|Medium|High) maximising macro-F1 on OOF predictions."""
    if len(pred) > max_rows:
        idx = np.random.default_rng(seed).choice(len(pred), max_rows, replace=False)
        pred, classes = pred[idx], classes[idx]
    qs = np.quantile(pred, np.linspace(0.02, 0.98, grid))
    best, best_f1 = [float(qs[grid // 3]), float(qs[2 * grid // 3])], -1.0
    for i, a in enumerate(qs):
        for b in qs[i + 1:]:
            f1 = fast_macro_f1(classes, np.digitize(pred, [a, b]))
            if f1 > best_f1:
                best, best_f1 = [float(a), float(b)], f1
    return best


# ------------------------------------------------------------------- stage
def base_model_names(cfg: Config) -> list[str]:
    names = []
    for name in cfg.models.enabled:
        if name in BASELINES:
            continue
        if load_predictions(cfg, name) is None:
            log.warning("No OOF predictions for %s — skipped in the ensemble.", name)
            continue
        names.append(name)
    return names


def run(cfg: Config) -> dict[str, Any]:
    seed_everything(cfg.project.seed)
    seed, paths, data = cfg.project.seed, run_paths(cfg), load_prepared(cfg)
    names = base_model_names(cfg)
    if not names:
        raise RuntimeError("No base-model predictions found — run the 'train' stage first.")
    y, y_test = data.train[data.target].to_numpy(), data.holdout[data.target].to_numpy()
    preds = {n: load_predictions(cfg, n) for n in names}
    P = np.column_stack([preds[n][0] for n in names])
    T = np.column_stack([preds[n][1] for n in names])
    rs = RowStatistics(list(cfg.features.factors), cfg.features.count_thresholds)
    meta_cols = list(cfg.stacking.meta_features)
    impute = lambda f: f[list(cfg.features.factors)].fillna(data.train[list(cfg.features.factors)].median())  # noqa: E731
    M_tr = rs.transform(impute(data.train))[meta_cols].to_numpy()
    M_te = rs.transform(impute(data.holdout))[meta_cols].to_numpy()

    candidates: dict[str, dict[str, Any]] = {}
    single_r2 = {n: float(r2_score(y, preds[n][0])) for n in names}
    best_single = max(single_r2, key=single_r2.get)
    i = names.index(best_single)
    candidates["best_single"] = {"oof": P[:, i], "test": T[:, i], "members": [best_single], "weights": [1.0]}

    w = nnls_weights(P, y)
    candidates["blend_nnls"] = {"oof": P @ w, "test": T @ w, "members": names, "weights": w.tolist()}
    wh = hill_climb_weights(P, y, int(cfg.stacking.hill_climbing_iters))
    candidates["blend_hill"] = {"oof": P @ wh, "test": T @ wh, "members": names, "weights": wh.tolist()}

    folds = int(cfg.stacking.meta_cv_folds)
    Z_tr, Z_te = meta_matrix(P, M_tr), meta_matrix(T, M_te)
    ridge = Pipeline([("s", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-4, 3, 30)))])
    ridge_full = clone(ridge).fit(Z_tr, y)
    coefs = ridge_full.named_steps["ridge"].coef_ / ridge_full.named_steps["s"].scale_
    candidates["stack_ridge"] = {"oof": stack_oof(ridge, Z_tr, y, folds, seed), "test": ridge_full.predict(Z_te),
                                 "members": names, "coefficients": dict(zip(names + meta_cols, coefs.round(6).tolist())),
                                 "meta": ridge_full}
    try:
        lgb = lgbm_meta(seed)
        lgb_full = clone(lgb).fit(Z_tr, y)
        candidates["stack_lgbm"] = {"oof": stack_oof(lgb, Z_tr, y, folds, seed), "test": lgb_full.predict(Z_te),
                                    "members": names, "meta": lgb_full,
                                    "importance": dict(zip(names + meta_cols, lgb_full.feature_importances_.tolist()))}
    except ImportError:
        log.warning("LightGBM missing — stack_lgbm skipped.")

    scores = {k: {"oof_r2": float(r2_score(y, c["oof"])), "test_r2": float(r2_score(y_test, c["test"]))}
              for k, c in candidates.items()}
    chosen = max(scores, key=lambda k: scores[k]["oof_r2"])  # chosen on OOF only
    for k, s in scores.items():
        log.info("%-12s OOF R² %.5f%s", k, s["oof_r2"], "  <- selected" if k == chosen else "")
    for k, c in candidates.items():
        np.save(paths.oof / f"ens_{k}_oof.npy", c["oof"])
        np.save(paths.oof / f"ens_{k}_test.npy", c["test"])

    # Soft-voting classifier over calibrated per-model class probabilities.
    cut_true = data.cutoffs
    cls_tr, cls_te = np.digitize(y, cut_true), np.digitize(y_test, cut_true)
    probs_tr, probs_te = [], []
    for n in names:
        p_oof, fitted = class_probabilities(preds[n][0], cls_tr, True, seed)
        p_te, _ = class_probabilities(preds[n][1], cls_te, False, seed, fitted)
        probs_tr.append(p_oof)
        probs_te.append(p_te)
    sv_tr, sv_te = np.mean(probs_tr, axis=0), np.mean(probs_te, axis=0)
    np.save(paths.oof / "softvote_oof_proba.npy", sv_tr)
    np.save(paths.oof / "softvote_test_proba.npy", sv_te)
    soft_vote = {"oof_macro_f1": float(f1_score(cls_tr, sv_tr.argmax(1), average="macro")),
                 "oof_log_loss": float(log_loss(cls_tr, sv_tr, labels=[0, 1, 2]))}

    final = candidates[chosen]
    pred_cut = tune_cutoffs(final["oof"], cls_tr, seed=seed)
    high_cal = LogisticRegression(max_iter=1000).fit(final["oof"].reshape(-1, 1), (cls_tr == 2).astype(int))
    residuals = y - final["oof"]

    pipeline, members_used, refit_info = build_deployable(cfg, data, chosen, final, names, meta_cols)
    deployed_test = pipeline.predict(data.X(data.holdout))
    deployed_r2 = float(r2_score(y_test, deployed_test))
    log.info("Deployed pipeline (refit on full training split): test R² %.5f", deployed_r2)

    factors = list(cfg.features.factors)
    bounds = {c: [int(np.nanmin(data.train[c])), int(np.nanmax(data.train[c]))] for c in factors}
    background = data.X(data.train).dropna().sample(n=min(int(cfg.explain.background), len(data.train)), random_state=seed)
    bundle = {
        "pipeline": pipeline, "strategy": chosen, "members": members_used, "factors": factors, "bounds": bounds,
        "risk_classes": list(cfg.risk.class_names), "true_cutoffs": cut_true, "prediction_cutoffs": pred_cut,
        "high_calibrator": high_cal, "residual_quantiles": {q: float(np.quantile(residuals, q)) for q in (0.05, 0.95)},
        "background": background, "themes": dict(cfg.features.themes), "version": cfg.project.version,
        "data_source": data.audit.get("source"), "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    model_path = paths.models / "floodguard_pipeline.joblib"
    joblib.dump(bundle, model_path, compress=3)

    card = {
        "name": "FloodGuard AI ensemble", "version": cfg.project.version, "task": "Regression of FloodProbability (0-1) + Low/Medium/High risk class",
        "data_source": data.audit.get("source"),
        "training_rows": int(len(data.train)), "holdout_rows": int(len(data.holdout)),
        "strategy": chosen, "strategy_display": {"best_single": f"Best single model ({display_name(best_single)})",
                                                 "blend_nnls": "NNLS weighted blend", "blend_hill": "Hill-climbing blend",
                                                 "stack_ridge": "Stacking (Ridge meta-learner)",
                                                 "stack_lgbm": "Stacking (LightGBM meta-learner)"}[chosen],
        "members": [{"name": m, "display_name": display_name(m)} for m in members_used],
        "weights": dict(zip(final["members"], final.get("weights", []))) if "weights" in final else None,
        "oof_r2": scores[chosen]["oof_r2"], "holdout_r2_cv_models": scores[chosen]["test_r2"],
        "holdout_r2_deployed_refit": deployed_r2, "feature_bounds": bounds, "risk_true_cutoffs": cut_true,
        "risk_prediction_cutoffs": pred_cut, "refit": refit_info,
        "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        "intended_use": "Decision support and education; not an official flood warning system.",
        "limitations": "Kaggle S4E5 data are synthetically generated from the original Flood Prediction Factors dataset; "
                       "factor scores are abstract 0-20 indices, not physical measurements.",
    }
    save_json(card, paths.models / "model_card.json")

    summary = {"members": names, "single_oof_r2": single_r2, "candidates": scores, "selected": chosen,
               "weights_nnls": dict(zip(names, w.round(6).tolist())), "weights_hill": dict(zip(names, wh.round(6).tolist())),
               "stack_ridge_coefficients": candidates["stack_ridge"]["coefficients"],
               "stack_lgbm_importance": candidates.get("stack_lgbm", {}).get("importance"),
               "soft_vote": soft_vote, "prediction_cutoffs": pred_cut, "deployed_holdout_r2": deployed_r2}
    save_json(summary, paths.metrics / "ensemble.json")

    if data.comp_test is not None and cfg.data.id_col in data.comp_test.columns:
        sub = pd.DataFrame({cfg.data.id_col: data.comp_test[cfg.data.id_col].astype(int),
                            data.target: np.clip(pipeline.predict(data.X(data.comp_test)), 0, 1)})
        sub.to_csv(paths.root / "submission.csv", index=False)
        log.info("Wrote Kaggle submission (%d rows) to %s", len(sub), paths.root / "submission.csv")

    tracker = Tracker(cfg)
    with tracker.run("ensemble", tags={"stage": "ensemble", "strategy": chosen}):
        tracker.log_params({"strategy": chosen, "members": ",".join(members_used)})
        tracker.log_metrics({f"{k}_oof_r2": v["oof_r2"] for k, v in scores.items()})
        tracker.log_metrics({"deployed_holdout_r2": deployed_r2})
        tracker.log_artifact(paths.models / "model_card.json")
        tracker.log_model(pipeline, data.X(data.holdout).head(50))
    return summary


def build_deployable(cfg: Config, data, chosen: str, final: dict, names: list[str], meta_cols: list[str]):
    """Refit the chosen members on the full training split; return one sklearn Pipeline."""
    if chosen.startswith("blend"):
        members = [n for n, w in zip(final["members"], final["weights"]) if w > 1e-6]
        weights = [w for w in final["weights"] if w > 1e-6]
        weights = list(np.asarray(weights) / np.sum(weights))
    elif chosen == "best_single":
        members, weights = final["members"], [1.0]
    else:
        members, weights = names, None
    variant = "ae" if "ae_lgbm" in members else "default"
    feat = make_feature_pipeline(cfg, data, variant)
    X = feat.fit_transform(data.X(data.train), data.train[data.target].to_numpy()).astype(np.float32)
    y = data.train[data.target].to_numpy()
    paths = run_paths(cfg)
    fitted, refit_info = [], {}
    for name in members:
        cols = [c for c in X.columns if not c.startswith("ae_")] if FEATURE_VARIANT.get(name) != "ae" else list(X.columns)
        cv_info = load_json(paths.metrics / f"cv_{name}.json", {})
        params = resolve_params(cfg, name)
        params.update(refit_params(name, cv_info.get("best_iterations", [])))
        model, _ = fit_model(name, make_model(name, params, cfg), X[cols], y, None, None, cfg, early_stopping=False)
        fitted.append((name, cols, model))
        refit_info[name] = {k: v for k, v in params.items() if k in ("n_estimators", "iterations", "epochs")}
    meta = final.get("meta") if chosen.startswith("stack") else None
    model = FloodGuardRegressor(fitted, "stack" if meta is not None else "blend", weights, meta,
                                meta_cols if meta is not None else ())
    keep = list(X.columns) + ([c for c in meta_cols if c not in X.columns] if meta is not None else [])
    select = ColumnSelector(keep)
    feat_steps = feat.steps[:-1]  # impute + features (selection handled below)
    pipeline = Pipeline(feat_steps + [("select", select), ("model", model)])
    return pipeline, members, refit_info


if __name__ == "__main__":
    from floodguard.utils.config import cli

    config, _ = cli(__doc__)
    run(config)
