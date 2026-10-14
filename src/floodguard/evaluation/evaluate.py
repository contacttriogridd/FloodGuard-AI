"""Hold-out evaluation, calibration, statistical tests, ablations, error analysis
and external validation.

The held-out test split is used here exactly once per model; every choice
(ensemble, risk cut-offs, calibrators) was made on out-of-fold predictions.

    python -m floodguard.evaluation.evaluate --override configs/smoke.yaml
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_recall_curve, roc_curve

from floodguard.data.prepare import FLAG, load_prepared
from floodguard.evaluation import metrics as M
from floodguard.models.cv import FoldCache, fit_full, load_predictions, run_cv
from floodguard.models.ensemble import tune_cutoffs
from floodguard.models.zoo import BASELINES, display_name
from floodguard.utils.common import get_logger, load_json, run_paths, save_json, seed_everything
from floodguard.utils.config import Config
from floodguard.utils.tracking import Tracker

log = get_logger("floodguard.evaluate")
ENSEMBLE_LABELS = {"best_single": "Best single model", "blend_nnls": "NNLS blend", "blend_hill": "Hill-climbing blend",
                   "stack_ridge": "Stacking (Ridge meta)", "stack_lgbm": "Stacking (LightGBM meta)"}


def registered_models(cfg: Config) -> list[dict[str, Any]]:
    """Every model with saved OOF + test predictions (baselines, base models, ensemble candidates)."""
    out = []
    for name in list(BASELINES) + [m for m in cfg.models.enabled if m not in BASELINES]:
        pr = load_predictions(cfg, name)
        if pr is not None:
            out.append({"key": name, "name": display_name(name), "kind": "baseline" if name in BASELINES else "single",
                        "oof": pr[0], "test": pr[1]})
    paths = run_paths(cfg)
    ens = load_json(paths.metrics / "ensemble.json", {})
    for cand in ENSEMBLE_LABELS:
        f_oof, f_test = paths.oof / f"ens_{cand}_oof.npy", paths.oof / f"ens_{cand}_test.npy"
        if f_oof.exists() and cand != "best_single":
            out.append({"key": f"ens_{cand}", "name": ENSEMBLE_LABELS[cand], "kind": "ensemble",
                        "oof": np.load(f_oof), "test": np.load(f_test), "selected": cand == ens.get("selected")})
    if ens.get("selected") == "best_single":
        best = max(ens["single_oof_r2"], key=ens["single_oof_r2"].get)
        for m in out:
            m["selected"] = m["key"] == best
    return out


def classification_view(oof: np.ndarray, test: np.ndarray, y: np.ndarray, y_test: np.ndarray, cut_true: list[float],
                        bins: int, seed: int) -> tuple[dict[str, float], dict[str, Any]]:
    """Cut-offs and High-risk calibration fitted on OOF; metrics on the test split."""
    cls_tr, cls_te = M.to_risk_class(y, cut_true), M.to_risk_class(y_test, cut_true)
    cut_pred = tune_cutoffs(oof, cls_tr, seed=seed)
    cal = LogisticRegression(max_iter=1000).fit(oof.reshape(-1, 1), (cls_tr == 2).astype(int))
    prob_high = cal.predict_proba(test.reshape(-1, 1))[:, 1]
    pred_cls = M.to_risk_class(test, cut_pred)
    metrics = M.classification_metrics(cls_te, pred_cls, (cls_te == 2).astype(int), prob_high, bins)
    return metrics, {"cutoffs": cut_pred, "pred_classes": pred_cls, "true_classes": cls_te, "prob_high": prob_high}


def run(cfg: Config) -> dict[str, Any]:
    seed_everything(cfg.project.seed)
    seed, paths, data, ecfg = cfg.project.seed, run_paths(cfg), load_prepared(cfg), cfg.evaluation
    y, y_test = data.train[data.target].to_numpy(), data.holdout[data.target].to_numpy()
    cut_true, bins, n_boot = data.cutoffs, int(ecfg.calibration_bins), int(ecfg.bootstrap)
    models = registered_models(cfg)
    if not models:
        raise RuntimeError("No predictions found — run 'train' and 'ensemble' first.")

    # -------------------------------------------------------- per-model metrics
    rows, views = [], {}
    for m in models:
        reg = {k: M.bootstrap_ci(f, y_test, m["test"], n_boot, float(ecfg.ci), seed) for k, f in M.REGRESSION.items()}
        cls, view = classification_view(m["oof"], m["test"], y, y_test, cut_true, bins, seed)
        views[m["key"]] = view
        rows.append({"key": m["key"], "name": m["name"], "kind": m["kind"], "selected": bool(m.get("selected")),
                     "oof_r2": M.r2(y, m["oof"]), **{f"test_{k}": v for k, v in reg.items()},
                     **{f"test_{k}": v for k, v in cls.items()}})
        log.info("%-36s test R² %.5f [%.5f, %.5f]  macro-F1 %.3f", m["name"], reg["r2"]["value"], reg["r2"]["lo"],
                 reg["r2"]["hi"], cls["macro_f1"])

    final = next((m for m in models if m.get("selected")), max(models, key=lambda m: M.r2(y, m["oof"])))
    v = views[final["key"]]
    fpr, tpr, _ = roc_curve(v["true_classes"] == 2, v["prob_high"])
    prec, rec, _ = precision_recall_curve(v["true_classes"] == 2, v["prob_high"])
    step = lambda a: a[:: max(len(a) // 200, 1)].tolist()  # noqa: E731 - thin long curves
    final_detail = {
        "key": final["key"], "name": final["name"], "prediction_cutoffs": v["cutoffs"], "true_cutoffs": cut_true,
        "confusion_matrix": confusion_matrix(v["true_classes"], v["pred_classes"], labels=[0, 1, 2]).tolist(),
        "roc": {"fpr": step(fpr), "tpr": step(tpr)}, "pr": {"precision": step(prec), "recall": step(rec),
                                                            "prevalence": float((v["true_classes"] == 2).mean())},
        "reliability": M.reliability_curve((v["true_classes"] == 2).astype(int), v["prob_high"], bins),
    }
    sv_path = paths.oof / "softvote_test_proba.npy"
    if sv_path.exists():
        sv = np.load(sv_path)
        cls_te = M.to_risk_class(y_test, cut_true)
        final_detail["soft_vote"] = {"macro_f1": float(f1_score(cls_te, sv.argmax(1), average="macro")),
                                     "log_loss": float(log_loss(cls_te, sv, labels=[0, 1, 2])),
                                     **{k: val for k, val in M.classification_metrics(cls_te, sv.argmax(1), (cls_te == 2).astype(int), sv[:, 2], bins).items()
                                        if k != "macro_f1"}}

    # --------------------------------------------------------- statistical tests
    singles = [m for m in models if m["kind"] == "single"]
    best_single = max(singles, key=lambda m: M.r2(y, m["oof"]))
    folds = data.train["fold"].to_numpy()
    fold_scores = lambda p: [M.r2(y[folds == k], p[folds == k]) for k in np.unique(folds)]  # noqa: E731
    tests = {
        "ensemble": final["name"], "best_single": best_single["name"],
        "paired_bootstrap_r2": M.paired_bootstrap(M.r2, y_test, final["test"], best_single["test"], n_boot, float(ecfg.ci), seed),
        "wilcoxon_folds": {**M.wilcoxon_folds(fold_scores(final["oof"]), fold_scores(best_single["oof"])),
                           "ensemble_fold_r2": fold_scores(final["oof"]), "single_fold_r2": fold_scores(best_single["oof"])},
    }
    rng = np.random.default_rng(seed)
    blocks = np.array_split(rng.permutation(len(y_test)), int(ecfg.friedman_blocks))
    pool = [m for m in models if m["kind"] != "ensemble"] + ([final] if final["kind"] == "ensemble" else [])
    score_matrix = np.array([[M.r2(y_test[b], m["test"][b]) for m in pool] for b in blocks])
    tests["friedman_nemenyi"] = M.friedman_nemenyi(score_matrix, [m["name"] for m in pool])
    log.info("Paired bootstrap ΔR² (ensemble − best single) = %.5f, p = %.3f; Friedman p = %.3g",
             tests["paired_bootstrap_r2"]["diff"], tests["paired_bootstrap_r2"]["p_value"], tests["friedman_nemenyi"]["p_value"])

    # ------------------------------------------------------------------ ablations
    ablation = run_ablations(cfg, data, y_test, n_boot, seed)
    ens = load_json(paths.metrics / "ensemble.json", {})
    ablation["ensemble_strategies"] = ens.get("candidates", {})

    # ------------------------------------------------------------ error analysis
    errors = error_analysis(cfg, data, final["test"], v["pred_classes"], v["true_classes"], cut_true)

    # -------------------------------------------------------- external validation
    external = external_validation(cfg, data, n_boot, seed) if (cfg.data.external_validation and data.external is not None) else {}

    save_json({"models": rows, "final": final_detail, "test_rows": int(len(y_test)), "data_source": data.audit.get("source")},
              paths.metrics / "evaluation.json")
    save_json(tests, paths.metrics / "statistical_tests.json")
    save_json(ablation, paths.metrics / "ablation.json")
    save_json(errors, paths.metrics / "error_analysis.json")
    save_json(external, paths.metrics / "external_validation.json")

    tracker = Tracker(cfg)
    with tracker.run("evaluate", tags={"stage": "evaluate"}):
        for r in rows:
            tracker.log_metrics({f"{r['key']}_test_r2": r["test_r2"]["value"]})
        tracker.log_metrics({"final_test_r2": next(r for r in rows if r["key"] == final["key"])["test_r2"]["value"]})
        for name in ("evaluation", "statistical_tests", "ablation", "error_analysis", "external_validation"):
            tracker.log_artifact(paths.metrics / f"{name}.json")
    return {"models": rows, "tests": tests}


def run_ablations(cfg: Config, data, y_test: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    name = cfg.evaluation.ablation_model
    paths = run_paths(cfg)
    out: dict[str, Any] = {"model": display_name(name), "runs": []}
    ref = load_predictions(cfg, name)
    ref_cv = load_json(paths.metrics / f"cv_{name}.json", {})
    if ref is not None:
        out["runs"].append({"variant": "full (all features)", "oof_r2": ref_cv.get("oof_r2"),
                            "test_r2": M.bootstrap_ci(M.r2, y_test, ref[1], n_boot, seed=seed)})
    ae = load_predictions(cfg, "ae_lgbm")
    if ae is not None:
        out["runs"].append({"variant": "+ autoencoder embeddings", "oof_r2": load_json(paths.metrics / "cv_ae_lgbm.json", {}).get("oof_r2"),
                            "test_r2": M.bootstrap_ci(M.r2, y_test, ae[1], n_boot, seed=seed)})
    cache = FoldCache(cfg, data)
    for ab in cfg.evaluation.ablations:
        if ab == "without_original":
            if not cfg.data.merge_original or data.train[FLAG].sum() == 0:
                continue
            frame = data.train[data.train[FLAG] == 0].reset_index(drop=True)
            res = run_cv(name, cfg, cache, variant="default", frame=frame, key="s4e5_only", save=False)
        else:
            res = run_cv(name, cfg, cache, variant=ab, save=False)
        out["runs"].append({"variant": ab.replace("_", " "), "oof_r2": res.oof_r2,
                            "test_r2": M.bootstrap_ci(M.r2, y_test, res.test, n_boot, seed=seed)})
        log.info("Ablation %-22s OOF R² %.5f, test R² %.5f", ab, res.oof_r2, M.r2(y_test, res.test))
    return out


def error_analysis(cfg: Config, data, pred: np.ndarray, pred_cls: np.ndarray, true_cls: np.ndarray,
                   cut_true: list[float]) -> dict[str, Any]:
    ecfg = cfg.evaluation.error_analysis
    hold = data.holdout.reset_index(drop=True)
    y = hold[data.target].to_numpy()
    res = y - pred
    abs_res = np.abs(res)
    deciles = pd.qcut(pd.Series(y).rank(method="first"), 10, labels=False).to_numpy()
    by_decile = [{"decile": int(d) + 1, "target_mean": float(y[deciles == d].mean()), "mean_residual": float(res[deciles == d].mean()),
                  "mae": float(abs_res[deciles == d].mean()), "n": int((deciles == d).sum())} for d in range(10)]
    by_theme = {}
    for theme, cols in dict(cfg.features.themes).items():
        score = hold[list(cols)].mean(axis=1)
        q = pd.qcut(score.rank(method="first"), 5, labels=False).to_numpy()
        by_theme[theme] = [{"quintile": int(k) + 1, "mae": float(abs_res[q == k].mean()),
                            "mean_residual": float(res[q == k].mean()), "n": int((q == k).sum())} for k in range(5)]
    lo, hi = np.quantile(y, 0.01), np.quantile(y, 0.99)
    extreme = {"overall_mae": float(abs_res.mean()),
               "bottom_1pct_mae": float(abs_res[y <= lo].mean()), "top_1pct_mae": float(abs_res[y >= hi].mean()),
               "bottom_1pct_mean_residual": float(res[y <= lo].mean()), "top_1pct_mean_residual": float(res[y >= hi].mean())}
    outliers = {}
    for flag in ("outlier_iqr", "outlier_iforest"):
        if flag in hold.columns:
            m = hold[flag].to_numpy() == 1
            outliers[flag] = {"rows": int(m.sum()), "mae_flagged": float(abs_res[m].mean()) if m.any() else None,
                              "mae_other": float(abs_res[~m].mean())}
    worst_idx = np.argsort(-abs_res)[: int(ecfg.top_k)]
    worst = [{"holdout_index": int(i), "actual": float(y[i]), "predicted": float(pred[i]), "residual": float(res[i]),
              "factor_sum": float(hold.loc[i, list(cfg.features.factors)].sum())} for i in worst_idx]
    near = np.min(np.abs(y[:, None] - np.asarray(cut_true)[None, :]), axis=1) < float(ecfg.boundary_width)
    wrong = pred_cls != true_cls
    boundary = {"width": float(ecfg.boundary_width), "near_rows": int(near.sum()),
                "error_rate_near_boundary": float(wrong[near].mean()) if near.any() else None,
                "error_rate_elsewhere": float(wrong[~near].mean()) if (~near).any() else None,
                "share_of_errors_near_boundary": float((wrong & near).sum() / max(wrong.sum(), 1))}
    return {"by_target_decile": by_decile, "by_theme_quintile": by_theme, "extremes": extreme, "outliers": outliers,
            "worst_predictions": worst, "class_boundaries": boundary}


def external_validation(cfg: Config, data, n_boot: int, seed: int) -> dict[str, Any]:
    """Train on S4E5 rows only, test on the original (naiyakhalid) dataset."""
    name = cfg.evaluation.ablation_model
    s4e5 = data.train[data.train[FLAG] == 0].reset_index(drop=True)
    pipe, model = fit_full(name, cfg, data, frame=s4e5)
    ext = data.external
    pred = np.asarray(model.predict(pipe.transform(data.X(ext)).astype(np.float32)), dtype=float)
    y_ext = ext[data.target].to_numpy()
    reg = {k: M.bootstrap_ci(f, y_ext, pred, n_boot, seed=seed) for k, f in M.REGRESSION.items()}
    cls_true, cls_pred = M.to_risk_class(y_ext, data.cutoffs), M.to_risk_class(pred, data.cutoffs)
    result = {"model": display_name(name), "trained_on": "S4E5 training split only", "tested_on": "original dataset",
              "rows": int(len(ext)), **reg,
              "macro_f1_at_true_cutoffs": float(f1_score(cls_true, cls_pred, average="macro")),
              "target_mean_train": float(s4e5[data.target].mean()), "target_mean_external": float(y_ext.mean())}
    log.info("External validation (%s): R² %.4f [%.4f, %.4f]", name, reg["r2"]["value"], reg["r2"]["lo"], reg["r2"]["hi"])
    return result


if __name__ == "__main__":
    from floodguard.utils.config import cli

    config, _ = cli(__doc__)
    run(config)
