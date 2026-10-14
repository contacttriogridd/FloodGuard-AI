"""Explainability: SHAP (global, local, interactions), permutation importance,
PDP/ICE and LIME.

* Global SHAP / interactions: TreeSHAP on a LightGBM model in the engineered
  feature space (figures 14, 15).
* Model-agnostic views of the *deployed* ensemble over the 20 raw factors:
  permutation importance, PDP/ICE (16), LIME (17) and local Shapley values for
  the 20 worst test predictions (used by the error analysis).

    python -m floodguard.explain.explainability --override configs/smoke.yaml
"""
from __future__ import annotations

from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from floodguard.data.prepare import load_prepared  # noqa: E402
from floodguard.models.cv import fit_full  # noqa: E402
from floodguard.models.zoo import refit_params, resolve_params  # noqa: E402
from floodguard.utils.common import get_logger, load_json, run_paths, save_json, seed_everything  # noqa: E402
from floodguard.utils.config import Config  # noqa: E402
from floodguard.utils.tracking import Tracker  # noqa: E402
from floodguard.visualization import plots  # noqa: E402

log = get_logger("floodguard.explain")


def tree_shap(cfg: Config, data, paths) -> dict[str, Any]:
    import shap

    ecfg = cfg.explain
    cv_info = load_json(paths.metrics / "cv_lgbm.json", {})
    params = resolve_params(cfg, "lgbm")
    params.update(refit_params("lgbm", cv_info.get("best_iterations", [])))
    pipe, model = fit_full("lgbm", cfg, data, params=params)
    sample = data.holdout.sample(n=min(int(ecfg.shap_sample), len(data.holdout)), random_state=cfg.project.seed)
    F = pipe.transform(data.X(sample)).astype(np.float32)
    explainer = shap.TreeExplainer(model)
    sv = explainer(F)
    mean_abs = np.abs(sv.values).mean(axis=0)
    order = np.argsort(-mean_abs)

    plots.style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
    try:
        shap.plots.beeswarm(sv, max_display=15, show=False, ax=a1, plot_size=None, color_bar=True)
    except TypeError:  # older shap without ax support
        plt.sca(a1)
        shap.summary_plot(sv.values, F, max_display=15, show=False, plot_size=None)
    a1.set_title("SHAP beeswarm — LightGBM (engineered features)")
    top = order[:15][::-1]
    a2.barh([F.columns[i] for i in top], mean_abs[top], color=plots.SLOTS[0], height=0.55)
    a2.set(title="Mean |SHAP| (global importance)", xlabel="mean |SHAP value|")
    plots.save(fig, "14_shap_beeswarm_bar", paths)

    # Interactions + dependence for the two most important features.
    n_int = min(int(ecfg.interaction_sample), len(F))
    Fi = F.iloc[:n_int]
    inter = np.abs(explainer.shap_interaction_values(Fi)).mean(axis=0)
    np.fill_diagonal(inter, 0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    pairs = []
    for ax, i in zip(axes[:2], order[:2]):
        j = int(np.argmax(inter[i]))
        pairs.append({"feature": F.columns[i], "partner": F.columns[j], "mean_abs_interaction": float(inter[i, j])})
        sc = ax.scatter(F.iloc[:, i], sv.values[:, i], c=F.iloc[:, j], cmap=plots.SEQUENTIAL, s=10, edgecolors="none")
        fig.colorbar(sc, ax=ax, label=F.columns[j])
        ax.axhline(0, color=plots.INK, lw=0.8)
        ax.set(title=f"Dependence: {F.columns[i]}", xlabel=F.columns[i], ylabel="SHAP value")
    k = min(10, len(order))
    idx = order[:k]
    im = axes[2].imshow(inter[np.ix_(idx, idx)], cmap=plots.SEQUENTIAL)
    axes[2].set_xticks(range(k), [F.columns[i] for i in idx], rotation=90, fontsize=7)
    axes[2].set_yticks(range(k), [F.columns[i] for i in idx], fontsize=7)
    axes[2].grid(False)
    axes[2].set_title("Mean |SHAP interaction| (top features)")
    fig.colorbar(im, ax=axes[2])
    plots.save(fig, "15_shap_dependence_interactions", paths)

    return {"model": "LightGBM (refit on full training split)", "rows": int(len(F)),
            "base_value": float(np.ravel(sv.base_values)[0]),
            "importance": [{"feature": F.columns[i], "mean_abs_shap": float(mean_abs[i])} for i in order],
            "top_interactions": pairs}


def deployed_views(cfg: Config, data, paths, bundle: dict[str, Any]) -> dict[str, Any]:
    from sklearn.inspection import PartialDependenceDisplay, permutation_importance

    ecfg, seed = cfg.explain, cfg.project.seed
    pipeline, factors = bundle["pipeline"], bundle["factors"]
    hold = data.holdout.sample(n=min(int(ecfg.permutation_sample), len(data.holdout)), random_state=seed)
    X, y = data.X(hold), hold[data.target].to_numpy()
    perm = permutation_importance(pipeline, X, y, n_repeats=int(ecfg.permutation_repeats), random_state=seed, scoring="r2",
                                  n_jobs=1)
    perm_rows = sorted([{"feature": c, "importance_mean": float(m), "importance_std": float(s)}
                        for c, m, s in zip(X.columns, perm.importances_mean, perm.importances_std) if c in factors],
                       key=lambda r: -r["importance_mean"])

    top = [r["feature"] for r in perm_rows[: int(ecfg.pdp_features)]]
    ice = X.sample(n=min(int(ecfg.ice_sample), len(X)), random_state=seed)
    plots.style()
    fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 3.8), squeeze=False)
    PartialDependenceDisplay.from_estimator(pipeline, ice, features=top, kind="both", subsample=len(ice),
                                            grid_resolution=15, random_state=seed, ax=axes[0],
                                            ice_lines_kw={"color": plots.SLOTS[0], "alpha": 0.08, "linewidth": 0.6},
                                            pd_line_kw={"color": plots.SLOTS[1], "linewidth": 2.2})
    fig.suptitle("Partial dependence (orange) and ICE curves (blue) — deployed ensemble", fontweight="semibold")
    plots.save(fig, "16_pdp_ice", paths)

    lime_out: dict[str, Any] = {}
    try:
        from lime.lime_tabular import LimeTabularExplainer

        train_f = data.train[factors].dropna().sample(n=min(5000, len(data.train)), random_state=seed).to_numpy()
        predict_fn = lambda A: pipeline.predict(pd.DataFrame(A, columns=factors).assign(is_original=0.0))  # noqa: E731
        preds = pipeline.predict(X)
        case = int(np.argsort(preds)[int(0.95 * (len(preds) - 1))])  # a high-risk test case
        lime_exp = LimeTabularExplainer(train_f, feature_names=factors, mode="regression", random_state=seed)
        exp = lime_exp.explain_instance(X.iloc[case][factors].to_numpy(dtype=float), predict_fn,
                                        num_features=int(ecfg.lime_num_features))
        weights = exp.as_list()
        plots.lime_figure(weights, f"LIME — test case predicted {preds[case]:.3f} (actual {y[case]:.3f})", paths)
        intercept = exp.intercept  # a {label: value} dict in regression mode
        intercept = next(iter(intercept.values())) if isinstance(intercept, dict) else np.ravel(intercept)[0]
        lime_out = {"case_index": case, "predicted": float(preds[case]), "actual": float(y[case]),
                    "weights": [{"rule": r, "weight": float(w)} for r, w in weights], "intercept": float(intercept)}
    except ImportError:
        log.warning("LIME not installed — figure 17 skipped.")
    return {"permutation_importance": perm_rows, "pdp_features": top, "lime": lime_out}


def worst_case_explanations(cfg: Config, data, paths, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    from floodguard.inference.predictor import FloodGuardPredictor

    errors = load_json(paths.metrics / "error_analysis.json", {})
    worst = errors.get("worst_predictions", [])
    if not worst:
        return []
    predictor = FloodGuardPredictor(bundle, local_max_evals=int(cfg.explain.local_max_evals))
    hold = data.holdout.reset_index(drop=True)
    out = []
    for w in worst:
        record = hold.loc[w["holdout_index"], bundle["factors"]].to_dict()
        exp = predictor.explain(record, top_k=5)
        out.append({**w, "top_drivers": exp["top_drivers"], "base_value": exp["base_value"]})
    return out


def run(cfg: Config) -> dict[str, Any]:
    seed_everything(cfg.project.seed)
    paths, data = run_paths(cfg), load_prepared(cfg)
    bundle = joblib.load(paths.models / "floodguard_pipeline.joblib")
    result = {"tree_shap": tree_shap(cfg, data, paths)}
    result.update(deployed_views(cfg, data, paths, bundle))
    result["worst_predictions"] = worst_case_explanations(cfg, data, paths, bundle)
    save_json(result, paths.explain / "explanations.json")
    tracker = Tracker(cfg)
    with tracker.run("explain", tags={"stage": "explain"}):
        tracker.log_dict({r["feature"]: r["mean_abs_shap"] for r in result["tree_shap"]["importance"][:30]},
                         "shap_importance.json")
        for name in ("14_shap_beeswarm_bar", "15_shap_dependence_interactions", "16_pdp_ice", "17_lime_explanation"):
            tracker.log_artifact(paths.figures / f"{name}.png")
    log.info("Explanations written (top SHAP feature: %s).", result["tree_shap"]["importance"][0]["feature"])
    return result


if __name__ == "__main__":
    from floodguard.utils.config import cli

    config, _ = cli(__doc__)
    run(config)
