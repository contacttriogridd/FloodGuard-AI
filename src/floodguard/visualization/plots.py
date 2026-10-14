"""All report figures (PNG) — written to <run_dir>/figures and report/figures.

    python -m floodguard.visualization.plots --override configs/smoke.yaml

Figures 14–17 (SHAP, dependence, PDP/ICE, LIME) are produced by the explain
stage using the helpers at the bottom of this module.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from floodguard.utils.common import RunPaths, get_logger, load_json, run_paths  # noqa: E402
from floodguard.utils.config import Config  # noqa: E402

log = get_logger("floodguard.plots")

# Validated categorical palette (fixed order), ordinal ramp for Low/Medium/High, diverging map.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CLASS_COLORS = ["#86b6ef", "#2a78d6", "#104281"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list("fg_div", ["#1c5cab", "#86b6ef", "#f0efec", "#f1a3a2", "#c73636"])
SEQUENTIAL = LinearSegmentedColormap.from_list("fg_seq", ["#e4effc", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 170, "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "lines.linewidth": 2.0, "axes.prop_cycle": plt.cycler(color=SLOTS),
    })


def save(fig: plt.Figure, name: str, paths: RunPaths) -> Path:
    fig.tight_layout()
    out = paths.figures / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy(out, paths.report_figures / out.name)
    return out


def _class_names(cfg: Config) -> list[str]:
    return list(cfg.risk.class_names)


# ------------------------------------------------------------------ data
def fig01_target(cfg: Config, paths: RunPaths) -> Path:
    train = pd.read_parquet(paths.data / "train.parquet")
    audit = load_json(paths.audit / "data_audit.json")
    y = train.loc[train["is_original"] == 0, cfg.data.target]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.hist(y, bins=60, color=SLOTS[0], edgecolor="white", linewidth=0.4)
    names = _class_names(cfg)
    for c in audit["risk"]["cutoffs"]:
        ax.axvline(c, color=INK, lw=1)
    edges = [y.min(), *audit["risk"]["cutoffs"], y.max()]
    for i, n in enumerate(names):
        ax.text((edges[i] + edges[i + 1]) / 2, ax.get_ylim()[1] * 0.92, n, ha="center", color=INK, fontweight="bold")
    ax.set(title="FloodProbability distribution and risk-class thresholds", xlabel="FloodProbability", ylabel="Rows")
    return save(fig, "01_target_distribution", paths)


def fig02_distributions(cfg: Config, paths: RunPaths) -> Path:
    train = pd.read_parquet(paths.data / "train.parquet")
    factors = list(cfg.features.factors)
    complete = train[factors].dropna()
    sample = complete.sample(n=min(20000, len(complete)), random_state=0)
    fig, axes = plt.subplots(4, 5, figsize=(12, 8.5), sharey=True)
    for ax, f in zip(axes.ravel(), factors):
        parts = ax.violinplot(sample[f], showmedians=True, widths=0.8)
        for b in parts["bodies"]:
            b.set_facecolor(SLOTS[0])
            b.set_alpha(0.35)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[key].set_color(SLOTS[0])
            parts[key].set_linewidth(1)
        ax.set_title(f, fontsize=8.5)
        ax.set_xticks([])
    fig.suptitle("Factor distributions (violin plots, training split)", fontweight="bold")
    return save(fig, "02_feature_distributions", paths)


def fig03_correlation(cfg: Config, paths: RunPaths) -> Path:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    corr = load_json(paths.features / "correlation.json")
    cols, mat = corr["columns"], np.asarray(corr["matrix"])
    dist = np.clip(1 - np.abs(mat), 0, None)
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist, checks=False), method="average")
    fig = plt.figure(figsize=(12, 6.2))
    # The dendrogram's leaf labels name the heatmap rows; both axes share the same height.
    ax_d = fig.add_axes([0.02, 0.18, 0.16, 0.74])
    dn = dendrogram(link, orientation="left", labels=cols, ax=ax_d, color_threshold=0, above_threshold_color=MUTED,
                    leaf_font_size=7)
    ax_d.grid(False)
    ax_d.spines[["left", "bottom"]].set_visible(False)
    ax_d.set_xticks([])
    order = dn["leaves"][::-1]
    ax_h = fig.add_axes([0.37, 0.18, 0.50, 0.74])
    im = ax_h.imshow(mat[np.ix_(order, order)], cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax_h.set_xticks(range(len(cols)), [cols[i] for i in order], rotation=90, fontsize=6.5)
    ax_h.set_yticks([])
    ax_h.grid(False)
    fig.colorbar(im, ax=ax_h, fraction=0.04, label="Spearman ρ")
    r = corr["target_vs_row_sum"]["spearman"]
    ax_h.set_title(f"Spearman correlation (clustered). ρ(target, factor sum) = {r:.3f}")
    ax_d.set_title("Clustering (1 − |ρ|)")
    out = paths.figures / "03_correlation_dendrogram.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy(out, paths.report_figures / out.name)
    return out


def fig04_missingness(cfg: Config, paths: RunPaths) -> Path:
    mask = np.load(paths.audit / "missing_mask_sample.npy")
    comp = load_json(paths.audit / "imputer_comparison.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    a1.imshow(~mask, aspect="auto", cmap=LinearSegmentedColormap.from_list("m", [INK, "#f0efec"]), interpolation="nearest")
    a1.set(title=f"Injected MCAR missingness ({comp['inject_rate']:.0%}) — first {len(mask)} rows",
           xlabel="Factor index", ylabel="Row")
    a1.grid(False)
    names = list(comp["results"])
    means = [comp["results"][n]["cv_r2_mean"] for n in names]
    stds = [comp["results"][n]["cv_r2_std"] for n in names]
    colors = [SLOTS[0] if n == comp["selected"] else "#a9b4c2" for n in names]
    a2.barh(names, means, xerr=stds, color=colors, height=0.5, error_kw={"ecolor": MUTED, "lw": 1})
    for i, m in enumerate(means):
        a2.text(m, i, f"  {m:.4f}", va="center", color=INK, fontsize=8.5)
    a2.set(title="Imputer comparison (downstream CV R²)", xlabel="CV R²", xlim=(min(means) - 0.02, max(means) + 0.02))
    return save(fig, "04_missingness_imputers", paths)


def fig05_pca(cfg: Config, paths: RunPaths) -> Path:
    summ = load_json(paths.reduction / "reduction_summary.json")["pca"]
    pts = np.load(paths.reduction / "pca_points.npz", allow_pickle=True)
    evr, cum = np.asarray(summ["explained_variance_ratio"]), np.asarray(summ["cumulative"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    k = np.arange(1, len(evr) + 1)
    a1.bar(k, evr, color=SLOTS[0], width=0.7, label="Explained variance")
    a1.plot(k, cum, color=SLOTS[1], marker="o", ms=4, label="Cumulative")
    a1.axhline(0.9, color=MUTED, lw=1)
    a1.set(title="PCA scree plot", xlabel="Component", ylabel="Variance ratio")
    a1.legend()
    P, cls = pts["points"], pts["classes"]
    for c, name in enumerate(_class_names(cfg)):
        m = cls == c
        a2.scatter(P[m, 0], P[m, 1], s=7, color=CLASS_COLORS[c], alpha=0.6, label=name, edgecolors="none")
    load, factors = pts["loadings"], pts["factors"]
    top = np.argsort(-np.hypot(load[0], load[1]))[:6]
    scale = np.abs(P).max() * 0.8 / max(np.abs(load[:, top]).max(), 1e-9)
    for i in top:
        a2.arrow(0, 0, load[0, i] * scale, load[1, i] * scale, color=INK, width=0.002, head_width=0.08)
        a2.text(load[0, i] * scale * 1.08, load[1, i] * scale * 1.08, str(factors[i]), fontsize=7, color=INK)
    a2.set(title="PCA biplot (PC1 vs PC2)", xlabel="PC1", ylabel="PC2")
    a2.legend(markerscale=2)
    return save(fig, "05_pca_scree_biplot", paths)


def fig06_projections(cfg: Config, paths: RunPaths) -> Path:
    methods = [m for m in ("tsne", "umap") if (paths.reduction / f"{m}.npz").exists()]
    fig, axes = plt.subplots(1, max(len(methods), 1), figsize=(5.6 * max(len(methods), 1), 4.6), squeeze=False)
    for ax, m in zip(axes[0], methods):
        d = np.load(paths.reduction / f"{m}.npz")
        for c, name in enumerate(_class_names(cfg)):
            sel = d["classes"] == c
            ax.scatter(d["embedding"][sel, 0], d["embedding"][sel, 1], s=7, color=CLASS_COLORS[c], alpha=0.7,
                       label=name, edgecolors="none")
        ax.set(title=f"{'t-SNE' if m == 'tsne' else 'UMAP'} projection of the 20 factors", xticks=[], yticks=[])
        ax.legend(markerscale=2, title="Risk class")
    return save(fig, "06_tsne_umap", paths)


def fig07_autoencoder(cfg: Config, paths: RunPaths) -> Path:
    hist = load_json(paths.reduction / "ae_history.json")
    emb = np.load(paths.reduction / "ae_embedding.npz")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist["train_loss"], label="Train")
    a1.plot(hist["val_loss"], label="Validation")
    a1.set(title="Autoencoder reconstruction loss", xlabel="Epoch", ylabel="MSE (standardised)")
    a1.legend()
    E = emb["embedding"]
    for c, name in enumerate(_class_names(cfg)):
        sel = emb["classes"] == c
        a2.scatter(E[sel, 0], E[sel, 1], s=7, color=CLASS_COLORS[c], alpha=0.7, label=name, edgecolors="none")
    a2.set(title="Bottleneck embedding (dims 1–2)", xlabel="ae_0", ylabel="ae_1")
    a2.legend(markerscale=2)
    return save(fig, "07_autoencoder", paths)


# ---------------------------------------------------------------- models
def fig08_training_curves(cfg: Config, paths: RunPaths) -> Path:
    curves = [(n, load_json(paths.metrics / f"cv_{n}.json", {}).get("curve_fold0")) for n in ("lgbm", "xgb", "mlp")]
    curves = [(n, c) for n, c in curves if c]
    fig, axes = plt.subplots(1, len(curves), figsize=(4.6 * len(curves), 3.8), squeeze=False)
    labels = {"lgbm": "LightGBM (RMSE)", "xgb": "XGBoost (RMSE)", "mlp": "MLP (standardised MSE)"}
    for ax, (n, c) in zip(axes[0], curves):
        if c.get("train"):
            ax.plot(c["train"], label="Train", color=SLOTS[0])
        ax.plot(c["valid"], label="Validation fold", color=SLOTS[1])
        ax.set(title=f"Learning curve — {labels[n]}", xlabel="Iteration / epoch", ylabel="Loss")
        ax.legend()
    return save(fig, "08_training_curves", paths)


def fig09_optuna(cfg: Config, paths: RunPaths) -> Path:
    models = [m for m in cfg.models.tune if (paths.optuna / f"{m}_history.json").exists()]
    imp_model = next((m for m in ("lgbm", "xgb", "catboost") + tuple(models)
                      if load_json(paths.optuna / f"{m}_importance.json", {})), None)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4))
    for i, m in enumerate(models):
        h = [t for t in load_json(paths.optuna / f"{m}_history.json") if t["best"] is not None]
        a1.step([t["number"] + 1 for t in h], [t["best"] for t in h], where="post", label=m, color=SLOTS[i % len(SLOTS)])
    a1.set(title="Optuna optimisation history (best CV R² so far)", xlabel="Trial", ylabel="CV R²")
    a1.legend(ncol=2, fontsize=8)
    if imp_model:
        imp = load_json(paths.optuna / f"{imp_model}_importance.json")
        keys = list(imp)[::-1]
        a2.barh(keys, [imp[k] for k in keys], color=SLOTS[0], height=0.55)
        a2.set(title=f"Hyper-parameter importance — {imp_model}", xlabel="fANOVA importance")
    else:
        a2.text(0.5, 0.5, "Needs ≥ 2 completed trials", ha="center", transform=a2.transAxes, color=MUTED)
        a2.set_axis_off()
    return save(fig, "09_optuna", paths)


def _final_oof(cfg: Config, paths: RunPaths) -> tuple[np.ndarray, str]:
    ens = load_json(paths.metrics / "ensemble.json")
    sel = ens["selected"]
    if sel == "best_single":
        best = max(ens["single_oof_r2"], key=ens["single_oof_r2"].get)
        return np.load(paths.oof / f"{best}_oof.npy"), best
    return np.load(paths.oof / f"ens_{sel}_oof.npy"), sel


def fig10_oof(cfg: Config, paths: RunPaths) -> Path:
    train = pd.read_parquet(paths.data / "train.parquet")
    y = train[cfg.data.target].to_numpy()
    p, name = _final_oof(cfg, paths)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    hb = a1.hexbin(y, p, gridsize=45, cmap=SEQUENTIAL, mincnt=1)
    lim = [min(y.min(), p.min()), max(y.max(), p.max())]
    a1.plot(lim, lim, color=INK, lw=1)
    a1.set(title=f"OOF predicted vs actual ({name})", xlabel="Actual FloodProbability", ylabel="Predicted")
    fig.colorbar(hb, ax=a1, label="Rows")
    a2.hexbin(p, y - p, gridsize=45, cmap=SEQUENTIAL, mincnt=1)
    a2.axhline(0, color=INK, lw=1)
    a2.set(title="Residuals vs prediction", xlabel="Predicted", ylabel="Residual (actual − predicted)")
    return save(fig, "10_oof_pred_vs_actual", paths)


def fig11_confusion(cfg: Config, paths: RunPaths) -> Path:
    final = load_json(paths.metrics / "evaluation.json")["final"]
    cm = np.asarray(final["confusion_matrix"])
    share = cm / cm.sum(axis=1, keepdims=True)
    names = _class_names(cfg)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > 0.5 else INK)
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.grid(False)
    ax.set(title=f"Confusion matrix — {final['name']} (test)", xlabel="Predicted class", ylabel="Actual class")
    return save(fig, "11_confusion_matrix", paths)


def fig12_roc_pr(cfg: Config, paths: RunPaths) -> Path:
    ev = load_json(paths.metrics / "evaluation.json")
    final = ev["final"]
    row = next(r for r in ev["models"] if r["key"] == final["key"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    a1.plot(final["roc"]["fpr"], final["roc"]["tpr"], color=SLOTS[0])
    a1.fill_between(final["roc"]["fpr"], final["roc"]["tpr"], color=SLOTS[0], alpha=0.1)
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1)
    a1.set(title=f"ROC — High-risk (AUC {row.get('test_roc_auc', float('nan')):.3f})", xlabel="False positive rate",
           ylabel="True positive rate")
    a2.plot(final["pr"]["recall"], final["pr"]["precision"], color=SLOTS[0])
    a2.axhline(final["pr"]["prevalence"], color=MUTED, lw=1)
    a2.set(title=f"Precision–recall — High-risk (AP {row.get('test_pr_auc', float('nan')):.3f})", xlabel="Recall",
           ylabel="Precision")
    return save(fig, "12_roc_pr", paths)


def fig13_calibration(cfg: Config, paths: RunPaths) -> Path:
    ev = load_json(paths.metrics / "evaluation.json")
    rel = ev["final"]["reliability"]
    row = next(r for r in ev["models"] if r["key"] == ev["final"]["key"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.2, 5.6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1, label="Perfect calibration")
    a1.plot(rel["confidence"], rel["accuracy"], marker="o", ms=5, color=SLOTS[0], label="FloodGuard")
    a1.set(title=f"Reliability curve — P(High) (ECE {row.get('test_ece', float('nan')):.3f})", ylabel="Observed frequency")
    a1.legend()
    a2.bar(rel["confidence"], rel["count"], width=0.06, color=SLOTS[0])
    a2.set(xlabel="Predicted probability", ylabel="Rows")
    return save(fig, "13_calibration", paths)


def fig18_model_comparison(cfg: Config, paths: RunPaths) -> Path:
    """Dot-and-whisker (not bars): the interesting differences are far from zero."""
    every = load_json(paths.metrics / "evaluation.json")["models"]
    trivial = [r for r in every if r["test_r2"]["value"] < 0.2]  # e.g. the mean predictor — noted, not plotted
    rows = sorted([r for r in every if r not in trivial], key=lambda r: r["test_r2"]["value"])
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(rows) + 1.4))
    for i, r in enumerate(rows):
        v, lo, hi = r["test_r2"]["value"], r["test_r2"]["lo"], r["test_r2"]["hi"]
        color = SLOTS[0] if r["selected"] else (MUTED if r["kind"] != "baseline" else "#a9a8a1")
        ax.plot([lo, hi], [i, i], color=color, lw=2, solid_capstyle="round")
        ax.scatter([v], [i], s=46 if r["selected"] else 30, color=color, zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(hi, i, f"  {v:.4f}", va="center", fontsize=8, color=INK, fontweight="bold" if r["selected"] else "normal")
    ax.set_yticks(range(len(rows)), [r["name"] + ("  (baseline)" if r["kind"] == "baseline" else "") for r in rows])
    span = max(r["test_r2"]["hi"] for r in rows) - min(r["test_r2"]["lo"] for r in rows)
    ax.set_xlim(min(r["test_r2"]["lo"] for r in rows) - 0.05 * span, max(r["test_r2"]["hi"] for r in rows) + 0.25 * span)
    note = "; ".join(f"{r['name']}: {r['test_r2']['value']:.3f}" for r in trivial)
    ax.set(title="Test R² with 95% bootstrap CIs (blue = selected FloodGuard ensemble)",
           xlabel="R² (hold-out test split)" + (f"\nNot shown: {note}" if note else ""))
    return save(fig, "18_model_comparison", paths)


def fig19_ensemble(cfg: Config, paths: RunPaths) -> Path:
    ens = load_json(paths.metrics / "ensemble.json")
    names = list(ens["weights_nnls"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in [(axes[0], "weights_nnls", "NNLS blend weights"), (axes[1], "weights_hill", "Hill-climbing weights")]:
        w = ens[key]
        ax.barh(names, [w[n] for n in names], color=SLOTS[0], height=0.55)
        ax.set(title=title, xlim=(0, 1))
    coefs = ens["stack_ridge_coefficients"]
    keys = list(coefs)
    axes[2].barh(keys, [coefs[k] for k in keys], color=[SLOTS[0] if coefs[k] >= 0 else SLOTS[7] for k in keys], height=0.55)
    axes[2].axvline(0, color=INK, lw=0.8)
    axes[2].set(title="Stacking (Ridge) coefficients")
    fig.suptitle(f"Ensemble layer — selected strategy: {ens['selected']}", fontweight="bold")
    return save(fig, "19_ensemble_weights", paths)


def fig20_errors(cfg: Config, paths: RunPaths) -> Path:
    err = load_json(paths.metrics / "error_analysis.json")
    dec = err["by_target_decile"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    x = [d["decile"] for d in dec]
    # Mean residual and MAE share the target's units, so one y-axis serves both.
    a1.bar(x, [d["mean_residual"] for d in dec], color=[SLOTS[0] if d["mean_residual"] >= 0 else SLOTS[7] for d in dec],
           width=0.6, label="Mean residual")
    a1.plot(x, [d["mae"] for d in dec], color=INK, marker="o", ms=4, lw=1.5, label="MAE")
    a1.axhline(0, color=INK, lw=0.8)
    a1.set(title="Errors by target decile", xlabel="Target decile (1 = lowest risk)", ylabel="FloodProbability units")
    a1.legend()
    for i, (theme, rows) in enumerate(err["by_theme_quintile"].items()):
        a2.plot([r["quintile"] for r in rows], [r["mae"] for r in rows], marker="o", ms=4, label=theme.replace("_", " "),
                color=SLOTS[i])
    a2.set(title="MAE by theme-score quintile", xlabel="Theme quintile (1 = lowest)", ylabel="MAE")
    a2.legend(fontsize=8)
    a1.set_xticks(range(1, 11))
    a2.set_xticks(range(1, 6))
    return save(fig, "20_error_analysis", paths)


def fig21_cd(cfg: Config, paths: RunPaths) -> Path:
    fr = load_json(paths.metrics / "statistical_tests.json")["friedman_nemenyi"]
    ranks = dict(sorted(fr["average_ranks"].items(), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(8, 0.33 * len(ranks) + 1.4))
    names = list(ranks)
    ax.scatter(list(ranks.values()), range(len(names)), color=SLOTS[0], s=40, zorder=3)
    best = min(ranks.values())
    ax.axvspan(best, best + fr["critical_difference"], color=SLOTS[0], alpha=0.1,
               label=f"CD = {fr['critical_difference']:.2f} (Nemenyi, α = {fr['alpha']})")
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set(title=f"Average rank over {fr['n_blocks']} test blocks (Friedman p = {fr['p_value']:.2g})", xlabel="Average rank (1 = best)")
    ax.legend(loc="lower right")
    return save(fig, "21_critical_difference", paths)


FIGURES: list[Callable[[Config, RunPaths], Path]] = [
    fig01_target, fig02_distributions, fig03_correlation, fig04_missingness, fig05_pca, fig06_projections,
    fig07_autoencoder, fig08_training_curves, fig09_optuna, fig10_oof, fig11_confusion, fig12_roc_pr,
    fig13_calibration, fig18_model_comparison, fig19_ensemble, fig20_errors, fig21_cd,
]


def run(cfg: Config) -> list[str]:
    style()
    paths = run_paths(cfg)
    done = []
    for fn in FIGURES:
        try:
            done.append(str(fn(cfg, paths)))
        except (FileNotFoundError, KeyError, TypeError, ValueError, StopIteration) as exc:
            log.warning("Figure %s skipped: %s", fn.__name__, exc)
    log.info("Wrote %d figures to %s", len(done), paths.figures)
    return done


# ---------------------------------------------------- helpers for explain
def save_current(name: str, paths: RunPaths) -> Path:
    return save(plt.gcf(), name, paths)


def lime_figure(weights: list[tuple[str, float]], title: str, paths: RunPaths) -> Path:
    style()
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(weights) + 1.2))
    labels = [w[0] for w in weights][::-1]
    vals = [w[1] for w in weights][::-1]
    ax.barh(labels, vals, color=[SLOTS[7] if v > 0 else SLOTS[0] for v in vals], height=0.55)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set(title=title, xlabel="LIME weight (effect on FloodProbability)")
    return save(fig, "17_lime_explanation", paths)


if __name__ == "__main__":
    from floodguard.utils.config import cli

    config, _ = cli(__doc__)
    run(config)

__all__: list[Any] = ["run", "style", "save", "lime_figure", "SLOTS", "CLASS_COLORS"]
