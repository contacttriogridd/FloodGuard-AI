"""Write LaTeX tables and \\newcommand macros for report/FloodGuard_IEEE.tex.

Every number in the report comes from these generated files.

    python -m floodguard.evaluation.report_assets --override configs/smoke.yaml
"""
from __future__ import annotations

import shutil
from typing import Any

from floodguard.utils.common import environment_info, get_logger, load_json, run_paths
from floodguard.utils.config import Config

log = get_logger("floodguard.report")


def tex(s: Any) -> str:
    text = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                 ("→", r"$\rightarrow$"), ("×", r"$\times$"), ("²", r"$^2$"), ("≥", r"$\geq$"), ("−", "-"),
                 ("–", "--")]:
        text = text.replace(a, b)
    return text


def f(x: Any, nd: int = 4) -> str:
    return "--" if x is None else f"{float(x):.{nd}f}"


def table(caption: str, label: str, header: list[str], rows: list[list[str]], align: str | None = None,
          wide: bool = False, note: str = "") -> str:
    env = "table*" if wide else "table"
    align = align or "l" + "c" * (len(header) - 1)
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    note_tex = f"\n\\par\\footnotesize {note}" if note else ""
    return (f"\\begin{{{env}}}[!t]\n\\centering\\scriptsize\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}{note_tex}\n\\end{{{env}}}\n")


def _cardinality(sel: dict) -> str:
    pairs, factors = sel.get("pair_cardinality") or {}, sel.get("factor_cardinality") or {}
    if not pairs:
        return "--"
    return (f"{len(pairs)} keys, {min(pairs.values())}--{max(pairs.values())} levels "
            f"(single factors: {min(factors.values())}--{max(factors.values())})")


def environment_table(env: dict, seconds: dict) -> list[list[str]]:
    libs = env.get("libraries", {})
    lib_text = ", ".join(f"{name} {libs[key]}" for key, name in
                         [("scikit-learn", "scikit-learn"), ("xgboost", "XGBoost"), ("lightgbm", "LightGBM"),
                          ("catboost", "CatBoost"), ("torch", "PyTorch"), ("optuna", "Optuna"), ("shap", "SHAP"),
                          ("mlflow", "MLflow")] if libs.get(key))
    stage_text = ", ".join(f"{k} {v:.0f}\\,s" for k, v in seconds.items() if k != "total" and v >= 1)
    return [["CPU", tex(f"{env.get('cpu', 'unknown')} ({env.get('logical_cpus', '--')} logical cores)")],
            ["RAM", f"{env['ram_gb']} GB" if env.get("ram_gb") else "--"], ["GPU", tex(env.get("gpu", "none"))],
            ["OS / Python", tex(f"{env.get('os', '--')} / Python {env.get('python', '--')}")],
            ["Libraries", tex(lib_text)],
            ["Pipeline runtime", f"{seconds.get('total', 0):.0f}\\,s total"],
            ["Per stage", stage_text]]


def run(cfg: Config) -> list[str]:
    paths = run_paths(cfg)
    out = paths.report_tables
    m = paths.metrics
    ev, tests, abl = load_json(m / "evaluation.json"), load_json(m / "statistical_tests.json", {}), load_json(m / "ablation.json", {})
    bench, ext = load_json(m / "benchmark.json", {}), load_json(m / "external_validation.json", {})
    audit, sel = load_json(paths.audit / "data_audit.json", {}), load_json(paths.features / "selected_features.json", {})
    corr, ens = load_json(paths.features / "correlation.json", {}), load_json(m / "ensemble.json", {})
    if not ev:
        raise RuntimeError("Run the 'evaluate' stage first.")
    written: list[str] = []
    synthetic = ev.get("data_source") == "synthetic"
    src_note = "Synthetic smoke-test data --- not reportable results." if synthetic else "Kaggle S4E5 hold-out split."

    def write(name: str, content: str) -> None:
        (out / f"{name}.tex").write_text(content, encoding="utf-8")
        written.append(name)

    # --- model comparison -------------------------------------------------
    rows = sorted(ev["models"], key=lambda r: -r["test_r2"]["value"])
    mc = []
    for r in rows:
        name = tex(r["name"]) + (" $\\star$" if r["selected"] else "")
        ci = f"{f(r['test_r2']['value'])} [{f(r['test_r2']['lo'])}, {f(r['test_r2']['hi'])}]"
        mc.append([name, f(r["oof_r2"]), ci, f(r["test_rmse"]["value"], 5), f(r["test_mae"]["value"], 5),
                   f(r.get("test_macro_f1"), 3), f(r.get("test_roc_auc"), 3), f(r.get("test_pr_auc"), 3), f(r.get("test_ece"), 3)])
    write("model_comparison", table("Hold-out results with 95\\% bootstrap CIs (" + str(cfg.evaluation.bootstrap) +
                                    " resamples). $\\star$ = selected FloodGuard ensemble.", "tab:models",
                                    ["Model", "OOF $R^2$", "Test $R^2$ [95\\% CI]", "RMSE", "MAE", "Macro-F1", "ROC-AUC", "PR-AUC", "ECE"],
                                    mc, wide=True, note=src_note))

    # --- benchmark ----------------------------------------------------------
    kag = bench.get("kaggle", {})
    br = [[tex(r["name"]), f(r["test_r2"]), f(r["test_rmse"], 5)] for r in bench.get("table", [])
          if r["kind"] == "baseline" or r["selected"]]
    br.append(["Kaggle S4E5 private LB --- top", f(kag.get("private_top_r2")), "--"])
    br.append(["Kaggle S4E5 private LB --- median", f(kag.get("private_median_r2")), "--"])
    write("benchmark", table("Comparison with baselines and the Kaggle S4E5 leaderboard.", "tab:benchmark",
                             ["Approach", "$R^2$", "RMSE"], br,
                             note="Leaderboard values from configs/benchmarks.yaml (``--'' = not filled in)."))

    # --- published studies (different datasets) ----------------------------------
    sel_row = next((r for r in ev["models"] if r["selected"]), None)
    lr = [[tex(lit.get("study", "")) + " \\cite{" + tex(lit.get("bibkey", "")) + "}", tex(lit.get("dataset", "")),
           tex(lit.get("model", "")), f"{f(lit.get('value'), 3)} ({tex(lit.get('metric', ''))})"]
          for lit in bench.get("literature", []) if lit.get("verified")]
    if sel_row is not None:
        lr.append(["FloodGuard AI (this work)", "Kaggle S4E5 factor scores", tex(sel_row["name"]) + ", High-risk flag",
                   f"{f(sel_row.get('test_roc_auc'), 3)} (ROC-AUC)"])
    if lr:
        write("literature", table("Published flood-susceptibility ensembles.", "tab:literature",
                                  ["Study", "Data", "Model", "Reported metric"], lr,
                                  align=">{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{4.2cm}"
                                        ">{\\raggedright\\arraybackslash}p{6.2cm}l", wide=True,
                                  note="Different datasets and tasks (GIS susceptibility maps vs.\\ tabular factor scores) "
                                       "--- indicative only, not a like-for-like comparison."))

    # --- ablation -----------------------------------------------------------
    ar = [[tex(r["variant"]), f(r.get("oof_r2")), f"{f(r['test_r2']['value'])} [{f(r['test_r2']['lo'])}, {f(r['test_r2']['hi'])}]"]
          for r in abl.get("runs", [])]
    for k, v in (abl.get("ensemble_strategies") or {}).items():
        ar.append([tex("ensemble: " + k), f(v["oof_r2"]), f(v["test_r2"])])
    write("ablation", table(f"Ablation study ({tex(abl.get('model', ''))} unless stated).", "tab:ablation",
                            ["Variant", "OOF $R^2$", "Test $R^2$ [95\\% CI]"], ar))

    # --- statistical tests ----------------------------------------------------
    pb, wx, fr = tests.get("paired_bootstrap_r2", {}), tests.get("wilcoxon_folds", {}), tests.get("friedman_nemenyi", {})
    sr = [["Paired bootstrap $\\Delta R^2$ (ensemble $-$ best single)", f"{f(pb.get('diff'), 5)} [{f(pb.get('lo'), 5)}, {f(pb.get('hi'), 5)}]",
           f(pb.get("p_value"), 3)],
          ["Wilcoxon signed-rank over CV folds", f(wx.get("statistic"), 1), f(wx.get("p_value"), 3)],
          [f"Friedman ({fr.get('k', '--')} models, {fr.get('n_blocks', '--')} blocks)", f(fr.get("statistic"), 2), f(fr.get("p_value"), 4)],
          ["Nemenyi critical difference", f(fr.get("critical_difference"), 3), "--"]]
    write("stat_tests", table(f"Statistical tests: {tex(tests.get('ensemble', ''))} vs {tex(tests.get('best_single', ''))}.",
                              "tab:stats", ["Test", "Statistic / effect", "$p$"], sr,
                              align=">{\\raggedright\\arraybackslash}p{3.3cm}>{\\raggedright\\arraybackslash}p{3.2cm}c"))

    # --- data audit -------------------------------------------------------------
    clean = audit.get("cleaning", {}).get("s4e5_train", {})
    sp, imp = audit.get("splits", {}), audit.get("imputation", {})
    out_tr = audit.get("outliers", {}).get("train", {})
    dr = [["Rows (raw / after cleaning)", f"{clean.get('rows_in', '--')} / {clean.get('rows_out', '--')}"],
          ["Exact duplicates removed", str(clean.get("duplicates_removed", "--"))],
          ["Original-dataset rows merged", str(sp.get("train_original_rows", "--"))],
          ["Train / hold-out / external rows", f"{sp.get('train_rows', '--')} / {sp.get('holdout_rows', '--')} / {sp.get('external_rows', '--')}"],
          ["Selected imputer (MCAR study)", tex(imp.get("selected", "--"))],
          ["IQR / IsolationForest outlier rows (train)", f"{out_tr.get('iqr_rows', '--')} / {out_tr.get('iforest_rows', '--')}"],
          ["Engineered features / selected $k$", f"{sel.get('n_engineered', '--')} / {sel.get('selected_k', '--')}"],
          ["High-cardinality pair keys (levels vs.\\ per factor)", _cardinality(sel)],
          ["Risk-class cut-offs (tertiles)", ", ".join(f(c, 3) for c in audit.get("risk", {}).get("cutoffs", []))]]
    write("data_audit", table("Data audit and preprocessing summary.", "tab:audit", ["Item", "Value"], dr, align="lr"))

    # --- hyper-parameters ---------------------------------------------------------
    hr = []
    for name in cfg.models.tune:
        b = load_json(paths.optuna / f"{name}_best.json")
        if b:
            params = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in b["params"].items())
            hr.append([tex(b["display_name"]), str(b["n_trials"]), f(b["value"]), "\\parbox[t]{8cm}{\\raggedright " + tex(params) + "}"])
    write("hyperparams", table("Optuna (TPE) results: best cross-validated $R^2$ and parameters.", "tab:hpo",
                               ["Model", "Trials", "CV $R^2$", "Best parameters"], hr, align="lccl", wide=True))

    sc = load_json(paths.optuna / "search_comparison.json", [])
    if sc:
        write("search_comparison", table("Search strategies on identical rows and folds.", "tab:search",
                                         ["Model", "Grid $R^2$ (evals)", "Optuna $R^2$ (trials)"],
                                         [[tex(r["model"]), f"{f(r['grid_best_cv_r2'])} ({r['grid_evaluations']})",
                                           f"{f(r.get('optuna_best_cv_r2'))} ({r.get('optuna_trials', '--')})"] for r in sc]))

    if ext:
        write("external", table("External validation: trained on S4E5, tested on the original dataset.", "tab:external",
                                ["Metric", "Value [95\\% CI]"],
                                [["$R^2$", f"{f(ext['r2']['value'])} [{f(ext['r2']['lo'])}, {f(ext['r2']['hi'])}]"],
                                 ["RMSE", f(ext["rmse"]["value"], 5)], ["MAE", f(ext["mae"]["value"], 5)],
                                 ["Macro-F1 (true cut-offs)", f(ext["macro_f1_at_true_cutoffs"], 3)]], align="lr"))

    # --- environment / computational setup ---------------------------------------
    run_info = load_json(paths.root / "pipeline_run.json", {})
    env = run_info.get("environment") or environment_info()
    write("environment", table("Computational environment of the reported run.", "tab:env", ["Item", "Value"],
                               environment_table(env, run_info.get("seconds", {})),
                               align="l>{\\raggedright\\arraybackslash}p{6.0cm}"))

    # --- macros ------------------------------------------------------------------------
    final = next(r for r in ev["models"] if r["selected"]) if any(r["selected"] for r in ev["models"]) else rows[0]
    singles = [r for r in ev["models"] if r["kind"] == "single"]
    best_single = max(singles, key=lambda r: r["oof_r2"]) if singles else final
    sum_base = next((r for r in ev["models"] if r["key"] == "base_linear_sum"), None)
    macros = {
        "FGDataSource": "synthetic smoke-test data" if synthetic else "Kaggle Playground S4E5",
        "FGFinalModel": tex(final["name"]), "FGFinalRtwo": f(final["test_r2"]["value"]),
        "FGFinalRtwoLo": f(final["test_r2"]["lo"]), "FGFinalRtwoHi": f(final["test_r2"]["hi"]),
        "FGFinalRMSE": f(final["test_rmse"]["value"], 5), "FGFinalMAE": f(final["test_mae"]["value"], 5),
        "FGFinalOOF": f(final["oof_r2"]), "FGMacroFone": f(final.get("test_macro_f1"), 3), "FGRocAuc": f(final.get("test_roc_auc"), 3),
        "FGPrAuc": f(final.get("test_pr_auc"), 3), "FGECE": f(final.get("test_ece"), 3), "FGBrier": f(final.get("test_brier"), 4),
        "FGBestSingle": tex(best_single["name"]), "FGBestSingleRtwo": f(best_single["test_r2"]["value"]),
        "FGDeltaRtwo": f(pb.get("diff"), 5), "FGPairedP": f(pb.get("p_value"), 3), "FGFriedmanP": f(fr.get("p_value"), 4),
        "FGSumBaselineRtwo": f(sum_base["test_r2"]["value"]) if sum_base else "--",
        "FGTrainRows": str(sp.get("train_rows", "--")), "FGHoldoutRows": str(sp.get("holdout_rows", "--")),
        "FGExternalRows": str(sp.get("external_rows", "--")), "FGNumEngineered": str(sel.get("n_engineered", "--")),
        "FGSelectedK": str(sel.get("selected_k", "--")), "FGImputer": tex(imp.get("selected", "--")),
        "FGRhoSum": f(corr.get("target_vs_row_sum", {}).get("spearman"), 3),
        "FGExternalRtwo": f(ext.get("r2", {}).get("value")) if ext else "--",
        "FGKaggleTop": f(kag.get("private_top_r2")), "FGEnsembleStrategy": tex(ens.get("selected", "--")),
        "FGBootstrap": str(cfg.evaluation.bootstrap), "FGFolds": str(cfg.cv.n_folds),
    }
    (out / "macros.tex").write_text("% Generated by floodguard.evaluation.report_assets — do not edit.\n" +
                                    "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n", encoding="utf-8")
    written.append("macros")

    for png in paths.figures.glob("*.png"):
        shutil.copy(png, paths.report_figures / png.name)
    log.info("Report assets: %d tables + macros in %s", len(written) - 1, out)
    return written


if __name__ == "__main__":
    from floodguard.utils.config import cli

    config, _ = cli(__doc__)
    run(config)
