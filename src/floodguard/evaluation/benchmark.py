"""Benchmarking: baselines vs tuned models vs the FloodGuard ensemble, plus
external references (Kaggle S4E5 leaderboard, published studies).

External numbers are read from configs/benchmarks.yaml and are never guessed.

    python -m floodguard.evaluation.benchmark --override configs/smoke.yaml
    python -m floodguard.evaluation.benchmark --fetch-leaderboard     # needs Kaggle credentials
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import yaml

from floodguard.data.download import credentials_available
from floodguard.utils.common import get_logger, load_json, run_paths, save_json
from floodguard.utils.config import Config, resolve_path

log = get_logger("floodguard.benchmark")
BENCHMARKS = "configs/benchmarks.yaml"


def load_benchmarks() -> dict[str, Any]:
    with open(resolve_path(BENCHMARKS), encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def fetch_leaderboard(cfg: Config) -> dict[str, Any]:
    """Download the leaderboard through the Kaggle API and summarise it.

    The API may return the public leaderboard; verify on the website and copy
    the private top / median R² into configs/benchmarks.yaml by hand.
    """
    if not credentials_available():
        raise SystemExit("Kaggle credentials not found — see `python -m floodguard.data.download` for instructions.")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    out_dir = run_paths(cfg).metrics / "leaderboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    api.competition_leaderboard_download(cfg.data.competition, path=str(out_dir))
    import zipfile

    for z in out_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out_dir)
    csv = next(out_dir.glob("*.csv"))
    board = pd.read_csv(csv)
    scores = pd.to_numeric(board["Score"], errors="coerce").dropna()
    snapshot = {"top_r2": float(scores.max()), "median_r2": float(scores.median()), "teams": int(len(scores)),
                "fetched_on": dt.date.today().isoformat(), "file": str(csv),
                "note": "From the Kaggle API — check on the website whether this is the public or private leaderboard."}
    save_json(snapshot, run_paths(cfg).metrics / "leaderboard_snapshot.json")
    log.info("Leaderboard snapshot: top %.5f, median %.5f over %d teams (verify public vs private).",
             snapshot["top_r2"], snapshot["median_r2"], snapshot["teams"])
    return snapshot


def run(cfg: Config) -> dict[str, Any]:
    paths = run_paths(cfg)
    evaluation = load_json(paths.metrics / "evaluation.json")
    if not evaluation:
        raise RuntimeError("Run the 'evaluate' stage first.")
    rows = evaluation["models"]
    table = [{
        "name": r["name"], "kind": r["kind"], "selected": r["selected"], "oof_r2": r["oof_r2"],
        "test_r2": r["test_r2"]["value"], "test_r2_lo": r["test_r2"]["lo"], "test_r2_hi": r["test_r2"]["hi"],
        "test_rmse": r["test_rmse"]["value"], "test_mae": r["test_mae"]["value"],
        "macro_f1": r.get("test_macro_f1"), "roc_auc": r.get("test_roc_auc"), "pr_auc": r.get("test_pr_auc"),
        "log_loss": r.get("test_log_loss"), "brier": r.get("test_brier"), "ece": r.get("test_ece"),
    } for r in rows]
    table.sort(key=lambda r: r["test_r2"], reverse=True)
    final = next((r for r in table if r["selected"]), table[0])
    baselines = {r["name"]: r["test_r2"] for r in table if r["kind"] == "baseline"}

    refs = load_benchmarks()
    lb = refs.get("kaggle_leaderboard", {}) or {}
    snapshot = load_json(paths.metrics / "leaderboard_snapshot.json")
    kaggle = {"source_url": lb.get("source_url"), "private_top_r2": lb.get("private_top_r2"),
              "private_median_r2": lb.get("private_median_r2"), "api_snapshot": snapshot}
    if lb.get("private_top_r2") is not None:
        kaggle["gap_to_top"] = final["test_r2"] - float(lb["private_top_r2"])
    if lb.get("private_median_r2") is not None:
        kaggle["gap_to_median"] = final["test_r2"] - float(lb["private_median_r2"])
    if lb.get("private_top_r2") is None:
        kaggle["note"] = "Leaderboard values not filled in configs/benchmarks.yaml — comparison omitted (never guessed)."
    if evaluation.get("data_source") == "synthetic":
        kaggle["note"] = "Synthetic smoke-test data — leaderboard comparison is not meaningful."

    literature = [{**entry, "note": "different dataset — indicative only"} for entry in refs.get("literature", []) or []]
    result = {
        "data_source": evaluation.get("data_source"), "final": final, "table": table,
        "improvement_over_baselines": {k: final["test_r2"] - v for k, v in baselines.items()},
        "kaggle": kaggle, "literature": literature,
        "external_validation": load_json(paths.metrics / "external_validation.json", {}),
        "search_comparison": load_json(paths.optuna / "search_comparison.json", []),
    }
    save_json(result, paths.metrics / "benchmark.json")
    log.info("Benchmark: %s test R² %.5f; vs baselines %s", final["name"], final["test_r2"],
             {k: round(v, 4) for k, v in result["improvement_over_baselines"].items()})
    return result


if __name__ == "__main__":
    from floodguard.utils.config import cli

    config, args = cli(__doc__, lambda p: p.add_argument("--fetch-leaderboard", action="store_true"))
    if args.fetch_leaderboard:
        fetch_leaderboard(config)
    else:
        run(config)
