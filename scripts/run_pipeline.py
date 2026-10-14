"""FloodGuard AI end-to-end pipeline.

    python scripts/run_pipeline.py --stages all                                   # full Kaggle run
    python scripts/run_pipeline.py --override configs/smoke.yaml --stages all --skip download
    python scripts/run_pipeline.py --stages tune,train --override optuna.n_trials=20

Stages (in order):
    download  prepare  features  reduce  tune  train  ensemble
    evaluate  benchmark  explain  plots  report
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floodguard.utils.common import environment_info, get_logger, load_json, run_paths, save_json, seed_everything  # noqa: E402
from floodguard.utils.config import add_config_args, load_config  # noqa: E402

log = get_logger("floodguard.pipeline")


def _stage(module: str) -> Callable:
    def call(cfg):
        mod = __import__(module, fromlist=["run"])
        return mod.run(cfg)
    return call


STAGES: dict[str, Callable] = {
    "download": _stage("floodguard.data.download"),
    "prepare": _stage("floodguard.data.prepare"),
    "features": _stage("floodguard.features.engineering"),
    "reduce": _stage("floodguard.features.reduction"),
    "tune": _stage("floodguard.tuning.optuna_search"),
    "train": _stage("floodguard.models.cv"),
    "ensemble": _stage("floodguard.models.ensemble"),
    "evaluate": _stage("floodguard.evaluation.evaluate"),
    "benchmark": _stage("floodguard.evaluation.benchmark"),
    "explain": _stage("floodguard.explain.explainability"),
    "plots": _stage("floodguard.visualization.plots"),
    "report": _stage("floodguard.evaluation.report_assets"),
}


def parse_list(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--stages", default="all", help="comma-separated stages or 'all'")
    parser.add_argument("--skip", default="", help="comma-separated stages to skip")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    seed_everything(cfg.project.seed)
    stages = list(STAGES) if args.stages == "all" else parse_list(args.stages)
    unknown = [s for s in stages + parse_list(args.skip) if s not in STAGES]
    if unknown:
        parser.error(f"Unknown stage(s): {unknown}. Choose from {list(STAGES)}")
    stages = [s for s in stages if s not in parse_list(args.skip)]

    paths = run_paths(cfg)
    log.info("FloodGuard AI pipeline | data=%s | run_dir=%s | stages=%s", cfg.data.source, paths.root, ",".join(stages))
    timings: dict[str, float] = {}
    start = time.perf_counter()
    for stage in stages:
        log.info("━━━ stage: %s ━━━", stage)
        t = time.perf_counter()
        STAGES[stage](cfg)
        timings[stage] = round(time.perf_counter() - t, 1)
        log.info("stage %s done in %.1fs", stage, timings[stage])
    timings["total"] = round(time.perf_counter() - start, 1)
    # Keep timings of stages run in earlier invocations (e.g. `--stages report` after a full run).
    previous = load_json(paths.root / "pipeline_run.json", {}).get("seconds", {})
    all_timings = {**{k: v for k, v in previous.items() if k != "total"}, **{k: v for k, v in timings.items() if k != "total"}}
    all_timings["total"] = round(sum(all_timings.values()), 1)
    save_json({"stages": stages, "seconds": all_timings, "environment": environment_info(), "config": cfg.to_dict()},
              paths.root / "pipeline_run.json")
    log.info("Pipeline finished in %.1fs. Artifacts: %s", timings["total"], paths.root)
    if cfg.data.source == "synthetic":
        log.warning("This run used SYNTHETIC data — use it to check the code, never as reported results.")


if __name__ == "__main__":
    main()
