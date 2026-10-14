"""Bayesian hyper-parameter optimisation with Optuna (TPE + MedianPruner, SQLite storage).

Each trial is cross-validated on a stratified subsample of the training split;
the per-fold R² is reported so unpromising trials are pruned. Studies are
resumable (same SQLite file + study name). Best parameters are written to
``<run_dir>/optuna/<model>_best.json`` and loaded automatically by training.
A GridSearchCV baseline on the *same* rows and folds compares search strategies.

    python -m floodguard.tuning.optuna_search --override configs/smoke.yaml --model lgbm
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline

from floodguard.data.prepare import assign_folds, load_prepared
from floodguard.models.cv import FoldCache, make_feature_pipeline
from floodguard.models.zoo import FEATURE_VARIANT, available, display_name, fit_model, make_model, predict
from floodguard.utils.common import get_logger, run_paths, save_json, seed_everything
from floodguard.utils.config import Config, load_search_space, resolve_path
from floodguard.utils.tracking import Tracker

log = get_logger("floodguard.optuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if name == "grid" or not isinstance(spec, dict) or "type" not in spec:
            continue
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search-space type '{kind}' for {name}")
    return params


def _storage(cfg: Config) -> str:
    path = resolve_path(cfg.optuna.storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def tuning_frame(cfg: Config, data):
    n = cfg.optuna.get("subsample")
    frame = data.train if not n or int(n) >= len(data.train) else data.train.sample(n=int(n), random_state=cfg.project.seed)
    return frame.reset_index(drop=True)


def tune_model(name: str, cfg: Config, cache: FoldCache, frame, tracker: Tracker) -> dict[str, Any]:
    space = load_search_space(name)
    variant = FEATURE_VARIANT.get(name, "default")
    folds = cache.folds(variant, frame=frame, n_folds=int(cfg.optuna.cv_folds), key="tuning", with_test=False)
    ocfg = cfg.optuna
    sampler = optuna.samplers.TPESampler(seed=cfg.project.seed, multivariate=bool(ocfg.multivariate),
                                         n_startup_trials=int(ocfg.n_startup_trials))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=int(ocfg.n_startup_trials)) if ocfg.pruner == "median" else optuna.pruners.NopPruner()
    study = optuna.create_study(study_name=f"floodguard-{name}", direction="maximize", sampler=sampler, pruner=pruner,
                                storage=_storage(cfg), load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, space)
        scores, iters = [], []
        with tracker.run(f"optuna/{name}/trial-{trial.number}", nested=True, tags={"stage": "tune", "model": name}):
            tracker.log_params(params)
            for k, fold in enumerate(folds):
                model, info = fit_model(name, make_model(name, params, cfg), fold.Xtr, fold.ytr, fold.Xva, fold.yva, cfg)
                score = float(r2_score(fold.yva, predict(model, fold.Xva)))
                scores.append(score)
                iters.append(info["best_iteration"])
                tracker.log_metrics({"fold_r2": score}, step=k)
                trial.report(score, k)
                if trial.should_prune():
                    tracker.set_tags({"state": "pruned"})
                    raise optuna.TrialPruned()
            tracker.log_metrics({"cv_r2": float(np.mean(scores))})
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("best_iterations", iters)
        return float(np.mean(scores))

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(int(ocfg.n_trials) - done, 0)
    start = time.perf_counter()
    with tracker.run(f"tune/{name}", tags={"stage": "tune", "model": name}):
        if remaining:
            study.optimize(objective, n_trials=remaining, timeout=int(ocfg.timeout))
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise RuntimeError(f"No completed Optuna trials for {name}")
        best = study.best_trial
        tracker.log_params({f"best_{k}": v for k, v in best.params.items()})
        tracker.log_metrics({"best_cv_r2": best.value, "n_trials": len(study.trials)})
    seconds = round(time.perf_counter() - start, 1)

    paths = run_paths(cfg)
    best_so_far, history = -np.inf, []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            best_so_far = max(best_so_far, t.value)
        history.append({"number": t.number, "state": t.state.name, "value": t.value,
                        "best": best_so_far if np.isfinite(best_so_far) else None})
    try:
        importance = optuna.importance.get_param_importances(study) if len(complete) >= 2 else {}
    except Exception:  # noqa: BLE001 - importance needs variation across trials
        importance = {}
    summary = {"model": name, "display_name": display_name(name), "params": best.params, "value": best.value,
               "fold_scores": best.user_attrs.get("fold_scores"), "n_trials": len(study.trials),
               "n_complete": len(complete), "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
               "seconds": seconds, "rows": len(frame), "cv_folds": int(ocfg.cv_folds), "sampler": "TPE (multivariate)",
               "pruner": ocfg.pruner}
    save_json(summary, paths.optuna / f"{name}_best.json")
    save_json(history, paths.optuna / f"{name}_history.json")
    save_json({k: float(v) for k, v in importance.items()}, paths.optuna / f"{name}_importance.json")
    log.info("Optuna %-12s best CV R² %.5f after %d trials (%d pruned) in %.0fs", name, best.value,
             summary["n_trials"], summary["n_pruned"], seconds)
    return summary


def grid_search_baseline(name: str, cfg: Config, data, frame, optuna_summary: dict | None) -> dict[str, Any]:
    """GridSearchCV over the same rows and the same fold assignment as Optuna."""
    space = load_search_space(name)
    grid = space.get("grid")
    if not grid:
        return {}
    sub_n = int(cfg.optuna.grid_search_baseline.subsample)
    sub = frame.iloc[: min(sub_n, len(frame))].reset_index(drop=True)
    fold_ids = assign_folds(sub, data.target, int(cfg.optuna.cv_folds), int(cfg.cv.stratify_bins), cfg.project.seed)
    prefix = "model__ridge__" if name == "ridge" else "model__"
    pipe = Pipeline([("features", make_feature_pipeline(cfg, data, FEATURE_VARIANT.get(name, "default"))),
                     ("model", make_model(name, {}, cfg))])
    search = GridSearchCV(pipe, {prefix + k: v for k, v in grid.items()}, scoring="r2", cv=PredefinedSplit(fold_ids), n_jobs=1)
    start = time.perf_counter()
    search.fit(data.X(sub), sub[data.target].to_numpy())
    result = {"model": name, "grid_best_cv_r2": float(search.best_score_),
              "grid_best_params": {k.replace(prefix, ""): v for k, v in search.best_params_.items()},
              "grid_evaluations": int(len(search.cv_results_["params"])), "grid_seconds": round(time.perf_counter() - start, 1),
              "rows": len(sub)}
    if optuna_summary:
        result.update({"optuna_best_cv_r2": optuna_summary["value"], "optuna_trials": optuna_summary["n_trials"],
                       "optuna_seconds": optuna_summary["seconds"], "optuna_rows": optuna_summary["rows"]})
    log.info("GridSearchCV %-8s best CV R² %.5f over %d combinations", name, result["grid_best_cv_r2"], result["grid_evaluations"])
    return result


def run(cfg: Config, models: list[str] | None = None) -> dict[str, Any]:
    seed_everything(cfg.project.seed)
    data = load_prepared(cfg)
    tracker = Tracker(cfg)
    cache = FoldCache(cfg, data)
    frame = tuning_frame(cfg, data)
    summaries: dict[str, Any] = {}
    for name in models or list(cfg.models.tune):
        if name not in cfg.models.enabled or not available(name):
            log.warning("Skipping tuning of %s (disabled or dependency missing).", name)
            continue
        summaries[name] = tune_model(name, cfg, cache, frame, tracker)
    gcfg = cfg.optuna.grid_search_baseline
    if gcfg.enabled:
        comparison = [grid_search_baseline(m, cfg, data, frame, summaries.get(m)) for m in gcfg.models]
        save_json([c for c in comparison if c], run_paths(cfg).optuna / "search_comparison.json")
    return summaries


if __name__ == "__main__":
    from floodguard.utils.config import cli

    config, args = cli(__doc__, lambda p: p.add_argument("--model", action="append", default=None))
    run(config, args.model)
