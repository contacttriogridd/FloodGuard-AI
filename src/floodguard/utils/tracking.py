"""Experiment tracking: MLflow when available, JSON-lines fallback otherwise.

Every Optuna trial and every trained model is recorded as a run with its
parameters, CV fold scores, metrics and artifacts.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from floodguard.utils.common import get_logger
from floodguard.utils.config import Config, resolve_path

log = get_logger("floodguard.tracking")


def _sqlite_uri(uri: str) -> str:
    """Make a relative ``sqlite:///runs/x.db`` URI absolute (portable across CWDs)."""
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        path = uri[len(prefix):]
        if not Path(path).is_absolute():
            target = resolve_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            return prefix + target.as_posix()
    return uri


class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backend = cfg.tracking.get("backend", "jsonl")
        self._mlflow = None
        self._stack: list[dict[str, Any]] = []
        self.jsonl_path = resolve_path(cfg.tracking.jsonl_path)
        if self.backend == "mlflow":
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            try:
                import mlflow

                mlflow.set_tracking_uri(_sqlite_uri(cfg.tracking.mlflow_uri))
                if mlflow.get_experiment_by_name(cfg.tracking.experiment) is None:
                    # Keep artefacts next to the run (default would be ./mlruns in the CWD).
                    store = resolve_path(cfg.project.run_dir) / "mlartifacts"
                    store.mkdir(parents=True, exist_ok=True)
                    mlflow.create_experiment(cfg.tracking.experiment, artifact_location=store.as_uri())
                mlflow.set_experiment(cfg.tracking.experiment)
                self._mlflow = mlflow
            except Exception as exc:  # pragma: no cover - environment dependent
                log.warning("MLflow unavailable (%s); falling back to JSON-lines tracking.", exc)
                self.backend = "jsonl"

    # ------------------------------------------------------------------ runs
    @contextmanager
    def run(self, name: str, nested: bool = False, tags: dict[str, str] | None = None) -> Iterator["Tracker"]:
        if self._mlflow is not None:
            with self._mlflow.start_run(run_name=name, nested=nested or bool(self._stack), tags=tags or {}):
                self._stack.append({"name": name})
                try:
                    yield self
                finally:
                    self._stack.pop()
            return
        record = {
            "run_id": uuid.uuid4().hex[:12], "name": name, "parent": self._stack[-1]["run_id"] if self._stack else None,
            "started": time.time(), "tags": tags or {}, "params": {}, "metrics": [], "artifacts": [],
        }
        self._stack.append(record)
        try:
            yield self
        finally:
            record = self._stack.pop()
            record["ended"] = time.time()
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")

    # --------------------------------------------------------------- logging
    def log_params(self, params: dict[str, Any]) -> None:
        clean = {str(k): (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in params.items()}
        if self._mlflow is not None:
            self._mlflow.log_params({k: str(v)[:500] for k, v in clean.items()})
        elif self._stack:
            self._stack[-1]["params"].update(clean)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if self._mlflow is not None:
            self._mlflow.log_metrics(clean, step=step)
        elif self._stack:
            self._stack[-1]["metrics"].append({"step": step, **clean})

    def set_tags(self, tags: dict[str, str]) -> None:
        if self._mlflow is not None:
            self._mlflow.set_tags(tags)
        elif self._stack:
            self._stack[-1]["tags"].update(tags)

    def log_artifact(self, path: str | Path) -> None:
        if not Path(path).exists():
            return
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(path))
        elif self._stack:
            self._stack[-1]["artifacts"].append(str(path))

    def log_dict(self, obj: dict[str, Any], name: str) -> None:
        if self._mlflow is not None:
            self._mlflow.log_dict(obj, name)
        elif self._stack:
            self._stack[-1]["artifacts"].append(name)

    def log_model(self, model: Any, X_example: Any) -> None:
        """Log a fitted sklearn pipeline with an inferred input/output signature."""
        if self._mlflow is None:
            return
        try:
            import mlflow.sklearn
            from mlflow.models import infer_signature

            X_example = X_example.astype(float)
            signature = infer_signature(X_example, model.predict(X_example))
            # cloudpickle (not skops) because the pipeline contains custom FloodGuard classes;
            # code_paths ships the package so the logged model loads anywhere.
            mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=X_example.head(3),
                                     serialization_format="cloudpickle",
                                     code_paths=[str(Path(__file__).resolve().parents[1])])
        except Exception as exc:  # pragma: no cover - optional
            log.warning("Could not log model to MLflow: %s", exc)
