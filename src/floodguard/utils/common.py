"""Seeding, logging, JSON helpers, timers and run-directory layout."""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from floodguard.utils.config import Config, resolve_path

_LOGGING_READY = False


def get_logger(name: str = "floodguard", level: str = "INFO") -> logging.Logger:
    global _LOGGING_READY
    if not _LOGGING_READY:
        for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        for noisy in ("PIL", "numba", "alembic", "urllib3", "httpx", "mlflow"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)  # font-weight fallbacks inside shap plots
        _LOGGING_READY = True
    return logging.getLogger(name)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _clean_nan(obj: Any) -> Any:
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    return obj


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean_nan(obj), fh, cls=_Encoder, indent=2, ensure_ascii=False)
    return path


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[dict[str, float]]:
    record: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["seconds"] = round(time.perf_counter() - start, 2)
        (logger or get_logger()).info("%s finished in %.1fs", label, record["seconds"])


@dataclass(frozen=True)
class RunPaths:
    """Directory layout of one pipeline run (``project.run_dir``)."""

    root: Path
    report: Path

    def _d(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data(self) -> Path: return self._d("data")
    @property
    def audit(self) -> Path: return self._d("audit")
    @property
    def features(self) -> Path: return self._d("features")
    @property
    def reduction(self) -> Path: return self._d("reduction")
    @property
    def optuna(self) -> Path: return self._d("optuna")
    @property
    def oof(self) -> Path: return self._d("oof")
    @property
    def metrics(self) -> Path: return self._d("metrics")
    @property
    def models(self) -> Path: return self._d("models")
    @property
    def explain(self) -> Path: return self._d("explain")
    @property
    def figures(self) -> Path: return self._d("figures")

    @property
    def report_figures(self) -> Path:
        p = self.report / "figures"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def report_tables(self) -> Path:
        p = self.report / "tables"
        p.mkdir(parents=True, exist_ok=True)
        return p


def run_paths(cfg: Config) -> RunPaths:
    return RunPaths(root=resolve_path(cfg.project.run_dir), report=resolve_path(cfg.project.report_dir))


def gpu_available() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    if shutil.which("nvidia-smi"):
        try:
            return subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
    return False


LIBRARIES = ("scikit-learn", "xgboost", "lightgbm", "catboost", "torch", "optuna", "shap", "mlflow", "lime",
             "umap-learn", "pandas", "numpy", "fastapi", "gradio")


def environment_info() -> dict[str, Any]:
    """Hardware + software description for the report (cheap: no heavy imports)."""
    import platform
    from importlib import metadata

    info: dict[str, Any] = {
        "python": platform.python_version(), "os": platform.platform(terse=True),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpus": os.cpu_count(), "ram_gb": None, "gpu": "none",
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
        elif sys.platform == "win32":
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = _MemoryStatus(dwLength=ctypes.sizeof(_MemoryStatus))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                info["ram_gb"] = round(status.ullTotalPhys / 1e9, 1)
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                info["gpu"] = "; ".join(line.strip() for line in out.stdout.strip().splitlines())
        except (OSError, subprocess.TimeoutExpired):
            pass
    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = None
    info["libraries"] = versions
    return info


def use_gpu(cfg: Config) -> bool:
    setting = cfg.models.get("use_gpu", "auto")
    if isinstance(setting, bool):
        return setting and gpu_available()
    return str(setting).lower() == "auto" and gpu_available()
