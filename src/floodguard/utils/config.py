"""YAML configuration: loading, deep-merged overrides and attribute access.

    cfg = load_config("configs/default.yaml", ["configs/smoke.yaml", "optuna.n_trials=5"])
    cfg.optuna.n_trials  # -> 5
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class Config(dict):
    """A dict with recursive attribute access (``cfg.data.target``)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(f"Config has no key '{key}'") from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            super().__setitem__(key, value)
        return value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return self[key] if key in self else default

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(_plain(self))


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(_plain(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _dotted(key: str, value: Any) -> dict:
    node: dict = {}
    cursor = node
    parts = key.split(".")
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return node


def parse_override(item: str) -> dict:
    """A YAML file path, or ``dotted.key=value`` (value parsed as YAML)."""
    path = resolve_path(item)
    if item.endswith((".yaml", ".yml")):
        if not path.exists():
            raise FileNotFoundError(f"Override file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    if "=" in item:
        key, raw = item.split("=", 1)
        return _dotted(key.strip(), yaml.safe_load(raw))
    raise ValueError(f"Cannot parse override '{item}' (expected a .yaml file or key=value)")


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> Config:
    with open(resolve_path(path) if path else DEFAULT_CONFIG, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    for item in overrides:
        data = deep_merge(data, parse_override(item))
    return Config(data)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_search_space(model: str) -> dict[str, Any]:
    path = ROOT / "configs" / "models" / f"{model}.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="base YAML config")
    parser.add_argument(
        "--override", action="append", default=[],
        help="YAML file or dotted key=value; may be given several times",
    )
    return parser


def cli(description: str, extra: callable | None = None) -> tuple[Config, argparse.Namespace]:
    """Standard CLI used by every ``python -m floodguard.<module>`` entry point."""
    parser = add_config_args(argparse.ArgumentParser(description=description))
    if extra:
        extra(parser)
    args = parser.parse_args()
    return load_config(args.config, args.override), args
