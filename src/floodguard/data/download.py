"""Download Kaggle S4E5 and the original Flood Prediction Factors dataset.

    python -m floodguard.data.download --config configs/default.yaml
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

from floodguard.utils.common import get_logger
from floodguard.utils.config import Config, cli, resolve_path

log = get_logger("floodguard.download")

INSTRUCTIONS = """
Kaggle credentials were not found.

1. Sign in at https://www.kaggle.com -> Settings -> API -> "Create New Token".
   This downloads kaggle.json.
2. Put it in place:
     Linux / macOS / Colab : ~/.kaggle/kaggle.json   (chmod 600 ~/.kaggle/kaggle.json)
     Windows               : C:\\Users\\<you>\\.kaggle\\kaggle.json
   or set the environment variables KAGGLE_USERNAME and KAGGLE_KEY.
3. Join the competition once in the browser (accept the rules):
     https://www.kaggle.com/competitions/playground-series-s4e5
4. Re-run:  python scripts/run_pipeline.py --stages download

For a quick offline smoke test use synthetic data instead:
     python scripts/make_synthetic_flood.py
     python scripts/run_pipeline.py --override configs/smoke.yaml --stages all --skip download
"""


def credentials_available() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (config_dir / "kaggle.json").exists()


def _unzip_all(folder: Path) -> None:
    for archive in folder.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(folder)
        archive.unlink()


def run(cfg: Config) -> dict[str, str]:
    if cfg.data.source != "kaggle":
        log.info("data.source=%s — nothing to download.", cfg.data.source)
        return {}
    if not credentials_available():
        log.error(INSTRUCTIONS)
        raise SystemExit(2)

    from kaggle.api.kaggle_api_extended import KaggleApi  # imported late: it authenticates on import

    api = KaggleApi()
    api.authenticate()

    train_path = resolve_path(cfg.data.train_path)
    comp_dir = train_path.parent
    comp_dir.mkdir(parents=True, exist_ok=True)
    if not train_path.exists():
        log.info("Downloading competition %s -> %s", cfg.data.competition, comp_dir)
        api.competition_download_files(cfg.data.competition, path=str(comp_dir), quiet=False)
        _unzip_all(comp_dir)

    original_path = resolve_path(cfg.data.original_path)
    if not original_path.exists():
        original_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading dataset %s -> %s", cfg.data.original_dataset, original_path.parent)
        api.dataset_download_files(cfg.data.original_dataset, path=str(original_path.parent), unzip=True, quiet=False)
        csvs = sorted(original_path.parent.glob("*.csv"))
        if csvs and not original_path.exists():
            csvs[0].rename(original_path)

    for p in (train_path, resolve_path(cfg.data.test_path), original_path):
        log.info("%-60s %s", p, "OK" if p.exists() else "MISSING")
    return {"train": str(train_path), "original": str(original_path)}


if __name__ == "__main__":
    config, _ = cli(__doc__)
    run(config)
