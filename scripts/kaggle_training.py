"""Run the full FloodGuard AI training on Kaggle's free cloud GPU — nothing heavy runs locally.

    python scripts/kaggle_training.py setup      # install kaggle.json from ~/Downloads (or KAGGLE_* env)
    python scripts/kaggle_training.py check      # credentials OK + competition rules accepted?
    python scripts/kaggle_training.py launch     # push deploy/kaggle_kernel as a private GPU notebook
    python scripts/kaggle_training.py status     # queued / running / complete / error
    python scripts/kaggle_training.py fetch      # download outputs -> runs/ + report/
    python scripts/kaggle_training.py submit     # late submission of runs/submission.csv (private LB score)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "playground-series-s4e5"
ORIGINAL_DATASET = "naiyakhalid/flood-prediction-dataset"
SLUG = "floodguard-ai-training"
KAGGLE_DIR = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))


def kaggle_cli() -> str:
    exe = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    return str(exe) if exe.exists() else "kaggle"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([kaggle_cli(), *args], capture_output=True, text=True, check=check)


def username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg = KAGGLE_DIR / "kaggle.json"
    if not cfg.exists():
        sys.exit("No Kaggle credentials. Download kaggle.json (Kaggle -> Settings -> API) and run `setup`.")
    return json.loads(cfg.read_text())["username"]


def kernel_id() -> str:
    return f"{username()}/{SLUG}"


def setup() -> None:
    target = KAGGLE_DIR / "kaggle.json"
    if target.exists():
        print(f"Credentials already installed for user '{username()}'.")
        return
    candidates = sorted((Path.home() / "Downloads").glob("kaggle*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        sys.exit("kaggle.json not found in Downloads. Kaggle -> Settings -> API -> 'Create Legacy API Key'.")
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(candidates[0], target)
    if os.name != "nt":
        target.chmod(0o600)
    print(f"Installed {candidates[0].name} -> {target} (user '{username()}').")


def check() -> None:
    print(f"User: {username()}")
    files = run("competitions", "files", COMPETITION, check=False)
    if files.returncode != 0 or "403" in files.stdout + files.stderr:
        sys.exit("Competition rules NOT accepted yet: open https://www.kaggle.com/competitions/playground-series-s4e5/rules "
                 "and click 'I Understand and Accept'.")
    print("Competition data accessible (rules accepted).")


def launch() -> None:
    check()
    build = Path(tempfile.mkdtemp(prefix="floodguard_kernel_"))
    shutil.copy(ROOT / "deploy" / "kaggle_kernel" / "run_floodguard.py", build / "run_floodguard.py")
    meta = {
        "id": kernel_id(), "title": "FloodGuard AI training", "code_file": "run_floodguard.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [ORIGINAL_DATASET], "competition_sources": [COMPETITION], "kernel_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    out = run("kernels", "push", "-p", str(build), check=False)
    print(out.stdout.strip() or out.stderr.strip())
    if out.returncode != 0:
        sys.exit("Push failed (internet access in Kaggle notebooks requires a phone-verified account).")
    print(f"Running on Kaggle: https://www.kaggle.com/code/{kernel_id()}")


def status() -> None:
    out = run("kernels", "status", kernel_id(), check=False)
    print((out.stdout or out.stderr).strip())


def fetch() -> None:
    dest = Path(tempfile.mkdtemp(prefix="floodguard_out_"))
    out = run("kernels", "output", kernel_id(), "-p", str(dest), check=False)
    print((out.stdout or out.stderr).strip()[-500:])
    archive = dest / "floodguard_outputs.zip"
    if not archive.exists():
        sys.exit("floodguard_outputs.zip not found — is the run complete? (`status`)")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(ROOT)  # -> runs/ and report/
    print(f"Extracted outputs into {ROOT / 'runs'} and {ROOT / 'report'}")


def submit() -> None:
    sub = ROOT / "runs" / "submission.csv"
    if not sub.exists():
        sys.exit("runs/submission.csv not found — run `fetch` first.")
    out = run("competitions", "submit", COMPETITION, "-f", str(sub), "-m", "FloodGuard AI ensemble", check=False)
    print((out.stdout or out.stderr).strip())
    print(run("competitions", "submissions", COMPETITION, check=False).stdout[:1500])


if __name__ == "__main__":
    actions = {"setup": setup, "check": check, "launch": launch, "status": status, "fetch": fetch, "submit": submit}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=list(actions))
    actions[parser.parse_args().action]()
