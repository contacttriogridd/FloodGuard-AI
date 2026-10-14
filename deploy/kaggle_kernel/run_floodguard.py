"""Runs INSIDE a Kaggle Notebook (pushed by scripts/kaggle_training.py).

Clones the GitHub repo, installs the pinned requirements, runs the full pipeline on the
attached competition + original datasets and leaves everything in /kaggle/working:
    runs/            models, metrics, figures, Optuna + MLflow databases, submission.csv
    report/          LaTeX tables + figures for report/FloodGuard_IEEE.tex
    floodguard_outputs.zip
"""
import glob
import os
import shutil
import subprocess
import sys
import time

REPO = "https://github.com/contacttriogridd/FloodGuard-AI.git"
WORK = "/kaggle/working"
CODE = f"{WORK}/FloodGuard-AI"


def sh(cmd, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


start = time.time()
sh(["nvidia-smi"]) if shutil.which("nvidia-smi") else print("No GPU visible")
if not os.path.isdir(CODE):
    sh(["git", "clone", "--depth", "1", REPO, CODE])
os.chdir(CODE)
# torch is preinstalled with CUDA on Kaggle; keep it, install everything else pinned.
reqs = [line for line in open("requirements.txt").read().splitlines() if line and not line.startswith(("#", "torch"))]
sh([sys.executable, "-m", "pip", "install", "-q", *reqs])
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."])

original = sorted(glob.glob("/kaggle/input/**/flood*.csv", recursive=True))
original = [p for p in original if "playground-series" not in p]
overrides = ["--override", "configs/kaggle.yaml"]
if original:
    overrides += ["--override", f"data.original_path={original[0]}"]
else:
    overrides += ["--override", "data.merge_original=false", "--override", "data.external_validation=false"]
print("Original dataset:", original[0] if original else "not attached")

try:
    sh([sys.executable, "scripts/run_pipeline.py", *overrides, "--stages", "all", "--skip", "download"])
finally:
    # Always package whatever was produced, so partial results survive a failure.
    for d in ("runs", "report"):
        if os.path.isdir(f"{WORK}/{d}"):
            print(d, "->", sum(len(f) for _, _, f in os.walk(f"{WORK}/{d}")), "files")
    keep = ["models", "metrics", "figures", "optuna", "audit", "features", "explain", "reduction"]
    staging = f"{WORK}/_outputs"
    os.makedirs(staging, exist_ok=True)
    for k in keep:
        src = f"{WORK}/runs/{k}"
        if os.path.isdir(src):
            shutil.copytree(src, f"{staging}/runs/{k}", dirs_exist_ok=True)
    for f in ("pipeline_run.json", "submission.csv", "mlflow.db"):
        if os.path.exists(f"{WORK}/runs/{f}"):
            shutil.copy(f"{WORK}/runs/{f}", f"{staging}/runs/{f}")
    if os.path.isdir(f"{WORK}/report"):
        shutil.copytree(f"{WORK}/report", f"{staging}/report", dirs_exist_ok=True)
    shutil.make_archive(f"{WORK}/floodguard_outputs", "zip", staging)
    # Keep only the zip as kernel output: small download, and the competition data
    # (runs/data, OOF arrays) is never re-shared.
    for path in (staging, CODE, f"{WORK}/runs", f"{WORK}/report"):
        shutil.rmtree(path, ignore_errors=True)
    print(f"Finished in {(time.time() - start) / 3600:.2f} h")
