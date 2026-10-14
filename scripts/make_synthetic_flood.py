"""Write a small SYNTHETIC dataset with the Kaggle S4E5 schema (smoke tests / CI only).

    python scripts/make_synthetic_flood.py                # data/synthetic/{s4e5,original}
    python scripts/make_synthetic_flood.py --rows 6000

The target is a noisy, clipped linear function of the factor sum plus one
interaction, mimicking the structure reported for S4E5. These rows are NOT
real observations; results obtained on them must never be reported as findings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floodguard.utils.config import load_config  # noqa: E402


def _factors(rng: np.random.Generator, n: int, k: int, lo: float, hi: float) -> np.ndarray:
    lam = rng.uniform(lo, hi, size=k)
    return np.clip(rng.poisson(lam, size=(n, k)), 0, 18)


def _target(rng: np.random.Generator, X: np.ndarray, names: list[str], noise: float) -> np.ndarray:
    s = X.sum(axis=1)
    urb, drain = X[:, names.index("Urbanization")], X[:, names.index("DrainageSystems")]
    y = 0.5 + 0.0056 * (s - 100) + 0.00025 * (urb * drain - 25) + rng.normal(0, noise, len(s))
    return np.clip(np.round(y, 3), 0.285, 0.725)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--test-rows", type=int, default=2000)
    parser.add_argument("--original-rows", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/synthetic")
    args = parser.parse_args()

    cfg = load_config()
    names: list[str] = list(cfg.features.factors)
    target = cfg.data.target
    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out

    X = _factors(rng, args.rows, len(names), 4.0, 6.0)
    train = pd.DataFrame(X, columns=names)
    train.insert(0, "id", np.arange(len(train)))
    train[target] = _target(rng, X, names, noise=0.022)
    # A few duplicates and impossible values so the cleaning code paths are exercised.
    dupes = train.sample(n=20, random_state=1).copy()
    dupes["id"] = np.arange(len(train), len(train) + len(dupes))
    train = pd.concat([train, dupes], ignore_index=True)
    train.loc[train.sample(n=5, random_state=2).index, "Urbanization"] = -1

    Xt = _factors(rng, args.test_rows, len(names), 4.0, 6.0)
    test = pd.DataFrame(Xt, columns=names)
    test.insert(0, "id", np.arange(len(train), len(train) + len(test)))

    Xo = _factors(rng, args.original_rows, len(names), 3.5, 6.5)  # slightly shifted distribution
    original = pd.DataFrame(Xo, columns=names)
    original[target] = _target(rng, Xo, names, noise=0.02)

    (out / "s4e5").mkdir(parents=True, exist_ok=True)
    (out / "original").mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "s4e5" / "train.csv", index=False)
    test.to_csv(out / "s4e5" / "test.csv", index=False)
    original.to_csv(out / "original" / "flood.csv", index=False)
    (out / "README.txt").write_text("SYNTHETIC data written by scripts/make_synthetic_flood.py — not real observations.\n")
    print(f"Wrote synthetic data to {out}: train={len(train)}, test={len(test)}, original={len(original)}")


if __name__ == "__main__":
    main()
