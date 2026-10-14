"""Fill the README results table from runs/metrics/benchmark.json (never by hand).

    python scripts/update_readme_results.py [--run-dir runs] [--kaggle-private 0.86xx]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def fmt(x, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--kaggle-private", type=float, default=None, help="private LB R² of runs/submission.csv")
    args = parser.parse_args()

    bench = json.loads((ROOT / args.run_dir / "metrics" / "benchmark.json").read_text(encoding="utf-8"))
    if bench.get("data_source") == "synthetic":
        raise SystemExit("benchmark.json comes from SYNTHETIC smoke data — refusing to publish it as results.")
    lines = ["| Model | OOF R² | Test R² [95 % CI] | RMSE | Macro-F1 | ROC-AUC (High) |", "|---|---|---|---|---|---|"]
    for r in bench["table"]:
        name = f"**{r['name']}** (deployed)" if r["selected"] else r["name"] + (" *(baseline)*" if r["kind"] == "baseline" else "")
        lines.append(f"| {name} | {fmt(r['oof_r2'])} | {fmt(r['test_r2'])} [{fmt(r['test_r2_lo'])}, {fmt(r['test_r2_hi'])}] | "
                     f"{fmt(r['test_rmse'], 5)} | {fmt(r['macro_f1'], 3)} | {fmt(r['roc_auc'], 3)} |")
    kag = bench.get("kaggle", {})
    if args.kaggle_private is not None:
        lines.append(f"| FloodGuard ensemble — Kaggle **private leaderboard** (late submission) | — | {args.kaggle_private:.5f} | — | — | — |")
    lines.append(f"| Kaggle S4E5 private LB — top / median | — | {fmt(kag.get('private_top_r2'), 5)} / "
                 f"{fmt(kag.get('private_median_r2'), 5)} | — | — | — |")
    ext = bench.get("external_validation") or {}
    note = ("\nKaggle S4E5 hold-out split (15 %); 95 % CIs from bootstrap resampling. External validation "
            f"(S4E5-trained LightGBM tested on the original dataset): R² = {fmt(ext.get('r2', {}).get('value'))}.\n")
    block = START + "\n" + "\n".join(lines) + "\n" + note + END

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START in text:
        text = re.sub(re.escape(START) + ".*?" + re.escape(END), block, text, flags=re.S)
    else:
        raise SystemExit(f"Markers {START} / {END} not found in README.md")
    readme.write_text(text, encoding="utf-8")
    print(f"README results table updated ({len(bench['table'])} models).")


if __name__ == "__main__":
    main()
