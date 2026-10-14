"""Package the IEEE report for Overleaf (New Project -> Upload Project -> this zip).

    python scripts/package_report.py                       # report/ -> report_overleaf.zip
    python scripts/package_report.py --report-dir runs/smoke/report --out smoke_report.zip

The zip contains FloodGuard_IEEE.tex plus the generated tables/ and figures/.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="report", help="folder with tables/ and figures/ (run's report_dir)")
    parser.add_argument("--out", default="report_overleaf.zip")
    args = parser.parse_args()

    assets = ROOT / args.report_dir
    tex = ROOT / "report" / "FloodGuard_IEEE.tex"
    tables, figures = sorted((assets / "tables").glob("*.tex")), sorted((assets / "figures").glob("*.png"))
    macros = assets / "tables" / "macros.tex"
    if not macros.exists():
        print(f"WARNING: {macros} not found — the PDF will show placeholders. Run the pipeline's 'report' stage first.")
    elif "synthetic smoke-test data" in macros.read_text(encoding="utf-8"):
        print("WARNING: these assets come from SYNTHETIC smoke-test data — do not submit this report.")

    out = ROOT / args.out
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tex, "FloodGuard_IEEE.tex")
        for path in tables:
            zf.write(path, f"tables/{path.name}")
        for path in figures:
            zf.write(path, f"figures/{path.name}")
    print(f"Wrote {out} ({len(tables)} tables, {len(figures)} figures). "
          "Overleaf: New Project -> Upload Project -> select the zip -> Recompile (pdfLaTeX).")


if __name__ == "__main__":
    main()
