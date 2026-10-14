"""Gradio interface for FloodGuard AI (mounted inside FastAPI at /ui).

Run standalone:  python app/gradio_ui.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

THEME_TITLES = {
    "hydro_climatic": "Hydro-climatic", "urban_infrastructure": "Urban & infrastructure",
    "environmental": "Environmental", "governance": "Governance",
}
BADGE = {"Low": "🟢 LOW", "Medium": "🟠 MEDIUM", "High": "🔴 HIGH"}


def build_demo(get_predictor: Callable[[], Any], factors: list[str], bounds: dict[str, list[int]],
               themes: dict[str, list[str]]) -> gr.Blocks:
    def run(*values: float):
        p = get_predictor()
        if p is None:
            return "### Model not loaded\nTrain the pipeline or set `FG_MODEL_REPO`.", [], ""
        record = {f: int(v) for f, v in zip(ordered, values)}
        r = p.predict_and_explain(record)
        lo, hi = r["confidence_interval"]
        summary = (f"## {BADGE.get(r['risk_class'], r['risk_class'])} flood risk\n"
                   f"**Flood probability:** {r['flood_probability']:.3f}  \n"
                   f"**Ensemble spread (95%):** {lo:.3f} – {hi:.3f}  \n"
                   f"**P(High risk):** {r['high_risk_probability']:.0%}")
        drivers = [[d["factor"], d["value"], round(d["shap"], 4), d["direction"]] for d in r["top_drivers"]]
        actions = "\n".join(f"- {a['action']}" for a in r["recommended_actions"])
        return summary, drivers, "### Recommended actions\n" + actions

    ordered = [f for cols in themes.values() for f in cols]
    ordered += [f for f in factors if f not in ordered]
    sliders: list[gr.Slider] = []
    with gr.Blocks(title="FloodGuard AI") as demo:
        gr.Markdown("# FloodGuard AI\nEnsemble learning framework for urban flood risk prediction. "
                    "Set the 20 factor scores and press **Predict**.")
        with gr.Row():
            with gr.Column(scale=3):
                for theme, cols in themes.items():
                    with gr.Accordion(THEME_TITLES.get(theme, theme), open=theme == "hydro_climatic"):
                        for f in cols:
                            lo, hi = bounds[f]
                            sliders.append(gr.Slider(lo, hi, value=min(max(5, lo), hi), step=1, label=f))
                button = gr.Button("Predict flood risk", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown("Result will appear here.")
                drivers = gr.Dataframe(headers=["Factor", "Value", "SHAP", "Effect"], label="Top-5 drivers (SHAP)",
                                       interactive=False)
                actions = gr.Markdown()
        button.click(run, inputs=sliders, outputs=[summary, drivers, actions])
    return demo


if __name__ == "__main__":
    from floodguard.inference.predictor import FloodGuardPredictor
    from floodguard.utils.config import load_config

    cfg = load_config()
    predictor = FloodGuardPredictor.from_env()
    build_demo(lambda: predictor, list(cfg.features.factors), predictor.bounds, dict(cfg.features.themes)).launch()
