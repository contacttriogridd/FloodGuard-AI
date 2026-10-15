"""FloodGuard AI — Streamlit app (entrypoint for Streamlit Community Cloud).

Run locally from the repository root (Community Cloud also runs from the root):
    pip install -r streamlit_app/requirements.txt
    streamlit run streamlit_app/app.py

The trained model is read from ``deploy/model_artifacts`` (or ``FG_ARTIFACTS_DIR``). If it is
not there and ``FG_MODEL_REPO`` is set — as an env var or a Streamlit secret — the artefacts are
downloaded from that Hugging Face model repo at start-up.
"""
from __future__ import annotations

import io
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floodguard.inference.predictor import FloodGuardPredictor  # noqa: E402
from floodguard.utils.config import load_config  # noqa: E402

st.set_page_config(page_title="FloodGuard AI", page_icon="🌊", layout="wide")

CFG = load_config()
MAX_CSV_ROWS = int(os.environ.get("FG_MAX_CSV_ROWS", CFG.app.max_csv_rows))
MAX_CSV_BYTES = int(os.environ.get("FG_MAX_CSV_BYTES", CFG.app.max_csv_bytes))
GITHUB_URL = "https://github.com/contacttriogridd/FloodGuard-AI"

THEME_TITLES = {
    "hydro_climatic": "Hydro-climatic", "urban_infrastructure": "Urban & infrastructure",
    "environmental": "Environmental", "governance": "Governance",
}
RISK_STYLE = {"Low": ("green", "🟢"), "Medium": ("orange", "🟠"), "High": ("red", "🔴")}
# Diverging pair (validated for colour-blind separation on Streamlit's light and dark backgrounds).
RAISES, LOWERS = "#e34948", "#2a78d6"

Scenario = Callable[[str, int, int], float]
SCENARIOS: dict[str, Scenario] = {
    "Balanced city": lambda f, lo, hi: 5,
    "Monsoon megacity": lambda f, lo, hi: {
        "MonsoonIntensity": 10, "Urbanization": 10, "DrainageSystems": 9, "DeterioratingInfrastructure": 9,
        "Encroachments": 9, "PopulationScore": 9, "ClimateChange": 8, "InadequatePlanning": 8}.get(f, 5),
    "Coastal delta": lambda f, lo, hi: {
        "CoastalVulnerability": 10, "Siltation": 9, "WetlandLoss": 9, "Watersheds": 9, "ClimateChange": 9,
        "MonsoonIntensity": 8, "RiverManagement": 8, "TopographyDrainage": 8}.get(f, 5),
    "Resilient city": lambda f, lo, hi: 2 if len(f) % 2 else 3,
    "Random": lambda f, lo, hi: random.randint(lo, lo + min(hi - lo, 10)),
}


# ------------------------------------------------------------------ model
def _secrets_to_env() -> None:
    """Expose Streamlit secrets (Community Cloud → App settings → Secrets) to the predictor."""
    for key in ("FG_MODEL_REPO", "HF_TOKEN"):
        if os.environ.get(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:  # noqa: BLE001 - no secrets.toml when running locally
            value = None
        if value:
            os.environ[key] = str(value)


@st.cache_resource(show_spinner="Loading the FloodGuard ensemble…")
def load_predictor() -> FloodGuardPredictor:
    # Failures raise and are not cached, so adding a secret and rerunning retries the load.
    directory = Path(os.environ.get("FG_ARTIFACTS_DIR") or ROOT / "deploy" / "model_artifacts")
    return FloodGuardPredictor.load(directory, int(os.environ.get("FG_LOCAL_MAX_EVALS", "300")))


def pretty(factor: str) -> str:
    words = re.findall(r"[A-Z][a-z]*", factor)
    return " ".join([words[0]] + [w.lower() for w in words[1:]]) if words else factor


def slider_key(factor: str) -> str:
    return f"factor_{factor}"


def apply_scenario(name: str, bounds: dict[str, list[int]]) -> None:
    for f, (lo, hi) in bounds.items():
        st.session_state[slider_key(f)] = int(np.clip(round(SCENARIOS[name](f, lo, hi)), lo, hi))


# ------------------------------------------------------------------ charts
def drivers_chart(drivers: list[dict[str, Any]]) -> alt.Chart:
    df = pd.DataFrame(drivers)
    df["Factor"] = df["factor"].map(pretty)
    df["Effect"] = np.where(df["shap"] >= 0, "Raises risk", "Lowers risk")
    bars = alt.Chart(df).mark_bar(size=16, cornerRadiusEnd=4).encode(
        x=alt.X("shap:Q", title="Change in flood probability (SHAP)", axis=alt.Axis(format="+.3f", tickCount=5)),
        y=alt.Y("Factor:N", sort=alt.EncodingSortField(field="abs", order="descending"), title=None,
                axis=alt.Axis(labelLimit=240)),
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RAISES, LOWERS]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("Factor:N"), alt.Tooltip("value:Q", title="Score"),
                 alt.Tooltip("shap:Q", title="SHAP", format="+.4f"), alt.Tooltip("Effect:N")],
    ).transform_calculate(abs="abs(datum.shap)")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeWidth=1, opacity=0.5).encode(x="x:Q")
    return (bars + zero).properties(height=48 * len(df) + 30)


def weights_chart(weights: dict[str, float], names: dict[str, str]) -> alt.Chart:
    df = pd.DataFrame([{"Model": names.get(k, k), "Weight": w} for k, w in weights.items() if w > 1e-6])
    return alt.Chart(df).mark_bar(size=16, cornerRadiusEnd=4, color=LOWERS).encode(
        x=alt.X("Weight:Q", axis=alt.Axis(format="%", tickCount=5), scale=alt.Scale(domain=[0, 1]), title="Blend weight"),
        y=alt.Y("Model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Weight:Q", format=".1%")],
    ).properties(height=48 * len(df) + 30)


# ------------------------------------------------------------------ CSV
def validate_csv(df: pd.DataFrame, factors: list[str], bounds: dict[str, list[int]]) -> list[dict[str, Any]]:
    """Same rules as the API's /predict/csv: integer scores inside the training-data range."""
    problems = []
    for f in factors:
        col = pd.to_numeric(df[f], errors="coerce")
        lo, hi = bounds[f]
        bad = col.isna() | (col % 1 != 0) | (col < lo) | (col > hi)
        if bad.any():
            problems.append({"column": f, "rows (incl. header line)": ", ".join(map(str, (np.where(bad)[0] + 2)[:10])),
                             "expected": f"integer {lo}–{hi}"})
    return problems


def template_csv(factors: list[str], bounds: dict[str, list[int]]) -> bytes:
    rows = []
    for name in ("Balanced city", "Monsoon megacity", "Resilient city"):
        row: dict[str, Any] = {"site": name}
        row.update({f: int(np.clip(SCENARIOS[name](f, *bounds[f]), *bounds[f])) for f in factors})
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode()


# ------------------------------------------------------------------ pages
def sidebar(p: FloodGuardPredictor) -> None:
    card = p.card
    with st.sidebar:
        st.title("🌊 FloodGuard AI")
        st.caption("Ensemble learning framework for urban flood risk prediction.")
        st.markdown(f"**Model:** {card.get('strategy_display', p.bundle.get('strategy'))}  \n"
                    f"**Version:** {p.bundle.get('version')}")
        if card.get("holdout_r2_deployed_refit") is not None:
            st.markdown(f"**Hold-out R²:** {card['holdout_r2_deployed_refit']:.3f}")
        if p.bundle.get("data_source") == "synthetic":
            st.warning("Demo model trained on synthetic smoke-test data: predictions are illustrative only.", icon="⚠️")
        st.caption(card.get("intended_use", "Decision support and education; not an official flood warning system."))
        st.markdown(f"[Source code on GitHub]({GITHUB_URL})")


def predict_tab(p: FloodGuardPredictor, groups: dict[str, list[str]]) -> None:
    st.caption("Each factor is an abstract integer index from the Kaggle flood-prediction data, not a physical "
               "measurement. Higher scores generally mean higher risk. Start from a scenario or set the sliders.")
    cols = st.columns(len(SCENARIOS))
    for col, name in zip(cols, SCENARIOS):
        col.button(name, on_click=apply_scenario, args=(name, p.bounds), width="stretch")

    left, right = st.columns([3, 2], gap="large")
    with left:
        for tab, (theme, factors) in zip(st.tabs([THEME_TITLES.get(t, pretty(t)) for t in groups]), groups.items()):
            with tab:
                pair = st.columns(2)
                for i, f in enumerate(factors):
                    lo, hi = p.bounds[f]
                    pair[i % 2].slider(pretty(f), int(lo), int(hi), step=1, key=slider_key(f))

    record = {f: int(st.session_state[slider_key(f)]) for f in p.factors}
    result = p.predict([record])[0]
    with right:
        color, icon = RISK_STYLE.get(result["risk_class"], ("gray", "⚪"))
        st.markdown(f"### {icon} :{color}[{result['risk_class']} flood risk]")
        m1, m2 = st.columns(2)
        m1.metric("Flood probability", f"{result['flood_probability']:.3f}",
                  help="Predicted FloodProbability (0–1) from the ensemble.")
        m2.metric("Chance of High risk", f"{result['high_risk_probability']:.0%}",
                  help="Calibrated probability that this area falls in the High risk class.")
        lo, hi = result["confidence_interval"]
        q_lo, q_hi = result["prediction_interval_90"]
        c1, c2 = p.bundle["prediction_cutoffs"]
        st.caption(f"Ensemble spread (95%): {lo:.3f}–{hi:.3f} · 90% prediction interval: {q_lo:.3f}–{q_hi:.3f}  \n"
                   f"Risk classes: Low < {c1:.3f} ≤ Medium < {c2:.3f} ≤ High")

        try:
            with st.spinner("Explaining the prediction…"):
                exp = p.explain(record)
        except Exception as exc:  # noqa: BLE001 - the prediction is still useful without SHAP
            st.warning(f"Explanation unavailable: {exc}")
            return
        st.markdown("#### What drives this prediction")
        st.caption(f"Top factors pushing the probability away from the average prediction ({exp['base_value']:.3f}).")
        st.altair_chart(drivers_chart(exp["top_drivers"]), width="stretch")

        st.markdown("#### Recommended actions")
        actions = exp["recommended_actions"] or [
            {"action": "No factor raises the risk above average; keep routine maintenance and monitoring."}]
        st.markdown("\n".join(f"- {a['action']}" for a in actions))

    with st.expander("All 20 factor contributions"):
        contrib = pd.DataFrame({"Factor": [pretty(f) for f in p.factors], "Score": [record[f] for f in p.factors],
                                "SHAP": [exp["all_contributions"][f] for f in p.factors]})
        st.dataframe(contrib.sort_values("SHAP", key=np.abs, ascending=False), hide_index=True,
                     column_config={"SHAP": st.column_config.NumberColumn(format="%+.4f")})


def batch_tab(p: FloodGuardPredictor) -> None:
    st.markdown(f"Upload a CSV with the 20 factor columns (up to {MAX_CSV_ROWS:,} rows). Extra columns, such as a "
                "site name, are kept in the output.")
    st.download_button("Download a template CSV", template_csv(p.factors, p.bounds), "floodguard_template.csv",
                       mime="text/csv")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is None:
        return
    if upload.size > MAX_CSV_BYTES:
        st.error(f"The file is larger than {MAX_CSV_BYTES / 1e6:.1f} MB.")
        return
    try:
        df = pd.read_csv(io.BytesIO(upload.getvalue()))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the CSV: {exc}")
        return
    if df.empty:
        st.error("The CSV has no rows.")
        return
    if len(df) > MAX_CSV_ROWS:
        st.error(f"At most {MAX_CSV_ROWS:,} rows per file (got {len(df):,}).")
        return
    missing = [f for f in p.factors if f not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return
    problems = validate_csv(df, p.factors, p.bounds)
    if problems:
        st.error("Some values are invalid. Fix them and upload again.")
        st.dataframe(pd.DataFrame(problems), hide_index=True)
        return

    with st.spinner(f"Scoring {len(df):,} rows…"):
        preds = p.predict(df[p.factors].astype(int).to_dict(orient="records"))
    out = df.copy()
    out["flood_probability"] = [r["flood_probability"] for r in preds]
    out["risk_class"] = [r["risk_class"] for r in preds]
    out["high_risk_probability"] = [r["high_risk_probability"] for r in preds]
    out["ci_low"] = [r["confidence_interval"][0] for r in preds]
    out["ci_high"] = [r["confidence_interval"][1] for r in preds]

    counts = out["risk_class"].value_counts()
    tiles = st.columns(len(p.bundle["risk_classes"]))
    for tile, cls in zip(tiles, p.bundle["risk_classes"]):
        tile.metric(f"{RISK_STYLE.get(cls, ('', ''))[1]} {cls} risk", f"{counts.get(cls, 0):,}",
                    help=f"{counts.get(cls, 0) / len(out):.0%} of rows")
    st.dataframe(out, hide_index=True,
                 column_config={"flood_probability": st.column_config.ProgressColumn(
                     "flood_probability", format="%.3f", min_value=0.0, max_value=1.0)})
    st.download_button("Download predictions", out.to_csv(index=False).encode(), "floodguard_predictions.csv",
                       mime="text/csv", type="primary")


def model_tab(p: FloodGuardPredictor) -> None:
    card = p.card
    m1, m2, m3 = st.columns(3)
    for col, key, label in ((m1, "oof_r2", "Out-of-fold R²"), (m2, "holdout_r2_cv_models", "Hold-out R² (CV models)"),
                            (m3, "holdout_r2_deployed_refit", "Hold-out R² (deployed)")):
        col.metric(label, f"{card[key]:.4f}" if card.get(key) is not None else "–")

    names = {m["name"]: m["display_name"] for m in card.get("members", [])}
    if card.get("weights"):
        st.markdown(f"#### Ensemble members · {card.get('strategy_display', p.bundle.get('strategy'))}")
        st.altair_chart(weights_chart(card["weights"], names), width="stretch")

    st.markdown("#### Training")
    details = {"Data source": p.bundle.get("data_source"), "Training rows": card.get("training_rows"),
               "Hold-out rows": card.get("holdout_rows"), "Trained at": p.bundle.get("trained_at"),
               **{f"Library: {k}": v for k, v in card.get("libraries", {}).items()}}
    st.dataframe(pd.DataFrame([{"Item": k, "Value": str(v)} for k, v in details.items() if v is not None]),
                 hide_index=True)
    if card.get("limitations"):
        st.markdown("#### Limitations")
        st.write(card["limitations"])


# ------------------------------------------------------------------ main
_secrets_to_env()
try:
    predictor = load_predictor()
except Exception as exc:  # noqa: BLE001
    st.title("🌊 FloodGuard AI")
    st.error(f"The model could not be loaded: {exc}")
    st.info("Commit the trained artefacts to `deploy/model_artifacts/`, or add `FG_MODEL_REPO` (and `HF_TOKEN` for a "
            "private repo) under the app's **Settings → Secrets** on Streamlit Community Cloud.")
    st.stop()

themes = predictor.bundle.get("themes") or {"factors": predictor.factors}
groups = {t: [f for f in fs if f in predictor.bounds] for t, fs in themes.items()}
leftover = [f for f in predictor.factors if not any(f in fs for fs in groups.values())]
if leftover:
    groups["other"] = leftover
for factor in predictor.factors:
    lo, hi = predictor.bounds[factor]
    st.session_state.setdefault(slider_key(factor), int(np.clip(5, lo, hi)))

sidebar(predictor)
st.title("Flood risk prediction")
predict, batch, model = st.tabs(["Predict", "Batch CSV", "Model"])
with predict:
    predict_tab(predictor, groups)
with batch:
    batch_tab(predictor)
with model:
    model_tab(predictor)
