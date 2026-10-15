"""FloodGuard AI — FastAPI inference service.

    uvicorn api.main:app --host 0.0.0.0 --port 7860

Endpoints
    GET  /health            liveness + model status
    GET  /model-info        features, valid ranges, metrics, version
    POST /predict           one record -> probability, risk class, CI, top-5 SHAP drivers, actions
    POST /predict/batch     up to app.max_batch records (no SHAP, fast)
    POST /predict/csv       CSV upload -> CSV with predictions
    POST /explain           SHAP contributions for one record
    GET  /                  HTML UI           /ui  Gradio UI           /docs  OpenAPI
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, create_model

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from floodguard.inference.predictor import CARD, FloodGuardPredictor, _download_from_hub, artifacts_dir  # noqa: E402
from floodguard.utils.common import get_logger, load_json  # noqa: E402
from floodguard.utils.config import load_config  # noqa: E402

log = get_logger("floodguard.api")
CFG = load_config()
FACTORS: list[str] = list(CFG.features.factors)
MAX_BATCH = int(os.environ.get("FG_MAX_BATCH", CFG.app.max_batch))
MAX_CSV_ROWS = int(os.environ.get("FG_MAX_CSV_ROWS", CFG.app.max_csv_rows))
MAX_CSV_BYTES = int(os.environ.get("FG_MAX_CSV_BYTES", CFG.app.max_csv_bytes))
STATIC = Path(__file__).parent / "static"


# ---------------------------------------------------------------- schema
def _bounds() -> dict[str, list[int]]:
    """Valid ranges come from the training data (model card); config range as fallback."""
    directory = artifacts_dir()
    if not (directory / CARD).exists():
        try:
            _download_from_hub(directory)
        except Exception as exc:  # noqa: BLE001 - surfaced through /health
            log.warning("Model download failed: %s", exc)
    card = load_json(directory / CARD, {})
    lo, hi = CFG.data.factor_range
    return {f: card.get("feature_bounds", {}).get(f, [int(lo), int(hi)]) for f in FACTORS}


BOUNDS = _bounds()
EXAMPLE = {f: int(round((BOUNDS[f][0] + BOUNDS[f][1]) / 3)) for f in FACTORS}

FloodFactors = create_model(
    "FloodFactors",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE}),
    **{f: (int, Field(..., ge=BOUNDS[f][0], le=BOUNDS[f][1], description=f"{f} score ({BOUNDS[f][0]}–{BOUNDS[f][1]})"))
       for f in FACTORS},
)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[FloodFactors]  # type: ignore[valid-type]


class Driver(BaseModel):
    factor: str
    value: float
    shap: float
    direction: str


class Action(BaseModel):
    factor: str | None
    action: str


class Prediction(BaseModel):
    flood_probability: float = Field(..., description="Predicted FloodProbability (0–1)")
    risk_class: str = Field(..., description="Low / Medium / High (cut-offs tuned on validation predictions)")
    high_risk_probability: float = Field(..., description="Calibrated probability of the High class")
    confidence_interval: list[float] = Field(..., description="95% spread across ensemble members")
    prediction_interval_90: list[float] = Field(..., description="90% interval from out-of-fold residuals")


class ExplainedPrediction(Prediction):
    top_drivers: list[Driver]
    recommended_actions: list[Action]
    model_version: str | None = None


# ------------------------------------------------------------------- app
STATE: dict[str, Any] = {"predictor": None, "error": None}


def load_predictor() -> None:
    try:
        STATE["predictor"] = FloodGuardPredictor.from_env()
        STATE["error"] = None
        log.info("Model loaded (%s, %s).", STATE["predictor"].bundle.get("strategy"), STATE["predictor"].bundle.get("version"))
    except Exception as exc:  # noqa: BLE001
        STATE["predictor"], STATE["error"] = None, str(exc)
        log.warning("Model not loaded: %s", exc)


def predictor() -> FloodGuardPredictor:
    if STATE["predictor"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train it (scripts/run_pipeline.py) or set FG_MODEL_REPO.")
    return STATE["predictor"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_predictor()
    port = os.environ.get("PORT", "7860")
    # Uvicorn only prints the 0.0.0.0 bind address; in Docker the port must also be published (-p).
    log.info("Open http://localhost:%s  (Docker: run with -p %s:%s or `docker compose up`)", port, port, port)
    yield


app = FastAPI(title="FloodGuard AI", version=CFG.project.version, lifespan=lifespan,
              description="Ensemble learning framework for urban flood risk prediction (Kaggle S4E5 schema).")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("FG_CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": "Invalid input", "errors": errors})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    p = STATE["predictor"]
    return {"status": "ok" if p else "degraded", "model_loaded": p is not None,
            "version": p.bundle.get("version") if p else None, "data_source": p.bundle.get("data_source") if p else None,
            "message": None if p else STATE["error"]}


@app.get("/model-info", tags=["model"])
def model_info() -> dict[str, Any]:
    p = predictor()
    info = p.info()
    info.update({"limits": {"max_batch": MAX_BATCH, "max_csv_rows": MAX_CSV_ROWS, "max_csv_bytes": MAX_CSV_BYTES},
                 "model_card": p.card})
    return info


@app.post("/predict", response_model=ExplainedPrediction, tags=["prediction"])
def predict(record: FloodFactors) -> dict[str, Any]:  # type: ignore[valid-type]
    p = predictor()
    out = p.predict_and_explain(record.model_dump())
    out["model_version"] = p.bundle.get("version")
    return out


@app.post("/predict/batch", response_model=list[Prediction], tags=["prediction"])
def predict_batch(request: BatchRequest) -> list[dict[str, Any]]:
    if len(request.records) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"At most {MAX_BATCH} records per batch (got {len(request.records)}).")
    if not request.records:
        raise HTTPException(status_code=400, detail="'records' is empty.")
    return predictor().predict([r.model_dump() for r in request.records])


@app.post("/predict/csv", tags=["prediction"], response_class=StreamingResponse)
async def predict_csv(file: UploadFile = File(..., description="CSV with the 20 factor columns (extra columns kept)")):
    raw = await file.read(MAX_CSV_BYTES + 1)
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail=f"CSV larger than {MAX_CSV_BYTES} bytes.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    if len(df) > MAX_CSV_ROWS:
        raise HTTPException(status_code=413, detail=f"At most {MAX_CSV_ROWS} rows per CSV (got {len(df)}).")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows.")
    missing = [f for f in FACTORS if f not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    problems = []
    for f in FACTORS:
        col = pd.to_numeric(df[f], errors="coerce")
        lo, hi = BOUNDS[f]
        bad = col.isna() | (col % 1 != 0) | (col < lo) | (col > hi)
        if bad.any():
            problems.append({"column": f, "rows": (np.where(bad)[0] + 2).tolist()[:10], "expected": f"integer {lo}–{hi}"})
    if problems:
        raise HTTPException(status_code=400, detail={"message": "Invalid values (row numbers include the header line)",
                                                     "problems": problems})
    preds = predictor().predict(df[FACTORS].astype(int).to_dict(orient="records"))
    out = df.copy()
    out["flood_probability"] = [p["flood_probability"] for p in preds]
    out["risk_class"] = [p["risk_class"] for p in preds]
    out["high_risk_probability"] = [p["high_risk_probability"] for p in preds]
    out["ci_low"] = [p["confidence_interval"][0] for p in preds]
    out["ci_high"] = [p["confidence_interval"][1] for p in preds]
    buffer = io.StringIO()
    out.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=floodguard_predictions.csv"})


@app.post("/explain", tags=["explainability"])
def explain(record: FloodFactors) -> dict[str, Any]:  # type: ignore[valid-type]
    return predictor().explain(record.model_dump())


# ------------------------------------------------------------------ Gradio
try:
    import gradio as gr

    from app.gradio_ui import build_demo

    app = gr.mount_gradio_app(app, build_demo(lambda: STATE["predictor"], FACTORS, BOUNDS, dict(CFG.features.themes)), path="/ui")
except ImportError as exc:  # pragma: no cover - Gradio is optional
    log.warning("Gradio UI not mounted (%s).", exc)
