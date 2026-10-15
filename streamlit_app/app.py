"""FloodGuard AI — Streamlit app (entrypoint for Streamlit Community Cloud).

Shows the same web UI as the FastAPI deployment (``api/static/index.html``), full screen, as a Streamlit
custom component. The page's API calls are passed to Python by ``streamlit_app/bridge.js`` and answered by
the FastAPI app itself (in-process, via its test client), so validation, results and errors are identical
and no separate API server is needed.

Run locally from the repository root (Community Cloud also runs from the root):
    pip install -r streamlit_app/requirements.txt
    streamlit run streamlit_app/app.py

The trained model is read from ``deploy/model_artifacts`` (or ``FG_ARTIFACTS_DIR``). If it is not there and
``FG_MODEL_REPO`` is set — as an env var or a Streamlit secret — it is downloaded from that Hugging Face repo.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
API_PATHS = {"/health", "/model-info", "/predict", "/predict/batch", "/predict/csv", "/explain"}
# Public FastAPI deployment: target of the page's Gradio / API-docs links and of its code snippets.
DEFAULT_API_URL = "https://floodguard-ai-e81o.onrender.com"

st.set_page_config(page_title="FloodGuard AI — Urban Flood Risk Intelligence", page_icon="🌊", layout="wide",
                   initial_sidebar_state="collapsed")


def _secrets_to_env() -> None:
    """Expose Streamlit secrets (Community Cloud → App settings → Secrets) as environment variables."""
    for key in ("FG_MODEL_REPO", "HF_TOKEN", "FG_API_URL"):
        if os.environ.get(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:  # noqa: BLE001 - no secrets.toml when running locally
            value = None
        if value:
            os.environ[key] = str(value)


@st.cache_resource(show_spinner="Loading the FloodGuard model…")
def api_client():
    """The FastAPI app with its model loaded. Failures raise and are not cached, so a rerun retries."""
    from fastapi.testclient import TestClient

    os.environ.setdefault("FG_ARTIFACTS_DIR", str(ROOT / "deploy" / "model_artifacts"))
    import api.main as fg_api  # noqa: PLC0415 - needs FG_ARTIFACTS_DIR set first

    fg_api.load_predictor()
    if fg_api.STATE["predictor"] is None:
        raise RuntimeError(fg_api.STATE["error"])
    return TestClient(fg_api.app)


@st.cache_resource
def ui_component(api_url: str):
    """Declare the component from index.html with the bridge script injected first in <head>."""
    page = (ROOT / "api" / "static" / "index.html").read_text(encoding="utf-8")
    bridge = (Path(__file__).parent / "bridge.js").read_text(encoding="utf-8").replace("__API_BASE__", json.dumps(api_url))
    page = page.replace("<head>", f"<head>\n<script>\n{bridge}</script>", 1)
    out = Path(tempfile.gettempdir()) / f"floodguard_ui_{hashlib.sha1(page.encode()).hexdigest()[:12]}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    return components.declare_component("floodguard_ui", path=str(out))


def answer(client, req: dict[str, Any]) -> dict[str, Any]:
    """Run one request from the page through the FastAPI app and return it in a JSON-safe form."""
    path, method = str(req.get("path", "")), str(req.get("method", "GET")).upper()
    try:
        if path not in API_PATHS:
            raise ValueError(f"Unknown API path: {path}")
        if req.get("files"):
            files = {f["name"]: (f["filename"], base64.b64decode(f["b64"]), f["type"]) for f in req["files"]}
            res = client.request(method, path, files=files)
        else:
            body = req.get("body")
            res = client.request(method, path, content=body,
                                 headers={"Content-Type": "application/json"} if body is not None else None)
        return {"id": req.get("id"), "status": res.status_code,
                "content_type": res.headers.get("content-type", "application/json"), "body": res.text}
    except Exception as exc:  # noqa: BLE001 - reported to the page as an API error
        return {"id": req.get("id"), "status": 500, "content_type": "application/json",
                "body": json.dumps({"detail": f"Internal error: {exc}"})}


# The page fills the whole viewport; hide Streamlit's own header and padding around it.
st.markdown("""<style>
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], footer { display: none !important; }
[data-testid="stMainBlockContainer"], .block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stMain"], [data-testid="stAppViewContainer"] { overflow: hidden !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
.stElementContainer:has(> .stMarkdown style) { display: none; }
iframe { display: block; border: 0; }
</style>""", unsafe_allow_html=True)

_secrets_to_env()
try:
    client = api_client()
except Exception as exc:  # noqa: BLE001
    st.title("🌊 FloodGuard AI")
    st.error(f"The model could not be loaded: {exc}")
    st.info("Commit the trained artefacts to `deploy/model_artifacts/`, or add `FG_MODEL_REPO` (and `HF_TOKEN` for a "
            "private repo) under the app's **Settings → Secrets** on Streamlit Community Cloud.")
    st.stop()

KEY = "floodguard_ui"
request = st.session_state.get(KEY)
if isinstance(request, dict) and request.get("id") and request["id"] != st.session_state.get("fg_answered"):
    st.session_state["fg_response"] = answer(client, request)
    st.session_state["fg_answered"] = request["id"]

ui_component(os.environ.get("FG_API_URL", DEFAULT_API_URL))(response=st.session_state.get("fg_response"), key=KEY,
                                                             default=None)
