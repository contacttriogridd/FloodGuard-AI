# FloodGuard AI — inference image (FastAPI + HTML UI + Gradio at /ui).
#   docker build -t floodguard-ai .
#   docker run -p 7860:7860 floodguard-ai                                   # model baked from deploy/model_artifacts
#   docker run -p 7860:7860 -e FG_MODEL_REPO=<user>/floodguard-ai-model floodguard-ai   # model pulled from the Hub
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    PYTHONPATH=/app/src:/app \
    FG_ARTIFACTS_DIR=/app/model \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/tmp/huggingface \
    GRADIO_ANALYTICS_ENABLED=False

# libgomp: OpenMP runtime needed by LightGBM / XGBoost wheels.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

COPY --chown=user configs/ configs/
COPY --chown=user src/ src/
COPY --chown=user api/ api/
COPY --chown=user app/ app/
# Trained artefacts (floodguard_pipeline.joblib + model_card.json). The folder may be
# empty: the app then downloads them from the Hub repo named in FG_MODEL_REPO.
COPY --chown=user deploy/model_artifacts/ /app/model/

USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '7860'), timeout=8)"
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
