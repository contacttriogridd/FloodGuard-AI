# FloodGuard AI — common commands.  (Windows without make: run the commands after each colon.)
PY ?= python
CONFIG ?= configs/default.yaml
export PYTHONPATH := src:.

.PHONY: help install install-cpu synthetic smoke data pipeline tune train test api gradio mlflow \
        report-assets report docker-build docker-run compose hf-deploy leaderboard clean

help:
	@grep -E '^[a-z-]+:' Makefile | sed 's/:.*//' | tr '\n' ' '; echo

install:            ## full environment (GPU PyTorch on Linux/Colab)
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

install-cpu:        ## CPU-only PyTorch (laptops)
	$(PY) -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu && $(PY) -m pip install -e .

synthetic:          ## small synthetic dataset with the S4E5 schema
	$(PY) scripts/make_synthetic_flood.py

smoke: synthetic    ## whole pipeline on synthetic data, CPU, a few minutes
	$(PY) scripts/run_pipeline.py --override configs/smoke.yaml --stages all --skip download

data:               ## download Kaggle S4E5 + original dataset (needs ~/.kaggle/kaggle.json)
	$(PY) scripts/run_pipeline.py --config $(CONFIG) --stages download

pipeline:           ## full run on the real data (use a GPU runtime)
	$(PY) scripts/run_pipeline.py --config $(CONFIG) --stages all

tune:
	$(PY) scripts/run_pipeline.py --config $(CONFIG) --stages tune

train:
	$(PY) scripts/run_pipeline.py --config $(CONFIG) --stages train,ensemble,evaluate,benchmark

test:
	$(PY) -m pytest -q

api:                ## local API + UI on http://localhost:7860 (docs at /docs, Gradio at /ui)
	FG_ARTIFACTS_DIR=$${FG_ARTIFACTS_DIR:-runs/models} uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload

gradio:
	$(PY) app/gradio_ui.py

mlflow:             ## experiment tracking UI on http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///runs/mlflow.db --port 5000

mlflow-smoke:
	mlflow ui --backend-store-uri sqlite:///runs/smoke/mlflow.db --port 5000

report-assets:
	$(PY) scripts/run_pipeline.py --config $(CONFIG) --stages report

report: report-assets   ## needs a LaTeX distribution (latexmk + IEEEtran)
	cd report && latexmk -pdf FloodGuard_IEEE.tex

overleaf: report-assets ## zip for Overleaf: New Project -> Upload Project
	$(PY) scripts/package_report.py

docker-build:       ## bakes runs/models into the image
	mkdir -p deploy/model_artifacts && cp runs/models/floodguard_pipeline.joblib runs/models/model_card.json deploy/model_artifacts/
	docker build -t floodguard-ai .

docker-run:
	docker run --rm -p 7860:7860 floodguard-ai

compose:
	docker compose up --build

hf-deploy:          ## needs HF_TOKEN and HF_USER
	bash deploy/push_to_hf_space.sh

leaderboard:        ## Kaggle leaderboard snapshot (verify public vs private!)
	$(PY) -m floodguard.evaluation.benchmark --config $(CONFIG) --fetch-leaderboard

clean:
	rm -rf runs/smoke data/synthetic .pytest_cache
