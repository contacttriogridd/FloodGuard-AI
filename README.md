# FloodGuard AI — An Ensemble Learning Framework for Urban Flood Risk Prediction

M.Sc. AI & ML — **19MAM56 Advanced Machine Learning Lab, CAT II**

FloodGuard AI predicts urban **flood probability** (regression, the Kaggle S4E5 metric R²) and a
**Low / Medium / High risk class** with a binary High-risk flag from 20 environmental, infrastructural and
socio-economic factors. It trains ten base learners through one out-of-fold engine, tunes them with Optuna, combines
them with blending/stacking, tests the gain statistically, explains predictions with SHAP / PDP / LIME and serves the
model through a validated FastAPI + Gradio app in Docker on Hugging Face Spaces.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contacttriogridd/FloodGuard-AI/blob/main/notebooks/FloodGuard_Colab.ipynb)

> **Links** — GitHub: <https://github.com/contacttriogridd/FloodGuard-AI> ·
> Hugging Face Space: <https://huggingface.co/spaces/Kiransarathi/floodguard-ai> (live after the Colab run deploys it) ·
> Model repo: <https://huggingface.co/Kiransarathi/floodguard-ai-model>

---

## Architecture

```
 Kaggle S4E5 (1.1 M rows) ─┐
 original dataset (50 k) ──┴─► prepare ─► features ─► reduce ─► tune ─► train ─► ensemble ─► evaluate ─► benchmark ─► explain ─► plots ─► report
                               audit       row stats    PCA       Optuna   10 models  NNLS/hill   CIs, tests  baselines   SHAP      21 PNGs   LaTeX
                               dedupe      themes       t-SNE     TPE +    5-fold OOF blend,      ablation    leaderboard PDP/ICE             tables +
                               MCAR study  freq/TE      UMAP      pruning  (shared    stacking,   errors      literature  LIME                macros
                               outliers    sorted       autoenc.  SQLite   engine)    soft vote   external
                                  │                                   └──────── MLflow tracking (every trial + model) ────────┘
                                  ▼
             deployed sklearn Pipeline (imputer → features → selection → FloodGuardRegressor)
                                  ▼
             FastAPI (/predict, /predict/batch, /predict/csv, /explain, /model-info, /health) + HTML UI (/) + Gradio (/ui)
                                  ▼
                     Docker (python:3.11-slim, uid 1000, port 7860)  →  Hugging Face Docker Space
```

## Rubric → files

| Rubric item (marks) | What satisfies it | Files |
|---|---|---|
| **1. Data engineering & visualisation (8)** | Schema/range audit, de-duplication, MCAR injection + median/KNN/iterative imputer comparison, IQR + IsolationForest flags, decile-stratified hold-out and folds; row statistics, theme aggregates, interactions, frequency encoding, **high-cardinality factor-pair keys** (hundreds of levels, frequency + cross-fitted target encoding), sorted values, out-of-fold target encoding, MI + permutation selection with top-k by CV; PCA, t-SNE, UMAP, PyTorch autoencoder; 21 figures | `src/floodguard/data/prepare.py`, `src/floodguard/features/engineering.py`, `src/floodguard/features/reduction.py`, `src/floodguard/visualization/plots.py` |
| **2. Advanced modelling & pipeline optimisation (10)** | Ridge, ElasticNet, RF, ExtraTrees, HistGB, XGBoost, LightGBM, CatBoost, PyTorch MLP, Autoencoder→LightGBM through one K-fold OOF engine with early stopping; NNLS + hill-climbing blending, Ridge/LightGBM stacking, soft voting; Optuna TPE (multivariate) + MedianPruner + SQLite, GridSearchCV baseline; MLflow tracking of every trial and model; one serialised sklearn Pipeline + model card; GPU for XGBoost / CatBoost / MLP (auto) and LightGBM (opt-in `models.lgbm_device`, CPU fallback) | `src/floodguard/models/{zoo,cv,ensemble}.py`, `src/floodguard/tuning/optuna_search.py`, `src/floodguard/utils/tracking.py`, `configs/models/*.yaml` |
| **3. Benchmarking & comparative analysis (10)** | R², RMSE, MAE, macro-F1, ROC-AUC, PR-AUC, log-loss, Brier, ECE with 1000-resample bootstrap CIs; paired bootstrap + Wilcoxon (ensemble vs best single), Friedman + Nemenyi CD; baselines (mean, linear raw, linear on factor sum); Kaggle leaderboard (to fill in) + published ensemble studies (Ren et al. 2024, Rahimi et al. 2026 — numbers verified from the abstracts) in `configs/benchmarks.yaml`, never guessed; external validation on the original dataset; ablations; error analysis by decile/theme/extremes/outliers/class boundaries + SHAP of the 20 worst predictions | `src/floodguard/evaluation/{metrics,evaluate,benchmark}.py`, `src/floodguard/explain/explainability.py` |
| **4a. FastAPI app (4)** | Pydantic validation (20 required integers, bounds from training data, `extra="forbid"`), single/batch/CSV prediction, SHAP explanation, mitigation actions, CI from ensemble spread, 422/413/400 errors, HTML UI + Gradio | `api/main.py`, `api/static/index.html`, `app/gradio_ui.py`, `src/floodguard/inference/predictor.py` |
| **4b. Docker & cloud (4)** | `python:3.11-slim`, non-root uid 1000, port 7860, HEALTHCHECK, docker-compose; HF Docker Space with model artefacts pulled from an HF model repo (`FG_MODEL_REPO`) | `Dockerfile`, `docker-compose.yml`, `deploy/hf_space/README.md`, `deploy/push_to_hf_space.sh` |
| **5. Documentation & report (4)** | IEEE report whose numbers are generated from the metrics JSON, incl. a computational-environment table (hardware, library versions, stage runtimes); modular package with type hints, docstrings, logging and per-module CLIs; tests | `report/FloodGuard_IEEE.tex`, `src/floodguard/evaluation/report_assets.py`, `tests/`, this README |

## Repository layout

```
configs/default.yaml        all paths + hyper-parameters        configs/smoke.yaml    tiny synthetic CPU run
configs/models/*.yaml       Optuna search spaces                configs/benchmarks.yaml  leaderboard + literature (fill in)
src/floodguard/             utils/ data/ features/ models/ tuning/ evaluation/ explain/ visualization/ inference/
scripts/run_pipeline.py     stage runner                        scripts/make_synthetic_flood.py
api/                        FastAPI service + static UI         app/gradio_ui.py
tests/                      pytest (features, metrics, ensemble, API contract)
notebooks/FloodGuard_Colab.ipynb                                report/FloodGuard_IEEE.tex
deploy/                     HF Space README + push script       Dockerfile, docker-compose.yml, Makefile
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu   # CPU PyTorch
pip install -e .
```

### Kaggle credentials
Kaggle → *Settings → API → Create New Token* downloads `kaggle.json`. Put it in `~/.kaggle/kaggle.json`
(Windows `C:\Users\<you>\.kaggle\kaggle.json`, `chmod 600` on Linux) and accept the
[competition rules](https://www.kaggle.com/competitions/playground-series-s4e5) once in the browser.

## Running

| Goal | Command |
|---|---|
| Smoke test (synthetic data, CPU, a few minutes) | `python scripts/make_synthetic_flood.py && python scripts/run_pipeline.py --override configs/smoke.yaml --stages all --skip download` |
| Download real data | `python scripts/run_pipeline.py --stages download` |
| Full run (GPU recommended) | `python scripts/run_pipeline.py --stages all --skip download` |
| Subset of stages / overrides | `python scripts/run_pipeline.py --stages tune,train --override optuna.n_trials=20` |
| One module | `python -m floodguard.models.cv --override configs/smoke.yaml --model lgbm` |
| Tests | `pytest -q` |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///runs/mlflow.db` (smoke: `runs/smoke/mlflow.db`) |
| API + UI | `FG_ARTIFACTS_DIR=runs/models uvicorn api.main:app --port 7860` → http://localhost:7860 (`/docs`, `/ui`) |
| Kaggle leaderboard snapshot | `python -m floodguard.evaluation.benchmark --fetch-leaderboard` |

`make` shortcuts: `make smoke`, `make data`, `make pipeline`, `make test`, `make api`, `make mlflow`, `make docker-build`,
`make docker-run`, `make hf-deploy`, `make report`.

### Kaggle cloud GPU (launched from your laptop — no local compute)
`configs/kaggle.yaml` sizes the run for Kaggle's 12-hour session; `deploy/kaggle_kernel/run_floodguard.py` runs inside
a private Kaggle notebook with the competition and original datasets attached.

```bash
python scripts/kaggle_training.py setup    # installs ~/Downloads/kaggle.json
python scripts/kaggle_training.py launch   # needs accepted competition rules + phone-verified Kaggle account (internet)
python scripts/kaggle_training.py status   # until "complete"
python scripts/kaggle_training.py fetch    # -> runs/ (models, metrics, figures) + report/ (tables, figures)
python scripts/kaggle_training.py submit   # late submission -> the ensemble's own private-leaderboard R²
```

### Colab
Open `notebooks/FloodGuard_Colab.ipynb`, select a **GPU** runtime, set `REPO_URL`, upload `kaggle.json` and run all
cells. XGBoost, CatBoost and the MLP use the GPU automatically (`models.use_gpu: auto`). For a shorter run add
`--override optuna.n_trials=15 --override data.sample_rows=400000`. Optuna studies live in SQLite, so an interrupted
tuning stage resumes.

## Deployment

```bash
# local container (bakes the trained model into the image)
mkdir -p deploy/model_artifacts && cp runs/models/floodguard_pipeline.joblib runs/models/model_card.json deploy/model_artifacts/
docker build -t floodguard-ai . && docker run -p 7860:7860 floodguard-ai

# Hugging Face: model repo + Docker Space (the Space downloads the model via FG_MODEL_REPO)
export HF_TOKEN=hf_... HF_USER=Kiransarathi
bash deploy/push_to_hf_space.sh

# Streamlit app — the same web UI as the FastAPI deployment (api/static/index.html), shown full screen as a
# Streamlit component; its API calls are answered by the FastAPI app in-process (streamlit_app/bridge.js)
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

**Streamlit Community Cloud:** *Create app* → repository `contacttriogridd/FloodGuard-AI`, branch `main`, main file
`streamlit_app/app.py`; under *Advanced settings* pick **Python 3.11** (the model is pickled with 3.11 /
scikit-learn 1.8.0). Cloud installs `streamlit_app/requirements.txt` (not the root training requirements) and
`packages.txt`. The app needs the model either committed in `deploy/model_artifacts/` or downloaded from a Hugging
Face model repo — add `FG_MODEL_REPO = "Kiransarathi/floodguard-ai-model"` (and `HF_TOKEN` if private) under
*Settings → Secrets*.

API: `POST /predict` (one record → `flood_probability`, `risk_class`, `high_risk_probability`, `confidence_interval`
(95 % spread across ensemble members), `prediction_interval_90`, top-5 SHAP drivers, mitigation actions),
`POST /predict/batch` (≤ 1000 records), `POST /predict/csv` (CSV in → CSV out, size/row limits → 413),
`POST /explain`, `GET /model-info`, `GET /health`. Invalid values, missing or unknown fields → **422**.

## Results

Filled in after the full Kaggle run (`runs/metrics/benchmark.json`, `report/tables/model_comparison.tex`).
Numbers from the synthetic smoke run are **not** results and must not be reported.

<!-- RESULTS:START -->
| Model | OOF R² | Test R² [95 % CI] | RMSE | Macro-F1 | ROC-AUC (High) |
|---|---|---|---|---|---|
| Mean predictor | | | | | |
| Linear regression (factor sum) | | | | | |
| Best single model | | | | | |
| **FloodGuard ensemble** | | | | | |
| Kaggle S4E5 private LB — top / median | — | from `configs/benchmarks.yaml` | | | |
<!-- RESULTS:END -->

## Honest limitations

* **S4E5 is synthetic**: it was generated by a model trained on the original Flood Prediction Factors dataset; factors
  are abstract scores, not physical measurements. Results do not transfer to a specific city without real data.
* **Benchmarks are never guessed**: the two literature entries were copied from the papers' abstracts (sources listed in
  `configs/benchmarks.yaml`). The official Kaggle private top / median R² could not be confirmed from a public source, so
  they stay `null` — copy them from the leaderboard page (or `--fetch-leaderboard`) and submit `runs/submission.csv` as a
  late submission to get the ensemble's own private-leaderboard score.
* Verify every reference in `report/FloodGuard_IEEE.tex` against the original paper before submission.
