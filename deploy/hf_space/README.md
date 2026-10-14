---
title: FloodGuard AI
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Ensemble learning for urban flood risk prediction
---

# FloodGuard AI — urban flood risk prediction

An ensemble (gradient boosting + bagging + PyTorch MLP, blended/stacked on out-of-fold
predictions) trained on the Kaggle Playground Series S4E5 flood dataset.

* **Web UI** — this page (20 factor sliders, risk class, SHAP drivers, mitigation actions)
* **Gradio UI** — [`/ui`](./ui)
* **REST API** — [`/docs`](./docs): `POST /predict`, `/predict/batch`, `/predict/csv`, `/explain`, `GET /model-info`, `/health`

Model artefacts are downloaded at start-up from the Hugging Face model repository named in
the Space variable `FG_MODEL_REPO` ([Kiransarathi/floodguard-ai-model](https://huggingface.co/Kiransarathi/floodguard-ai-model)).
Source code: [github.com/contacttriogridd/FloodGuard-AI](https://github.com/contacttriogridd/FloodGuard-AI).

> Decision-support tool for education and planning — not an official flood warning system.
> The S4E5 data are synthetically generated from the original Flood Prediction Factors dataset.
