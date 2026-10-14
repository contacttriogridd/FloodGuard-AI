"""Load the deployed FloodGuard pipeline and serve predictions with explanations.

    from floodguard.inference.predictor import FloodGuardPredictor
    p = FloodGuardPredictor.from_env()
    p.predict([{...20 factors...}])
    p.explain({...})

Artefacts are read from ``FG_ARTIFACTS_DIR`` (default: ``app.artifacts_dir`` of
the config). If they are missing and ``FG_MODEL_REPO`` names a Hugging Face model
repository, they are downloaded at start-up.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from floodguard.utils.common import get_logger, load_json
from floodguard.utils.config import load_config, resolve_path

log = get_logger("floodguard.predictor")
BUNDLE = "floodguard_pipeline.joblib"
CARD = "model_card.json"

MITIGATIONS: dict[str, str] = {
    "MonsoonIntensity": "Prepare for intense monsoon rain: desilt drains before the season and activate early-warning alerts.",
    "TopographyDrainage": "Map natural drainage paths and keep low-lying flow corridors free of obstructions.",
    "RiverManagement": "Strengthen river management: maintain embankments, dredge channels and regulate flows.",
    "Deforestation": "Restore tree cover in the catchment to slow runoff and reduce erosion.",
    "Urbanization": "Offset urban runoff with permeable pavements, green roofs and retention basins.",
    "ClimateChange": "Adopt climate-adaptive design standards for drainage and flood defences.",
    "DamsQuality": "Inspect and upgrade dams and reservoirs; review spillway capacity and operating rules.",
    "Siltation": "Desilt rivers, canals and storm drains to restore their carrying capacity.",
    "AgriculturalPractices": "Promote contour farming, cover crops and buffer strips to reduce runoff.",
    "Encroachments": "Clear encroachments from floodplains, lakes and drainage channels.",
    "IneffectiveDisasterPreparedness": "Improve preparedness: evacuation plans, flood shelters and community drills.",
    "DrainageSystems": "Improve storm-water drainage capacity and clear blocked drains before heavy rain.",
    "CoastalVulnerability": "Protect coastal zones with mangroves, sea walls and storm-surge warnings.",
    "Landslides": "Stabilise slopes and monitor landslide-prone areas during heavy rainfall.",
    "Watersheds": "Implement integrated watershed management with check dams and recharge structures.",
    "DeterioratingInfrastructure": "Repair ageing culverts, bridges and drainage infrastructure.",
    "PopulationScore": "Plan shelter and evacuation capacity for densely populated neighbourhoods.",
    "WetlandLoss": "Protect and restore wetlands that store floodwater.",
    "InadequatePlanning": "Enforce flood-zone land-use planning and flood-resilient building codes.",
    "PoliticalFactors": "Strengthen flood governance, funding and inter-agency coordination.",
}


def artifacts_dir() -> Path:
    env = os.environ.get("FG_ARTIFACTS_DIR")
    if env:
        return Path(env)
    return resolve_path(load_config().app.artifacts_dir)


def _download_from_hub(target: Path) -> None:
    repo = os.environ.get("FG_MODEL_REPO")
    if not repo:
        return
    from huggingface_hub import snapshot_download

    log.info("Downloading model artefacts from Hugging Face repo %s", repo)
    snapshot_download(repo_id=repo, local_dir=str(target), token=os.environ.get("HF_TOKEN"))


class FloodGuardPredictor:
    def __init__(self, bundle: dict[str, Any], card: dict[str, Any] | None = None, local_max_evals: int = 500):
        self.bundle = bundle
        self.card = card or {}
        self.pipeline = bundle["pipeline"]
        self.factors: list[str] = bundle["factors"]
        self.bounds: dict[str, list[int]] = bundle["bounds"]
        self.local_max_evals = max(local_max_evals, 2 * len(self.factors) + 1)
        self._explainer = None

    # ----------------------------------------------------------- loading
    @classmethod
    def load(cls, directory: str | Path, local_max_evals: int = 500) -> "FloodGuardPredictor":
        directory = Path(directory)
        if not (directory / BUNDLE).exists():
            _download_from_hub(directory)
        if not (directory / BUNDLE).exists():
            raise FileNotFoundError(f"No trained model at {directory / BUNDLE}. Run the pipeline or set FG_MODEL_REPO.")
        return cls(joblib.load(directory / BUNDLE), load_json(directory / CARD, {}), local_max_evals)

    @classmethod
    def from_env(cls) -> "FloodGuardPredictor":
        evals = int(os.environ.get("FG_LOCAL_MAX_EVALS", "300"))
        return cls.load(artifacts_dir(), evals)

    # --------------------------------------------------------- prediction
    def frame(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)[self.factors].astype(float)
        df["is_original"] = 0.0
        return df

    def _member_spread(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        model = self.pipeline.steps[-1][1]
        Xt = self.pipeline[:-1].transform(X)
        P = model.member_predictions(Xt)
        w = np.asarray(model.weights if model.weights is not None else np.full(P.shape[1], 1 / P.shape[1]))
        mean = P @ w
        std = np.sqrt(np.maximum(((P - mean[:, None]) ** 2) @ w, 0))
        return mean, std

    def risk_class(self, values: np.ndarray) -> list[str]:
        names = self.bundle["risk_classes"]
        return [names[i] for i in np.digitize(values, self.bundle["prediction_cutoffs"])]

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        X = self.frame(records)
        pred = np.clip(self.pipeline.predict(X), 0, 1)
        _, std = self._member_spread(X)
        p_high = self.bundle["high_calibrator"].predict_proba(pred.reshape(-1, 1))[:, 1]
        q = self.bundle["residual_quantiles"]
        classes = self.risk_class(pred)
        out = []
        for i in range(len(X)):
            out.append({
                "flood_probability": round(float(pred[i]), 5),
                "risk_class": classes[i],
                "high_risk_probability": round(float(p_high[i]), 4),
                "confidence_interval": [round(float(np.clip(pred[i] - 1.96 * std[i], 0, 1)), 5),
                                        round(float(np.clip(pred[i] + 1.96 * std[i], 0, 1)), 5)],
                "prediction_interval_90": [round(float(np.clip(pred[i] + q[0.05], 0, 1)), 5),
                                           round(float(np.clip(pred[i] + q[0.95], 0, 1)), 5)],
            })
        return out

    # -------------------------------------------------------- explanation
    def _predict_factors(self, V: np.ndarray) -> np.ndarray:
        X = pd.DataFrame(np.asarray(V, dtype=float), columns=self.factors)
        X["is_original"] = 0.0
        return self.pipeline.predict(X)

    @property
    def explainer(self):
        if self._explainer is None:
            import shap

            bg = self.bundle["background"][self.factors].to_numpy(dtype=float)
            self._explainer = shap.explainers.Permutation(self._predict_factors, shap.maskers.Independent(bg, max_samples=len(bg)),
                                                          seed=0)
        return self._explainer

    @lru_cache(maxsize=256)
    def _shap(self, values: tuple[float, ...]) -> tuple[tuple[float, ...], float]:
        exp = self.explainer(np.array([values]), max_evals=self.local_max_evals, silent=True)
        return tuple(float(v) for v in exp.values[0]), float(np.ravel(exp.base_values)[0])

    def explain(self, record: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
        values = tuple(float(record[f]) for f in self.factors)
        contrib, base = self._shap(values)
        order = np.argsort(-np.abs(contrib))
        drivers = [{"factor": self.factors[i], "value": values[i], "shap": round(contrib[i], 5),
                    "direction": "raises" if contrib[i] >= 0 else "lowers"} for i in order[:top_k]]
        actions = [{"factor": d["factor"], "action": MITIGATIONS[d["factor"]]} for d in drivers if d["shap"] > 0]
        return {"base_value": round(base, 5), "top_drivers": drivers, "recommended_actions": actions,
                "all_contributions": {self.factors[i]: round(contrib[i], 5) for i in range(len(self.factors))}}

    def predict_and_explain(self, record: dict[str, Any]) -> dict[str, Any]:
        result = self.predict([record])[0]
        exp = self.explain(record)
        result.update({"top_drivers": exp["top_drivers"], "recommended_actions": exp["recommended_actions"]})
        if not exp["recommended_actions"]:
            result["recommended_actions"] = [{"factor": None, "action": "No factor raises the risk above average; keep routine maintenance and monitoring."}]
        return result

    # ------------------------------------------------------------- info
    def info(self) -> dict[str, Any]:
        return {"version": self.bundle.get("version"), "strategy": self.bundle.get("strategy"),
                "members": self.bundle.get("members"), "factors": self.factors, "bounds": self.bounds,
                "risk_classes": self.bundle["risk_classes"], "prediction_cutoffs": self.bundle["prediction_cutoffs"],
                "data_source": self.bundle.get("data_source"), "trained_at": self.bundle.get("trained_at"),
                "themes": self.bundle.get("themes"), "metrics": {k: self.card.get(k) for k in
                                                                 ("oof_r2", "holdout_r2_cv_models", "holdout_r2_deployed_refit")}}
