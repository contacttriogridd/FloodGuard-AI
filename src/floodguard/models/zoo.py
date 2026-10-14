"""Model factory: linear baselines, bagging, gradient boosting and a PyTorch MLP.

    make_model("lgbm", params, cfg)        -> unfitted estimator
    fit_model(name, model, Xtr, ytr, Xva, yva, cfg) -> fitted model + training curve

Best Optuna parameters in ``<run_dir>/optuna/<model>_best.json`` are merged
automatically on top of the defaults in the config.
"""
from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from floodguard.utils.common import load_json, run_paths, use_gpu

MODEL_NAMES = {
    "ridge": "Ridge", "elasticnet": "ElasticNet", "rf": "Random Forest", "et": "Extra Trees",
    "histgb": "HistGradientBoosting", "xgb": "XGBoost", "lgbm": "LightGBM", "catboost": "CatBoost",
    "mlp": "Tabular MLP (PyTorch)", "ae_lgbm": "Autoencoder → LightGBM",
    "base_mean": "Mean predictor", "base_linear_raw": "Linear regression (raw factors)",
    "base_linear_sum": "Linear regression (factor sum only)",
}
BASELINES = ("base_mean", "base_linear_raw", "base_linear_sum")
FEATURE_VARIANT = {"ae_lgbm": "ae", "base_linear_raw": "raw", "base_linear_sum": "sum_only"}
EARLY_STOPPING = {"xgb", "lgbm", "ae_lgbm", "catboost", "mlp"}


def display_name(name: str) -> str:
    return MODEL_NAMES.get(name, name)


# ------------------------------------------------------------------ MLP
class TabularMLPRegressor(RegressorMixin, BaseEstimator):
    """PyTorch MLP for tabular regression (BatchNorm, dropout, cosine LR, early stopping).

    Trained with PyTorch; predictions use a NumPy forward pass with the stored
    weights, so the fitted model can be served without PyTorch.
    """

    def __init__(self, hidden: Sequence[int] = (256, 128, 64), dropout: float = 0.1, lr: float = 2e-3,
                 weight_decay: float = 1e-5, epochs: int = 60, batch_size: int = 2048, patience: int = 8,
                 val_fraction: float = 0.1, device: str = "cpu", random_state: int = 42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.device = device
        self.random_state = random_state

    def fit(self, X: Any, y: Any, eval_set: tuple[Any, Any] | None = None) -> "TabularMLPRegressor":
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.x_mean_, self.x_std_ = X.mean(0), X.std(0) + 1e-6
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() + 1e-9)
        if eval_set is None:
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(len(X))
            n_val = max(int(self.val_fraction * len(X)), 1)
            eval_set = (X[idx[:n_val]], y[idx[:n_val]])
            X, y = X[idx[n_val:]], y[idx[n_val:]]
        Xv, yv = np.asarray(eval_set[0], dtype=np.float32), np.asarray(eval_set[1], dtype=np.float32)
        dev = torch.device(self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu")
        to_t = lambda a: torch.tensor(a, device=dev)  # noqa: E731
        Xt, yt = to_t((X - self.x_mean_) / self.x_std_), to_t((y - self.y_mean_) / self.y_std_)
        Xvt, yvt = to_t((Xv - self.x_mean_) / self.x_std_), to_t((yv - self.y_mean_) / self.y_std_)

        layers: list[nn.Module] = []
        width_in = X.shape[1]
        for width in self.hidden:
            layers += [nn.Linear(width_in, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(self.dropout)]
            width_in = width
        layers.append(nn.Linear(width_in, 1))
        net = nn.Sequential(*layers).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.epochs, 1))
        loss_fn = nn.MSELoss()
        self.history_ = {"train_loss": [], "val_loss": []}
        best, best_state, wait = np.inf, None, 0
        for epoch in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xt), device=dev)
            total = 0.0
            for start in range(0, len(Xt), self.batch_size):
                b = perm[start:start + self.batch_size]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = loss_fn(net(Xt[b]).squeeze(1), yt[b])
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            sched.step()
            net.eval()
            with torch.no_grad():
                val = loss_fn(net(Xvt).squeeze(1), yvt).item()
            self.history_["train_loss"].append(total / len(Xt))
            self.history_["val_loss"].append(val)
            if val < best - 1e-6:
                best, wait, self.best_iteration_ = val, 0, epoch + 1
                best_state = copy.deepcopy(net.state_dict())
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.layers_ = self._export(net)
        return self

    @staticmethod
    def _export(net) -> list[dict[str, np.ndarray]]:
        from torch import nn

        out: list[dict[str, np.ndarray]] = []
        modules = list(net)
        for i, m in enumerate(modules):
            if isinstance(m, nn.Linear):
                entry = {"W": m.weight.detach().cpu().numpy().T.copy(), "b": m.bias.detach().cpu().numpy().copy()}
                if i + 1 < len(modules) and isinstance(modules[i + 1], nn.BatchNorm1d):
                    bn = modules[i + 1]
                    scale = bn.weight.detach().cpu().numpy() / np.sqrt(bn.running_var.cpu().numpy() + bn.eps)
                    entry["scale"] = scale
                    entry["shift"] = bn.bias.detach().cpu().numpy() - bn.running_mean.cpu().numpy() * scale
                out.append(entry)
        return out

    def predict(self, X: Any) -> np.ndarray:
        h = (np.asarray(X, dtype=np.float32) - self.x_mean_) / self.x_std_
        for i, layer in enumerate(self.layers_):
            h = h @ layer["W"] + layer["b"]
            if "scale" in layer:
                h = h * layer["scale"] + layer["shift"]
            if i < len(self.layers_) - 1:
                h = np.maximum(h, 0)
        return h[:, 0] * self.y_std_ + self.y_mean_


# --------------------------------------------------------------- factory
def best_params(cfg, name: str) -> dict[str, Any]:
    tuned = load_json(run_paths(cfg).optuna / f"{name if name != 'ae_lgbm' else 'lgbm'}_best.json", {})
    return dict(tuned.get("params", {})) if tuned else {}


def resolve_params(cfg, name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(cfg.models.defaults.get(name, {}) or {})
    merged.update(best_params(cfg, name) if params is None else params)
    return merged


def _mlp_params(p: dict[str, Any]) -> dict[str, Any]:
    p = dict(p)
    if "width" in p:  # Optuna parameterisation: width of first layer + depth (halving)
        width, depth = int(p.pop("width")), int(p.pop("depth", 2))
        p["hidden"] = [max(width // (2 ** i), 16) for i in range(depth)]
    return p


def make_model(name: str, params: dict[str, Any] | None, cfg) -> Any:
    seed = cfg.project.seed
    gpu = use_gpu(cfg)
    p = resolve_params(cfg, name, params)
    if name == "base_mean":
        return DummyRegressor(strategy="mean")
    if name in ("base_linear_raw", "base_linear_sum"):
        return LinearRegression()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**p))
    if name == "elasticnet":
        return make_pipeline(StandardScaler(), ElasticNet(random_state=seed, **p))
    if name == "rf":
        return RandomForestRegressor(n_jobs=-1, random_state=seed, **p)
    if name == "et":
        return ExtraTreesRegressor(n_jobs=-1, random_state=seed, **p)
    if name == "histgb":
        return HistGradientBoostingRegressor(early_stopping=True, validation_fraction=0.1, random_state=seed, **p)
    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(tree_method="hist", device="cuda" if gpu else "cpu", random_state=seed, n_jobs=-1, **p)
    if name in ("lgbm", "ae_lgbm"):
        from lightgbm import LGBMRegressor

        # Opt-in: "gpu" (OpenCL) or "cuda" need a GPU-enabled LightGBM build; fit_model falls back to CPU.
        device = str(cfg.models.get("lgbm_device", "cpu")).lower()
        if device != "cpu" and gpu:
            p.setdefault("device_type", device)
        return LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **p)
    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(random_seed=seed, verbose=0, task_type="GPU" if gpu else "CPU",
                                 allow_writing_files=False, **p)
    if name == "mlp":
        return TabularMLPRegressor(random_state=seed, device="cuda" if gpu else "cpu", **_mlp_params(p))
    raise ValueError(f"Unknown model '{name}'")


def available(name: str) -> bool:
    """Whether the optional dependency of a model is installed."""
    modules = {"xgb": "xgboost", "lgbm": "lightgbm", "ae_lgbm": "lightgbm", "catboost": "catboost", "mlp": "torch"}
    if name not in modules:
        return True
    try:
        __import__(modules[name])
        return True
    except ImportError:
        return False


# ------------------------------------------------------------- training
def _fit_lgbm(model: Any, Xtr, ytr, Xva, yva, es: int, use_es: bool, info: dict[str, Any]) -> None:
    if not use_es:
        model.fit(Xtr, ytr)
        return
    import inspect

    import lightgbm as lgb

    # LightGBM >= 4.7 takes eval_X / eval_y; older versions take eval_set.
    evals = ({"eval_X": (Xtr, Xva), "eval_y": (ytr, yva)} if "eval_X" in inspect.signature(model.fit).parameters
             else {"eval_set": [(Xtr, ytr), (Xva, yva)]})
    model.fit(Xtr, ytr, eval_names=["train", "valid"], eval_metric="rmse",
              callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)], **evals)
    res = model.evals_result_
    info["curve"] = {"train": res["train"]["rmse"], "valid": res["valid"]["rmse"]}
    info["best_iteration"] = int(model.best_iteration_ or model.n_estimators)


def fit_model(name: str, model: Any, Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame | None, yva: np.ndarray | None,
              cfg, early_stopping: bool = True) -> tuple[Any, dict[str, Any]]:
    """Fit with early stopping on the validation fold where supported.

    Returns the model and a dict with ``best_iteration`` and the training curve.
    """
    import warnings

    from scipy.linalg import LinAlgWarning

    warnings.filterwarnings("ignore", category=LinAlgWarning)  # collinear engineered features (Ridge handles them)
    es = int(cfg.cv.early_stopping_rounds)
    info: dict[str, Any] = {"best_iteration": None, "curve": None}
    use_es = early_stopping and Xva is not None and name in EARLY_STOPPING
    if name == "xgb":
        if use_es:
            model.set_params(early_stopping_rounds=es, eval_metric="rmse")
            model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
            res = model.evals_result()
            info["curve"] = {"train": res["validation_0"]["rmse"], "valid": res["validation_1"]["rmse"]}
            info["best_iteration"] = int(model.best_iteration) + 1
        else:
            model.fit(Xtr, ytr, verbose=False)
    elif name in ("lgbm", "ae_lgbm"):
        try:
            _fit_lgbm(model, Xtr, ytr, Xva, yva, es, use_es, info)
        except Exception as exc:  # noqa: BLE001 - GPU build / driver missing
            if model.get_params().get("device_type", "cpu") == "cpu":
                raise
            from floodguard.utils.common import get_logger

            get_logger("floodguard.zoo").warning("LightGBM GPU training failed (%s) — falling back to CPU.", exc)
            model.set_params(device_type="cpu")
            _fit_lgbm(model, Xtr, ytr, Xva, yva, es, use_es, info)
    elif name == "catboost":
        if use_es:
            model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=es, use_best_model=True)
            info["best_iteration"] = int(model.get_best_iteration()) + 1
            res = model.get_evals_result()
            if "validation" in res:
                info["curve"] = {"valid": res["validation"].get("RMSE", [])}
        else:
            model.fit(Xtr, ytr)
    elif name == "mlp":
        model.fit(Xtr.to_numpy(), ytr, eval_set=(Xva.to_numpy(), yva) if use_es else None)
        info["best_iteration"] = getattr(model, "best_iteration_", None)
        info["curve"] = {"train": model.history_["train_loss"], "valid": model.history_["val_loss"]}
    else:
        model.fit(Xtr, ytr)
    return model, info


def refit_params(name: str, best_iterations: list[int | None]) -> dict[str, Any]:
    """Iteration count for a refit on all training rows (no validation fold)."""
    its = [b for b in best_iterations if b]
    if not its:
        return {}
    n = int(np.ceil(np.mean(its) * 1.1))
    return {"xgb": {"n_estimators": n}, "lgbm": {"n_estimators": n}, "ae_lgbm": {"n_estimators": n},
            "catboost": {"iterations": n}, "mlp": {"epochs": n}}.get(name, {})


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X.to_numpy() if isinstance(model, TabularMLPRegressor) else X), dtype=float)
