"""Dimensionality reduction: PCA, t-SNE, UMAP and a PyTorch autoencoder.

The autoencoder is trained with PyTorch, but its encoder weights are stored as
NumPy arrays, so serving (and the GBMs that use the embeddings) never needs
PyTorch. It is unsupervised and fitted on the training split only.

    python -m floodguard.features.reduction --override configs/smoke.yaml
"""
from __future__ import annotations

from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from floodguard.utils.common import get_logger, run_paths, save_json, seed_everything

log = get_logger("floodguard.reduction")


class AutoencoderEmbedder(TransformerMixin, BaseEstimator):
    """Symmetric MLP autoencoder; ``transform`` returns the bottleneck embedding.

    ``weights`` is a constructor parameter so that ``sklearn.clone`` keeps a
    trained encoder (fit is a no-op once weights exist).
    """

    def __init__(self, factors: Sequence[str] = (), hidden: Sequence[int] = (32, 16), bottleneck: int = 4,
                 epochs: int = 40, batch_size: int = 2048, lr: float = 2e-3, patience: int = 6,
                 max_rows: int = 400000, random_state: int = 42, weights: dict | None = None):
        self.factors = factors
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.max_rows = max_rows
        self.random_state = random_state
        self.weights = weights

    # -- training (PyTorch) ---------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Any = None) -> "AutoencoderEmbedder":
        if self.weights is not None:
            return self
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        V = X[list(self.factors)].to_numpy(dtype=np.float32)
        if len(V) > self.max_rows:
            V = V[np.random.default_rng(self.random_state).choice(len(V), self.max_rows, replace=False)]
        mean, std = V.mean(0), V.std(0) + 1e-6
        Z = (V - mean) / std
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(Z))
        n_val = max(int(0.1 * len(Z)), 1)
        val, tr = torch.tensor(Z[idx[:n_val]]), torch.tensor(Z[idx[n_val:]])

        dims = [Z.shape[1], *self.hidden, self.bottleneck]
        enc: list[nn.Module] = []
        for i in range(len(dims) - 1):
            enc.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                enc.append(nn.ReLU())
        dec: list[nn.Module] = []
        rdims = dims[::-1]
        for i in range(len(rdims) - 1):
            dec.append(nn.Linear(rdims[i], rdims[i + 1]))
            if i < len(rdims) - 2:
                dec.append(nn.ReLU())
        encoder, decoder = nn.Sequential(*enc), nn.Sequential(*dec)
        model = nn.Sequential(encoder, decoder)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            model.train()
            perm = torch.randperm(len(tr))
            total = 0.0
            for start in range(0, len(tr), self.batch_size):
                batch = tr[perm[start:start + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            model.eval()
            with torch.no_grad():
                v = loss_fn(model(val), val).item()
            history["train_loss"].append(total / len(tr))
            history["val_loss"].append(v)
            if v < best - 1e-5:
                best, wait = v, 0
                best_state = {k: t.detach().clone() for k, t in encoder.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        encoder.load_state_dict(best_state)
        layers = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                  for m in encoder if isinstance(m, nn.Linear)]
        self.weights = {"mean": mean, "std": std, "layers": layers}
        self.history_ = history
        return self

    # -- inference (NumPy) ----------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.weights is None:
            raise RuntimeError("AutoencoderEmbedder is not fitted")
        h = (X[list(self.factors)].to_numpy(dtype=np.float32) - self.weights["mean"]) / self.weights["std"]
        layers = self.weights["layers"]
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < len(layers) - 1:
                h = np.maximum(h, 0)
        return pd.DataFrame(h, columns=[f"ae_{i}" for i in range(h.shape[1])], index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray([f"ae_{i}" for i in range(self.bottleneck)], dtype=object)


def load_autoencoder(cfg) -> AutoencoderEmbedder | None:
    path = run_paths(cfg).reduction / "autoencoder.joblib"
    return joblib.load(path) if path.exists() else None


def _classes(y: np.ndarray, cutoffs: Sequence[float]) -> np.ndarray:
    return np.digitize(y, cutoffs)


def run(cfg) -> dict:
    from floodguard.data.prepare import load_prepared

    seed_everything(cfg.project.seed)
    paths, data, rcfg, seed = run_paths(cfg), load_prepared(cfg), cfg.reduction, cfg.project.seed
    factors = list(cfg.features.factors)
    train = data.train
    X = train[factors].fillna(train[factors].median())
    y = train[data.target].to_numpy()
    classes = _classes(y, data.cutoffs)
    scaler = StandardScaler().fit(X)
    rng = np.random.default_rng(seed)
    summary: dict[str, Any] = {}

    if rcfg.pca:
        pca = PCA(random_state=seed).fit(scaler.transform(X))
        idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
        proj = pca.transform(scaler.transform(X.iloc[idx]))[:, :2]
        summary["pca"] = {
            "explained_variance_ratio": pca.explained_variance_ratio_.round(5),
            "cumulative": np.cumsum(pca.explained_variance_ratio_).round(5),
            "components_for_90pct": int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1),
            "loadings_pc1": dict(zip(factors, pca.components_[0].round(4))),
            "loadings_pc2": dict(zip(factors, pca.components_[1].round(4))),
        }
        np.savez(paths.reduction / "pca_points.npz", points=proj, classes=classes[idx], target=y[idx],
                 loadings=pca.components_[:2], factors=np.array(factors))
        log.info("PCA: %d components explain 90%% of variance (PC1 %.1f%%).",
                 summary["pca"]["components_for_90pct"], 100 * pca.explained_variance_ratio_[0])

    for method in ("tsne", "umap"):
        mcfg = rcfg[method]
        if not mcfg.enabled:
            continue
        idx = rng.choice(len(X), size=min(int(mcfg.sample), len(X)), replace=False)
        Z = scaler.transform(X.iloc[idx])
        try:
            if method == "tsne":
                emb = TSNE(n_components=2, perplexity=float(mcfg.perplexity), init="pca", learning_rate="auto",
                           random_state=seed).fit_transform(Z)
            else:
                import umap

                emb = umap.UMAP(n_neighbors=int(mcfg.n_neighbors), min_dist=float(mcfg.get("min_dist", 0.1)),
                                random_state=seed).fit_transform(Z)
        except ImportError:
            log.warning("%s skipped (package not installed).", method.upper())
            continue
        np.savez(paths.reduction / f"{method}.npz", embedding=emb, classes=classes[idx], target=y[idx])
        summary[method] = {"rows": int(len(idx))}
        log.info("%s projection computed on %d rows.", method.upper(), len(idx))

    acfg = rcfg.autoencoder
    if acfg.enabled:
        try:
            ae = AutoencoderEmbedder(factors=factors, hidden=list(acfg.hidden), bottleneck=int(acfg.bottleneck),
                                     epochs=int(acfg.epochs), batch_size=int(acfg.batch_size), lr=float(acfg.lr),
                                     patience=int(acfg.patience), max_rows=int(acfg.max_rows), random_state=seed)
            ae.fit(X)
            joblib.dump(ae, paths.reduction / "autoencoder.joblib")
            idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
            emb = ae.transform(X.iloc[idx]).to_numpy()
            np.savez(paths.reduction / "ae_embedding.npz", embedding=emb, classes=classes[idx], target=y[idx])
            save_json(ae.history_, paths.reduction / "ae_history.json")
            summary["autoencoder"] = {"epochs_run": len(ae.history_["val_loss"]),
                                      "best_val_mse": float(min(ae.history_["val_loss"])), "bottleneck": int(acfg.bottleneck)}
            log.info("Autoencoder: %d epochs, best val MSE %.4f.", summary["autoencoder"]["epochs_run"],
                     summary["autoencoder"]["best_val_mse"])
        except ImportError:
            log.warning("Autoencoder skipped (PyTorch not installed).")

    save_json(summary, paths.reduction / "reduction_summary.json")
    return summary


if __name__ == "__main__":
    from floodguard.utils.config import cli

    config, _ = cli(__doc__)
    run(config)
