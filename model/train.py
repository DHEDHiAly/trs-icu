"""
model/train.py

Training, validation, and evaluation utilities for the TRS-ICU GRU model.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from .gru_model import TRSModel


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _make_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def _split_dataset(
    dataset: TensorDataset,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    return random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )


# ---------------------------------------------------------------------------
# MAP normalisation
# ---------------------------------------------------------------------------

def compute_map_stats(X: np.ndarray) -> Tuple[float, float]:
    """Return (mean, std) of the MAP channel (column 0) in X."""
    map_vals = X[:, :, 0].flatten()
    return float(map_vals.mean()), float(map_vals.std() + 1e-8)


def normalise(
    X: np.ndarray,
    y: np.ndarray,
    mean: float,
    std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score normalise the MAP channel of X and all of y."""
    X_norm = X.copy()
    X_norm[:, :, 0] = (X_norm[:, :, 0] - mean) / std
    y_norm = (y - mean) / std
    return X_norm, y_norm


def denormalise(y_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
    return y_norm * std + mean


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    hidden_size: int = 64,
    num_layers: int = 2,
    pred_len: int = 6,
    dropout: float = 0.2,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[TRSModel, Dict[str, float], float, float]:
    """Train the GRU model.

    Parameters
    ----------
    X : np.ndarray, shape (N, seq_len, 2)
    y : np.ndarray, shape (N, pred_len)
    ... (other hyperparameters)

    Returns
    -------
    model : trained TRSModel (on *device*)
    history : dict with 'train_loss' and 'val_loss' lists
    map_mean, map_std : normalisation statistics (needed for inference)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Normalise MAP channel
    map_mean, map_std = compute_map_stats(X)
    X_norm, y_norm = normalise(X, y, map_mean, map_std)

    dataset = _make_tensor_dataset(X_norm, y_norm)
    train_ds, val_ds = _split_dataset(dataset, val_fraction=val_fraction)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TRSModel(
        input_size=X.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        pred_len=pred_len,
        dropout=dropout,
    ).to(dev)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---- Validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        scheduler.step(v_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={t_loss:.4f}  val_loss={v_loss:.4f}")

    return model, history, map_mean, map_std


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: TRSModel,
    X: np.ndarray,
    y: np.ndarray,
    map_mean: float,
    map_std: float,
    device: Optional[str] = None,
    batch_size: int = 256,
) -> Dict[str, float]:
    """Compute MSE and RMSE on a dataset (in original mmHg units).

    Returns
    -------
    dict with keys 'mse' and 'rmse'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    X_norm, y_norm = normalise(X, y, map_mean, map_std)
    dataset = _make_tensor_dataset(X_norm, y_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds_norm, targets_norm = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(dev)
            preds_norm.append(model(xb).cpu().numpy())
            targets_norm.append(yb.numpy())

    preds = denormalise(np.concatenate(preds_norm), map_mean, map_std)
    targets = denormalise(np.concatenate(targets_norm), map_mean, map_std)

    mse = float(np.mean((preds - targets) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(preds - targets)))

    print(f"Evaluation — MSE={mse:.2f}  RMSE={rmse:.2f}  MAE={mae:.2f} (mmHg)")
    return {"mse": mse, "rmse": rmse, "mae": mae}
