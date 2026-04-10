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
    embed_dim: int = 8,
    hidden_size: int = 64,
    num_layers: int = 2,
    pred_len: int = 6,
    dropout: float = 0.2,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    cf_loss_weight: float = 0.05,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[TRSModel, Dict[str, float], float, float]:
    """Train the GRU model with treatment embedding and counterfactual consistency loss.

    Parameters
    ----------
    X : np.ndarray, shape (N, seq_len, 2)
        Column 0: MAP values, Column 1: treatment index (0/1/2) as float.
    y : np.ndarray, shape (N, pred_len)
    embed_dim : int
        Dimension of the learned treatment embedding (>= 4).
    cf_loss_weight : float
        Weight for the counterfactual consistency loss (encourages divergence
        between treatment trajectories).  Set to 0 to disable.
    ... (other hyperparameters)

    Returns
    -------
    model : trained TRSModel (on *device*)
    history : dict with 'train_loss' and 'val_loss' lists
    map_mean, map_std : normalisation statistics (needed for inference)
    """
    if embed_dim < 4:
        raise ValueError(f"embed_dim must be >= 4 for expressive treatment representations, got {embed_dim}")
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
        embed_dim=embed_dim,
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
            main_loss = criterion(pred, yb)

            # Counterfactual consistency loss: encourage divergence between arms.
            # Build a single 3x stacked batch using repeat (one copy per arm).
            if cf_loss_weight > 0:
                b = xb.shape[0]
                xb_stack = xb.repeat(3, 1, 1)
                # Set treatment column to 0, 1, 2 for the three segments
                treat_vals = torch.arange(3, dtype=xb.dtype, device=dev).repeat_interleave(b)
                xb_stack[:, :, 1] = treat_vals.unsqueeze(1)
                preds_all = model(xb_stack)
                pred_none, pred_fluid, pred_vaso = preds_all.chunk(3, dim=0)

                cf_loss = -torch.mean(
                    torch.abs(pred_vaso - pred_none)
                    + torch.abs(pred_fluid - pred_none)
                )
                loss = main_loss + cf_loss_weight * cf_loss
            else:
                loss = main_loss

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


def evaluate_counterfactual_effects(
    model: TRSModel,
    X: np.ndarray,
    map_mean: float,
    map_std: float,
    device: Optional[str] = None,
    n_samples: int = 512,
) -> Dict[str, float]:
    """Log average treatment effects and verify their ordering.

    Computes ΔMAP_fluids = mean(y_fluids - y_none) and
    ΔMAP_vaso = mean(y_vaso - y_none) for the first prediction horizon.

    Parameters
    ----------
    model : trained TRSModel
    X : np.ndarray, shape (N, seq_len, 2)  — unnormalised
    map_mean, map_std : normalisation statistics
    n_samples : max samples to use for the diagnostic

    Returns
    -------
    dict with keys 'delta_fluids_h1', 'delta_vaso_h1', 'ordering_ok'
    """
    # Use the model's current device rather than inferring from CUDA availability
    # so this works correctly for CPU-loaded models (e.g. via load_model).
    if device is None:
        dev = next(model.parameters()).device
    else:
        dev = torch.device(device)

    idx = np.random.default_rng(0).choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sub = X[idx].copy().astype(np.float32)
    X_sub[:, :, 0] = (X_sub[:, :, 0] - map_mean) / (map_std + 1e-8)

    x_tensor = torch.tensor(X_sub, dtype=torch.float32).to(dev)

    model.eval()
    with torch.no_grad():
        p_none = model.predict_with_treatment(x_tensor, 0).cpu().numpy() * map_std + map_mean
        p_fluid = model.predict_with_treatment(x_tensor, 1).cpu().numpy() * map_std + map_mean
        p_vaso = model.predict_with_treatment(x_tensor, 2).cpu().numpy() * map_std + map_mean

    delta_fluid = float(np.mean(p_fluid[:, 0] - p_none[:, 0]))
    delta_vaso = float(np.mean(p_vaso[:, 0] - p_none[:, 0]))
    ordering_ok = bool(delta_vaso > delta_fluid > 0)

    print(f"\n--- Counterfactual Treatment Effect Diagnostics (h+1) ---")
    print(f"  ΔMAP_fluids     = {delta_fluid:+.2f} mmHg")
    print(f"  ΔMAP_vasopressor= {delta_vaso:+.2f} mmHg")
    if ordering_ok:
        print("  ✓ Ordering satisfied: vasopressor > fluids > no-treatment")
    else:
        print(
            "  ✗ WARNING: Expected ΔMAP_vaso > ΔMAP_fluids > 0, "
            f"got {delta_vaso:.2f} > {delta_fluid:.2f} > 0"
        )
    print("-" * 55)

    return {
        "delta_fluids_h1": delta_fluid,
        "delta_vaso_h1": delta_vaso,
        "ordering_ok": ordering_ok,
    }
