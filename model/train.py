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


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 300,
    alpha: float = 0.05,
    seed: int = 42,
    bootstrap_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap CI for mean trajectory over the first axis of values (N, T).

    If bootstrap_indices is provided, it must have shape (n_boot, N) and will
    be reused so multiple metrics share identical resamples.
    """
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    boot_means = []
    if bootstrap_indices is None:
        bootstrap_indices = rng.integers(0, n, size=(n_boot, n))
    for idx in bootstrap_indices:
        boot_means.append(values[idx].mean(axis=0))
    boot = np.stack(boot_means, axis=0)
    mean = values.mean(axis=0)
    lo = np.quantile(boot, alpha / 2.0, axis=0)
    hi = np.quantile(boot, 1.0 - alpha / 2.0, axis=0)
    return mean, lo, hi


def _distribution_overlap_ratio(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    """Histogram overlap ratio in [0, 1] for one-dimensional samples."""
    low = float(min(a.min(), b.min()))
    high = float(max(a.max(), b.max()))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 1.0
    hist_a, edges = np.histogram(a, bins=bins, range=(low, high), density=True)
    hist_b, _ = np.histogram(b, bins=edges, density=True)
    widths = np.diff(edges)
    overlap = float(np.sum(np.minimum(hist_a, hist_b) * widths))
    return max(0.0, min(1.0, overlap))


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
    cf_margin_mmhg: float = 0.75,
    smooth_loss_weight: float = 0.02,
    calibration_loss_weight: float = 0.02,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[TRSModel, Dict[str, float], float, float]:
    """Train the recurrent model with ranking and temporal consistency losses.

    Parameters
    ----------
    X : np.ndarray, shape (N, seq_len, 2)
        Column 0: MAP values, Column 1: treatment index (0/1/2) as float.
    y : np.ndarray, shape (N, pred_len)
    embed_dim : int
        Dimension of the learned treatment embedding (>= 4).
    cf_loss_weight : float
        Weight for the hinge-style ranking counterfactual loss.
    cf_margin_mmhg : float
        Ranking margin in mmHg used in:
            max(0, y_none - y_fluids + margin)
            max(0, y_fluids - y_vasopressor + margin)
    smooth_loss_weight : float
        Weight for temporal smoothness penalty on counterfactual separation:
            mean(|Delta_t - Delta_{t-1}|),
        where Delta_t = |y_vaso(t) - y_none(t)|.
    calibration_loss_weight : float
        Weight for calibration term that keeps arm trajectories within observed
        MAP distribution statistics (mean and variance).
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

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "skipped_batches": []}

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        instability_events_epoch = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            pred = model(xb, check_stability=True, stability_mode="clamp_detach")
            instability_events_epoch += model.consume_instability_events()

            main_loss = criterion(pred, yb)
            if not torch.isfinite(main_loss):
                instability_events_epoch += 1
                continue

            # Counterfactual consistency loss: encourage divergence between arms.
            # Build a single 3x stacked batch using repeat (one copy per arm).
            if cf_loss_weight > 0:
                b = xb.shape[0]
                xb_stack = xb.repeat(3, 1, 1)
                # Set treatment column to 0, 1, 2 for the three segments
                treat_vals = torch.arange(3, dtype=xb.dtype, device=dev).repeat_interleave(b)
                xb_stack[:, :, 1] = treat_vals.unsqueeze(1)
                preds_all = model(
                    xb_stack,
                    check_stability=True,
                    stability_mode="clamp_detach",
                )
                instability_events_epoch += model.consume_instability_events()

                pred_none, pred_fluid, pred_vaso = preds_all.chunk(3, dim=0)

                # Margin is specified in mmHg; convert to normalized MAP space.
                margin = cf_margin_mmhg / (map_std + 1e-8)
                rank_loss = (
                    torch.relu(pred_none - pred_fluid + margin).mean()
                    + torch.relu(pred_fluid - pred_vaso + margin).mean()
                )

                if smooth_loss_weight > 0 and pred_vaso.shape[1] > 1:
                    delta = torch.abs(pred_vaso - pred_none)
                    smooth_loss = torch.abs(delta[:, 1:] - delta[:, :-1]).mean()
                else:
                    smooth_loss = torch.tensor(0.0, device=dev)

                # Keep treatment effects physiologically bounded using untreated arm only.
                yb_mean = yb.mean()
                yb_std = yb.std(unbiased=False)
                cal_loss = (
                    (pred_none.mean() - yb_mean).pow(2)
                    + (pred_none.std(unbiased=False) - yb_std).pow(2)
                )

                if not (torch.isfinite(rank_loss) and torch.isfinite(smooth_loss) and torch.isfinite(cal_loss)):
                    instability_events_epoch += 1
                    continue

                loss = (
                    main_loss
                    + cf_loss_weight * rank_loss
                    + smooth_loss_weight * smooth_loss
                    + calibration_loss_weight * cal_loss
                )
            else:
                loss = main_loss

            if not torch.isfinite(loss):
                instability_events_epoch += 1
                continue

            loss.backward()

            gate_grad_norms = model.intervention_grad_norms()
            if not (gate_grad_norms["Vz"] > 0.0 or gate_grad_norms["Vr"] > 0.0 or gate_grad_norms["Vh"] > 0.0):
                raise AssertionError(
                    "Treatment embedding does not influence any GRU gate: "
                    f"Vz={gate_grad_norms['Vz']:.3e}, "
                    f"Vr={gate_grad_norms['Vr']:.3e}, "
                    f"Vh={gate_grad_norms['Vh']:.3e}"
                )

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

        t_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        v_loss = float(np.mean(val_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["skipped_batches"].append(0.0)
        scheduler.step(v_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d}/{epochs}  train_loss={t_loss:.4f}  "
                f"val_loss={v_loss:.4f}  instability_events={instability_events_epoch}"
            )

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
        dev = next(model.parameters()).device
    else:
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
    assert_sanity: bool = True,

) -> Dict[str, object]:
    """Log full-horizon treatment effects and verify trajectory ordering.

    Computes DeltaMAP_fluids(t) = mean(y_fluids(t) - y_none(t)) and
    DeltaMAP_vaso(t) = mean(y_vaso(t) - y_none(t)) over all timesteps.
    Checks timestep-wise ordering rate for:
        y_vaso(t) >= y_fluids(t) >= y_none(t)
    and requires >= 80% sample-wise ordering for each timestep to count as
    ordered. The global check passes when at least 80% of timesteps are ordered.

    Parameters
    ----------
    model : trained TRSModel
    X : np.ndarray, shape (N, seq_len, 2)  — unnormalised
    map_mean, map_std : normalisation statistics
    n_samples : max samples to use for the diagnostic

    Returns
    -------
    dict containing delta curves, ordering rates, and explicit violations
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

    x_tensor = torch.tensor(X_sub, dtype=torch.float32, device=dev)

    model.eval()
    model.zero_grad(set_to_none=True)
    probe_batch = x_tensor[: min(64, x_tensor.shape[0])]
    probe_out = model.predict_with_treatment(probe_batch, treatment_label=2)
    probe_loss = probe_out.mean()
    probe_loss.backward()
    eval_gate_grad_norms = model.intervention_grad_norms()
    if not (
        eval_gate_grad_norms["Vz"] > 0.0
        or eval_gate_grad_norms["Vr"] > 0.0
        or eval_gate_grad_norms["Vh"] > 0.0
    ):
        raise AssertionError(
            "Eval-time gate gradients are zero: treatment embedding does not "
            "influence transition gates. "
            f"Vz={eval_gate_grad_norms['Vz']:.3e}, "
            f"Vr={eval_gate_grad_norms['Vr']:.3e}, "
            f"Vh={eval_gate_grad_norms['Vh']:.3e}"
        )
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        p_none = model.predict_with_treatment(x_tensor, 0).cpu().numpy() * map_std + map_mean
        p_fluid = model.predict_with_treatment(x_tensor, 1).cpu().numpy() * map_std + map_mean
        p_vaso = model.predict_with_treatment(x_tensor, 2).cpu().numpy() * map_std + map_mean

        p_none_zero = model.predict_with_treatment(
            x_tensor, 0, zero_treatment_embedding=True
        ).cpu().numpy() * map_std + map_mean
        p_fluid_zero = model.predict_with_treatment(
            x_tensor, 1, zero_treatment_embedding=True
        ).cpu().numpy() * map_std + map_mean
        p_vaso_zero = model.predict_with_treatment(
            x_tensor, 2, zero_treatment_embedding=True
        ).cpu().numpy() * map_std + map_mean

        h_none = model.last_hidden_with_treatment(x_tensor, 0).cpu().numpy()
        h_none_zero = model.last_hidden_with_treatment(
            x_tensor, 0, zero_treatment_embedding=True
        ).cpu().numpy()

    delta_fluid = p_fluid - p_none
    delta_vaso = p_vaso - p_none
    delta_fluid_curve = np.mean(delta_fluid, axis=0)
    delta_vaso_curve = np.mean(delta_vaso, axis=0)

    # Delta curves must reflect temporal dynamics, not constant offsets.
    fluid_temporal_var = float(np.std(np.diff(delta_fluid_curve)))
    vaso_temporal_var = float(np.std(np.diff(delta_vaso_curve)))
    if assert_sanity and max(fluid_temporal_var, vaso_temporal_var) <= 1e-4:
        raise AssertionError(
            "Counterfactual DeltaMAP is effectively a constant offset over time; "
            f"std(diff Delta fluids)={fluid_temporal_var:.3e}, "
            f"std(diff Delta vaso)={vaso_temporal_var:.3e}"
        )

    # Nonlinearity over time: ensure curve itself is not flat offset-like.
    if assert_sanity and max(float(np.std(delta_fluid_curve)), float(np.std(delta_vaso_curve))) <= 5e-2:
        raise AssertionError(
            "DeltaMAP curves are too flat across time; likely monotonic-offset shortcut."
        )

    # Ablation: when a_t is forced to zero, trajectories should collapse.
    ablation_gap = max(
        float(np.max(np.abs(p_none_zero - p_fluid_zero))),
        float(np.max(np.abs(p_none_zero - p_vaso_zero))),
        float(np.max(np.abs(p_fluid_zero - p_vaso_zero))),
    )
    if assert_sanity and ablation_gap > 1e-5:
        raise AssertionError(
            "Treatment ablation failed: trajectories do not collapse when a_t=0. "
            f"max_gap={ablation_gap:.3e} mmHg"
        )

    ordering_mask = (p_vaso >= p_fluid) & (p_fluid >= p_none)
    ordering_rate_by_t = np.mean(ordering_mask, axis=0)
    timestep_ordered = ordering_rate_by_t >= 0.80
    ordered_timestep_ratio = float(np.mean(timestep_ordered))
    ordering_ok = bool(ordered_timestep_ratio >= 0.80)

    # Bootstrap confidence intervals on DeltaMAP curves using shared row-level
    # resampling to preserve paired treatment assignment structure per trajectory.
    rng_boot = np.random.default_rng(42)
    boot_idx = rng_boot.integers(0, delta_fluid.shape[0], size=(300, delta_fluid.shape[0]))
    delta_fluid_mean, delta_fluid_lo, delta_fluid_hi = _bootstrap_mean_ci(
        delta_fluid,
        bootstrap_indices=boot_idx,
    )
    delta_vaso_mean, delta_vaso_lo, delta_vaso_hi = _bootstrap_mean_ci(
        delta_vaso,
        bootstrap_indices=boot_idx,
    )

    # Variance of intervention effects.
    delta_fluid_var = float(np.var(delta_fluid))
    delta_vaso_var = float(np.var(delta_vaso))

    # Overlap ratio between treatment distributions per timestep.
    overlap_nf = [
        _distribution_overlap_ratio(p_none[:, t], p_fluid[:, t])
        for t in range(p_none.shape[1])
    ]
    overlap_fv = [
        _distribution_overlap_ratio(p_fluid[:, t], p_vaso[:, t])
        for t in range(p_none.shape[1])
    ]
    overlap_nv = [
        _distribution_overlap_ratio(p_none[:, t], p_vaso[:, t])
        for t in range(p_none.shape[1])
    ]

    # Sanity assertions required for demo robustness.
    patient_ordering_ratio = np.mean(ordering_mask, axis=1)
    patients_ok_ratio = float(np.mean(patient_ordering_ratio >= 0.70))
    if assert_sanity and patients_ok_ratio < 0.70:
        raise AssertionError(
            "Ordering sanity check failed: <70% patients satisfy "
            "vasopressor>=fluids>=none for >=70% timesteps. "
            f"patients_ok_ratio={patients_ok_ratio:.3f}"
        )

    variance_threshold = 1e-3
    if assert_sanity and max(delta_fluid_var, delta_vaso_var) <= variance_threshold:
        raise AssertionError(
            "DeltaMAP variance is too small; counterfactual simulator is collapsed. "
            f"var_fluids={delta_fluid_var:.3e}, var_vaso={delta_vaso_var:.3e}"
        )

    baseline_gap = float(np.max(np.abs(p_none - p_none_zero)))
    baseline_hidden_gap = float(np.max(np.abs(h_none - h_none_zero)))
    if assert_sanity and (baseline_gap > 1e-5 or baseline_hidden_gap > 1e-5):
        raise AssertionError(
            "Baseline-only trajectory drifted from no-treatment beyond tolerance. "
            f"output_gap={baseline_gap:.3e}, hidden_gap={baseline_hidden_gap:.3e}"
        )

    # Rank stability across early/late horizons to avoid single-offset shortcut.
    split = max(1, ordering_rate_by_t.shape[0] // 2)
    early_order_rate = float(np.mean(ordering_rate_by_t[:split]))
    late_order_rate = float(np.mean(ordering_rate_by_t[split:]))
    if assert_sanity and (early_order_rate < 0.60 or late_order_rate < 0.60):
        raise AssertionError(
            "Rank stability check failed across horizons. "
            f"early={early_order_rate:.3f}, late={late_order_rate:.3f}"
        )

    print("\n--- Counterfactual Treatment Effect Diagnostics (full horizon) ---")
    for t in range(p_none.shape[1]):
        print(
            f"  h+{t+1}: DeltaMAP_fluids={delta_fluid_curve[t]:+6.2f} mmHg, "
            f"DeltaMAP_vasopressor={delta_vaso_curve[t]:+6.2f} mmHg, "
            f"ordering_rate={100.0 * ordering_rate_by_t[t]:5.1f}%"
        )
        print(
            f"       95%CI_fluids=[{delta_fluid_lo[t]:+6.2f}, {delta_fluid_hi[t]:+6.2f}] "
            f"95%CI_vaso=[{delta_vaso_lo[t]:+6.2f}, {delta_vaso_hi[t]:+6.2f}]"
        )

    violating_ts = np.where(~timestep_ordered)[0]
    if len(violating_ts) == 0:
        print("  ✓ Ordering satisfied for all timesteps at >=80% sample rate")
    else:
        print("  ✗ Ordering violations (timestep-level):")
        for t in violating_ts.tolist():
            n_viol = int((~ordering_mask[:, t]).sum())
            print(
                f"    h+{t+1}: {n_viol}/{ordering_mask.shape[0]} samples violate "
                "vasopressor >= fluids >= no-treatment"
            )

    print(f"  Ordered timesteps: {100.0 * ordered_timestep_ratio:.1f}%")
    print(f"  Patients satisfying ordering >=70% horizon: {100.0 * patients_ok_ratio:.1f}%")
    print(f"  Rank stability (early horizon): {100.0 * early_order_rate:.1f}%")
    print(f"  Rank stability (late horizon): {100.0 * late_order_rate:.1f}%")
    print(
        "  Eval gate grad norms: "
        f"Vz={eval_gate_grad_norms['Vz']:.3e}, "
        f"Vr={eval_gate_grad_norms['Vr']:.3e}, "
        f"Vh={eval_gate_grad_norms['Vh']:.3e}"
    )
    print(f"  Temporal Delta variability (fluids): {fluid_temporal_var:.3e}")
    print(f"  Temporal Delta variability (vasopressor): {vaso_temporal_var:.3e}")
    print(f"  Delta variance (fluids): {delta_fluid_var:.3e}")
    print(f"  Delta variance (vasopressor): {delta_vaso_var:.3e}")
    print(f"  Overlap ratio mean (none-fluids): {float(np.mean(overlap_nf)):.3f}")
    print(f"  Overlap ratio mean (fluids-vasopressor): {float(np.mean(overlap_fv)):.3f}")
    print(f"  Overlap ratio mean (none-vasopressor): {float(np.mean(overlap_nv)):.3f}")
    print(f"  Baseline/no-treatment max output gap: {baseline_gap:.3e} mmHg")
    print(f"  Baseline/no-treatment max hidden gap: {baseline_hidden_gap:.3e}")
    print(f"  Treatment-ablation collapse max gap: {ablation_gap:.3e} mmHg")
    print("-" * 70)

    return {
        "delta_fluids_curve": delta_fluid_curve.tolist(),
        "delta_vaso_curve": delta_vaso_curve.tolist(),
        "delta_fluids_ci95": {
            "mean": delta_fluid_mean.tolist(),
            "low": delta_fluid_lo.tolist(),
            "high": delta_fluid_hi.tolist(),
        },
        "delta_vaso_ci95": {
            "mean": delta_vaso_mean.tolist(),
            "low": delta_vaso_lo.tolist(),
            "high": delta_vaso_hi.tolist(),
        },
        "ordering_rate_by_t": ordering_rate_by_t.tolist(),
        "ordered_timestep_ratio": ordered_timestep_ratio,
        "violation_timesteps": (violating_ts + 1).tolist(),
        "patients_ordering_ok_ratio": patients_ok_ratio,
        "early_order_rate": early_order_rate,
        "late_order_rate": late_order_rate,
        "eval_gate_grad_norms": eval_gate_grad_norms,
        "delta_temporal_var_fluids": fluid_temporal_var,
        "delta_temporal_var_vaso": vaso_temporal_var,
        "delta_variance_fluids": delta_fluid_var,
        "delta_variance_vaso": delta_vaso_var,
        "overlap_ratio_none_fluids": overlap_nf,
        "overlap_ratio_fluids_vaso": overlap_fv,
        "overlap_ratio_none_vaso": overlap_nv,
        "baseline_no_treatment_gap": baseline_gap,
        "baseline_no_treatment_hidden_gap": baseline_hidden_gap,
        "ablation_collapse_max_gap": ablation_gap,
        "ordering_ok": ordering_ok,
    }
