"""
generate_visual.py
==================
Train the TRS-ICU GRU model on synthetic data, run counterfactual inference,
and save a multi-panel PNG to ``results_visual.png``.

Usage
-----
    python generate_visual.py              # saves results_visual.png
    python generate_visual.py --out my.png # custom output path
    python generate_visual.py --patients 8 # show 8 cohort patients
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Allow imports from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import make_synthetic_sequences, sequence_from_map_values, TREATMENT_LABELS
from model.train import train_model, evaluate_model
from inference.counterfactual import predict_counterfactuals

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "no_treatment": "#636EFA",
    "fluids":       "#00CC96",
    "vasopressor":  "#EF553B",
    "history":      "#888888",
    "threshold":    "#FF6B6B",
}

ARMS = ["no_treatment", "fluids", "vasopressor"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_pipeline(n_patients: int = 300, epochs: int = 20, seed: int = 42):
    """Train model and return (model, map_mean, map_std, X, y)."""
    X, y, _, _ = make_synthetic_sequences(n_patients=n_patients, seed=seed)
    model, _, map_mean, map_std = train_model(
        X, y,
        hidden_size=32,
        num_layers=1,
        epochs=epochs,
        batch_size=64,
        verbose=False,
    )
    model.eval()
    return model, map_mean, map_std, X, y


def _counterfactuals_for(X, indices, model, map_mean, map_std):
    """Return list of CounterfactualResult for given sample indices."""
    results = []
    for idx in indices:
        past_map = X[idx, :, 0].astype(float).tolist()
        seq = sequence_from_map_values(past_map, treatment_label=0)
        results.append(
            predict_counterfactuals(seq, model, map_mean=map_mean, map_std=map_std)
        )
    return results


# ── individual trajectory panels ─────────────────────────────────────────────

def _draw_trajectory(ax, past_map, result, patient_no, best_color):
    seq_len = len(past_map)
    pred_len = len(next(iter(result.trajectories.values())))
    x_hist = list(range(-seq_len, 0))
    x_pred = list(range(0, pred_len))

    ax.plot(x_hist, past_map, color=C["history"], lw=1.5, ls="--",
            marker="o", markersize=4, label="Historical MAP")
    ax.axhline(65, color=C["threshold"], ls=":", lw=1.2, alpha=0.7)
    ax.axvline(0, color="#cccccc", ls="--", lw=0.8)

    for arm in ARMS:
        traj = result.trajectories[arm]
        is_best = arm == result.best_treatment
        ax.plot(x_pred, traj,
                color=C[arm],
                lw=2.5 if is_best else 1.5,
                ls="solid" if is_best else "dashed",
                marker="o" if is_best else None,
                markersize=4,
                label=("★ " if is_best else "") + TREATMENT_LABELS[arm],
                zorder=3 if is_best else 2)

    ax.set_title(f"Patient {patient_no}", fontsize=9, pad=3)
    ax.set_ylabel("MAP (mmHg)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-(seq_len + 0.5), pred_len - 0.5)
    ax.set_xticks(list(range(-seq_len, pred_len)))
    ax.set_xticklabels(
        [f"h{t}" for t in range(-seq_len, 0)] + [f"+{t}h" for t in range(1, pred_len + 1)],
        fontsize=6, rotation=45, ha="right"
    )
    ax.set_ylim(50, 100)
    ax.grid(axis="y", alpha=0.3)


# ── mean trajectory ± CI panel ────────────────────────────────────────────────

def _draw_mean_trajectories(ax, results):
    pred_len = len(next(iter(results[0].trajectories.values())))
    x = np.arange(pred_len)

    for arm in ARMS:
        mat = np.stack([r.trajectories[arm] for r in results], axis=0)  # (N, T)
        mean = mat.mean(0)
        ci_lo = np.percentile(mat, 5, axis=0)
        ci_hi = np.percentile(mat, 95, axis=0)
        ax.fill_between(x, ci_lo, ci_hi, color=C[arm], alpha=0.15)
        ax.plot(x, mean, color=C[arm], lw=2.2, label=TREATMENT_LABELS[arm], marker="o", markersize=5)

    ax.axhline(65, color=C["threshold"], ls=":", lw=1.4, alpha=0.8, label="MAP target ≥65 mmHg")
    ax.set_title("Mean Predicted MAP Trajectories (cohort, 90% CI)", fontsize=10)
    ax.set_xlabel("Prediction horizon (hours)", fontsize=9)
    ax.set_ylabel("MAP (mmHg)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"+{t+1}h" for t in x])
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)


# ── treatment effect (Delta MAP) panel ───────────────────────────────────────

def _draw_delta_map(ax, results):
    pred_len = len(next(iter(results[0].trajectories.values())))
    x = np.arange(pred_len)

    for arm in ["fluids", "vasopressor"]:
        deltas = np.stack(
            [r.trajectories[arm] - r.trajectories["no_treatment"] for r in results], axis=0
        )
        mean_d = deltas.mean(0)
        ci_lo = np.percentile(deltas, 10, axis=0)
        ci_hi = np.percentile(deltas, 90, axis=0)
        ax.fill_between(x, ci_lo, ci_hi, color=C[arm], alpha=0.2)
        ax.plot(x, mean_d, color=C[arm], lw=2.2, label=f"Δ {TREATMENT_LABELS[arm]}", marker="s", markersize=5)

    ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_title("Treatment Effect vs. No Treatment (ΔMAP)", fontsize=10)
    ax.set_xlabel("Prediction horizon (hours)", fontsize=9)
    ax.set_ylabel("ΔMAP (mmHg)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"+{t+1}h" for t in x])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


# ── training loss curve panel ─────────────────────────────────────────────────

def _draw_loss_curve(ax, history):
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color="#636EFA", lw=2, label="Train loss")
    ax.plot(epochs, history["val_loss"],   color="#EF553B", lw=2, ls="--", label="Val loss")
    ax.set_title("Training / Validation Loss", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("MSE Loss (normalised)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


# ── best-treatment distribution panel ────────────────────────────────────────

def _draw_recommendation_bar(ax, results):
    counts = {arm: 0 for arm in ARMS}
    for r in results:
        counts[r.best_treatment] = counts.get(r.best_treatment, 0) + 1

    labels = [TREATMENT_LABELS[a] for a in ARMS]
    values = [counts[a] for a in ARMS]
    colors = [C[a] for a in ARMS]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Best Treatment Recommendation per Patient", fontsize=10)
    ax.set_ylabel("Number of patients", fontsize=9)
    ax.set_ylim(0, max(values) * 1.25 + 1)
    ax.grid(axis="y", alpha=0.3)


# ── metrics table panel ───────────────────────────────────────────────────────

def _draw_metrics_table(ax, metrics, map_mean, map_std, n_patients, epochs):
    ax.axis("off")
    data = [
        ["Metric", "Value"],
        ["Patients (synthetic)", str(n_patients)],
        ["Training epochs", str(epochs)],
        ["MAP mean (train)", f"{map_mean:.1f} mmHg"],
        ["MAP std (train)", f"{map_std:.1f} mmHg"],
        ["RMSE", f"{metrics['rmse']:.2f} mmHg"],
        ["MAE", f"{metrics['mae']:.2f} mmHg"],
        ["MSE", f"{metrics['mse']:.2f} mmHg²"],
    ]
    tbl = ax.table(cellText=data[1:], colLabels=data[0],
                   cellLoc="center", loc="center",
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F0F4FF")
        cell.set_edgecolor("#cccccc")
    ax.set_title("Model Summary", fontsize=10, pad=6)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate TRS-ICU results visual")
    parser.add_argument("--out", default="results_visual.png", help="Output PNG path")
    parser.add_argument("--patients", type=int, default=6, help="Cohort patients to display (max 8)")
    parser.add_argument("--n-train", type=int, default=300, help="Synthetic patients for training")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    n_cohort = min(max(args.patients, 2), 8)

    print("=" * 60)
    print("TRS-ICU · Generating results visual")
    print(f"  Training patients : {args.n_train}")
    print(f"  Epochs            : {args.epochs}")
    print(f"  Cohort patients   : {n_cohort}")
    print(f"  Output            : {args.out}")
    print("=" * 60)

    # ── 1. Train model ──────────────────────────────────────────────────────
    print("\n[1/4] Training GRU model on synthetic data …")
    model, map_mean, map_std, X, y = _run_pipeline(
        n_patients=args.n_train, epochs=args.epochs, seed=args.seed
    )

    # ── 2. Evaluate ─────────────────────────────────────────────────────────
    print("[2/4] Evaluating model …")
    metrics = evaluate_model(model, X, y, map_mean, map_std)
    print(f"  RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  MSE={metrics['mse']:.2f}")

    # Retrieve training history for loss curve
    _, history, _, _ = train_model(
        X, y,
        hidden_size=32, num_layers=1,
        epochs=args.epochs, batch_size=64,
        verbose=False,
    )

    # ── 3. Counterfactual inference on cohort ───────────────────────────────
    print("[3/4] Running counterfactual inference …")
    rng = np.random.default_rng(args.seed)
    cohort_idx = rng.choice(len(X), size=n_cohort, replace=False).tolist()
    cf_results = _counterfactuals_for(X, cohort_idx, model, map_mean, map_std)

    # Also run on a larger sample for aggregate panels
    big_idx = rng.choice(len(X), size=min(100, len(X)), replace=False).tolist()
    big_results = _counterfactuals_for(X, big_idx, model, map_mean, map_std)

    # ── 4. Build figure ──────────────────────────────────────────────────────
    print("[4/4] Building figure …")

    # Layout:
    #   • Row(s) 0..nrows_traj-1 : individual patient trajectory panels (3-up per row)
    #   • Row  nrows_traj         : mean trajectory (left) + ΔMAP (right)
    #   • Row  nrows_traj+1       : loss curve (left) + bar (centre) + metrics table (right)

    ncols_traj = 3
    nrows_traj = (n_cohort + ncols_traj - 1) // ncols_traj

    # Use 12 logical columns so trajectory (4-wide each) and analysis (6-wide each) align neatly
    NCOLS = 12
    analysis_row  = nrows_traj        # mean/delta row
    bottom_row    = nrows_traj + 1    # loss/bar/table row
    total_rows    = nrows_traj + 2

    # Row heights: trajectory rows = 1 unit each, analysis rows = 1.2 units
    height_ratios = [1.0] * nrows_traj + [1.2, 1.2]

    fig = plt.figure(figsize=(16, 4.5 * nrows_traj + 5.5 * 2))
    gs = gridspec.GridSpec(
        total_rows, NCOLS,
        figure=fig,
        hspace=0.45,
        wspace=0.5,
        height_ratios=height_ratios,
        top=0.96, bottom=0.04, left=0.06, right=0.97,
    )

    fig.suptitle(
        "TRS-ICU · Treatment Response Simulator — Results Summary",
        fontsize=14, fontweight="bold",
    )

    # — Trajectory panels ——————————————————————————————————————————————————
    traj_col_width = NCOLS // ncols_traj   # 4 columns each
    handles, labels = None, None
    for i, (idx, res) in enumerate(zip(cohort_idx, cf_results)):
        tr, tc = divmod(i, ncols_traj)
        c0 = tc * traj_col_width
        ax = fig.add_subplot(gs[tr, c0 : c0 + traj_col_width])
        _draw_trajectory(ax, X[idx, :, 0].tolist(), res, i + 1, C[res.best_treatment])
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    # Shared legend tucked inside the figure (no bbox overflow)
    if handles:
        fig.legend(
            handles, labels,
            loc="upper right",
            fontsize=8,
            title="Legend",
            title_fontsize=8,
            framealpha=0.9,
            borderpad=0.6,
        )

    # — Middle row: mean trajectory + delta ——————————————————————————————
    ax_mean = fig.add_subplot(gs[analysis_row, :6])
    ax_delta = fig.add_subplot(gs[analysis_row, 6:])
    _draw_mean_trajectories(ax_mean, big_results)
    _draw_delta_map(ax_delta, big_results)

    # — Bottom row: loss + bar + metrics ————————————————————————————————
    ax_loss = fig.add_subplot(gs[bottom_row, :4])
    ax_bar  = fig.add_subplot(gs[bottom_row, 4:8])
    ax_tbl  = fig.add_subplot(gs[bottom_row, 8:])
    _draw_loss_curve(ax_loss, history)
    _draw_recommendation_bar(ax_bar, big_results)
    _draw_metrics_table(ax_tbl, metrics, map_mean, map_std, args.n_train, args.epochs)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out) \
        if not os.path.isabs(args.out) else args.out
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Visual saved to '{out_path}'")


if __name__ == "__main__":
    main()
