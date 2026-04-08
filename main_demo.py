"""
main_demo.py
============
Hackathon-ready end-to-end demo for TRS-ICU.

Runs the *full* pipeline on synthetic data — no real eICU files required:

  Step 1 – Auto-generate minimal synthetic eICU CSVs (if data/eicu/ is empty)
  Step 2 – Load CSVs with data/loader.py
  Step 3 – Preprocess into sequences with data/preprocessor.py  → X (N,6,2), y (N,6)
  Step 4 – Train a demo GRU model with z-score MAP normalisation
  Step 5 – Evaluate model on training set (sanity check metrics)
  Step 6 – Run counterfactual inference (no treatment / fluids / vasopressor)
  Step 7 – Determine best treatment and print recommendations
  Step 8 – Print instructions for the Streamlit app

Usage
-----
    python main_demo.py                  # full demo, 200 synthetic patients
    python main_demo.py --n-patients 50  # faster, smaller dataset
    python main_demo.py --epochs 5       # fewer epochs (quick iteration)
    python main_demo.py --save-model demo_model.pt   # persist trained weights
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRS-ICU hackathon demo — runs end-to-end on synthetic data."
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=200,
        metavar="N",
        help="Number of synthetic patients to generate (default: 200).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64).",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="GRU hidden units (default: 32 — small for fast demo).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Path to eICU CSV directory. Defaults to data/eicu/ "
             "(auto-generated if empty).",
    )
    parser.add_argument(
        "--save-model",
        metavar="PATH",
        default=None,
        help="Save trained model weights + normalisation stats to PATH (.pt).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation (default: 42).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Banner helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _step(n: int, desc: str) -> None:
    print(f"\n[Step {n}] {desc}")
    print("-" * 48)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve data directory relative to this script
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(repo_root, "data", "eicu")

    _banner("TRS-ICU · Hackathon Demo")
    print("Running the full pipeline on SYNTHETIC data.")
    print(f"  Synthetic patients : {args.n_patients}")
    print(f"  Training epochs    : {args.epochs}")
    print(f"  GRU hidden size    : {args.hidden_size}")
    print(f"  Data directory     : {data_dir}")

    # ------------------------------------------------------------------ #
    # Step 1 – Generate synthetic eICU CSVs (if data/eicu/ is empty)     #
    # ------------------------------------------------------------------ #
    _step(1, "Generate synthetic eICU CSVs (if data/eicu/ is empty)")

    from data.synthetic import generate_synthetic_eicu_csvs

    # Check whether CSVs already exist
    existing_csvs = [
        f for f in os.listdir(data_dir)
        if f.endswith((".csv", ".csv.gz")) and os.path.isfile(os.path.join(data_dir, f))
    ] if os.path.isdir(data_dir) else []

    if existing_csvs:
        print(f"Found {len(existing_csvs)} CSV file(s) in '{data_dir}' — skipping generation.")
    else:
        print(f"No CSV files found in '{data_dir}'. Generating synthetic data …")
        generate_synthetic_eicu_csvs(
            data_dir=data_dir,
            n_patients=args.n_patients,
            seed=args.seed,
        )

    # ------------------------------------------------------------------ #
    # Step 2 – Load CSVs with data/loader.py                              #
    # ------------------------------------------------------------------ #
    _step(2, "Load CSVs with data/loader.py")

    from data.loader import load_eicu_data

    # load_eicu_data will auto-generate synthetic CSVs if dir is still empty
    dataframes = load_eicu_data(data_dir=data_dir)

    print(f"\nLoaded {len(dataframes)} DataFrames:")
    for name, df in dataframes.items():
        print(f"  {name:20s}: shape={df.shape}")

    # ------------------------------------------------------------------ #
    # Step 3 – Preprocess into sequences with data/preprocessor.py        #
    #          Output: X (N, 6, 2) — [MAP, treatment], y (N, 6) — MAP    #
    # ------------------------------------------------------------------ #
    _step(3, "Preprocess sequences  →  X (N, 6, 2)  /  y (N, 6)")

    from data.preprocessor import preprocess_all

    X, y, treatment_labels, patient_ids, patient_info = preprocess_all(dataframes)

    print(f"\nSequence arrays:")
    print(f"  X shape          : {X.shape}   (samples × seq_len × features)")
    print(f"  y shape          : {y.shape}   (samples × pred_len)")
    print(f"  treatment counts : {np.bincount(treatment_labels, minlength=3)}"
          "  [no_treatment, fluids, vasopressor]")
    print(f"  MAP range (X)    : {X[:, :, 0].min():.1f} – {X[:, :, 0].max():.1f} mmHg")
    print(f"  MAP range (y)    : {y.min():.1f} – {y.max():.1f} mmHg")
    print(f"  Patients in info : {len(patient_info)}")

    assert X.shape[1] == 6, f"Expected seq_len=6, got {X.shape[1]}"
    assert y.shape[1] == 6, f"Expected pred_len=6, got {y.shape[1]}"
    assert X.shape[2] == 2, f"Expected 2 features [MAP, treatment], got {X.shape[2]}"
    print("\n✓ Shape assertions passed.")

    # ------------------------------------------------------------------ #
    # Step 4 – Train a demo GRU model with z-score MAP normalisation      #
    # ------------------------------------------------------------------ #
    _step(4, "Train demo GRU model (z-score MAP normalisation)")

    from model.train import train_model

    model, history, map_mean, map_std = train_model(
        X, y,
        hidden_size=args.hidden_size,
        num_layers=1,          # single GRU layer — lightweight for demo
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
    )

    print(f"\nNormalisation stats  — MAP mean={map_mean:.2f} mmHg, std={map_std:.2f} mmHg")
    final_train = history["train_loss"][-1]
    final_val   = history["val_loss"][-1]
    print(f"Final losses         — train={final_train:.4f}  val={final_val:.4f}")

    # Optionally save the trained model
    if args.save_model:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "map_mean": map_mean,
                "map_std": map_std,
                "config": {
                    "input_size": model.gru.input_size,
                    "hidden_size": model.hidden_size,
                    "num_layers": model.num_layers,
                    "pred_len": model.pred_len,
                },
            },
            args.save_model,
        )
        print(f"Model saved to '{args.save_model}'.")

    # ------------------------------------------------------------------ #
    # Step 5 – Evaluate on training set (sanity-check metrics)            #
    # ------------------------------------------------------------------ #
    _step(5, "Evaluate model (sanity-check on training set)")

    from model.train import evaluate_model

    metrics = evaluate_model(model, X, y, map_mean, map_std)
    print(f"  MSE  = {metrics['mse']:.2f} mmHg²")
    print(f"  RMSE = {metrics['rmse']:.2f} mmHg")
    print(f"  MAE  = {metrics['mae']:.2f} mmHg")

    # ------------------------------------------------------------------ #
    # Step 6 – Counterfactual inference                                   #
    #          Three arms: no treatment / fluids / vasopressor            #
    # ------------------------------------------------------------------ #
    _step(6, "Counterfactual inference — 3 treatment arms")

    from inference.counterfactual import predict_counterfactuals
    from utils.helpers import sequence_from_map_values, TREATMENT_LABELS

    # Representative demo patient: descending MAP (hypotensive trend)
    demo_map = [72.0, 70.0, 68.0, 66.0, 64.0, 62.0]
    print(f"\nDemo patient — past 6 h MAP (mmHg): {demo_map}")

    sequence = sequence_from_map_values(demo_map, treatment_label=0)
    result = predict_counterfactuals(sequence, model, map_mean=map_mean, map_std=map_std)

    print("\nPredicted MAP trajectories (next 6 hours):")
    header = f"  {'Treatment':15s}  " + "  ".join(f"+{i+1}h" for i in range(6))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, traj in result.trajectories.items():
        label = TREATMENT_LABELS.get(name, name)
        traj_str = "  ".join(f"{v:5.1f}" for v in traj)
        marker = " ★" if name == result.best_treatment else ""
        print(f"  {label:15s}  {traj_str}{marker}")

    # ------------------------------------------------------------------ #
    # Step 7 – Determine and print best treatment recommendation          #
    # ------------------------------------------------------------------ #
    _step(7, "Treatment recommendation")

    best_label = TREATMENT_LABELS.get(result.best_treatment, result.best_treatment)
    print(f"\n  → Recommended treatment : {best_label}")
    print(f"     Predicted mean MAP    : {result.best_mean_map:.1f} mmHg")
    if result.best_mean_map >= 65.0:
        print("     Status               : ✓ Above 65 mmHg clinical target")
    else:
        print("     Status               : ⚠ Below 65 mmHg — consider escalation")

    # ------------------------------------------------------------------ #
    # Step 8 – Streamlit app launch instructions                          #
    # ------------------------------------------------------------------ #
    _step(8, "Launch the interactive Streamlit app")

    print(
        "\n  The Streamlit app provides:\n"
        "    • Sidebar sliders for past MAP input (synthetic defaults)\n"
        "    • Interactive Plotly chart — 3 counterfactual trajectories\n"
        "    • Treatment recommendation callout\n"
        "    • 65 mmHg MAP threshold reference line\n"
        "\n  Run with:\n"
        "    streamlit run app/streamlit_app.py\n"
    )

    _banner("Demo complete!")
    print("All steps passed.  The TRS-ICU pipeline is fully operational.\n")


if __name__ == "__main__":
    main()
