"""
main.py

Orchestrates the full TRS-ICU pipeline:
  1. Data loading (real eICU or synthetic demo)
  2. Preprocessing / sequence building
  3. Model training
  4. Evaluation
  5. Counterfactual inference demo

Usage
-----
# Quick synthetic demo (no data needed):
    python main.py --demo

# Real eICU data (from a local directory):
    python main.py --data-dir /path/to/eicu/csvs

# Real eICU data downloaded from Google Drive on the fly:
    python main.py --download

# Limit to N patients to save memory:
    python main.py --data-dir /path/to/eicu --sample-patients 500

# Save / load trained model:
    python main.py --demo --save-model model.pt
    python main.py --demo --load-model model.pt
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
        description="TRS-ICU: Treatment Response Simulator for ICU"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--demo",
        action="store_true",
        help="Run on synthetic data (no eICU files required).",
    )
    mode.add_argument(
        "--data-dir",
        metavar="DIR",
        help="Path to a directory containing eICU CSV/CSV.GZ files.",
    )
    mode.add_argument(
        "--download",
        action="store_true",
        help="Download eICU data from Google Drive (requires gdown).",
    )
    parser.add_argument(
        "--sample-patients",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N patients from the dataset (for speed).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size (default: 128).",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="GRU hidden size (default: 64).",
    )
    parser.add_argument(
        "--save-model",
        metavar="PATH",
        default=None,
        help="Save trained model + normalisation stats to this .pt file.",
    )
    parser.add_argument(
        "--load-model",
        metavar="PATH",
        default=None,
        help="Load a pre-trained model from this .pt file (skips training).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_model(path: str, model, map_mean: float, map_std: float) -> None:
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
        path,
    )
    print(f"Model saved to '{path}'.")


def load_model(path: str):
    from model.gru_model import TRSModel

    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    model = TRSModel(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Model loaded from '{path}'.")
    return model, ckpt["map_mean"], ckpt["map_std"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Build sequences
    # ------------------------------------------------------------------ #
    if args.load_model:
        # If loading a pre-trained model we only need a small demo sequence
        X = y = treatment_labels = patient_ids = patient_info = None
    elif args.demo or (not args.data_dir and not args.download):
        print("=" * 60)
        print("Running in DEMO mode with synthetic data.")
        print("=" * 60)
        from utils.helpers import make_synthetic_sequences

        X, y, treatment_labels, patient_ids = make_synthetic_sequences(
            n_patients=args.sample_patients or 300,
            seed=42,
        )
        patient_info = None
        print(f"Synthetic sequences: X={X.shape}, y={y.shape}")
    else:
        print("=" * 60)
        print("Loading eICU data …")
        print("=" * 60)
        from data.loader import load_eicu_data
        from data.preprocessor import preprocess_all

        # sample_n caps rows per CSV file to limit memory; sample_patients
        # then selects a subset of unique patients from the loaded rows.
        dataframes = load_eicu_data(
            data_dir=args.data_dir,
            download=args.download,
            sample_n=50_000 if args.sample_patients else None,
        )
        X, y, treatment_labels, patient_ids, patient_info = preprocess_all(
            dataframes,
            sample_patients=args.sample_patients,
        )

    # ------------------------------------------------------------------ #
    # 2. Train or load model
    # ------------------------------------------------------------------ #
    if args.load_model:
        model, map_mean, map_std = load_model(args.load_model)
    else:
        print("\nTraining GRU model …")
        from model.train import train_model

        model, history, map_mean, map_std = train_model(
            X, y,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        if args.save_model:
            save_model(args.save_model, model, map_mean, map_std)

    # ------------------------------------------------------------------ #
    # 3. Evaluate
    # ------------------------------------------------------------------ #
    if X is not None:
        print("\nEvaluating on training set (as a sanity check) …")
        from model.train import evaluate_model

        evaluate_model(model, X, y, map_mean, map_std)

    # ------------------------------------------------------------------ #
    # 4. Counterfactual demo
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Counterfactual inference demo")
    print("=" * 60)

    from inference.counterfactual import predict_counterfactuals
    from utils.helpers import sequence_from_map_values, TREATMENT_LABELS

    demo_map = [72.0, 70.0, 68.0, 66.0, 64.0, 62.0]
    print(f"Input MAP sequence (mmHg): {demo_map}")

    sequence = sequence_from_map_values(demo_map, treatment_label=0)
    result = predict_counterfactuals(sequence, model, map_mean=map_mean, map_std=map_std)

    print("\nPredicted MAP trajectories (next 6 hours):")
    for name, traj in result.trajectories.items():
        label = TREATMENT_LABELS.get(name, name)
        traj_str = "  ".join(f"{v:.1f}" for v in traj)
        print(f"  {label:15s}: {traj_str}")

    best_label = TREATMENT_LABELS.get(result.best_treatment, result.best_treatment)
    print(f"\n→ Best treatment: {best_label} (mean MAP ≈ {result.best_mean_map:.1f} mmHg)")

    print("\nDone! To launch the Streamlit app run:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
