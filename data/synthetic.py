"""
data/synthetic.py

Generate minimal synthetic eICU-compatible CSV files for hackathon demos
and development without real patient data.

Three files are written to *data_dir* that mimic the eICU schema used by
``data/loader.py`` and ``data/preprocessor.py``:

  vitalPeriodic.csv  — hourly MAP readings (systemicmean)
  infusionDrug.csv   — vasopressor / fluid infusions
  patient.csv        — demographics

Usage
-----
    from data.synthetic import generate_synthetic_eicu_csvs
    generate_synthetic_eicu_csvs("data/eicu/", n_patients=200)

Or from the command line:
    python -m data.synthetic --data-dir data/eicu/ --n-patients 200
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

# Treatment effect parameters are defined in utils.helpers and imported here so
# both synthetic CSV generation and make_synthetic_sequences share a single
# source of truth, preventing silent drift between the two.
from utils.helpers import (
    _FLUID_DECAY_RANGE,
    _FLUID_EFFECT_RANGE,
    _VASO_DECAY_RANGE,
    _VASO_EFFECT_RANGE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default number of synthetic patients to generate
DEFAULT_N_PATIENTS = 200

# Vasopressor drug names (match the keywords in preprocessor.py)
_VASO_DRUGS = [
    "norepinephrine",
    "epinephrine",
    "dopamine",
    "vasopressin",
]

# Fluid drug names
_FLUID_DRUGS = [
    "Normal Saline",
    "Lactated Ringer's",
    "Albumin 5%",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_eicu_csvs(
    data_dir: str,
    n_patients: int = DEFAULT_N_PATIENTS,
    seed: int = 42,
) -> None:
    """Write three synthetic eICU CSV files to *data_dir*.

    Parameters
    ----------
    data_dir:
        Directory in which to write the CSV files.  Created if it does not
        already exist.
    n_patients:
        Number of synthetic patients.  Use >= 20 to guarantee enough
        sequences after the sliding-window step.
    seed:
        NumPy random seed for reproducibility.

    Generated files
    ---------------
    vitalPeriodic.csv
        24 hourly MAP readings per patient modelled as a mean-reverting
        AR(1) process.  A quarter of patients start with hypotensive
        baselines (45-65 mmHg) to make the treatment effect meaningful.
    infusionDrug.csv
        Treatment groups are randomly assigned using the seeded RNG:
        vasopressors receive norepinephrine/etc. for the first 12 hours;
        fluids receive saline/Ringer's for the first 8 hours.  All three
        treatment labels appear in the training set.
    patient.csv
        Synthetic demographics (age, gender, ethnicity, height, weight,
        unit type, discharge status).
    """
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    pids = list(range(1, n_patients + 1))

    # Randomise treatment groups using the RNG for realistic assignment
    groups = rng.integers(0, 3, size=n_patients).tolist()  # 0=none, 1=vaso, 2=fluids

    # ------------------------------------------------------------------
    # 1. vitalPeriodic — hourly MAP readings (AR-1 process per patient)
    # ------------------------------------------------------------------
    vital_rows = []
    for pid, group in zip(pids, groups):
        # First quarter of patients are hypotensive for demo effect
        if pid <= n_patients // 4:
            base_map = float(rng.uniform(45, 65))
        else:
            base_map = float(rng.uniform(65, 90))

        # Determine treatment effect so the MAP trajectory reflects the
        # causal effect: vasopressor → larger MAP lift; fluids → moderate lift
        if group == 1:   # vasopressor: strong, fast effect
            effect_mag = float(rng.uniform(*_VASO_EFFECT_RANGE))
            decay_rate = float(rng.uniform(*_VASO_DECAY_RANGE))
            treat_start_hour = 1   # treatment begins at hour 1
            noise_std = 2.5
        elif group == 2:  # fluids: moderate, slower effect
            effect_mag = float(rng.uniform(*_FLUID_EFFECT_RANGE))
            decay_rate = float(rng.uniform(*_FLUID_DECAY_RANGE))
            treat_start_hour = 1
            noise_std = 2.0
        else:             # no treatment
            effect_mag = 0.0
            decay_rate = 0.0
            treat_start_hour = 999  # never
            noise_std = 2.0

        # 24 hourly observations; observationoffset in minutes
        for hour in range(24):
            # Mean-reverting AR(1) process centred at 75 mmHg
            base_map = 0.9 * base_map + 0.1 * 75.0 + float(rng.normal(0, 2))
            # Add decaying treatment effect from treat_start_hour onwards
            if hour >= treat_start_hour:
                hours_since = hour - treat_start_hour
                effect = effect_mag * float(np.exp(-decay_rate * hours_since))
            else:
                effect = 0.0
            val = float(np.clip(base_map + effect + rng.normal(0, noise_std), 20, 200))
            vital_rows.append(
                {
                    "patientunitstayid": pid,
                    "observationoffset": hour * 60,   # minutes from admission
                    "systemicmean": round(val, 1),
                }
            )

    vital_df = pd.DataFrame(vital_rows)

    # ------------------------------------------------------------------
    # 2. infusionDrug — vasopressors / fluids
    # ------------------------------------------------------------------
    infusion_rows = []
    for pid, group in zip(pids, groups):
        if group == 1:
            # Vasopressor group: norepinephrine for the first 12 hours
            drug = _VASO_DRUGS[pid % len(_VASO_DRUGS)]
            for offset in range(60, 60 * 13, 60):
                infusion_rows.append(
                    {
                        "patientunitstayid": pid,
                        "infusionoffset": offset,
                        "drugname": drug,
                    }
                )
        elif group == 2:
            # Fluids group: saline / ringers for the first 8 hours
            drug = _FLUID_DRUGS[pid % len(_FLUID_DRUGS)]
            for offset in range(60, 60 * 9, 60):
                infusion_rows.append(
                    {
                        "patientunitstayid": pid,
                        "infusionoffset": offset,
                        "drugname": drug,
                    }
                )
        # group == 0: no treatment (no rows written)

    infusion_df = pd.DataFrame(
        infusion_rows if infusion_rows else
        [{"patientunitstayid": 0, "infusionoffset": 0, "drugname": ""}]
    )

    # ------------------------------------------------------------------
    # 3. patient — synthetic demographics
    # ------------------------------------------------------------------
    ethnicities = ["Caucasian", "African American", "Asian", "Hispanic"]
    unit_types = ["MICU", "SICU", "CCU"]
    discharge_opts = ["Alive", "Alive", "Alive", "Expired"]   # 75 % survival

    patient_df = pd.DataFrame(
        {
            "patientunitstayid": pids,
            "uniquepid": [f"SYNTH{p:04d}" for p in pids],
            "age": rng.integers(40, 86, size=n_patients).tolist(),
            "gender": rng.choice(["Male", "Female"], size=n_patients).tolist(),
            "ethnicity": rng.choice(ethnicities, size=n_patients).tolist(),
            "admissionheight": rng.uniform(155, 190, size=n_patients).round(1).tolist(),
            "admissionweight": rng.uniform(55, 110, size=n_patients).round(1).tolist(),
            "unittype": rng.choice(unit_types, size=n_patients).tolist(),
            "unitdischargestatus": rng.choice(discharge_opts, size=n_patients).tolist(),
            "hospitaldischargestatus": rng.choice(discharge_opts, size=n_patients).tolist(),
        }
    )

    # ------------------------------------------------------------------
    # 4. Write to disk
    # ------------------------------------------------------------------
    vital_path     = os.path.join(data_dir, "vitalPeriodic.csv")
    infusion_path  = os.path.join(data_dir, "infusionDrug.csv")
    patient_path   = os.path.join(data_dir, "patient.csv")

    vital_df.to_csv(vital_path, index=False)
    infusion_df.to_csv(infusion_path, index=False)
    patient_df.to_csv(patient_path, index=False)

    print(f"Synthetic eICU CSVs written to '{data_dir}':")
    print(f"  vitalPeriodic.csv : {len(vital_df):,} rows  ({n_patients} patients × 24 h)")
    print(f"  infusionDrug.csv  : {len(infusion_df):,} rows  (vasopressor/fluids groups)")
    print(f"  patient.csv       : {len(patient_df):,} rows  (demographics)")


# ---------------------------------------------------------------------------
# CLI self-test  (python -m data.synthetic)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="Generate synthetic eICU CSV files.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Target directory (default: a temp dir for self-test).",
    )
    parser.add_argument(
        "--n-patients", type=int, default=50,
        help="Number of synthetic patients (default: 50).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target = args.data_dir
    if target is None:
        tmp = tempfile.mkdtemp()
        target = tmp
        print(f"No --data-dir given; writing to temp dir: {tmp}")

    generate_synthetic_eicu_csvs(target, n_patients=args.n_patients, seed=args.seed)

    # Quick smoke-test: re-load the generated files
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.loader import load_eicu_data

    dfs = load_eicu_data(data_dir=target)
    for name, df in dfs.items():
        print(f"  Loaded '{name}': {df.shape}")

    print("\n✓ data/synthetic self-test passed.")
