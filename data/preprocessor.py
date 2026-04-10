"""
data/preprocessor.py

Extract and preprocess MAP time-series, vasopressor treatments, and patient
demographics from the eICU dataset DataFrames.  Produces labelled sequences
suitable for training the GRU model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQ_LEN = 6          # number of historical timesteps fed as input
PRED_LEN = 6         # number of future timesteps to predict
RESAMPLE_FREQ = "1h" # hourly resampling

# Treatment labels
TREAT_NONE = 0
TREAT_FLUID = 1
TREAT_VASOPRESSOR = 2

# Vasopressor drug names (case-insensitive partial match)
VASOPRESSOR_KEYWORDS = [
    "norepinephrine", "epinephrine", "dopamine", "vasopressin",
    "phenylephrine", "dobutamine", "milrinone",
]

# Fluid drug names (case-insensitive partial match)
FLUID_KEYWORDS = [
    "normal saline", "0.9% sodium chloride", "lactated ringer",
    "albumin", "plasmalyte", "hartmann", "dextrose",
]

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_map_series(
    vital_df: pd.DataFrame,
    patient_ids: Optional[List[int]] = None,
) -> Dict[int, pd.DataFrame]:
    """Return a dict ``{patientunitstayid: df}`` with hourly MAP series.

    The input DataFrame should come from ``vitalPeriodic.csv.gz`` and contain
    at least ``patientunitstayid``, ``observationoffset`` (minutes from ICU
    admission), and ``systemicmean`` (MAP proxy in mmHg).
    """
    required = {"patientunitstayid", "observationoffset", "systemicmean"}
    missing = required - set(vital_df.columns)
    if missing:
        raise ValueError(f"vitalPeriodic DataFrame missing columns: {missing}")

    df = vital_df[["patientunitstayid", "observationoffset", "systemicmean"]].copy()
    df = df.rename(columns={"systemicmean": "map"})
    df = df.dropna(subset=["map"])
    df["map"] = pd.to_numeric(df["map"], errors="coerce")
    df = df.dropna(subset=["map"])

    # Filter to plausible MAP range (20–200 mmHg)
    df = df[(df["map"] >= 20) & (df["map"] <= 200)]

    if patient_ids is not None:
        df = df[df["patientunitstayid"].isin(patient_ids)]

    # Convert offset (minutes) → timedelta, resample to hourly mean
    series_by_patient: Dict[int, pd.DataFrame] = {}
    for pid, grp in df.groupby("patientunitstayid"):
        grp = grp.sort_values("observationoffset").drop_duplicates("observationoffset")
        grp.index = pd.to_timedelta(grp["observationoffset"], unit="min")
        grp = grp[["map"]]
        # Resample to hourly, interpolate gaps ≤ 4 h then forward-fill
        hourly = grp.resample(RESAMPLE_FREQ).mean()
        hourly["map"] = hourly["map"].interpolate(method="time", limit=4)
        hourly["map"] = hourly["map"].ffill().bfill()
        if len(hourly) >= SEQ_LEN + PRED_LEN:
            series_by_patient[int(pid)] = hourly
    return series_by_patient


def extract_treatment_series(
    infusion_df: pd.DataFrame,
    patient_ids: Optional[List[int]] = None,
) -> Dict[int, pd.Series]:
    """Return a dict ``{patientunitstayid: hourly_treatment_series}``.

    Values are integer treatment labels:
      TREAT_NONE (0)        — no recognised infusion
      TREAT_FLUID (1)       — IV fluid detected
      TREAT_VASOPRESSOR (2) — vasopressor detected (takes priority over fluids)

    The input DataFrame should come from ``infusionDrug.csv.gz``.
    """
    required = {"patientunitstayid", "infusionoffset", "drugname"}
    missing = required - set(infusion_df.columns)
    if missing:
        raise ValueError(f"infusionDrug DataFrame missing columns: {missing}")

    df = infusion_df[["patientunitstayid", "infusionoffset", "drugname"]].copy()
    df = df.dropna(subset=["drugname"])
    drug_lower = df["drugname"].str.lower()

    vaso_pattern = "|".join(VASOPRESSOR_KEYWORDS)
    fluid_pattern = "|".join(FLUID_KEYWORDS)

    df["is_vaso"] = drug_lower.str.contains(vaso_pattern, na=False)
    df["is_fluid"] = drug_lower.str.contains(fluid_pattern, na=False) & ~df["is_vaso"]
    df = df[df["is_vaso"] | df["is_fluid"]]

    if patient_ids is not None:
        df = df[df["patientunitstayid"].isin(patient_ids)]

    treatment_by_patient: Dict[int, pd.Series] = {}
    for pid, grp in df.groupby("patientunitstayid"):
        grp = grp.sort_values("infusionoffset")
        grp.index = pd.to_timedelta(grp["infusionoffset"], unit="min")

        # Assign label: vasopressor takes priority (2 > 1 > 0)
        label_series = pd.Series(TREAT_NONE, index=grp.index, name="treatment")
        label_series[grp["is_fluid"]] = TREAT_FLUID
        label_series[grp["is_vaso"]] = TREAT_VASOPRESSOR

        # Resample to hourly: max label per hour preserves priority ordering
        hourly = label_series.resample(RESAMPLE_FREQ).max().fillna(TREAT_NONE)
        treatment_by_patient[int(pid)] = hourly.astype(int)
    return treatment_by_patient


def extract_vasopressor_series(
    infusion_df: pd.DataFrame,
    patient_ids: Optional[List[int]] = None,
) -> Dict[int, pd.Series]:
    """Backward-compatible alias for ``extract_treatment_series``.

    .. deprecated::
        Use ``extract_treatment_series`` which also detects IV fluids.
    """
    return extract_treatment_series(infusion_df, patient_ids=patient_ids)


def extract_patient_info(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with key demographic fields from ``patient.csv.gz``."""
    keep_cols = [
        "patientunitstayid",
        "uniquepid",
        "age",
        "gender",
        "ethnicity",
        "admissionheight",
        "admissionweight",
        "unittype",
        "unitdischargestatus",
        "hospitaldischargestatus",
    ]
    available = [c for c in keep_cols if c in patient_df.columns]
    info = patient_df[available].copy()

    # Coerce age (eICU uses "> 89" for oldest patients)
    if "age" in info.columns:
        info["age"] = (
            info["age"]
            .astype(str)
            .str.replace("> 89", "90", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
    return info.set_index("patientunitstayid") if "patientunitstayid" in info.columns else info


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def build_sequences(
    map_series: Dict[int, pd.DataFrame],
    treatment_series: Optional[Dict[int, pd.Series]] = None,
    seq_len: int = SEQ_LEN,
    pred_len: int = PRED_LEN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Slide a window over each patient's MAP series to build model inputs.

    Parameters
    ----------
    map_series:
        ``{patientunitstayid: hourly_map_df}`` from *extract_map_series*.
    treatment_series:
        ``{patientunitstayid: hourly_treatment_series}`` from
        *extract_vasopressor_series*.  Optional; defaults to TREAT_NONE.
    seq_len:
        Number of historical timesteps in each input window.
    pred_len:
        Number of future timesteps to predict.

    Returns
    -------
    X : np.ndarray shape (N, seq_len, 2)
        Input sequences – last dimension is [MAP, treatment_label].
    y : np.ndarray shape (N, pred_len)
        Target MAP values for the next *pred_len* hours.
    treatment_labels : np.ndarray shape (N,)
        Dominant treatment label for the input window (for stratification).
    patient_ids : list of int
        Patient ID for each sample (for traceability).
    """
    treatment_series = treatment_series or {}

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    treat_list: List[int] = []
    pid_list: List[int] = []

    for pid, map_df in map_series.items():
        map_vals = map_df["map"].values.astype(np.float32)
        n = len(map_vals)
        if n < seq_len + pred_len:
            continue

        # Align treatment labels with MAP time index
        treat_vals = np.zeros(n, dtype=np.int32)
        if pid in treatment_series:
            t_series = treatment_series[pid]
            # Re-index treatment onto the MAP time index
            aligned = t_series.reindex(map_df.index, method="nearest", tolerance=pd.Timedelta("1h"))
            aligned = aligned.fillna(TREAT_NONE)
            treat_vals = aligned.values.astype(np.int32)

        for start in range(n - seq_len - pred_len + 1):
            x_map = map_vals[start : start + seq_len]
            x_treat = treat_vals[start : start + seq_len].astype(np.float32)
            y_map = map_vals[start + seq_len : start + seq_len + pred_len]

            X_list.append(np.stack([x_map, x_treat], axis=-1))  # (seq_len, 2)
            y_list.append(y_map)
            # Dominant treatment in input window
            treat_list.append(int(np.bincount(treat_vals[start : start + seq_len]).argmax()))
            pid_list.append(pid)

    if not X_list:
        raise ValueError("No sequences could be built. Check that your data is loaded correctly.")

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(treat_list, dtype=np.int32),
        pid_list,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def preprocess_all(
    dataframes: Dict[str, pd.DataFrame],
    sample_patients: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], pd.DataFrame]:
    """Run the full preprocessing pipeline on the loaded eICU DataFrames.

    Parameters
    ----------
    dataframes:
        Dictionary returned by ``load_eicu_data``.
    sample_patients:
        If given, randomly sample this many patients to speed up development.

    Returns
    -------
    X, y, treatment_labels, patient_ids, patient_info
    """
    if "patient" not in dataframes:
        raise KeyError("'patient' DataFrame not found. Load patient.csv.gz first.")
    if "vitalPeriodic" not in dataframes:
        raise KeyError("'vitalPeriodic' DataFrame not found. Load vitalPeriodic.csv.gz first.")

    patient_info = extract_patient_info(dataframes["patient"])

    all_pids = patient_info.index.tolist()
    if sample_patients is not None and len(all_pids) > sample_patients:
        rng = np.random.default_rng(42)
        all_pids = rng.choice(all_pids, size=sample_patients, replace=False).tolist()
        patient_info = patient_info.loc[patient_info.index.isin(all_pids)]

    print(f"Building MAP series for {len(all_pids)} patients …")
    map_series = extract_map_series(dataframes["vitalPeriodic"], patient_ids=all_pids)
    print(f"  → {len(map_series)} patients with sufficient MAP data.")

    treatment_series: Dict[int, pd.Series] = {}
    if "infusionDrug" in dataframes:
        print("Building treatment series (vasopressor + fluids) …")
        treatment_series = extract_treatment_series(
            dataframes["infusionDrug"], patient_ids=all_pids
        )
        print(f"  → {len(treatment_series)} patients with treatment data.")

    print("Building sequences …")
    X, y, treatment_labels, patient_ids = build_sequences(map_series, treatment_series)
    print(f"  → {len(X)} sequences (X shape={X.shape}, y shape={y.shape}).")

    return X, y, treatment_labels, patient_ids, patient_info


# ---------------------------------------------------------------------------
# Self-test (python -m data.preprocessor)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    print("Running preprocessor self-test with synthetic eICU data …\n")

    # Build minimal synthetic DataFrames that mimic eICU schema
    rng = np.random.default_rng(0)
    n_patients = 20
    pids = list(range(1, n_patients + 1))

    # vitalPeriodic: 24 hourly readings per patient
    vital_rows = []
    for pid in pids:
        base = float(rng.uniform(60, 90))
        for minute in range(0, 24 * 60, 60):
            val = float(np.clip(base + rng.normal(0, 4), 20, 200))
            vital_rows.append(
                {"patientunitstayid": pid, "observationoffset": minute, "systemicmean": val}
            )
    vital_df = pd.DataFrame(vital_rows)

    # infusionDrug: vasopressor for half the patients
    infusion_rows = [
        {"patientunitstayid": pid, "infusionoffset": 120, "drugname": "norepinephrine"}
        for pid in pids[:n_patients // 2]
    ]
    infusion_df = pd.DataFrame(infusion_rows)

    # patient demographics
    patient_df = pd.DataFrame(
        {
            "patientunitstayid": pids,
            "uniquepid": [f"U{p:04d}" for p in pids],
            "age": rng.integers(40, 85, size=n_patients).tolist(),
            "gender": rng.choice(["Male", "Female"], size=n_patients).tolist(),
            "ethnicity": rng.choice(["Caucasian", "African American", "Asian"], size=n_patients).tolist(),
            "admissionheight": rng.uniform(155, 185, size=n_patients).tolist(),
            "admissionweight": rng.uniform(55, 110, size=n_patients).tolist(),
            "unittype": ["MICU"] * n_patients,
            "unitdischargestatus": ["Alive"] * n_patients,
            "hospitaldischargestatus": ["Alive"] * n_patients,
        }
    )

    dataframes = {
        "vitalPeriodic": vital_df,
        "infusionDrug": infusion_df,
        "patient": patient_df,
    }

    X, y, treatment_labels, patient_ids, patient_info = preprocess_all(dataframes)

    # Verification summary
    print("\n--- Verification ---")
    print(f"X shape           : {X.shape}   (samples × seq_len × features)")
    print(f"y shape           : {y.shape}   (samples × pred_len)")
    print(f"treatment_labels  : {np.bincount(treatment_labels, minlength=3)} (counts per label 0/1/2)")
    print(f"patient_ids count : {len(patient_ids)} sample–patient mappings")
    print(f"patient_info      : {patient_info.shape} rows × columns")
    print(f"MAP range in X    : {X[:, :, 0].min():.1f} – {X[:, :, 0].max():.1f} mmHg")
    print(f"MAP range in y    : {y.min():.1f} – {y.max():.1f} mmHg")
    assert X.shape[1] == SEQ_LEN, f"Expected seq_len={SEQ_LEN}"
    assert y.shape[1] == PRED_LEN, f"Expected pred_len={PRED_LEN}"
    assert X.shape[2] == 2, "Expected 2 features: [MAP, treatment]"
    print("\n✓ Preprocessor self-test passed.")
