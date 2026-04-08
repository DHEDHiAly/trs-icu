"""
data/loader.py

Load eICU CSV and CSV.GZ files into Pandas DataFrames.

Supports loading from:
  - A local directory path
  - A Google Drive folder (via gdown)

Each DataFrame is stored in a dictionary keyed by the file name (without extension).
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List, Optional

import pandas as pd

from data.synthetic import generate_synthetic_eicu_csvs


# ---------------------------------------------------------------------------
# Google Drive folder download helper
# ---------------------------------------------------------------------------

GDRIVE_FOLDER_ID = "12b_rL9mTCUYBDaWU_rwJRqsrNQHaAPJ3"


def _download_gdrive_folder(dest_dir: str) -> str:
    """Download the eICU dataset folder from Google Drive using gdown."""
    try:
        import gdown  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download data from Google Drive. "
            "Install it with: pip install gdown"
        ) from exc

    import gdown

    os.makedirs(dest_dir, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
    gdown.download_folder(url=url, output=dest_dir, quiet=False, use_cookies=False)
    return dest_dir


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_eicu_data(
    data_dir: Optional[str] = None,
    download: bool = False,
    sample_n: Optional[int] = None,
    usecols: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load all CSV / CSV.GZ files in *data_dir* into Pandas DataFrames.

    Parameters
    ----------
    data_dir:
        Path to the directory containing eICU CSV files.  If *None* and
        *download* is *True*, the files are downloaded from Google Drive into
        ``./data/raw``.
    download:
        When *True*, download the dataset folder from Google Drive before
        loading.  Requires ``gdown``.
    sample_n:
        If given, randomly sample at most *sample_n* rows from every file to
        reduce memory usage.
    usecols:
        Mapping of ``{file_stem: [col, ...]}`` to load only a subset of
        columns from specific files.  Files not present in the mapping are
        loaded fully (subject to *sample_n*).

    Returns
    -------
    dict
        ``{file_stem: DataFrame}`` for every CSV/CSV.GZ file found.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "eicu")

    if download:
        _download_gdrive_folder(data_dir)

    data_dir = pathlib.Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist. "
            "Pass download=True to fetch from Google Drive, or provide a valid path."
        )

    # Collect all CSV / CSV.GZ files
    csv_files = sorted(
        list(data_dir.glob("*.csv")) + list(data_dir.glob("*.csv.gz"))
    )

    if not csv_files:
        # Auto-generate synthetic eICU CSVs so the demo works without real data
        print(
            f"No CSV files found in '{data_dir}'. "
            "Generating synthetic eICU CSVs for demo …"
        )
        generate_synthetic_eicu_csvs(str(data_dir))
        csv_files = sorted(
            list(data_dir.glob("*.csv")) + list(data_dir.glob("*.csv.gz"))
        )

    dataframes: Dict[str, pd.DataFrame] = {}
    usecols = usecols or {}

    for fpath in csv_files:
        # Build a clean stem: strip both ".csv" and optional ".gz"
        stem = fpath.name
        for suffix in (".csv.gz", ".csv"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        cols = usecols.get(stem, None)

        print(f"Loading '{fpath.name}' ...", end=" ", flush=True)
        try:
            df = pd.read_csv(fpath, usecols=cols, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            continue

        if sample_n is not None and len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

        dataframes[stem] = df
        print(f"shape={df.shape}  columns={list(df.columns)}")

    return dataframes


# ---------------------------------------------------------------------------
# Self-test (python -m data.loader)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import tempfile
    import io

    parser = argparse.ArgumentParser(description="Test the eICU data loader.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to directory with eICU CSV/CSV.GZ files. "
             "Defaults to data/eicu/ relative to this file.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        metavar="N",
        help="Load at most N rows per file (for quick testing).",
    )
    args = parser.parse_args()

    target_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "eicu")

    if not os.path.isdir(target_dir) or not any(
        f.endswith((".csv", ".csv.gz"))
        for f in os.listdir(target_dir)
        if os.path.isfile(os.path.join(target_dir, f))
    ):
        print(
            f"No CSV files found in '{target_dir}'.\n"
            "Generating a tiny synthetic CSV to demonstrate the loader …\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Build minimal synthetic CSVs that mimic the eICU schema
            vital = pd.DataFrame(
                {
                    "patientunitstayid": [1, 1, 1, 2, 2, 2] * 5,
                    "observationoffset": list(range(0, 360, 60)) * 5,
                    "systemicmean": [72, 70, 68, 66, 64, 62] * 5,
                }
            )
            infusion = pd.DataFrame(
                {
                    "patientunitstayid": [1, 2],
                    "infusionoffset": [60, 120],
                    "drugname": ["norepinephrine", "Normal Saline"],
                }
            )
            patient = pd.DataFrame(
                {
                    "patientunitstayid": [1, 2],
                    "uniquepid": ["A001", "A002"],
                    "age": [65, 72],
                    "gender": ["Male", "Female"],
                    "ethnicity": ["Caucasian", "African American"],
                    "admissionheight": [170, 165],
                    "admissionweight": [80, 75],
                    "unittype": ["MICU", "MICU"],
                    "unitdischargestatus": ["Alive", "Alive"],
                    "hospitaldischargestatus": ["Alive", "Alive"],
                }
            )
            vital.to_csv(os.path.join(tmpdir, "vitalPeriodic.csv"), index=False)
            infusion.to_csv(os.path.join(tmpdir, "infusionDrug.csv"), index=False)
            patient.to_csv(os.path.join(tmpdir, "patient.csv"), index=False)

            dfs = load_eicu_data(data_dir=tmpdir, sample_n=args.sample_n)
    else:
        dfs = load_eicu_data(data_dir=target_dir, sample_n=args.sample_n)

    _MAX_DISPLAY_COLS = 8  # keep output readable for wide tables
    print("\nLoaded DataFrames:")
    for name, df in dfs.items():
        print(f"  {name}: {df.shape}  columns={list(df.columns[:_MAX_DISPLAY_COLS])}")

    print("\n✓ Loader self-test passed.")
