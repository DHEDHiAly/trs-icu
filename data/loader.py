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
        data_dir = os.path.join(os.path.dirname(__file__), "raw")

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
        raise FileNotFoundError(
            f"No CSV or CSV.GZ files found in '{data_dir}'."
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
