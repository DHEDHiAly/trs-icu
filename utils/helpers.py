"""
utils/helpers.py

Miscellaneous helper functions for preprocessing, synthetic data generation,
plotting, and sequence construction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Treatment colour map used in plots
TREATMENT_COLORS = {
    "no_treatment": "#636EFA",
    "fluids": "#00CC96",
    "vasopressor": "#EF553B",
}

TREATMENT_LABELS = {
    "no_treatment": "No Treatment",
    "fluids": "Fluids",
    "vasopressor": "Vasopressor",
}


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_synthetic_sequences(
    n_patients: int = 200,
    seq_len: int = 6,
    pred_len: int = 6,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Generate synthetic MAP sequences for development / unit testing.

    For each base patient trajectory, **three counterfactual versions** are
    created—one per treatment arm—with the same observed MAP history but
    different future trajectories reflecting the causal treatment effect:

    * No treatment (0): pure AR(1) continuation around 75 mmHg.
    * Fluids (1): AR(1) plus a positive MAP effect (+1.5–3 mmHg) that decays
      exponentially over ~3–5 prediction steps.
    * Vasopressor (2): AR(1) plus a stronger immediate effect (+3–6 mmHg)
      with faster onset and faster decay, and slightly higher noise.

    This ensures that the same input MAP sequence paired with different
    treatment labels produces distinctly different future trajectories,
    forcing the model to learn causal treatment effects rather than relying
    solely on MAP level.

    Returns
    -------
    X : (N, seq_len, 2)  — column 0: MAP, column 1: treatment index (0/1/2)
    y : (N, pred_len)
    treatment_labels : (N,)
    patient_ids : list of int
    """
    rng = np.random.default_rng(seed)
    X_list, y_list, treat_list, pid_list = [], [], [], []

    for pid in range(n_patients):
        # Generate the observed historical MAP sequence (treatment-agnostic)
        n_history = seq_len + rng.integers(0, 6)
        base_map = float(rng.uniform(55, 85))
        history = [base_map]
        for _ in range(n_history - 1):
            next_val = 0.9 * history[-1] + 0.1 * 75.0 + float(rng.normal(0, 2))
            history.append(float(np.clip(next_val, 30, 150)))

        # Use the last seq_len steps as the common input window
        x_map = np.array(history[-seq_len:], dtype=np.float32)

        # Sample per-patient effect magnitudes and decay rates once, then
        # generate one prediction window per treatment arm
        fluid_mag = float(rng.uniform(1.5, 3.0))
        fluid_decay = float(rng.uniform(0.20, 0.35))   # slower: 3–5 steps
        vaso_mag = float(rng.uniform(3.0, 6.0))
        vaso_decay = float(rng.uniform(0.40, 0.60))    # faster: 1–3 steps

        for treat in range(3):  # 0=none, 1=fluids, 2=vasopressor
            x_treat = np.full(seq_len, float(treat), dtype=np.float32)

            # Simulate future MAP under this treatment
            y_map = []
            last = float(x_map[-1])
            for h in range(pred_len):
                ar_next = 0.9 * last + 0.1 * 75.0
                if treat == 1:
                    effect = fluid_mag * np.exp(-fluid_decay * h)
                    noise_std = 2.0
                elif treat == 2:
                    effect = vaso_mag * np.exp(-vaso_decay * h)
                    noise_std = 2.5
                else:
                    effect = 0.0
                    noise_std = 2.0
                next_val = ar_next + effect + float(rng.normal(0, noise_std))
                next_val = float(np.clip(next_val, 30, 150))
                y_map.append(next_val)
                last = next_val

            X_list.append(np.stack([x_map, x_treat], axis=-1))
            y_list.append(np.array(y_map, dtype=np.float32))
            treat_list.append(treat)
            pid_list.append(pid)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(treat_list, dtype=np.int32),
        pid_list,
    )


# ---------------------------------------------------------------------------
# Sequence construction from raw values
# ---------------------------------------------------------------------------

def sequence_from_map_values(
    map_values: List[float],
    treatment_label: int = 0,
    seq_len: int = 6,
) -> np.ndarray:
    """Build a model input sequence from a list of MAP readings.

    Parameters
    ----------
    map_values : list of floats, length == seq_len
    treatment_label : 0=none, 1=fluids, 2=vasopressor
    seq_len : expected length

    Returns
    -------
    np.ndarray of shape (1, seq_len, 2)  — ready for model forward pass
    """
    if len(map_values) != seq_len:
        raise ValueError(
            f"Expected {seq_len} MAP values, got {len(map_values)}."
        )
    x_map = np.array(map_values, dtype=np.float32)
    x_treat = np.full(seq_len, float(treatment_label), dtype=np.float32)
    seq = np.stack([x_map, x_treat], axis=-1)  # (seq_len, 2)
    return seq[np.newaxis, :, :]               # (1, seq_len, 2)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_counterfactuals(
    trajectories: Dict[str, np.ndarray],
    past_map: Optional[List[float]] = None,
    best_treatment: str = "",
    title: str = "Predicted MAP Trajectories",
) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """Return a Plotly figure comparing counterfactual MAP trajectories.

    Parameters
    ----------
    trajectories : dict  {treatment_name: np.ndarray (pred_len,)}
    past_map : optional list of historical MAP values to prepend
    best_treatment : name of the arm to highlight
    title : figure title

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for plotting. pip install plotly") from exc

    fig = go.Figure()

    # Historical MAP
    if past_map:
        x_hist = list(range(-len(past_map), 0))
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=past_map,
                mode="lines+markers",
                name="Historical MAP",
                line=dict(color="gray", dash="dot"),
                marker=dict(size=6),
            )
        )

    pred_len = max(len(v) for v in trajectories.values())
    x_future = list(range(0, pred_len))

    for name, traj in trajectories.items():
        color = TREATMENT_COLORS.get(name, "blue")
        label = TREATMENT_LABELS.get(name, name)
        is_best = name == best_treatment
        fig.add_trace(
            go.Scatter(
                x=x_future,
                y=traj.tolist(),
                mode="lines+markers",
                name=f"{'★ ' if is_best else ''}{label}",
                line=dict(
                    color=color,
                    width=3 if is_best else 1.5,
                    dash="solid" if is_best else "dash",
                ),
                marker=dict(size=7 if is_best else 5),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Hour (0 = now)",
        yaxis_title="MAP (mmHg)",
        legend_title="Treatment",
        template="plotly_white",
        hovermode="x unified",
    )

    # Reference line at 65 mmHg (clinical threshold)
    fig.add_hline(
        y=65,
        line_dash="dot",
        line_color="red",
        annotation_text="MAP target ≥ 65 mmHg",
        annotation_position="bottom right",
    )

    return fig


# ---------------------------------------------------------------------------
# Demographics formatting
# ---------------------------------------------------------------------------

def format_demographics(
    patient_info: "pd.DataFrame",  # type: ignore[name-defined]
    patient_id: int,
) -> Dict[str, str]:
    """Return a dict of formatted demographic strings for a patient."""
    try:
        row = patient_info.loc[patient_id]
    except KeyError:
        return {"Error": f"Patient {patient_id} not found in demographics."}

    fields: Dict[str, str] = {}
    field_map = {
        "uniquepid": "Unique Patient ID",
        "age": "Age",
        "gender": "Gender",
        "ethnicity": "Ethnicity",
        "admissionheight": "Height (cm)",
        "admissionweight": "Weight (kg)",
        "unittype": "ICU Unit Type",
        "unitdischargestatus": "ICU Discharge Status",
        "hospitaldischargestatus": "Hospital Discharge Status",
    }
    for col, label in field_map.items():
        if col in row.index and pd.notna(row[col]):
            fields[label] = str(row[col])
    return fields
