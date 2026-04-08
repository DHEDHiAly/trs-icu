"""
app/streamlit_app.py

TRS-ICU Streamlit application.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

# Allow imports from the project root when running via `streamlit run app/...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import torch

from inference.counterfactual import predict_counterfactuals
from model.gru_model import TRSModel
from utils.helpers import (
    TREATMENT_LABELS,
    format_demographics,
    make_synthetic_sequences,
    plot_counterfactuals,
    sequence_from_map_values,
)
from model.train import train_model

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TRS-ICU · Treatment Response Simulator",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training model on synthetic data …")
def _load_demo_model() -> tuple:
    """Train a quick demo model on synthetic data and return (model, mean, std)."""
    X, y, _, _ = make_synthetic_sequences(n_patients=300, seed=0)
    model, _, map_mean, map_std = train_model(
        X, y,
        hidden_size=32,
        num_layers=1,
        epochs=20,
        batch_size=64,
        verbose=False,
    )
    model.eval()
    return model, map_mean, map_std


# ---------------------------------------------------------------------------
# Sidebar – patient input
# ---------------------------------------------------------------------------

st.sidebar.header("🏥 TRS-ICU")
st.sidebar.subheader("Patient Input")

patient_id_input = st.sidebar.text_input("Patient ID (for display)", value="P-001")

st.sidebar.markdown("**Past 6 hours of MAP (mmHg)**")

default_maps = [72.0, 70.0, 68.0, 66.0, 64.0, 62.0]
map_inputs: list[float] = []
cols = st.sidebar.columns(2)
for i in range(6):
    col = cols[i % 2]
    val = col.number_input(
        f"Hour -{6 - i}",
        min_value=20.0,
        max_value=200.0,
        value=default_maps[i],
        step=0.5,
        key=f"map_{i}",
    )
    map_inputs.append(val)

current_treatment = st.sidebar.selectbox(
    "Current treatment",
    options=list(TREATMENT_LABELS.keys()),
    format_func=lambda k: TREATMENT_LABELS[k],
    index=0,
)

# Numeric label for current treatment
treatment_map = {"no_treatment": 0, "fluids": 1, "vasopressor": 2}
current_treat_label = treatment_map[current_treatment]

run_btn = st.sidebar.button("🔮 Predict Trajectories", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Demo mode: model trained on synthetic AR(1) MAP data. "
    "Connect real eICU data via `main.py` for clinical use."
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🏥 TRS-ICU · Treatment Response Simulator")
st.markdown(
    "Simulate ICU patient **MAP trajectories** under counterfactual treatments "
    "using a GRU-based deep learning model."
)

# Patient demographics (mock for demo)
demo_demographics = {
    "Patient ID": patient_id_input,
    "Age": "67",
    "Gender": "Male",
    "Unit Type": "MICU",
    "Admission Weight (kg)": "82",
}

col_demo, col_chart = st.columns([1, 3])

with col_demo:
    st.subheader("👤 Patient Demographics")
    for label, value in demo_demographics.items():
        st.markdown(f"**{label}:** {value}")

    st.markdown("---")
    st.subheader("📊 Current MAP trend")
    st.line_chart({"MAP (mmHg)": map_inputs})

if run_btn:
    with col_chart:
        st.subheader("🔮 Counterfactual MAP Predictions")

        model, map_mean, map_std = _load_demo_model()

        sequence = sequence_from_map_values(
            map_inputs,
            treatment_label=current_treat_label,
        )

        result = predict_counterfactuals(
            sequence, model, map_mean=map_mean, map_std=map_std
        )

        fig = plot_counterfactuals(
            result.trajectories,
            past_map=map_inputs,
            best_treatment=result.best_treatment,
            title=f"Predicted MAP – Next 6 Hours | Patient {patient_id_input}",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Best treatment callout
        best_label = TREATMENT_LABELS.get(result.best_treatment, result.best_treatment)
        st.success(
            f"⭐ **Recommended treatment:** {best_label} "
            f"(predicted mean MAP ≈ {result.best_mean_map:.1f} mmHg)"
        )

        # Trajectory table
        st.subheader("📋 Predicted MAP by Hour")
        import pandas as pd

        rows = {}
        for name, traj in result.trajectories.items():
            label = TREATMENT_LABELS.get(name, name)
            rows[label] = [f"{v:.1f}" for v in traj]

        pred_df = pd.DataFrame(rows, index=[f"+{i+1}h" for i in range(len(traj))])
        st.dataframe(pred_df, use_container_width=True)

else:
    with col_chart:
        st.info(
            "👈  Adjust the MAP readings in the sidebar and click "
            "**Predict Trajectories** to run counterfactual inference."
        )
