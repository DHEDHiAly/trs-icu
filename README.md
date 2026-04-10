

# TRS-ICU · Treatment Response Simulator for ICU

A treatment-conditioned deep learning system for simulating ICU patient **Mean Arterial Pressure (MAP)** trajectories under different interventions and generating **counterfactual physiological responses** using a GRU-based dynamical model.

The system supports both synthetic and real-world eICU data and produces three-way counterfactual predictions:

* No treatment
* Fluids
* Vasopressor

---

## Project Structure

```
trs-icu/
├── data/
│   ├── loader.py              # CSV / CSV.GZ ingestion (local + Google Drive support)
│   └── preprocessor.py        # MAP extraction, drug labeling, sequence construction
│
├── model/
│   ├── gru_model.py          # Treatment-conditioned GRU dynamics model
│   └── train.py              # Training loop, evaluation, and counterfactual diagnostics
│
├── inference/
│   └── counterfactual.py     # 3-arm counterfactual prediction API
│
├── app/
│   └── streamlit_app.py      # Interactive visualization interface
│
├── utils/
│   └── helpers.py            # Synthetic data generation + sequence utilities + plotting
│
├── main.py                   # CLI pipeline orchestrator (train / eval / demo)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run synthetic demo (no dataset required)

Trains and evaluates the model on synthetic MAP trajectories and prints counterfactual responses:

```bash
python main.py --demo
```

---

### 3. Launch Streamlit interface

```bash
streamlit run app/streamlit_app.py
```

Provides:

* Manual MAP input
* Real-time counterfactual simulation
* Treatment comparison plots

---

### 4. Run on real eICU data

Place dataset files in:

```
data/eicu/
```

Then run:

```bash
python main.py --data-dir data/eicu --sample-patients 500
```

---

### 5. Download dataset (Google Drive)

```bash
python main.py --download --sample-patients 500
```

Requires:

```bash
pip install gdown
```

---

## Key Features

| Component             | Description                                               |
| --------------------- | --------------------------------------------------------- |
| Data ingestion        | Loads CSV/CSV.GZ files from local storage or Google Drive |
| MAP processing        | Hourly resampling with robust missing-value handling      |
| Treatment labeling    | Rule-based extraction of vasopressors and IV fluids       |
| GRU model             | Sequence model conditioned on treatment embedding         |
| Counterfactual engine | Simulates 3 intervention arms per patient                 |
| Streamlit UI          | Interactive clinical visualization tool                   |
| Synthetic mode        | Fully generative AR-style MAP simulator for testing       |

---

## Counterfactual API

```python
from inference.counterfactual import predict_counterfactuals
from utils.helpers import sequence_from_map_values

# Build input sequence from last observed MAP values
sequence = sequence_from_map_values(
    [72, 70, 68, 66, 64, 62],
    treatment_label=0
)

# Run counterfactual inference
result = predict_counterfactuals(
    sequence,
    model,
    map_mean=mean,
    map_std=std
)

print(result.best_treatment)
print(result.trajectories["fluids"])   # shape: (T,)
```

---

## eICU Dataset

Dataset source:

[https://drive.google.com/drive/folders/12b_rL9mTCUYBDaWU_rwJRqsrNQHaAPJ3](https://drive.google.com/drive/folders/12b_rL9mTCUYBDaWU_rwJRqsrNQHaAPJ3)

Required files:

```
data/eicu/
├── vitalPeriodic.csv.gz
├── infusionDrug.csv.gz
└── patient.csv.gz
```

---

## Model Overview

The GRU-based model learns a conditional dynamics function:

* Input: MAP time series + treatment embedding
* Output: next-step MAP prediction
* Conditioning: learned embedding of intervention type

The model supports:

* Counterfactual rollouts under alternative treatments
* Multi-step trajectory simulation
* Treatment effect separation via learned latent perturbations

---

## Output Interpretation

Counterfactual outputs represent:

* Expected MAP trajectory under each intervention
* Relative treatment effect magnitude over time
* Model-derived treatment recommendation (argmax response)

---

## Notes

* Synthetic mode is intended for debugging and pipeline validation only
* Real-world performance depends on quality of treatment labeling in eICU
* Counterfactual outputs are predictive, not causal guarantees
