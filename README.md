# TRS-ICU · Treatment Response Simulator for ICU

A hackathon-ready clinical AI project that simulates ICU patient **Mean
Arterial Pressure (MAP)** trajectories under different treatments and provides
**counterfactual predictions** via a GRU-based deep-learning model.

---

## Project Structure

```
trs-icu/
├── data/
│   ├── loader.py          # Load CSV / CSV.GZ files (local or Google Drive)
│   └── preprocessor.py    # Extract MAP series, vasopressor labels, build sequences
├── model/
│   ├── gru_model.py       # PyTorch GRU model (treatment-conditioned)
│   └── train.py           # Training loop, evaluation (MSE / RMSE / MAE)
├── inference/
│   └── counterfactual.py  # predict_counterfactuals() – 3 treatment arms
├── app/
│   └── streamlit_app.py   # Streamlit visualisation interface
├── utils/
│   └── helpers.py         # Synthetic data, Plotly charts, sequence helpers
├── main.py                # CLI pipeline orchestrator
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the demo (no data required)

Trains on synthetic MAP data and shows counterfactual predictions in the
terminal:

```bash
python main.py --demo
```

### 3. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### 4. Use real eICU data

Place the eICU CSV/CSV.GZ files in a local directory (e.g. `data/raw/`) then:

```bash
python main.py --data-dir data/raw --sample-patients 500
```

Or download directly from Google Drive (requires `gdown`):

```bash
python main.py --download --sample-patients 500
```

---

## Key Features

| Feature | Details |
|---|---|
| **Data loading** | Loads all CSV/CSV.GZ from a directory; Google Drive download via gdown |
| **MAP extraction** | Hourly resampled MAP from `vitalPeriodic`; robust NaN handling |
| **Vasopressor labels** | Drug-name matching across 7 vasopressor families |
| **GRU model** | Configurable hidden size, layers, dropout; treatment feature in input |
| **Counterfactual inference** | Three arms: No Treatment / Fluids / Vasopressor |
| **Streamlit app** | Interactive MAP entry, Plotly trajectory chart, treatment recommendation |
| **Synthetic demo** | AR(1) MAP generator for zero-data development / testing |

---

## Counterfactual API

```python
from inference.counterfactual import predict_counterfactuals
from utils.helpers import sequence_from_map_values

# Build input from the last 6 MAP readings
sequence = sequence_from_map_values([72, 70, 68, 66, 64, 62], treatment_label=0)

# Run counterfactual inference
result = predict_counterfactuals(sequence, model, map_mean=mean, map_std=std)

print(result.best_treatment)          # e.g. "vasopressor"
print(result.trajectories["fluids"])  # np.ndarray shape (6,) in mmHg
```

---

## eICU Data Source

Dataset available at:
<https://drive.google.com/drive/folders/12b_rL9mTCUYBDaWU_rwJRqsrNQHaAPJ3>

Required files:
- `vitalPeriodic.csv.gz`
- `infusionDrug.csv.gz`
- `patient.csv.gz`
