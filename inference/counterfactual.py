"""
inference/counterfactual.py

Counterfactual MAP trajectory prediction under three treatment arms:
  0 – no treatment
  1 – fluids
  2 – vasopressor

Usage
-----
>>> result = predict_counterfactuals(sequence, model, map_mean, map_std)
>>> result["vasopressor"]   # np.ndarray shape (pred_len,) in mmHg
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from model.gru_model import TRSModel

# Treatment arm labels
TREATMENTS = {
    "no_treatment": 0,
    "fluids": 1,
    "vasopressor": 2,
}


@dataclass
class CounterfactualResult:
    """Holds predicted MAP trajectories for each treatment arm."""

    trajectories: Dict[str, np.ndarray] = field(default_factory=dict)
    best_treatment: str = ""
    best_mean_map: float = 0.0

    def __post_init__(self) -> None:
        if self.trajectories and not self.best_treatment:
            self.best_treatment = max(
                self.trajectories, key=lambda k: float(np.mean(self.trajectories[k]))
            )
            self.best_mean_map = float(np.mean(self.trajectories[self.best_treatment]))


def predict_counterfactuals(
    sequence: np.ndarray,
    model: TRSModel,
    map_mean: float = 0.0,
    map_std: float = 1.0,
    device: Optional[str] = None,
) -> CounterfactualResult:
    """Predict MAP trajectories for all three treatment arms.

    Parameters
    ----------
    sequence : np.ndarray, shape (seq_len, 2) or (1, seq_len, 2)
        Input sequence with columns [MAP, treatment_label].
        MAP values should be in original mmHg (they will be normalised
        internally if *map_mean* / *map_std* are provided).
    model : TRSModel
        Trained GRU model.
    map_mean, map_std :
        Normalisation statistics from training (``model/train.compute_map_stats``).
    device : str, optional
        'cpu' or 'cuda'.  Defaults to the model's current device.

    Returns
    -------
    CounterfactualResult
        ``.trajectories`` is a dict ``{treatment_name: np.ndarray (pred_len,)}``.
        ``.best_treatment`` is the name of the arm with the highest mean MAP.
    """
    if device is None:
        device = next(model.parameters()).device
    dev = torch.device(device) if isinstance(device, str) else device

    seq = np.array(sequence, dtype=np.float32)
    if seq.ndim == 2:
        seq = seq[np.newaxis, :, :]   # (1, seq_len, 2)

    # Normalise MAP channel (column 0)
    seq_norm = seq.copy()
    seq_norm[:, :, 0] = (seq_norm[:, :, 0] - map_mean) / (map_std + 1e-8)

    x_tensor = torch.tensor(seq_norm, dtype=torch.float32).to(dev)

    model.eval()
    trajectories: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, label in TREATMENTS.items():
            pred_norm = model.predict_with_treatment(x_tensor, label)
            pred_mmhg = pred_norm.cpu().numpy().squeeze() * map_std + map_mean
            trajectories[name] = pred_mmhg

    return CounterfactualResult(trajectories=trajectories)
