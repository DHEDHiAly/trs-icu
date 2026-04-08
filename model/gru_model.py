"""
model/gru_model.py

PyTorch GRU-based sequence model for treatment-conditioned MAP trajectory
prediction.

Architecture
------------
Input  : (batch, seq_len, input_size)  — [MAP, treatment_label] per timestep
GRU    : hidden_size units, num_layers stacked
Output : (batch, pred_len)             — predicted MAP for the next pred_len hours

The model supports **treatment conditioning at inference time**: the treatment
feature in the input sequence is replaced with the desired counterfactual label
before the forward pass, enabling the three-arm comparison used in
``inference/counterfactual.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TRSModel(nn.Module):
    """GRU-based MAP trajectory predictor.

    Parameters
    ----------
    input_size:
        Number of features per timestep (default 2: MAP + treatment label).
    hidden_size:
        Number of GRU hidden units per layer.
    num_layers:
        Number of stacked GRU layers.
    pred_len:
        Number of future timesteps to predict.
    dropout:
        Dropout probability between GRU layers (ignored when num_layers==1).
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        pred_len: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, input_size)

        Returns
        -------
        Tensor of shape (batch, pred_len)
        """
        out, _ = self.gru(x)          # (batch, seq_len, hidden_size)
        last = out[:, -1, :]          # (batch, hidden_size)
        return self.fc(last)          # (batch, pred_len)

    # ------------------------------------------------------------------
    def predict_with_treatment(
        self,
        x: torch.Tensor,
        treatment_label: int,
    ) -> torch.Tensor:
        """Run a counterfactual forward pass with a fixed treatment label.

        The treatment feature (last column of *x*) is overwritten with
        *treatment_label* for every timestep before the GRU forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, input_size)
        treatment_label : int  — 0=no treatment, 1=fluids, 2=vasopressor

        Returns
        -------
        Tensor of shape (batch, pred_len)
        """
        x_cf = x.clone()
        x_cf[:, :, -1] = float(treatment_label)
        return self.forward(x_cf)
