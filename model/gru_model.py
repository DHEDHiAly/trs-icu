"""
model/gru_model.py

PyTorch GRU-based sequence model for treatment-conditioned MAP trajectory
prediction.

Architecture
------------
Input  : (batch, seq_len, 2)  — [MAP_t, treatment_idx_t] per timestep
Embedding: treatment_idx → embed_dim vector via nn.Embedding (trainable)
GRU    : hidden_size units, num_layers stacked
         GRU input at each step: [MAP_t, treatment_embedding_t]
         (size: 1 + embed_dim)
FiLM   : h_last = gamma(embed_last) * h_last + beta(embed_last)
Output : (batch, pred_len)     — predicted MAP for the next pred_len hours

The model supports **treatment conditioning at inference time**: the treatment
index in column 1 of the input is replaced with the desired counterfactual
label before the forward pass, enabling the three-arm comparison used in
``inference/counterfactual.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TRSModel(nn.Module):
    """GRU-based MAP trajectory predictor with treatment embedding and FiLM conditioning.

    Parameters
    ----------
    num_treatments:
        Number of treatment categories (default 3: none, fluids, vasopressor).
    embed_dim:
        Dimension of the learned treatment embedding (>= 4; default 8).
    hidden_size:
        Number of GRU hidden units per layer.
    num_layers:
        Number of stacked GRU layers.
    pred_len:
        Number of future timesteps to predict.
    dropout:
        Dropout probability between GRU layers (ignored when num_layers==1).
    use_film:
        Whether to apply FiLM conditioning on the final GRU hidden state.
    """

    def __init__(
        self,
        num_treatments: int = 3,
        embed_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        pred_len: int = 6,
        dropout: float = 0.2,
        use_film: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.embed_dim = embed_dim
        self.use_film = use_film

        # Learned treatment embedding: integer index → embed_dim vector
        self.treatment_embedding = nn.Embedding(num_treatments, embed_dim)

        # GRU input: MAP (1 dim) concatenated with treatment embedding (embed_dim)
        gru_input_size = 1 + embed_dim
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # FiLM conditioning: learn per-treatment scale and shift of hidden state
        if use_film:
            self.film_gamma = nn.Linear(embed_dim, hidden_size)
            self.film_beta = nn.Linear(embed_dim, hidden_size)

        self.fc = nn.Linear(hidden_size, pred_len)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, 2)
            Column 0: MAP values (float, normalised).
            Column 1: treatment index stored as float (0.0 / 1.0 / 2.0).

        Returns
        -------
        Tensor of shape (batch, pred_len)
        """
        map_feat = x[:, :, 0:1]                       # (batch, seq_len, 1)
        treat_idx = x[:, :, 1].long()                 # (batch, seq_len)
        embed = self.treatment_embedding(treat_idx)    # (batch, seq_len, embed_dim)
        gru_in = torch.cat([map_feat, embed], dim=-1)  # (batch, seq_len, 1+embed_dim)

        out, _ = self.gru(gru_in)                      # (batch, seq_len, hidden_size)
        last = out[:, -1, :]                           # (batch, hidden_size)

        if self.use_film:
            embed_last = embed[:, -1, :]               # (batch, embed_dim)
            gamma = self.film_gamma(embed_last)        # (batch, hidden_size)
            beta = self.film_beta(embed_last)          # (batch, hidden_size)
            last = gamma * last + beta

        return self.fc(last)                           # (batch, pred_len)

    # ------------------------------------------------------------------
    def predict_with_treatment(
        self,
        x: torch.Tensor,
        treatment_label: int,
    ) -> torch.Tensor:
        """Run a counterfactual forward pass with a fixed treatment label.

        The treatment index (column 1 of *x*) is overwritten with
        *treatment_label* for every timestep before the forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, 2)
        treatment_label : int  — 0=no treatment, 1=fluids, 2=vasopressor

        Returns
        -------
        Tensor of shape (batch, pred_len)
        """
        x_cf = x.clone()
        x_cf[:, :, 1] = float(treatment_label)
        return self.forward(x_cf)
