"""PyTorch recurrent model with intervention-parameterized state transitions.

Each transition uses treatment-conditioned gates:
    z_t = sigmoid(Wz x_t + Uz h_{t-1} + Vz a_t)
    r_t = sigmoid(Wr x_t + Ur h_{t-1} + Vr a_t)
    h~_t = tanh(Wh x_t + Uh (r_t * h_{t-1}) + Vh a_t)
    h_t = (1 - z_t) * h_{t-1} + z_t * h~_t

where a_t is the treatment embedding at timestep t.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalTransitionGRUCell(nn.Module):
    """GRU-like cell with explicit intervention terms in all gates."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        treatment_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.Wz = nn.Linear(input_size, hidden_size, bias=False)
        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wr = nn.Linear(input_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wh = nn.Linear(input_size, hidden_size, bias=False)
        self.Uh = nn.Linear(hidden_size, hidden_size, bias=False)

        # Shared intervention projection (physiology backbone + perturbation).
        self.V_shared = nn.Linear(treatment_dim, hidden_size, bias=True)

        # Gate-specific perturbation scales in [0.05, 0.5], initialized near 0.2.
        init_raw = -0.6931  # sigmoid(init_raw)=1/3 -> 0.05 + 0.45*(1/3)=0.2
        self.lambda_z_raw = nn.Parameter(torch.full((hidden_size,), init_raw))
        self.lambda_r_raw = nn.Parameter(torch.full((hidden_size,), init_raw))
        self.lambda_h_raw = nn.Parameter(torch.full((hidden_size,), init_raw))

        # Aliases kept for training/eval gate-gradient checks.
        self.Vz = self.V_shared
        self.Vr = self.V_shared
        self.Vh = self.V_shared

    @staticmethod
    def _lambda_from_raw(raw: torch.Tensor) -> torch.Tensor:
        return 0.05 + 0.45 * torch.sigmoid(raw)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        a_t: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        intervention = self.V_shared(a_t)
        lambda_z = self._lambda_from_raw(self.lambda_z_raw)
        lambda_r = self._lambda_from_raw(self.lambda_r_raw)
        lambda_h = self._lambda_from_raw(self.lambda_h_raw)

        z_t = torch.sigmoid(self.Wz(x_t) + self.Uz(h_prev) + lambda_z * intervention)
        r_t = torch.sigmoid(self.Wr(x_t) + self.Ur(h_prev) + lambda_r * intervention)
        h_tilde = torch.tanh(self.Wh(x_t) + self.Uh(r_t * h_prev) + lambda_h * intervention)
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
        if not return_diagnostics:
            return h_t
        return h_t, {
            "z_t": z_t,
            "r_t": r_t,
            "h_tilde": h_tilde,
            "h_t": h_t,
        }


class TRSModel(nn.Module):
    """MAP predictor with treatment-conditioned recurrent transitions.

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
    """

    def __init__(
        self,
        num_treatments: int = 3,
        embed_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        pred_len: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.embed_dim = embed_dim

        # Learned treatment embedding; label 0 is fixed to all-zero baseline.
        self.treatment_embedding = nn.Embedding(num_treatments, embed_dim, padding_idx=0)
        with torch.no_grad():
            self.treatment_embedding.weight[0].zero_()

        # Layer-0 input follows [MAP || treatment_embedding].
        self.input_size = 1 + embed_dim
        self.gru_cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            # Higher layers also receive treatment embedding in x_t so treatment
            # influences both the input path and explicit gate terms throughout depth.
            layer_input_size = self.input_size if layer_idx == 0 else hidden_size + embed_dim
            self.gru_cells.append(
                ConditionalTransitionGRUCell(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    treatment_dim=embed_dim,
                )
            )

        self.layer_dropout = nn.Dropout(dropout) if num_layers > 1 and dropout > 0 else nn.Identity()
        self._instability_events = 0

        self.fc = nn.Linear(hidden_size, pred_len)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        check_stability: bool = False,
        stability_mode: str = "raise",
    ) -> torch.Tensor:
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
        map_feat = x[:, :, 0:1]
        # Clamp indices to valid embedding range before lookup.
        treat_idx = x[:, :, 1].long().clamp(
            0, self.treatment_embedding.num_embeddings - 1
        )
        embed = self.treatment_embedding(treat_idx)
        return self._forward_with_embeddings(
            map_feat=map_feat,
            treatment_embed=embed,
            check_stability=check_stability,
            stability_mode=stability_mode,
        )

    def _stabilize_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        timestep: int,
        layer_idx: int,
        stability_mode: str,
    ) -> torch.Tensor:
        if torch.isfinite(tensor).all():
            return tensor
        if stability_mode == "raise":
            raise FloatingPointError(
                f"Non-finite tensor in recurrent transition: {name} "
                f"at timestep={timestep}, layer={layer_idx}"
            )
        if stability_mode != "clamp_detach":
            raise ValueError(f"Unknown stability_mode: {stability_mode}")

        self._instability_events += 1
        stabilized = torch.nan_to_num(tensor, nan=0.0, posinf=20.0, neginf=-20.0)
        stabilized = torch.clamp(stabilized, min=-20.0, max=20.0)
        # Do not backprop through unstable timestep dynamics.
        return stabilized.detach()

    def consume_instability_events(self) -> int:
        n = int(self._instability_events)
        self._instability_events = 0
        return n

    def _forward_with_embeddings(
        self,
        map_feat: torch.Tensor,
        treatment_embed: torch.Tensor,
        check_stability: bool = False,
        stability_mode: str = "raise",
        return_last_hidden: bool = False,
    ) -> torch.Tensor:
        """Run recurrent rollout using externally provided treatment embeddings."""
        seq_len = map_feat.shape[1]
        device = map_feat.device
        batch_size = map_feat.shape[0]

        hidden_states = [
            torch.zeros(batch_size, self.hidden_size, device=device, dtype=map_feat.dtype)
            for _ in range(self.num_layers)
        ]

        for t in range(seq_len):
            treatment_embed_t = treatment_embed[:, t, :]
            step_input = torch.cat([map_feat[:, t, :], treatment_embed_t], dim=-1)

            for layer_idx, gru_cell in enumerate(self.gru_cells):
                if check_stability:
                    h_t, diag = gru_cell(
                        step_input,
                        hidden_states[layer_idx],
                        treatment_embed_t,
                        return_diagnostics=True,
                    )
                    for name, tensor in diag.items():
                        diag[name] = self._stabilize_tensor(
                            tensor=tensor,
                            name=name,
                            timestep=t,
                            layer_idx=layer_idx,
                            stability_mode=stability_mode,
                        )
                    h_t = diag["h_t"]
                else:
                    h_t = gru_cell(step_input, hidden_states[layer_idx], treatment_embed_t)

                if check_stability:
                    h_t = self._stabilize_tensor(
                        tensor=h_t,
                        name="h_t",
                        timestep=t,
                        layer_idx=layer_idx,
                        stability_mode=stability_mode,
                    )

                hidden_states[layer_idx] = h_t
                if layer_idx < self.num_layers - 1:
                    step_input = torch.cat([self.layer_dropout(h_t), treatment_embed_t], dim=-1)
                else:
                    step_input = h_t

        last = hidden_states[-1]
        if return_last_hidden:
            return last
        return self.fc(last)

    def last_hidden_with_treatment(
        self,
        x: torch.Tensor,
        treatment_label: int,
        zero_treatment_embedding: bool = False,
        check_stability: bool = False,
        stability_mode: str = "raise",
    ) -> torch.Tensor:
        x_cf = x.clone()
        x_cf[:, :, 1] = float(treatment_label)
        map_feat = x_cf[:, :, 0:1]
        if zero_treatment_embedding:
            embed = torch.zeros(
                x_cf.shape[0],
                x_cf.shape[1],
                self.embed_dim,
                device=x_cf.device,
                dtype=x_cf.dtype,
            )
        else:
            treat_idx = x_cf[:, :, 1].long().clamp(0, self.treatment_embedding.num_embeddings - 1)
            embed = self.treatment_embedding(treat_idx)
        return self._forward_with_embeddings(
            map_feat=map_feat,
            treatment_embed=embed,
            check_stability=check_stability,
            stability_mode=stability_mode,
            return_last_hidden=True,
        )

    def intervention_grad_norms(self) -> dict[str, float]:
        """Return gradient norms for intervention gate parameters Vz/Vr/Vh."""
        norms = {"Vz": 0.0, "Vr": 0.0, "Vh": 0.0}
        for cell in self.gru_cells:
            shared_grad = cell.V_shared.weight.grad
            if shared_grad is not None:
                shared_norm = float(torch.norm(shared_grad).item())
                norms["Vz"] += shared_norm
                norms["Vr"] += shared_norm
                norms["Vh"] += shared_norm
        return norms

    # ------------------------------------------------------------------
    def predict_with_treatment(
        self,
        x: torch.Tensor,
        treatment_label: int,
        zero_treatment_embedding: bool = False,
        check_stability: bool = False,
        stability_mode: str = "raise",
    ) -> torch.Tensor:
        """Run a counterfactual forward pass with a fixed treatment label.

        The treatment index (column 1 of *x*) is overwritten with
        *treatment_label* for every timestep before the forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, 2)
        treatment_label : int  — 0=no treatment, 1=fluids, 2=vasopressor

        zero_treatment_embedding : bool
            If True, ignores treatment indices and uses a_t=0 for every
            timestep (ablation mode).

        Returns
        -------
        Tensor of shape (batch, pred_len)
        """
        x_cf = x.clone()
        x_cf[:, :, 1] = float(treatment_label)
        if not zero_treatment_embedding:
            return self.forward(
                x_cf,
                check_stability=check_stability,
                stability_mode=stability_mode,
            )

        map_feat = x_cf[:, :, 0:1]
        batch_size, seq_len = x_cf.shape[0], x_cf.shape[1]
        embed = torch.zeros(
            batch_size,
            seq_len,
            self.embed_dim,
            device=x_cf.device,
            dtype=x_cf.dtype,
        )
        return self._forward_with_embeddings(
            map_feat=map_feat,
            treatment_embed=embed,
            check_stability=check_stability,
            stability_mode=stability_mode,
        )
