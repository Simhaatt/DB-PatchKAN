"""Physics-mediated KAN for small external battery-health datasets.

The model deliberately keeps the irreversible stress pathway separate from a
small additive KAN residual.  Positive weights on standardised cumulative-dose
features guarantee that predicted latent degradation is non-decreasing in each
of those inputs when all other inputs are fixed.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .kan import ConceptKAN


class PhysicsMediatedKAN(nn.Module):
    """Non-negative SOH-loss model with monotone physics and interaction KAN paths.

    ``behaviour`` in the public call is a legacy argument name.  In the primary
    scientific model it contains only named battery-stress/context concepts.
    Raw driver variables may enter this path only in a separately labelled
    direct-effect ablation.
    """

    def __init__(
        self,
        n_physics: int,
        n_behaviour: int,
        *,
        grid_size: int = 5,
        residual_limit_pp: float = 0.5,
    ) -> None:
        super().__init__()
        if n_physics < 1 or n_behaviour < 1:
            raise ValueError("Both physics and behaviour inputs are required")
        self.n_physics = int(n_physics)
        self.n_behaviour = int(n_behaviour)
        self.residual_limit_pp = float(residual_limit_pp)
        self.raw_physics_weights = nn.Parameter(torch.zeros(n_physics))
        self.bias = nn.Parameter(torch.tensor(0.0))
        # One output layer makes the KAN additive across named behaviour inputs.
        self.behaviour_kan = ConceptKAN(
            (n_behaviour, 1), grid_size=grid_size, spline_order=3
        )

    @property
    def positive_physics_weights(self) -> Tensor:
        return F.softplus(self.raw_physics_weights)

    def forward(self, physics: Tensor, behaviour: Tensor) -> dict[str, Tensor]:
        physics32 = physics.float()
        behaviour32 = behaviour.float()
        physics_logit = physics32 @ self.positive_physics_weights
        residual_raw = self.behaviour_kan(behaviour32).squeeze(-1)
        residual = self.residual_limit_pp * torch.tanh(residual_raw)
        latent = self.bias + physics_logit + residual
        loss_pp = F.softplus(latent)
        return {
            "loss_pp": loss_pp,
            "physics_logit": physics_logit,
            "behaviour_residual": residual,
            "interaction_residual": residual,
            "latent_loss": latent,
        }

    def regularization_loss(self) -> Tensor:
        return self.behaviour_kan.regularization_loss()


__all__ = ["PhysicsMediatedKAN"]
