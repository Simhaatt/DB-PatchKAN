"""Corrected baseline-residual physics hybrid for Nissan Stage 4N-B."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .kan import ConceptKAN
from .physics_patchkan import PhysicsInformedDBPatchKAN


CURRENT_MONOTONIC_FEATURES = (
    "cum_efc",
    "cum_electrothermal_dose_h",
    "cum_soc_high_dose_h",
    "cum_c_rate_transient_dose",
)
CURRENT_INTERACTION_FEATURES = (
    "c_rate_abs_p95",
    "pack_temp_max_c",
    "soc_high_hot_time_h",
    "bsi_rate",
    "bsi_interaction",
)
FORECAST_30D_FEATURES = (
    "hist_30d_efc",
    "hist_30d_electrothermal_dose_h",
    "hist_30d_soc_high_dose_h",
    "hist_30d_c_rate_transient_dose",
)
FORECAST_500KM_FEATURES = (
    "hist_90d_efc",
    "hist_90d_electrothermal_dose_h",
    "hist_90d_soc_high_dose_h",
    "hist_90d_c_rate_transient_dose",
)


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1.0e-6)
    return math.log(math.expm1(value)) if value < 20 else value


class BoundedMonotonicResidualHead(nn.Module):
    """Bounded correction around a separately fitted age/mileage SOH baseline."""

    def __init__(self, n_monotonic: int, n_interaction: int, *, limit_pp: float = 1.0) -> None:
        super().__init__()
        self.raw_weights = nn.Parameter(torch.full((n_monotonic,), -4.0))
        self.interaction_kan = ConceptKAN((n_interaction, 1), grid_size=5)
        self.bias = nn.Parameter(torch.zeros(()))
        self.limit_pp = float(limit_pp)

    @property
    def positive_weights(self) -> Tensor:
        return F.softplus(self.raw_weights)

    def forward(self, monotonic: Tensor, interaction: Tensor) -> dict[str, Tensor]:
        monotonic_logit = monotonic.float() @ self.positive_weights
        interaction_term = 0.25 * torch.tanh(
            self.interaction_kan(interaction.float()).squeeze(-1)
        )
        correction = self.limit_pp * torch.tanh(self.bias + monotonic_logit + interaction_term)
        return {
            "correction_pp": correction,
            "monotonic_logit": monotonic_logit,
            "interaction_term": interaction_term,
        }

    def regularization_loss(self) -> Tensor:
        return self.interaction_kan.regularization_loss()


class ScaledForecastHead(nn.Module):
    """Non-negative horizon loss calibrated to the empirical training-label scale."""

    def __init__(
        self,
        n_monotonic: int,
        n_response: int,
        *,
        output_scale_pp: float,
        initial_median_pp: float,
    ) -> None:
        super().__init__()
        self.raw_weights = nn.Parameter(torch.full((n_monotonic,), -4.0))
        self.response_kan = ConceptKAN((n_response, 1), grid_size=5)
        self.register_buffer("output_scale_pp", torch.tensor(float(output_scale_pp)))
        initial_ratio = max(float(initial_median_pp) / float(output_scale_pp), 1.0e-5)
        self.bias = nn.Parameter(torch.tensor(inverse_softplus(initial_ratio)))

    @property
    def positive_weights(self) -> Tensor:
        return F.softplus(self.raw_weights)

    def forward(self, monotonic: Tensor, response: Tensor) -> dict[str, Tensor]:
        monotonic_logit = monotonic.float() @ self.positive_weights
        interaction = 0.25 * torch.tanh(self.response_kan(response.float()).squeeze(-1))
        loss = self.output_scale_pp * F.softplus(self.bias + monotonic_logit + interaction)
        return {
            "loss_pp": loss,
            "monotonic_logit": monotonic_logit,
            "response_interaction": interaction,
        }

    def regularization_loss(self) -> Tensor:
        return self.response_kan.regularization_loss()


class CorrectedPhysicsHybrid(nn.Module):
    """Frozen temporal response model plus corrected physical SOH equations."""

    def __init__(
        self,
        temporal_model: PhysicsInformedDBPatchKAN,
        *,
        median_loss_30d_pp: float,
        median_loss_500km_pp: float,
        scale_loss_30d_pp: float = 0.10,
        scale_loss_500km_pp: float = 0.10,
    ) -> None:
        super().__init__()
        self.temporal_model = temporal_model
        n_response = len(temporal_model.response_center)
        self.current_head = BoundedMonotonicResidualHead(4, 5, limit_pp=0.5)
        self.loss_30d_head = ScaledForecastHead(
            4, n_response, output_scale_pp=scale_loss_30d_pp,
            initial_median_pp=median_loss_30d_pp,
        )
        self.loss_500km_head = ScaledForecastHead(
            4, n_response, output_scale_pp=scale_loss_500km_pp,
            initial_median_pp=median_loss_500km_pp,
        )
        self.freeze_temporal(True)

    def freeze_temporal(self, freeze: bool = True) -> None:
        for parameter in self.temporal_model.parameters():
            parameter.requires_grad = not freeze

    def temporal_response(
        self,
        driver: Tensor,
        battery: Tensor,
        *,
        segment_mask: Tensor,
        driver_mask: Tensor | None = None,
        battery_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        encoded = self.temporal_model.backbone.encode_trip_bag(
            driver, battery, segment_mask=segment_mask,
            driver_mask=driver_mask, battery_mask=battery_mask,
        )
        response_normalized = self.temporal_model.response_head(encoded["embedding"])
        return {
            **encoded,
            "response_predictions_normalized": response_normalized,
            "response_predictions": (
                response_normalized * self.temporal_model.response_scale
                + self.temporal_model.response_center
            ),
            "stress_class_logits": self.temporal_model.stress_class_head(encoded["embedding"]),
        }

    def forward_from_cached(
        self,
        *,
        baseline_soh: Tensor,
        current_monotonic: Tensor,
        current_interaction: Tensor,
        forecast_30d: Tensor,
        forecast_500km: Tensor,
        response_normalized: Tensor,
        analytic_only: bool = False,
    ) -> dict[str, Tensor]:
        response = torch.zeros_like(response_normalized) if analytic_only else response_normalized
        current = self.current_head(current_monotonic, current_interaction)
        loss_30d = self.loss_30d_head(forecast_30d, response)
        loss_500 = self.loss_500km_head(forecast_500km, response)
        current_soh = baseline_soh.float() - current["correction_pp"]
        return {
            "baseline_soh": baseline_soh,
            "current_correction_pp": current["correction_pp"],
            "current_soh": current_soh,
            "loss_30d": loss_30d["loss_pp"],
            "loss_500km": loss_500["loss_pp"],
            "future_soh_30d": current_soh - loss_30d["loss_pp"],
            "future_soh_500km": current_soh - loss_500["loss_pp"],
            "current_monotonic_logit": current["monotonic_logit"],
            "loss_30d_monotonic_logit": loss_30d["monotonic_logit"],
            "loss_500km_monotonic_logit": loss_500["monotonic_logit"],
        }

    def regularization_loss(self) -> Tensor:
        return torch.stack([
            self.current_head.regularization_loss(),
            self.loss_30d_head.regularization_loss(),
            self.loss_500km_head.regularization_loss(),
        ]).mean()


__all__ = [
    "CURRENT_INTERACTION_FEATURES", "CURRENT_MONOTONIC_FEATURES",
    "FORECAST_30D_FEATURES", "FORECAST_500KM_FEATURES",
    "BoundedMonotonicResidualHead", "CorrectedPhysicsHybrid", "ScaledForecastHead",
]
