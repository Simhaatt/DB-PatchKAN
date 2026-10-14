"""Physics-informed dual-stream Patch Transformer with monotonic concept KAN."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .patchkan import DBPatchKAN
from .physics_kan import PhysicsMediatedKAN


NISSAN_PHYSICS_CONCEPT_NAMES = (
    "calendar_days_since_start",
    "odometer_clean_km",
    "cum_ah_throughput",
    "cum_efc",
    "cum_c_rate_squared_dose_h",
    "cum_electrothermal_dose_h",
    "cum_joule_heat_estimate_kwh",
    "cum_soc_high_dose_h",
    "cum_c_rate_transient_dose",
    "cum_stress_dose",
    "c_rate_abs_p95",
    "c_rate_abs_peak",
    "pack_temp_max_c",
    "pack_temp_rise_c",
    "pack_temp_spread_p95_c",
    "high_c_rate_time_h",
    "soc_high_hot_time_h",
    "power_reversal_per_h",
    "bsi_rate",
    "bsi_interaction",
)

NISSAN_RESPONSE_TARGET_NAMES = (
    "pack_temp_rise_c",
    "ambient_adjusted_pack_temp_rise_c",
    "energy_kwh_per_100km",
    "c_rate_abs_peak",
    "pack_temp_max_c",
    "high_c_rate_time_h",
    "soc_high_hot_time_h",
)


class PhysicsInformedDBPatchKAN(nn.Module):
    """Pretrained temporal encoder plus measured/predicted physics bottleneck.

    The first ten concepts are cumulative harmful exposures.  Every SOH-loss
    head has non-negative weights on those standardised concepts.  The final
    ten concepts enter a bounded additive KAN for nonlinear stress interactions.
    No unrestricted latent embedding is connected to an SOH output.
    """

    def __init__(
        self,
        backbone: DBPatchKAN,
        *,
        concept_names: tuple[str, ...] = NISSAN_PHYSICS_CONCEPT_NAMES,
        n_monotonic: int = 10,
        grid_size: int = 5,
    ) -> None:
        super().__init__()
        if not 1 <= n_monotonic < len(concept_names):
            raise ValueError("Physics hybrid needs monotonic and interaction concepts")
        self.backbone = backbone
        self.concept_names = tuple(concept_names)
        self.n_monotonic = int(n_monotonic)
        n_concepts = len(self.concept_names)
        d_model = backbone.config.d_model
        self.physics_concept_predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(backbone.config.dropout),
            nn.Linear(d_model, n_concepts),
        )
        self.response_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Dropout(backbone.config.dropout),
            nn.Linear(d_model // 2, len(NISSAN_RESPONSE_TARGET_NAMES)),
        )
        self.stress_class_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 3)
        )
        n_interaction = n_concepts - n_monotonic
        self.current_degradation_head = PhysicsMediatedKAN(
            n_monotonic, n_interaction, grid_size=grid_size, residual_limit_pp=1.0
        )
        self.loss_30d_head = PhysicsMediatedKAN(
            n_monotonic, n_interaction, grid_size=grid_size, residual_limit_pp=0.20
        )
        self.loss_500km_head = PhysicsMediatedKAN(
            n_monotonic, n_interaction, grid_size=grid_size, residual_limit_pp=0.20
        )
        self.register_buffer("physics_center", torch.zeros(n_concepts), persistent=True)
        self.register_buffer("physics_scale", torch.ones(n_concepts), persistent=True)
        self.register_buffer(
            "response_center", torch.zeros(len(NISSAN_RESPONSE_TARGET_NAMES)), persistent=True
        )
        self.register_buffer(
            "response_scale", torch.ones(len(NISSAN_RESPONSE_TARGET_NAMES)), persistent=True
        )
        self.register_buffer("stress_thresholds", torch.zeros(2), persistent=True)

    def set_physics_normalization(self, center: Tensor, scale: Tensor) -> None:
        center = torch.as_tensor(center, dtype=torch.float32, device=self.physics_center.device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.physics_scale.device)
        if center.shape != self.physics_center.shape or scale.shape != self.physics_scale.shape:
            raise ValueError("Physics normalisation vectors have the wrong shape")
        self.physics_center.copy_(center)
        self.physics_scale.copy_(scale.clamp_min(1.0e-6))

    def set_response_normalization(self, center: Tensor, scale: Tensor) -> None:
        center = torch.as_tensor(center, dtype=torch.float32, device=self.response_center.device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.response_scale.device)
        if center.shape != self.response_center.shape or scale.shape != self.response_scale.shape:
            raise ValueError("Response normalisation vectors have the wrong shape")
        self.response_center.copy_(center)
        self.response_scale.copy_(scale.clamp_min(1.0e-6))

    def set_stress_thresholds(self, thresholds: Tensor) -> None:
        thresholds = torch.as_tensor(
            thresholds, dtype=torch.float32, device=self.stress_thresholds.device
        )
        if thresholds.shape != (2,) or thresholds[0] > thresholds[1]:
            raise ValueError("Stress thresholds must be two ordered values")
        self.stress_thresholds.copy_(thresholds)

    def freeze_backbone(self, freeze: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze

    def unfreeze_last_encoder_block(self) -> None:
        self.freeze_backbone(True)
        for stream in (self.backbone.driver_encoder, self.backbone.battery_encoder):
            for parameter in stream.encoder.layers[-1].parameters():
                parameter.requires_grad = True
        for module in (
            self.backbone.driver_to_battery,
            self.backbone.battery_to_driver,
            self.backbone.driver_cross_norm,
            self.backbone.battery_cross_norm,
            self.backbone.driver_pool,
            self.backbone.battery_pool,
            self.backbone.trip_pool,
            self.backbone.trip_driver_pool,
            self.backbone.fusion_projection,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True

    @staticmethod
    def _merge_concepts(
        predicted: Tensor,
        observed: Tensor | None,
        observed_mask: Tensor | None,
        source: str,
    ) -> Tensor:
        if source == "predicted":
            return predicted
        if source not in {"observed", "mixed"}:
            raise ValueError("concept_source must be predicted, observed, or mixed")
        if observed is None:
            if source == "observed":
                raise ValueError("Observed concepts were requested but not supplied")
            return predicted
        observed = observed.to(device=predicted.device, dtype=predicted.dtype)
        valid = torch.isfinite(observed)
        if observed_mask is not None:
            valid &= observed_mask.to(device=predicted.device, dtype=torch.bool)
        if observed.shape != predicted.shape or valid.shape != predicted.shape:
            raise ValueError("Observed physics concepts have the wrong shape")
        return torch.where(valid, observed, predicted)

    def forward_trip_bag(
        self,
        driver: Tensor,
        battery: Tensor,
        *,
        segment_mask: Tensor,
        driver_mask: Tensor | None = None,
        battery_mask: Tensor | None = None,
        observed_physics: Tensor | None = None,
        observed_physics_mask: Tensor | None = None,
        concept_source: str = "predicted",
    ) -> dict[str, Tensor]:
        encoded = self.backbone.encode_trip_bag(
            driver,
            battery,
            segment_mask=segment_mask,
            driver_mask=driver_mask,
            battery_mask=battery_mask,
        )
        predicted_normalized = self.physics_concept_predictor(encoded["embedding"])
        predicted = predicted_normalized * self.physics_scale + self.physics_center
        concept_input = self._merge_concepts(
            predicted, observed_physics, observed_physics_mask, concept_source
        )
        normalized = (concept_input - self.physics_center) / self.physics_scale
        monotonic = normalized[:, : self.n_monotonic]
        interactions = normalized[:, self.n_monotonic :]
        current = self.current_degradation_head(monotonic, interactions)
        loss_30d = self.loss_30d_head(monotonic, interactions)
        loss_500km = self.loss_500km_head(monotonic, interactions)
        current_soh = 100.0 - current["loss_pp"]
        response_normalized = self.response_head(encoded["embedding"])
        return {
            **encoded,
            "physics_concepts": predicted,
            "physics_concepts_normalized": predicted_normalized,
            "physics_concept_input": concept_input,
            "current_degradation_pp": current["loss_pp"],
            "current_soh": current_soh,
            "loss_30d": loss_30d["loss_pp"],
            "loss_500km": loss_500km["loss_pp"],
            "future_soh_30d": current_soh - loss_30d["loss_pp"],
            "future_soh_500km": current_soh - loss_500km["loss_pp"],
            "current_physics_logit": current["physics_logit"],
            "current_interaction_residual": current["interaction_residual"],
            "loss_30d_physics_logit": loss_30d["physics_logit"],
            "loss_30d_interaction_residual": loss_30d["interaction_residual"],
            "loss_500km_physics_logit": loss_500km["physics_logit"],
            "loss_500km_interaction_residual": loss_500km["interaction_residual"],
            "behavior_logits": self.backbone.behavior_head(encoded["driver_embedding"]),
            "response_predictions_normalized": response_normalized,
            "response_predictions": (
                response_normalized * self.response_scale + self.response_center
            ),
            "stress_class_logits": self.stress_class_head(encoded["embedding"]),
        }

    def regularization_loss(self) -> Tensor:
        return torch.stack([
            self.current_degradation_head.regularization_loss(),
            self.loss_30d_head.regularization_loss(),
            self.loss_500km_head.regularization_loss(),
        ]).mean()

    def monotonic_weights(self) -> Mapping[str, Tensor]:
        return {
            "current_degradation": self.current_degradation_head.positive_physics_weights,
            "loss_30d": self.loss_30d_head.positive_physics_weights,
            "loss_500km": self.loss_500km_head.positive_physics_weights,
        }


__all__ = [
    "NISSAN_PHYSICS_CONCEPT_NAMES", "NISSAN_RESPONSE_TARGET_NAMES",
    "PhysicsInformedDBPatchKAN",
]
