"""Losses for supervised DB-PatchKAN fine-tuning and SSL pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def masked_huber_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    *,
    delta: float = 1.0,
) -> Tensor:
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    valid = torch.isfinite(target) & torch.isfinite(prediction)
    if mask is not None:
        valid &= mask.to(device=prediction.device, dtype=torch.bool)
    if not torch.any(valid):
        return prediction.sum() * 0.0
    return F.huber_loss(prediction[valid], target[valid], reduction="mean", delta=delta)


def masked_cross_entropy(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int = -100,
    class_weight: Tensor | None = None,
) -> Tensor:
    target = target.to(device=logits.device, dtype=torch.long)
    valid = target != ignore_index
    if not torch.any(valid):
        return logits.sum() * 0.0
    weight = None if class_weight is None else class_weight.to(logits.device)
    return F.cross_entropy(logits[valid], target[valid], weight=weight)


def masked_patch_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    masked_patches: Tensor,
    element_valid: Tensor,
) -> Tensor:
    if prediction.shape != target.shape or prediction.shape != element_valid.shape:
        raise ValueError("Reconstruction, target, and element mask shapes must match")
    selected = (
        masked_patches.to(device=prediction.device, dtype=torch.bool)[:, :, None, None]
        & element_valid.to(device=prediction.device, dtype=torch.bool)
        & torch.isfinite(target)
    )
    if not torch.any(selected):
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(
        prediction[selected], target.to(prediction.device)[selected], reduction="mean"
    )


def nt_xent_loss(left: Tensor, right: Tensor, *, temperature: float = 0.1) -> Tensor:
    """Symmetric in-batch cross-modal contrastive alignment loss."""

    if left.shape != right.shape:
        raise ValueError("Aligned embedding tensors must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    left = F.normalize(left.float(), dim=-1)
    right = F.normalize(right.float(), dim=-1)
    if left.shape[0] == 1:
        return (1.0 - (left * right).sum(dim=-1)).mean()
    logits = left @ right.transpose(0, 1) / temperature
    labels = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)
    )


@dataclass(slots=True)
class LossWeights:
    concept: float = 0.2
    physics: float = 0.1
    driver_id: float = 0.1
    channel_l0: float = 1.0e-4
    kan_regularization: float = 1.0e-4
    ssl_alignment: float = 0.1


def physics_consistency_loss(outputs: dict[str, Tensor]) -> Tensor:
    """Soft physical bounds beyond the hard non-negative forecast loss."""

    current = outputs["current_soh"]
    future_30d = outputs["future_soh_30d"]
    future_500km = outputs["future_soh_500km"]
    penalties = (
        F.relu(-current),
        F.relu(current - 110.0),
        F.relu(-future_30d),
        F.relu(-future_500km),
        F.relu(future_30d - current),
        F.relu(future_500km - current),
    )
    return torch.stack([penalty.square().mean() for penalty in penalties]).mean()


def _first_present(batch: dict[str, Tensor], *names: str) -> Tensor | None:
    return next((batch[name] for name in names if name in batch), None)


class DBPatchKANMultiTaskLoss(nn.Module):
    """Learned-uncertainty supervised loss plus explicit concept/physics terms."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        *,
        driver_class_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.log_variances = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(()))
                for name in ("current_soh", "loss_30d", "loss_500km", "behavior")
            }
        )
        if driver_class_weight is None:
            self.register_buffer("driver_class_weight", None)
        else:
            self.register_buffer("driver_class_weight", driver_class_weight.float())

    def _uncertainty_weighted(self, name: str, value: Tensor) -> Tensor:
        log_variance = self.log_variances[name]
        return torch.exp(-log_variance) * value + log_variance

    def forward(
        self, outputs: dict[str, Tensor], batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        reference = outputs["current_soh"]
        components: dict[str, Tensor] = {}
        supervised_terms: list[Tensor] = []

        target_specs = (
            ("current_soh", ("current_soh_target", "soh_target", "soh_target_raw_pct")),
            ("loss_30d", ("loss_30d_target", "target_loss_30d_pp")),
            ("loss_500km", ("loss_500km_target", "target_loss_500km_pp")),
        )
        for output_name, target_names in target_specs:
            target = _first_present(batch, *target_names)
            if target is None:
                continue
            mask = _first_present(batch, f"{output_name}_mask", f"{target_names[0]}_mask")
            value = masked_huber_loss(outputs[output_name], target, mask, delta=1.0)
            components[output_name] = value
            valid = torch.isfinite(target)
            if mask is not None:
                valid &= mask.to(device=target.device, dtype=torch.bool)
            # Do not optimize a learned uncertainty scalar on an unlabeled
            # batch.  Adding only ``log_variance`` when the task loss is zero
            # drives the scalar toward -infinity and destabilizes the next
            # batch that actually contains a forecast label.
            if torch.any(valid):
                supervised_terms.append(self._uncertainty_weighted(output_name, value))

        behavior_target = _first_present(batch, "behavior_target", "behavior_level")
        if behavior_target is not None:
            behavior = masked_cross_entropy(outputs["behavior_logits"], behavior_target)
            components["behavior"] = behavior
            if torch.any(behavior_target != -100):
                supervised_terms.append(self._uncertainty_weighted("behavior", behavior))

        driver_target = _first_present(batch, "driver_target", "driver_id_target")
        if driver_target is not None:
            driver = masked_cross_entropy(
                outputs["driver_logits"],
                driver_target,
                class_weight=self.driver_class_weight,
            )
        else:
            driver = reference.sum() * 0.0
        components["driver_id"] = driver

        concept_target = _first_present(batch, "concept_targets", "concept_target")
        if concept_target is not None:
            concept_mask = _first_present(batch, "concept_mask", "concept_target_mask")
            concept_prediction = outputs["concepts"]
            if "concepts_normalized" in outputs:
                center = outputs["concept_center"].to(concept_target.device)
                scale = outputs["concept_scale"].to(concept_target.device).clamp_min(1.0e-6)
                concept_target = (concept_target - center) / scale
                concept_prediction = outputs["concepts_normalized"]
            concept = masked_huber_loss(
                concept_prediction, concept_target, concept_mask, delta=1.0
            )
        else:
            concept = reference.sum() * 0.0
        components["concept"] = concept

        physics = physics_consistency_loss(outputs)
        components["physics"] = physics
        l0 = outputs.get("l0_penalty", reference.sum() * 0.0)
        kan = outputs.get("kan_regularization", reference.sum() * 0.0)
        components["channel_l0"] = l0
        components["kan_regularization"] = kan

        supervised = (
            torch.stack(supervised_terms).sum()
            if supervised_terms
            else reference.sum() * 0.0
        )
        total = (
            supervised
            + self.weights.concept * concept
            + self.weights.physics * physics
            + self.weights.driver_id * driver
            + self.weights.channel_l0 * l0
            + self.weights.kan_regularization * kan
        )
        components["supervised"] = supervised
        components["total"] = total
        return components


class MaskedSelfSupervisedLoss(nn.Module):
    """Masked reconstruction plus driver/battery representation alignment."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        *,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.temperature = temperature

    def forward(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        driver = masked_patch_reconstruction_loss(
            outputs["driver_reconstruction"],
            outputs["driver_target_patches"],
            outputs["driver_masked_patches"],
            outputs["driver_element_valid"],
        )
        battery = masked_patch_reconstruction_loss(
            outputs["battery_reconstruction"],
            outputs["battery_target_patches"],
            outputs["battery_masked_patches"],
            outputs["battery_element_valid"],
        )
        alignment = nt_xent_loss(
            outputs["driver_embedding"],
            outputs["battery_embedding"],
            temperature=self.temperature,
        )
        l0 = outputs.get("l0_penalty", driver.sum() * 0.0)
        total = (
            driver
            + battery
            + self.weights.ssl_alignment * alignment
            + self.weights.channel_l0 * l0
        )
        return {
            "driver_reconstruction": driver,
            "battery_reconstruction": battery,
            "alignment": alignment,
            "channel_l0": l0,
            "total": total,
        }


__all__ = [
    "LossWeights",
    "masked_huber_loss",
    "masked_cross_entropy",
    "masked_patch_reconstruction_loss",
    "nt_xent_loss",
    "physics_consistency_loss",
    "DBPatchKANMultiTaskLoss",
    "MaskedSelfSupervisedLoss",
]
