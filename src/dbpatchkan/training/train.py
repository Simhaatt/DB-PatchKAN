"""Dependency-light training CLI for CPU, Kaggle T4, and Colab A100.

Examples
--------
Fast CPU verification::

    python -m dbpatchkan.training.train --dry-run

Full pretraining/fine-tuning::

    python -m dbpatchkan.training.train --data segments.npz --stage both --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset

from dbpatchkan.models import (
    DBPatchKAN,
    DBPatchKANConfig,
    DBPatchKANMultiTaskLoss,
    EVSegmentDataset,
    EVTripBagDataset,
    MaskedSelfSupervisedLoss,
    LossWeights,
    assert_disjoint_trip_bags,
    make_synthetic_dataset,
)


@dataclass(slots=True)
class TrainingConfig:
    stage: str = "both"
    device: str = "auto"
    precision: str = "auto"
    pretrain_batch_size: int = 32
    finetune_batch_size: int = 32
    gradient_accumulation: int = 1
    pretrain_epochs: int = 60
    finetune_epochs: int = 80
    pretrain_lr: float = 3.0e-4
    encoder_lr: float = 1.0e-5
    head_lr: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    gradient_clip: float = 1.0
    patience: int = 15
    mask_ratio: float = 0.4
    num_workers: int = 0
    seed: int = 42
    domain: str = "nissan"
    freeze_encoders: bool = False
    trip_bags: bool = False
    max_batches: int | None = None
    resume_checkpoint: str | None = None
    concept_weight: float = 0.2
    physics_weight: float = 0.1
    driver_id_weight: float = 0.1
    ssl_alignment_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.stage not in {"pretrain", "finetune", "both"}:
            raise ValueError("stage must be pretrain, finetune, or both")
        if self.gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be positive")
        if self.pretrain_batch_size < 1 or self.finetune_batch_size < 1:
            raise ValueError("Batch sizes must be positive")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is preferable for research folds; warn-only avoids failures
    # on GPU kernels without deterministic implementations.
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested == "auto":
        if device.type != "cuda":
            return "fp32"
        major, _ = torch.cuda.get_device_capability(device)
        return "bf16" if major >= 8 and torch.cuda.is_bf16_supported() else "fp16"
    if requested not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be auto, fp32, fp16, or bf16")
    if device.type == "cpu" and requested == "fp16":
        return "fp32"
    return requested


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _model_forward(model: DBPatchKAN, batch: dict[str, Any], domain: str) -> dict[str, Tensor]:
    if batch["driver"].ndim == 4:
        return model.forward_trip_bag(
            batch["driver"],
            batch["battery"],
            segment_mask=batch["segment_mask"],
            driver_mask=batch.get("driver_mask"),
            battery_mask=batch.get("battery_mask"),
            observed_concepts=batch.get("observed_concepts"),
            observed_concept_mask=batch.get("observed_concept_mask"),
            domain=domain,
        )
    return model(
        batch["driver"],
        batch["battery"],
        driver_mask=batch.get("driver_mask"),
        battery_mask=batch.get("battery_mask"),
        observed_concepts=batch.get("observed_concepts"),
        observed_concept_mask=batch.get("observed_concept_mask"),
        domain=domain,
    )


def _ssl_streams(batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Flatten only valid segments from an optional padded trip-bag batch."""

    driver = batch["driver"]
    battery = batch["battery"]
    driver_mask = batch.get("driver_mask")
    battery_mask = batch.get("battery_mask")
    if driver.ndim == 3:
        return driver, battery, driver_mask, battery_mask
    if driver.ndim != 4 or "segment_mask" not in batch:
        raise ValueError("Sequence batches must be segment [B,T,C] or trip [B,S,T,C]")
    batch_size, segments, time, driver_channels = driver.shape
    valid = batch["segment_mask"].reshape(-1).bool()
    driver = driver.reshape(batch_size * segments, time, driver_channels)[valid]
    battery = battery.reshape(batch_size * segments, time, battery.shape[-1])[valid]
    if driver_mask is not None:
        driver_mask = driver_mask.reshape(
            batch_size * segments, *driver_mask.shape[2:]
        )[valid]
    if battery_mask is not None:
        battery_mask = battery_mask.reshape(
            batch_size * segments, *battery_mask.shape[2:]
        )[valid]
    return driver, battery, driver_mask, battery_mask


def _optimizer_step(
    loss: Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    parameters: list[nn.Parameter],
    *,
    should_step: bool,
    gradient_clip: float,
) -> None:
    scaler.scale(loss).backward()
    if not should_step:
        return
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train_pretraining_epoch(
    model: DBPatchKAN,
    loader: DataLoader,
    criterion: MaskedSelfSupervisedLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    precision: str,
    mask_ratio: float,
    gradient_accumulation: int,
    gradient_clip: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    seen = 0
    parameters = [p for p in model.parameters() if p.requires_grad]
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break
        batch = move_batch(batch, device)
        driver, battery, driver_mask, battery_mask = _ssl_streams(batch)
        with autocast_context(device, precision):
            outputs = model.forward_ssl(
                driver,
                battery,
                driver_mask=driver_mask,
                battery_mask=battery_mask,
                mask_ratio=mask_ratio,
            )
            losses = criterion(outputs)
            scaled_loss = losses["total"] / gradient_accumulation
        should_step = (
            (batch_index + 1) % gradient_accumulation == 0
            or batch_index + 1 == total_batches
        )
        _optimizer_step(
            scaled_loss,
            optimizer,
            scaler,
            parameters,
            should_step=should_step,
            gradient_clip=gradient_clip,
        )
        if should_step:
            scheduler.step()
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach())
        seen += 1
    return {name: value / max(1, seen) for name, value in sums.items()}


def train_finetuning_epoch(
    model: DBPatchKAN,
    loader: DataLoader,
    criterion: DBPatchKANMultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    precision: str,
    domain: str,
    gradient_accumulation: int,
    gradient_clip: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    parameters = [
        parameter
        for module in (model, criterion)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    sums: dict[str, float] = {}
    seen = 0
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break
        batch = move_batch(batch, device)
        with autocast_context(device, precision):
            outputs = _model_forward(model, batch, domain)
            losses = criterion(outputs, batch)
            scaled_loss = losses["total"] / gradient_accumulation
        should_step = (
            (batch_index + 1) % gradient_accumulation == 0
            or batch_index + 1 == total_batches
        )
        _optimizer_step(
            scaled_loss,
            optimizer,
            scaler,
            parameters,
            should_step=should_step,
            gradient_clip=gradient_clip,
        )
        if should_step:
            scheduler.step()
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach())
        seen += 1
    return {name: value / max(1, seen) for name, value in sums.items()}


@torch.no_grad()
def evaluate_finetuning(
    model: DBPatchKAN,
    loader: DataLoader,
    criterion: DBPatchKANMultiTaskLoss,
    *,
    device: torch.device,
    precision: str,
    domain: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    criterion.eval()
    sums: dict[str, float] = {}
    regression_pairs: dict[str, list[tuple[Tensor, Tensor]]] = {
        "current_soh": [], "loss_30d": [], "loss_500km": []
    }
    class_pairs: dict[str, list[tuple[Tensor, Tensor]]] = {
        "behavior": [], "driver": []
    }
    seen = 0
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break
        batch = move_batch(batch, device)
        with autocast_context(device, precision):
            outputs = _model_forward(model, batch, domain)
            losses = criterion(outputs, batch)
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach())
        target_aliases = {
            "current_soh": ("current_soh_target", "soh_target", "soh_target_raw_pct"),
            "loss_30d": ("loss_30d_target", "target_loss_30d_pp"),
            "loss_500km": ("loss_500km_target", "target_loss_500km_pp"),
        }
        for name, aliases in target_aliases.items():
            target = next((batch[key] for key in aliases if key in batch), None)
            if target is None:
                continue
            finite = torch.isfinite(target) & torch.isfinite(outputs[name])
            if torch.any(finite):
                regression_pairs[name].append(
                    (outputs[name][finite].float().cpu(), target[finite].float().cpu())
                )
        for name, target_name, logits_name in (
            ("behavior", "behavior_target", "behavior_logits"),
            ("driver", "driver_target", "driver_logits"),
        ):
            target = batch.get(target_name)
            if target is None:
                continue
            valid = target != -100
            if torch.any(valid):
                class_pairs[name].append(
                    (outputs[logits_name].argmax(-1)[valid].cpu(), target[valid].cpu())
                )
        seen += 1
    metrics = {name: value / max(1, seen) for name, value in sums.items()}
    for name, pairs in regression_pairs.items():
        if not pairs:
            continue
        prediction = torch.cat([pair[0] for pair in pairs]).double()
        target = torch.cat([pair[1] for pair in pairs]).double()
        error = prediction - target
        prefix = "current_soh" if name == "current_soh" else name
        metrics[f"{prefix}_mae_pp"] = float(error.abs().mean())
        metrics[f"{prefix}_rmse_pp"] = float(error.square().mean().sqrt())
        metrics[f"{prefix}_bias_pp"] = float(error.mean())
        metrics[f"{prefix}_p95_error_pp"] = float(torch.quantile(error.abs(), 0.95))
        denominator = (target - target.mean()).square().sum()
        metrics[f"{prefix}_r2"] = (
            float(1.0 - error.square().sum() / denominator)
            if denominator > 0 else float("nan")
        )

    for name, pairs in class_pairs.items():
        if not pairs:
            continue
        prediction = torch.cat([pair[0] for pair in pairs]).long()
        target = torch.cat([pair[1] for pair in pairs]).long()
        labels = torch.unique(target)
        recalls: list[Tensor] = []
        f1_scores: list[Tensor] = []
        for label in labels:
            true_positive = ((prediction == label) & (target == label)).sum().float()
            false_positive = ((prediction == label) & (target != label)).sum().float()
            false_negative = ((prediction != label) & (target == label)).sum().float()
            recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
            precision_value = true_positive / (true_positive + false_positive).clamp_min(1.0)
            f1 = 2 * precision_value * recall / (precision_value + recall).clamp_min(1.0e-12)
            recalls.append(recall)
            f1_scores.append(f1)
            metrics[f"{name}_recall_class_{int(label)}"] = float(recall)
        metrics[f"{name}_accuracy"] = float((prediction == target).float().mean())
        metrics[f"{name}_balanced_accuracy"] = float(torch.stack(recalls).mean())
        metrics[f"{name}_macro_f1"] = float(torch.stack(f1_scores).mean())
    return metrics


def _make_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_fraction: float
) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, multiplier)


def _make_finetune_optimizer(
    model: DBPatchKAN,
    criterion: DBPatchKANMultiTaskLoss,
    config: TrainingConfig,
) -> AdamW:
    encoder_prefixes = (
        "driver_encoder",
        "battery_encoder",
        "driver_to_battery",
        "battery_to_driver",
        "driver_cross_norm",
        "battery_cross_norm",
    )
    encoder_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = encoder_parameters if name.startswith(encoder_prefixes) else head_parameters
        target.append(parameter)
    head_parameters.extend(p for p in criterion.parameters() if p.requires_grad)
    groups: list[dict[str, Any]] = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": config.encoder_lr})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": config.head_lr})
    return AdamW(groups, weight_decay=config.weight_decay)


def _chronological_split(dataset: Dataset, validation_fraction: float = 0.2):
    n = len(dataset)
    if n < 2:
        return dataset, dataset
    split = min(n - 1, max(1, int(round(n * (1.0 - validation_fraction)))))
    return Subset(dataset, range(split)), Subset(dataset, range(split, n))


def _adapt_model_config(
    config: DBPatchKANConfig, dataset: Dataset
) -> DBPatchKANConfig:
    """Match tensor/channel widths while preserving named default concepts."""

    if len(dataset) == 0:
        raise ValueError("Training dataset is empty")
    sample = dataset[0]
    driver_channels = int(sample["driver"].shape[-1])
    battery_channels = int(sample["battery"].shape[-1])
    segment_length = int(sample["driver"].shape[-2])
    concept_values = sample.get("concept_targets", sample.get("observed_concepts"))
    if concept_values is None:
        concept_names = config.concept_names
    else:
        n_concepts = int(concept_values.shape[-1])
        if n_concepts > 20:
            raise ValueError("NPZ contains more than the permitted 20 concepts")
        if n_concepts <= len(config.concept_names):
            concept_names = config.concept_names[:n_concepts]
        else:  # pragma: no cover - default configuration already defines 20
            concept_names = tuple(f"concept_{index}" for index in range(n_concepts))
    return replace(
        config,
        driver_channels=driver_channels,
        battery_channels=battery_channels,
        segment_length=segment_length,
        concept_names=concept_names,
        protected_driver_channels=tuple(
            index for index in config.protected_driver_channels if index < driver_channels
        ),
        protected_battery_channels=tuple(
            index for index in config.protected_battery_channels if index < battery_channels
        ),
    )


def _concept_statistics(dataset: Dataset, n_concepts: int) -> tuple[Tensor, Tensor]:
    rows: list[Tensor] = []
    for index in range(len(dataset)):
        item = dataset[index]
        value = item.get("concept_targets", item.get("observed_concepts"))
        if value is not None:
            rows.append(value.float())
    if not rows:
        return torch.zeros(n_concepts), torch.ones(n_concepts)
    values = torch.stack(rows)
    center = torch.zeros(n_concepts)
    scale = torch.ones(n_concepts)
    for column in range(n_concepts):
        finite = torch.isfinite(values[:, column])
        if torch.any(finite):
            center[column] = values[finite, column].mean()
            std = values[finite, column].std(unbiased=False)
            scale[column] = std if std > 1.0e-6 else 1.0
    return center, scale


def _stream_statistics(
    dataset: Dataset,
    value_key: str,
    mask_key: str,
    width: int,
    physical_width: int,
) -> tuple[Tensor, Tensor]:
    """Fit finite, mask-aware channel statistics on one training fold only."""

    count = torch.zeros(width, dtype=torch.float64)
    total = torch.zeros(width, dtype=torch.float64)
    total_square = torch.zeros(width, dtype=torch.float64)
    fitted_width = min(width, physical_width)
    for index in range(len(dataset)):
        item = dataset[index]
        values = item[value_key].detach().to(dtype=torch.float64)
        if values.ndim not in {2, 3} or values.shape[-1] != width:
            raise ValueError(f"{value_key} must have shape [T,{width}] or [S,T,{width}]")
        values = values.reshape(-1, width)[:, :fitted_width]
        valid = torch.isfinite(values)
        mask = item.get(mask_key)
        if mask is not None:
            mask = mask.detach().to(dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(-1).expand(-1, width)
            elif mask.ndim == 3:
                mask = mask.reshape(-1, width)
            valid &= mask[:, :fitted_width]
        safe = torch.where(valid, values, torch.zeros_like(values))
        count[:fitted_width] += valid.sum(dim=0)
        total[:fitted_width] += safe.sum(dim=0)
        total_square[:fitted_width] += safe.square().sum(dim=0)
    center = torch.zeros(width, dtype=torch.float32)
    scale = torch.ones(width, dtype=torch.float32)
    observed = count[:fitted_width] > 0
    means = torch.zeros(fitted_width, dtype=torch.float64)
    means[observed] = total[:fitted_width][observed] / count[:fitted_width][observed]
    variance = torch.zeros(fitted_width, dtype=torch.float64)
    variance[observed] = (
        total_square[:fitted_width][observed] / count[:fitted_width][observed]
        - means[observed].square()
    ).clamp_min(0.0)
    standard_deviation = variance.sqrt()
    standard_deviation = torch.where(
        standard_deviation > 1.0e-6, standard_deviation, torch.ones_like(standard_deviation)
    )
    center[:fitted_width] = means.float()
    scale[:fitted_width] = standard_deviation.float()
    return center, scale


def fit_input_normalization(
    dataset: Dataset,
    driver_channels: int,
    battery_channels: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return fold-fitted stats while leaving four shared metadata channels unchanged."""

    driver_physical = driver_channels - 4 if driver_channels >= 13 else driver_channels
    battery_physical = battery_channels - 4 if battery_channels >= 11 else battery_channels
    driver_center, driver_scale = _stream_statistics(
        dataset, "driver", "driver_mask", driver_channels, driver_physical
    )
    battery_center, battery_scale = _stream_statistics(
        dataset, "battery", "battery_mask", battery_channels, battery_physical
    )
    return driver_center, driver_scale, battery_center, battery_scale


def _driver_class_weights(dataset: Dataset, num_classes: int) -> Tensor | None:
    """Inverse-frequency weights from eligible training-fold Driver_ID labels."""

    counts = torch.zeros(num_classes, dtype=torch.float32)
    for index in range(len(dataset)):
        item = dataset[index]
        target = item.get("driver_target", item.get("driver_id_target"))
        if target is None:
            continue
        value = int(torch.as_tensor(target).item())
        if 0 <= value < num_classes:
            counts[value] += 1
    present = counts > 0
    if not torch.any(present):
        return None
    # Give an unseen-but-eligible validation class a neutral finite weight.
    # A zero weight makes PyTorch's mean cross-entropy divide by zero when a
    # validation batch contains only that class, yielding NaN smoke metrics.
    weights = torch.ones_like(counts)
    weights[present] = counts[present].sum() / (
        present.sum().float() * counts[present]
    )
    return weights


def _loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    workers: int,
    device: torch.device,
) -> DataLoader:
    def dataset_collate(current: Dataset):
        collate = getattr(current, "collate_fn", None)
        if collate is not None:
            return collate
        nested = getattr(current, "dataset", None)
        return None if nested is None else dataset_collate(nested)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
        collate_fn=dataset_collate(dataset),
    )


def _trip_bag_base(dataset: Dataset) -> EVTripBagDataset | None:
    """Find a trip-bag dataset through standard dataset view wrappers."""

    current: Any = dataset
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, EVTripBagDataset):
            return current
        nested = getattr(current, "dataset", None)
        if nested is None:
            break
        current = nested
    return None


def _save_checkpoint(
    path: Path,
    *,
    model: DBPatchKAN,
    criterion: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    stage: str,
    epoch: int,
    metrics: dict[str, float],
    training_config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "criterion": None if criterion is None else criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "stage": stage,
            "epoch": epoch,
            "metrics": metrics,
            "model_config": model.config.to_dict(),
            "training_config": asdict(training_config),
        },
        temporary,
    )
    os.replace(temporary, path)


def train(
    model: DBPatchKAN,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: TrainingConfig,
    *,
    output_dir: str | Path = "outputs/dbpatchkan",
) -> dict[str, Any]:
    """Run the selected stages and return JSON-serialisable final metrics."""

    seed_everything(config.seed)
    training_bags = _trip_bag_base(train_dataset)
    validation_bags = _trip_bag_base(validation_dataset)
    if (training_bags is None) != (validation_bags is None):
        raise ValueError("Training and validation must use the same segment/trip dataset kind")
    if training_bags is not None and validation_bags is not None:
        assert_disjoint_trip_bags(training_bags, validation_bags)
        # Persist the actual path in run_config/checkpoints even for callers
        # that constructed TrainingConfig programmatically.
        config.trip_bags = True
    device = resolve_device(config.device)
    precision = resolve_precision(config.precision, device)
    input_stats = fit_input_normalization(
        train_dataset, model.config.driver_channels, model.config.battery_channels
    )
    model.set_input_normalization(*input_stats)
    model = model.to(device)
    center, scale = _concept_statistics(train_dataset, model.config.n_concepts)
    model.set_concept_normalization(center.to(device), scale.to(device))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "model": model.config.to_dict(),
                "training": asdict(config),
                "resolved_device": str(device),
                "resolved_precision": precision,
                "input_normalization": {
                    "driver_center": input_stats[0].tolist(),
                    "driver_scale": input_stats[1].tolist(),
                    "battery_center": input_stats[2].tolist(),
                    "battery_scale": input_stats[3].tolist(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history: list[dict[str, Any]] = []
    scaler = torch.amp.GradScaler(
        device.type, enabled=device.type == "cuda" and precision == "fp16"
    )
    resume_state: dict[str, Any] | None = None
    if config.resume_checkpoint:
        resume_state = torch.load(
            config.resume_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(resume_state["model"])
        if resume_state.get("scaler"):
            scaler.load_state_dict(resume_state["scaler"])

    resume_stage = None if resume_state is None else resume_state.get("stage")
    if config.stage in {"pretrain", "both"} and resume_stage != "finetune":
        # Trip bags can contain hundreds of overlapping segments.  SSL has no
        # trip-level labels, so train it on the fold's underlying segments;
        # hierarchical collation is reserved for supervised fine-tuning.
        pretraining_dataset: Dataset = (
            training_bags.segments if training_bags is not None else train_dataset
        )
        train_loader = _loader(
            pretraining_dataset,
            config.pretrain_batch_size,
            shuffle=True,
            workers=config.num_workers,
            device=device,
        )
        ssl_criterion = MaskedSelfSupervisedLoss(
            LossWeights(ssl_alignment=config.ssl_alignment_weight)
        ).to(device)
        pretrain_optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.pretrain_lr,
            weight_decay=config.weight_decay,
        )
        batches = (
            len(train_loader)
            if config.max_batches is None
            else min(len(train_loader), config.max_batches)
        )
        total_steps = max(
            1,
            math.ceil(batches / config.gradient_accumulation) * config.pretrain_epochs,
        )
        pretrain_scheduler = _make_scheduler(
            pretrain_optimizer, total_steps, config.warmup_fraction
        )
        pretrain_start = 0
        if resume_state is not None and resume_stage == "pretrain":
            pretrain_optimizer.load_state_dict(resume_state["optimizer"])
            pretrain_scheduler.load_state_dict(resume_state["scheduler"])
            pretrain_start = int(resume_state.get("epoch", 0))
        for epoch in range(pretrain_start, config.pretrain_epochs):
            metrics = train_pretraining_epoch(
                model,
                train_loader,
                ssl_criterion,
                pretrain_optimizer,
                pretrain_scheduler,
                scaler,
                device=device,
                precision=precision,
                mask_ratio=config.mask_ratio,
                gradient_accumulation=config.gradient_accumulation,
                gradient_clip=config.gradient_clip,
                max_batches=config.max_batches,
            )
            record = {"stage": "pretrain", "epoch": epoch + 1, **metrics}
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            _save_checkpoint(
                output / "pretrain_latest.pt",
                model=model,
                criterion=ssl_criterion,
                optimizer=pretrain_optimizer,
                scheduler=pretrain_scheduler,
                scaler=scaler,
                stage="pretrain",
                epoch=epoch + 1,
                metrics=record,
                training_config=config,
            )

    if config.stage in {"finetune", "both"}:
        if config.freeze_encoders:
            model.freeze_patch_encoders(True)
        train_loader = _loader(
            train_dataset,
            config.finetune_batch_size,
            shuffle=True,
            workers=config.num_workers,
            device=device,
        )
        validation_loader = _loader(
            validation_dataset,
            config.finetune_batch_size,
            shuffle=False,
            workers=config.num_workers,
            device=device,
        )
        driver_class_weight = _driver_class_weights(
            train_dataset, model.config.num_drivers
        )
        criterion = DBPatchKANMultiTaskLoss(
            LossWeights(
                concept=config.concept_weight,
                physics=config.physics_weight,
                driver_id=config.driver_id_weight,
            ),
            driver_class_weight=driver_class_weight,
        ).to(device)
        optimizer = _make_finetune_optimizer(model, criterion, config)
        batches = (
            len(train_loader)
            if config.max_batches is None
            else min(len(train_loader), config.max_batches)
        )
        total_steps = max(
            1,
            math.ceil(batches / config.gradient_accumulation) * config.finetune_epochs,
        )
        scheduler = _make_scheduler(optimizer, total_steps, config.warmup_fraction)
        best = float("inf")
        stale = 0
        finetune_start = 0
        if resume_state is not None and resume_stage == "finetune":
            if resume_state.get("criterion"):
                criterion.load_state_dict(resume_state["criterion"])
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            finetune_start = int(resume_state.get("epoch", 0))
            best = float(
                resume_state.get("metrics", {}).get("validation_total", float("inf"))
            )
        for epoch in range(finetune_start, config.finetune_epochs):
            train_metrics = train_finetuning_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                device=device,
                precision=precision,
                domain=config.domain,
                gradient_accumulation=config.gradient_accumulation,
                gradient_clip=config.gradient_clip,
                max_batches=config.max_batches,
            )
            validation_metrics = evaluate_finetuning(
                model,
                validation_loader,
                criterion,
                device=device,
                precision=precision,
                domain=config.domain,
                max_batches=config.max_batches,
            )
            record = {
                "stage": "finetune",
                "epoch": epoch + 1,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            _save_checkpoint(
                output / "latest.pt",
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                stage="finetune",
                epoch=epoch + 1,
                metrics=record,
                training_config=config,
            )
            score = validation_metrics.get("total", float("inf"))
            if score < best:
                best = score
                stale = 0
                _save_checkpoint(
                    output / "best.pt",
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    stage="finetune",
                    epoch=epoch + 1,
                    metrics=record,
                    training_config=config,
                )
            else:
                stale += 1
                if stale >= config.patience:
                    break

    (output / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return {
        "device": str(device),
        "precision": precision,
        "epochs_completed": len(history),
        "last": history[-1] if history else {},
        "output_dir": str(output.resolve()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        ),
    }


def run_synthetic_smoke(
    *,
    device: str = "cpu",
    output_dir: str | Path = "outputs/dbpatchkan_smoke",
    fusion_head: str = "kan",
    seed: int = 42,
) -> dict[str, Any]:
    """One-batch CPU/GPU end-to-end smoke entry for CI and the root CLI."""

    model_cfg = DBPatchKANConfig(
        segment_length=32,
        patch_length=8,
        patch_stride=4,
        d_model=32,
        encoder_layers=1,
        n_heads=4,
        ff_dim=64,
        dropout=0.0,
        fusion_head=fusion_head,
    )
    dataset = make_synthetic_dataset(model_cfg, n_samples=6, seed=seed)
    training, validation = _chronological_split(dataset)
    training_cfg = TrainingConfig(
        stage="both",
        device=device,
        precision="fp32" if device == "cpu" else "auto",
        pretrain_batch_size=4,
        finetune_batch_size=4,
        pretrain_epochs=1,
        finetune_epochs=1,
        patience=1,
        seed=seed,
        max_batches=1,
    )
    return train(
        DBPatchKAN(model_cfg), training, validation, training_cfg, output_dir=output_dir
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="NPZ segment tensor file")
    parser.add_argument(
        "--validation-data", type=Path, help="Optional separate NPZ validation file"
    )
    parser.add_argument(
        "--trip-bags",
        action="store_true",
        help="Group overlapping segment windows by trip and optimize one label row per trip",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dbpatchkan"))
    parser.add_argument("--stage", choices=("pretrain", "finetune", "both"), default="both")
    parser.add_argument("--domain", choices=("nissan", "capacity"), default="nissan")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-samples", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fusion-head", choices=("kan", "mlp"), default="kan")
    parser.add_argument("--input-modality", choices=("both", "driver", "battery"), default="both")
    parser.add_argument(
        "--cross-attention-mode",
        choices=("concatenation", "one_way", "bidirectional"),
        default="bidirectional",
    )
    parser.add_argument("--prediction-input", choices=("concepts", "latent"), default="concepts")
    parser.add_argument("--segment-length", type=int, default=256)
    parser.add_argument("--patch-length", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--kan-grid", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--finetune-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--pretrain-lr", type=float, default=3.0e-4)
    parser.add_argument("--encoder-lr", type=float, default=1.0e-5)
    parser.add_argument("--head-lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--concept-weight", type=float, default=0.2)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--driver-id-weight", type=float, default=0.1)
    parser.add_argument("--ssl-alignment-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-encoders", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", type=Path, help="Resume from a latest/best checkpoint")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    if args.dry_run:
        result = run_synthetic_smoke(
            device=args.device if args.device != "auto" else "cpu",
            output_dir=args.output_dir,
            fusion_head=args.fusion_head,
            seed=args.seed,
        )
    else:
        model_cfg = DBPatchKANConfig(
            segment_length=args.segment_length,
            patch_length=args.patch_length,
            patch_stride=args.patch_stride,
            d_model=args.d_model,
            encoder_layers=args.encoder_layers,
            n_heads=args.heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            fusion_head=args.fusion_head,
            input_modality=args.input_modality,
            cross_attention_mode=args.cross_attention_mode,
            prediction_input=args.prediction_input,
            kan_grid_size=args.kan_grid,
        )
        dataset_type = EVTripBagDataset if args.trip_bags else EVSegmentDataset
        if args.data:
            full_dataset: Dataset = dataset_type.from_npz(args.data)
        else:
            full_dataset = make_synthetic_dataset(
                model_cfg, n_samples=args.synthetic_samples, seed=args.seed
            )
        if args.validation_data:
            train_dataset = full_dataset
            validation_dataset: Dataset = dataset_type.from_npz(args.validation_data)
            if args.trip_bags:
                assert_disjoint_trip_bags(train_dataset, validation_dataset)
        elif args.data:
            raise ValueError(
                "Production NPZ training requires --validation-data with a "
                "trip/vehicle-separated split. Automatic window-row splitting "
                "would leak overlapping segments across folds."
            )
        else:
            train_dataset, validation_dataset = _chronological_split(full_dataset)
        model_cfg = _adapt_model_config(model_cfg, train_dataset)
        training_cfg = TrainingConfig(
            stage=args.stage,
            device=args.device,
            precision=args.precision,
            pretrain_batch_size=args.pretrain_batch_size,
            finetune_batch_size=args.finetune_batch_size,
            gradient_accumulation=args.gradient_accumulation,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.epochs,
            pretrain_lr=args.pretrain_lr,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            warmup_fraction=args.warmup_fraction,
            gradient_clip=args.gradient_clip,
            patience=args.patience,
            mask_ratio=args.mask_ratio,
            num_workers=args.num_workers,
            seed=args.seed,
            domain=args.domain,
            freeze_encoders=args.freeze_encoders,
            trip_bags=args.trip_bags,
            resume_checkpoint=None if args.resume is None else str(args.resume),
            concept_weight=args.concept_weight,
            physics_weight=args.physics_weight,
            driver_id_weight=args.driver_id_weight,
            ssl_alignment_weight=args.ssl_alignment_weight,
        )
        result = train(
            DBPatchKAN(model_cfg),
            train_dataset,
            validation_dataset,
            training_cfg,
            output_dir=args.output_dir,
        )
    result["elapsed_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TrainingConfig",
    "seed_everything",
    "resolve_device",
    "resolve_precision",
    "train_pretraining_epoch",
    "train_finetuning_epoch",
    "evaluate_finetuning",
    "fit_input_normalization",
    "train",
    "run_synthetic_smoke",
    "build_parser",
    "main",
]
