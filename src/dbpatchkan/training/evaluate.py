"""One-way, checkpoint-only evaluation on an untouched NPZ split.

This module deliberately contains no optimiser, scheduler, fitting, calibration,
or checkpoint-writing path.  It restores all learned parameters and training-fold
normalisation buffers from a selected fine-tuning checkpoint, freezes the model,
and writes predictions plus metrics once to a fresh output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dbpatchkan.models import DBPatchKAN, DBPatchKANConfig, EVSegmentDataset
from dbpatchkan.training.train import (
    autocast_context,
    move_batch,
    resolve_device,
    resolve_precision,
)


_TUPLE_CONFIG_FIELDS = {
    "concept_names",
    "kan_hidden",
    "mlp_hidden",
    "protected_driver_channels",
    "protected_battery_channels",
    "calibration_domains",
}


class _InferenceConceptView(Dataset[dict[str, Tensor]]):
    """Slice checkpoint-selected concepts without requiring test labels."""

    _CONCEPT_KEYS = (
        "concept_targets",
        "concept_mask",
        "observed_concepts",
        "observed_concept_mask",
    )

    def __init__(self, dataset: Dataset[dict[str, Tensor]], n_concepts: int) -> None:
        if len(dataset) == 0:
            raise ValueError("Untouched-test dataset is empty")
        sample = dataset[0]
        for key in self._CONCEPT_KEYS:
            if key in sample and sample[key].shape[-1] < n_concepts:
                raise ValueError(
                    f"Test {key} has fewer concepts than the frozen checkpoint"
                )
        self.dataset = dataset
        self.n_concepts = n_concepts

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = dict(self.dataset[index])
        for key in self._CONCEPT_KEYS:
            if key in item:
                item[key] = item[key][..., : self.n_concepts]
        return item


def _sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(model: DBPatchKAN) -> str:
    """Hash parameters and persistent buffers in stable state-dict order."""

    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _restore_model_config(raw: dict[str, Any]) -> DBPatchKANConfig:
    values = dict(raw)
    for name in _TUPLE_CONFIG_FIELDS:
        if name in values:
            values[name] = tuple(values[name])
    return DBPatchKANConfig(**values)


def load_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[DBPatchKAN, dict[str, Any]]:
    """Restore and freeze a selected fine-tuning checkpoint without mutation."""

    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"model", "model_config"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing {sorted(missing)}")
    if checkpoint.get("stage") not in {None, "finetune"}:
        raise ValueError("Untouched-test evaluation requires a fine-tuning checkpoint")
    model = DBPatchKAN(_restore_model_config(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, checkpoint


def detect_npz_dataset_kind(path: str | Path) -> str:
    """Detect segment versus trip-bag exchange data from numeric NPZ structure."""

    with np.load(Path(path), allow_pickle=False) as arrays:
        names = set(arrays.files)
        if {"trip_offsets", "bag_offsets", "segment_mask"} & names:
            return "trip"
        driver_name = "driver" if "driver" in names else "driver_values"
        if driver_name not in names:
            raise ValueError("NPZ does not contain a driver tensor")
        rank = arrays[driver_name].ndim
    if rank == 4:
        return "trip"
    if rank == 3:
        return "segment"
    raise ValueError(f"Unsupported driver tensor rank {rank}; expected 3 or 4")


def _load_dataset(
    path: Path, kind: str
) -> tuple[Dataset[dict[str, Tensor]], Any | None, str]:
    resolved_kind = detect_npz_dataset_kind(path) if kind == "auto" else kind
    if resolved_kind == "segment":
        return EVSegmentDataset.from_npz(path), None, resolved_kind
    if resolved_kind != "trip":
        raise ValueError("dataset kind must be auto, segment, or trip")
    try:
        from dbpatchkan.models import EVTripBagDataset, collate_trip_bags
    except ImportError as error:
        raise RuntimeError(
            "Trip-bag NPZ detected, but the installed model package has no trip-bag API"
        ) from error
    return EVTripBagDataset.from_npz(path), collate_trip_bags, resolved_kind


def _decode_utf8_rows(encoded: np.ndarray) -> list[str]:
    if encoded.ndim != 2:
        raise ValueError("UTF-8 metadata must have shape [N, bytes]")
    decoded: list[str] = []
    for row in encoded.astype(np.uint8, copy=False):
        payload = bytes(row[row != 0].tolist())
        decoded.append(payload.decode("utf-8", errors="replace"))
    return decoded


def load_lineage_metadata(path: str | Path, expected_rows: int) -> pd.DataFrame:
    """Read identifiers for reporting only; they never enter model inputs."""

    columns: dict[str, Any] = {"sample_index": np.arange(expected_rows)}
    numeric_names = (
        "window_index",
        "window_start_unix_ms",
        "window_end_unix_ms",
        "segment_id",
        "coverage_pct",
        "trip_index",
        "segments_per_trip",
    )
    with np.load(Path(path), allow_pickle=False) as arrays:
        for name in numeric_names:
            if name in arrays and arrays[name].ndim == 1 and len(arrays[name]) == expected_rows:
                columns[name] = arrays[name]
        for name in arrays.files:
            if not name.endswith("_utf8"):
                continue
            if arrays[name].ndim == 2 and len(arrays[name]) == expected_rows:
                columns[name.removesuffix("_utf8")] = _decode_utf8_rows(arrays[name])
    return pd.DataFrame(columns)


def _forward(model: DBPatchKAN, batch: dict[str, Any], domain: str) -> dict[str, Tensor]:
    common = {
        "driver_mask": batch.get("driver_mask"),
        "battery_mask": batch.get("battery_mask"),
        "observed_concepts": batch.get("observed_concepts"),
        "observed_concept_mask": batch.get("observed_concept_mask"),
        "domain": domain,
    }
    if batch["driver"].ndim == 4:
        if not hasattr(model, "forward_trip_bag"):
            raise RuntimeError("Checkpoint model does not support trip-bag inference")
        return model.forward_trip_bag(
            batch["driver"],
            batch["battery"],
            segment_mask=batch["segment_mask"],
            **common,
        )
    if batch["driver"].ndim != 3:
        raise ValueError("Expected segment [B,T,C] or trip bag [B,S,T,C]")
    return model(batch["driver"], batch["battery"], **common)


def _target(batch: dict[str, Any], *names: str) -> Tensor | None:
    return next((batch[name] for name in names if name in batch), None)


def _regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid].astype(np.float64)
    predicted = predicted[valid].astype(np.float64)
    if not len(observed):
        return {}
    error = predicted - observed
    denominator = np.square(observed - observed.mean()).sum()
    return {
        "count": int(len(observed)),
        "mae_pp": float(np.abs(error).mean()),
        "rmse_pp": float(np.sqrt(np.square(error).mean())),
        "bias_pp": float(error.mean()),
        "p95_absolute_error_pp": float(np.quantile(np.abs(error), 0.95)),
        "r2": float(1.0 - np.square(error).sum() / denominator)
        if denominator > 0
        else float("nan"),
    }


def _classification_metrics(
    observed: np.ndarray, probabilities: np.ndarray, *, ignore_index: int = -100
) -> dict[str, Any]:
    finite = np.isfinite(observed)
    observed_integer = np.full(observed.shape, ignore_index, dtype=np.int64)
    observed_integer[finite] = observed[finite].astype(np.int64)
    valid = finite & (observed_integer != ignore_index)
    valid &= (observed_integer >= 0) & (observed_integer < probabilities.shape[1])
    if not np.any(valid):
        return {}
    observed = observed_integer[valid]
    probabilities = probabilities[valid].astype(np.float64)
    predicted = probabilities.argmax(axis=1)
    labels = np.unique(observed)
    recalls: list[float] = []
    f1_values: list[float] = []
    per_class: dict[str, float] = {}
    for label in labels:
        true_positive = np.sum((predicted == label) & (observed == label))
        false_positive = np.sum((predicted == label) & (observed != label))
        false_negative = np.sum((predicted != label) & (observed == label))
        recall = true_positive / max(1, true_positive + false_negative)
        precision = true_positive / max(1, true_positive + false_positive)
        f1 = 2 * precision * recall / max(1.0e-12, precision + recall)
        recalls.append(float(recall))
        f1_values.append(float(f1))
        per_class[f"recall_class_{int(label)}"] = float(recall)
    selected = np.clip(probabilities[np.arange(len(observed)), observed], 1.0e-12, 1.0)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[observed]
    return {
        "count": int(len(observed)),
        "accuracy": float(np.mean(predicted == observed)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "log_loss": float(-np.log(selected).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        **per_class,
    }


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    test_data_path: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    precision_name: str = "auto",
    domain: str = "auto",
    dataset_kind: str = "auto",
    batch_size: int = 128,
    num_workers: int = 0,
    acknowledge_final_evaluation: bool = False,
) -> dict[str, Any]:
    """Evaluate once without fitting, calibration, gradient, or weight updates."""

    if not acknowledge_final_evaluation:
        raise ValueError(
            "Final model/config must be frozen before untouched-test evaluation"
        )
    checkpoint_path = Path(checkpoint_path).resolve()
    test_data_path = Path(test_data_path).resolve()
    destination = Path(output_dir).resolve()
    predictions_path = destination / "untouched_test_predictions.csv"
    metrics_path = destination / "untouched_test_metrics.json"
    if predictions_path.exists() or metrics_path.exists():
        raise FileExistsError(
            "Untouched-test outputs already exist; use a new audit directory, not overwrite"
        )
    destination.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    precision = resolve_precision(precision_name, device)
    model, checkpoint = load_checkpoint_model(checkpoint_path, device=device)
    checkpoint_domain = checkpoint.get("training_config", {}).get("domain", "nissan")
    resolved_domain = checkpoint_domain if domain == "auto" else domain
    if resolved_domain not in model.config.calibration_domains:
        raise ValueError(f"Checkpoint has no {resolved_domain!r} calibration head")

    if dataset_kind == "auto":
        dataset_kind = (
            "trip"
            if checkpoint.get("training_config", {}).get("trip_bags")
            else "segment"
        )
    dataset, collate, resolved_kind = _load_dataset(test_data_path, dataset_kind)
    trip_group_ids = getattr(dataset, "group_ids", None)
    trip_segment_counts = (
        [int(dataset[index]["driver"].shape[0]) for index in range(len(dataset))]
        if trip_group_ids is not None
        else None
    )
    dataset = _InferenceConceptView(dataset, model.config.n_concepts)
    sample = dataset[0]
    expected_rank = 4 if resolved_kind == "trip" else 3
    if sample["driver"].ndim != expected_rank - 1:
        raise ValueError("Dataset sample rank does not match its declared evaluation grain")
    if sample["driver"].shape[-1] != model.config.driver_channels:
        raise ValueError("Test driver-channel count differs from the frozen checkpoint")
    if sample["battery"].shape[-1] != model.config.battery_channels:
        raise ValueError("Test battery-channel count differs from the frozen checkpoint")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )
    fingerprint_before = state_fingerprint(model)
    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with autocast_context(device, precision):
                outputs = _forward(model, batch, resolved_domain)
            count = int(outputs["current_soh"].shape[0])
            behavior_probability = torch.softmax(
                outputs["behavior_logits"].float(), dim=-1
            ).cpu()
            driver_probability = torch.softmax(
                outputs["driver_logits"].float(), dim=-1
            ).cpu()
            wanted_outputs = {
                "current_soh",
                "loss_30d",
                "loss_500km",
                "future_soh_30d",
                "future_soh_500km",
                "concepts",
                "trip_segment_attention",
            }
            output_cpu = {
                name: outputs[name].float().cpu()
                for name in wanted_outputs
                if name in outputs
            }
            targets = {
                "current_soh_observed_pct": _target(
                    batch, "current_soh_target", "soh_target", "soh_target_raw_pct"
                ),
                "loss_30d_observed_pp": _target(
                    batch, "loss_30d_target", "target_loss_30d_pp"
                ),
                "loss_500km_observed_pp": _target(
                    batch, "loss_500km_target", "target_loss_500km_pp"
                ),
                "behavior_observed": _target(batch, "behavior_target", "behavior_level"),
                "driver_observed": _target(batch, "driver_target", "driver_id_target"),
            }
            for index in range(count):
                row: dict[str, Any] = {
                    "sample_index": offset + index,
                    "split_role": "untouched_test",
                    "current_soh_pred_pct": float(output_cpu["current_soh"][index]),
                    "loss_30d_pred_pp": float(output_cpu["loss_30d"][index]),
                    "loss_500km_pred_pp": float(output_cpu["loss_500km"][index]),
                    "future_soh_30d_pred_pct": float(
                        output_cpu["future_soh_30d"][index]
                    ),
                    "future_soh_500km_pred_pct": float(
                        output_cpu["future_soh_500km"][index]
                    ),
                    "behavior_pred": int(behavior_probability[index].argmax()),
                    "driver_pred": int(driver_probability[index].argmax()),
                }
                for class_index, probability in enumerate(behavior_probability[index]):
                    row[f"behavior_probability_{class_index}"] = float(probability)
                for class_index, probability in enumerate(driver_probability[index]):
                    row[f"driver_probability_{class_index}"] = float(probability)
                for concept_index, concept_name in enumerate(model.config.concept_names):
                    row[f"concept_pred_{concept_name}"] = float(
                        output_cpu["concepts"][index, concept_index]
                    )
                for name, values in targets.items():
                    row[name] = (
                        float(values[index].detach().cpu())
                        if values is not None
                        else float("nan")
                    )
                if "trip_segment_attention" in output_cpu:
                    attention = output_cpu["trip_segment_attention"][index]
                    positive = attention[attention > 0]
                    row["trip_attention_max"] = float(attention.max())
                    row["trip_attention_entropy"] = float(
                        -(positive * positive.log()).sum()
                    )
                    row["segments_used"] = int((attention > 0).sum())
                rows.append(row)
            offset += count
    fingerprint_after = state_fingerprint(model)
    if fingerprint_before != fingerprint_after:
        raise RuntimeError("Frozen checkpoint state changed during untouched-test evaluation")

    predictions = pd.DataFrame(rows)
    lineage = load_lineage_metadata(test_data_path, len(predictions))
    if trip_group_ids is not None:
        lineage["dataset_id"] = [group[0] for group in trip_group_ids]
        lineage["vehicle_id"] = [group[1] for group in trip_group_ids]
        lineage["trip_key"] = [group[2] for group in trip_group_ids]
        lineage["segments_per_trip"] = trip_segment_counts
    predictions = lineage.merge(predictions, on="sample_index", how="right", validate="one_to_one")
    for observed_name, predicted_name, prefix in (
        ("current_soh_observed_pct", "current_soh_pred_pct", "current_soh"),
        ("loss_30d_observed_pp", "loss_30d_pred_pp", "loss_30d"),
        ("loss_500km_observed_pp", "loss_500km_pred_pp", "loss_500km"),
    ):
        predictions[f"{prefix}_residual_pp"] = (
            predictions[predicted_name] - predictions[observed_name]
        )
        predictions[f"{prefix}_absolute_error_pp"] = predictions[
            f"{prefix}_residual_pp"
        ].abs()
    regression = {
        "current_soh": _regression_metrics(
            predictions["current_soh_observed_pct"].to_numpy(),
            predictions["current_soh_pred_pct"].to_numpy(),
        ),
        "loss_30d": _regression_metrics(
            predictions["loss_30d_observed_pp"].to_numpy(),
            predictions["loss_30d_pred_pp"].to_numpy(),
        ),
        "loss_500km": _regression_metrics(
            predictions["loss_500km_observed_pp"].to_numpy(),
            predictions["loss_500km_pred_pp"].to_numpy(),
        ),
    }
    behavior_columns = sorted(
        column for column in predictions if column.startswith("behavior_probability_")
    )
    driver_columns = sorted(
        column for column in predictions if column.startswith("driver_probability_")
    )
    classification = {
        "behavior": _classification_metrics(
            predictions["behavior_observed"].to_numpy(),
            predictions[behavior_columns].to_numpy(),
        ),
        "driver": _classification_metrics(
            predictions["driver_observed"].to_numpy(),
            predictions[driver_columns].to_numpy(),
        ),
    }
    report = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "frozen_untouched_test_complete",
        "split_role": "untouched_test",
        "evaluation_grain": resolved_kind,
        "fit_operations": [],
        "optimizer_created": False,
        "gradients_enabled": False,
        "weights_updated": False,
        "state_fingerprint_before": fingerprint_before,
        "state_fingerprint_after": fingerprint_after,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection_metrics": checkpoint.get("metrics", {}),
        "test_data_path": str(test_data_path),
        "test_data_sha256": _sha256_file(test_data_path),
        "sample_count": len(predictions),
        "device": str(device),
        "precision": precision,
        "domain": resolved_domain,
        "model_config": model.config.to_dict(),
        "regression": regression,
        "classification": classification,
    }
    predictions_temporary = predictions_path.with_suffix(".csv.tmp")
    predictions.to_csv(predictions_temporary, index=False)
    os.replace(predictions_temporary, predictions_path)
    _atomic_write_text(metrics_path, json.dumps(_finite_json(report), indent=2))
    return {
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "sample_count": len(predictions),
        "weights_updated": False,
        "state_fingerprint": fingerprint_after,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-kind", choices=("auto", "segment", "trip"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto"
    )
    parser.add_argument(
        "--domain", choices=("auto", "nissan", "capacity"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--final-evaluation",
        action="store_true",
        help="Acknowledge that model/config selection is frozen before opening test data",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.final_evaluation:
        raise ValueError(
            "Pass --final-evaluation only after freezing model/config selection"
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        device_name=args.device,
        precision_name=args.precision,
        domain=args.domain,
        dataset_kind=args.dataset_kind,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        acknowledge_final_evaluation=True,
    )
    print(json.dumps(result, indent=2))
    return 0


evaluation_main = main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "state_fingerprint",
    "load_checkpoint_model",
    "detect_npz_dataset_kind",
    "load_lineage_metadata",
    "evaluate_checkpoint",
    "build_parser",
    "main",
    "evaluation_main",
]
