"""Train one leakage-safe PI-DB-PatchKAN seed on Nissan development data."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from dbpatchkan.models import (
    DBPatchKAN,
    DBPatchKANConfig,
    EVTripBagDataset,
    NISSAN_PHYSICS_CONCEPT_NAMES,
    NISSAN_RESPONSE_TARGET_NAMES,
    PhysicsInformedDBPatchKAN,
    collate_trip_bags,
)


def update(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.zeros(values.shape[1], dtype=np.float32)
    scale = np.ones(values.shape[1], dtype=np.float32)
    for column in range(values.shape[1]):
        current = values[:, column]
        finite = np.isfinite(current)
        if finite.any():
            center[column] = np.mean(current[finite])
            std = np.std(current[finite])
            scale[column] = std if std > 1.0e-6 else 1.0
    return center, scale


def fit_ambient_adjustment(fit_table: pd.DataFrame) -> Any:
    predictors = ["ambient_mean_c", "physics_duration_h", "hvac_energy_kwh"]
    target = pd.to_numeric(fit_table["pack_temp_rise_c"], errors="coerce")
    valid = target.notna()
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(fit_table.loc[valid, predictors], target.loc[valid])
    return model


def apply_ambient_adjustment(table: pd.DataFrame, model: Any) -> pd.DataFrame:
    output = table.copy()
    predictors = ["ambient_mean_c", "physics_duration_h", "hvac_energy_kwh"]
    expected = model.predict(output[predictors])
    output["ambient_expected_pack_temp_rise_c"] = expected
    output["ambient_adjusted_pack_temp_rise_c"] = (
        pd.to_numeric(output["pack_temp_rise_c"], errors="coerce") - expected
    )
    return output


class PhysicsTripDataset(Dataset[dict[str, Tensor]]):
    """Attach named physics/response targets to existing variable-size trip bags."""

    def __init__(self, bags: EVTripBagDataset, table: pd.DataFrame) -> None:
        self.bags = bags
        indexed = table.set_index("trip_key", drop=False)
        rows = []
        for _, _, trip_key in bags.group_ids:
            if trip_key not in indexed.index:
                raise KeyError(f"Physics table has no row for {trip_key}")
            row = indexed.loc[trip_key]
            if isinstance(row, pd.DataFrame):
                raise ValueError(f"Duplicate physics rows for {trip_key}")
            rows.append(row)
        self.table = pd.DataFrame(rows).reset_index(drop=True)
        self.physics = torch.tensor(
            self.table.loc[:, list(NISSAN_PHYSICS_CONCEPT_NAMES)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float32)
        )
        self.responses = torch.tensor(
            self.table.loc[:, list(NISSAN_RESPONSE_TARGET_NAMES)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float32)
        )
        bsi = pd.to_numeric(self.table["bsi_intensity"], errors="coerce").to_numpy(np.float32)
        self.bsi = torch.tensor(bsi)
        self.current_soh = torch.tensor(
            pd.to_numeric(self.table["current_soh_target"], errors="coerce").to_numpy(np.float32)
        )
        self.loss_30d = torch.tensor(
            pd.to_numeric(self.table["target_loss_30d_pp"], errors="coerce").to_numpy(np.float32)
        )
        self.loss_500km = torch.tensor(
            pd.to_numeric(self.table["target_loss_500km_pp"], errors="coerce").to_numpy(np.float32)
        )
        self.stress_thresholds = torch.zeros(2)

    def set_stress_thresholds(self, thresholds: np.ndarray) -> None:
        self.stress_thresholds = torch.tensor(thresholds, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = dict(self.bags[index])
        physics = self.physics[index]
        responses = self.responses[index]
        bsi = self.bsi[index]
        stress_class = torch.bucketize(bsi, self.stress_thresholds) if torch.isfinite(bsi) else torch.tensor(-100)
        item.update({
            "observed_physics": physics,
            "observed_physics_mask": torch.isfinite(physics),
            "response_targets": responses,
            "response_mask": torch.isfinite(responses),
            "stress_class_target": stress_class.long(),
            "current_soh_physics_target": self.current_soh[index],
            "loss_30d_physics_target": self.loss_30d[index],
            "loss_500km_physics_target": self.loss_500km[index],
        })
        return item


def _load_backbone(checkpoint_path: Path, device: torch.device) -> DBPatchKAN:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = DBPatchKANConfig(**checkpoint["model_config"])
    backbone = DBPatchKAN(config)
    backbone.load_state_dict(checkpoint["model"], strict=True)
    return backbone.to(device)


def _forward(model: PhysicsInformedDBPatchKAN, batch: dict[str, Tensor], source: str):
    return model.forward_trip_bag(
        batch["driver"], batch["battery"], segment_mask=batch["segment_mask"],
        driver_mask=batch["driver_mask"], battery_mask=batch["battery_mask"],
        observed_physics=batch["observed_physics"],
        observed_physics_mask=batch["observed_physics_mask"],
        concept_source=source,
    )


def _masked_huber(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    valid = torch.isfinite(target) & torch.isfinite(prediction)
    if mask is not None:
        valid &= mask.bool()
    if not torch.any(valid):
        return prediction.new_zeros(())
    return F.huber_loss(prediction[valid], target[valid], delta=1.0)


def calculate_loss(
    model: PhysicsInformedDBPatchKAN,
    output: dict[str, Tensor],
    batch: dict[str, Tensor],
) -> tuple[Tensor, dict[str, float]]:
    current = _masked_huber(output["current_soh"], batch["current_soh_physics_target"])
    loss_30d = _masked_huber(output["loss_30d"], batch["loss_30d_physics_target"])
    loss_500 = _masked_huber(output["loss_500km"], batch["loss_500km_physics_target"])
    concept_target = (batch["observed_physics"] - model.physics_center) / model.physics_scale
    concept = _masked_huber(
        output["physics_concepts_normalized"], concept_target,
        batch["observed_physics_mask"],
    )
    response_target = (batch["response_targets"] - model.response_center) / model.response_scale
    response = _masked_huber(
        output["response_predictions_normalized"], response_target, batch["response_mask"]
    )
    stress_valid = batch["stress_class_target"].ne(-100)
    stress = (
        F.cross_entropy(output["stress_class_logits"][stress_valid], batch["stress_class_target"][stress_valid])
        if torch.any(stress_valid) else current.new_zeros(())
    )
    regularization = model.regularization_loss()
    total = (
        current + loss_30d + loss_500 + 0.20 * concept + 0.20 * response
        + 0.10 * stress + 1.0e-4 * regularization
    )
    values = {
        "total": float(total.detach()), "current_soh": float(current.detach()),
        "loss_30d": float(loss_30d.detach()), "loss_500km": float(loss_500.detach()),
        "concept": float(concept.detach()), "response": float(response.detach()),
        "stress_class": float(stress.detach()),
    }
    return total, values


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
            for key, value in batch.items()}


def run_epoch(
    model: PhysicsInformedDBPatchKAN,
    loader: DataLoader,
    optimizer: AdamW,
    *,
    device: torch.device,
    concept_source: str,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            output = _forward(model, batch, concept_source)
            loss, components = calculate_loss(model, output, batch)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + value
        count += 1
    return {name: value / max(1, count) for name, value in sums.items()}


@torch.no_grad()
def predict(
    model: PhysicsInformedDBPatchKAN,
    dataset: PhysicsTripDataset,
    indices: list[int],
    *,
    device: torch.device,
    concept_source: str,
    split_name: str,
    batch_size: int,
) -> pd.DataFrame:
    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        collate_fn=collate_trip_bags, num_workers=0,
    )
    model.eval()
    records: list[dict[str, Any]] = []
    offset = 0
    for batch in loader:
        batch = move_batch(batch, device)
        output = _forward(model, batch, concept_source)
        n = len(batch["current_soh_physics_target"])
        response = output["response_predictions"].float().cpu().numpy()
        stress_prob = torch.softmax(output["stress_class_logits"], dim=-1).float().cpu().numpy()
        for row in range(n):
            dataset_index = indices[offset + row]
            source_row = dataset.table.iloc[dataset_index]
            record: dict[str, Any] = {
                "split": split_name, "concept_source": concept_source,
                "trip_key": source_row["trip_key"], "Trip": source_row["Trip"],
                "current_soh_true": float(batch["current_soh_physics_target"][row].cpu()),
                "current_soh_pred": float(output["current_soh"][row].cpu()),
                "loss_30d_true": float(batch["loss_30d_physics_target"][row].cpu()),
                "loss_30d_pred": float(output["loss_30d"][row].cpu()),
                "loss_500km_true": float(batch["loss_500km_physics_target"][row].cpu()),
                "loss_500km_pred": float(output["loss_500km"][row].cpu()),
                "stress_class_true": int(batch["stress_class_target"][row].cpu()),
                "stress_class_pred": int(np.argmax(stress_prob[row])),
            }
            for klass, label in enumerate(("low", "moderate", "high")):
                record[f"stress_probability_{label}"] = float(stress_prob[row, klass])
            for column, name in enumerate(NISSAN_RESPONSE_TARGET_NAMES):
                record[f"{name}_true"] = float(batch["response_targets"][row, column].cpu())
                record[f"{name}_pred"] = float(response[row, column])
            records.append(record)
        offset += n
    return pd.DataFrame(records)


def metric_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for target in ("current_soh", "loss_30d", "loss_500km"):
        true = pd.to_numeric(predictions[f"{target}_true"], errors="coerce")
        pred = pd.to_numeric(predictions[f"{target}_pred"], errors="coerce")
        valid = true.notna() & pred.notna()
        error = pred[valid] - true[valid]
        metrics[target] = {
            "n": int(valid.sum()), "mae_pp": float(error.abs().mean()),
            "rmse_pp": float(np.sqrt(np.mean(error.pow(2)))) if valid.any() else math.nan,
            "bias_pp": float(error.mean()),
        }
    valid = predictions["stress_class_true"].ge(0)
    metrics["stress_class_accuracy"] = float(
        predictions.loc[valid, "stress_class_true"].eq(
            predictions.loc[valid, "stress_class_pred"]
        ).mean()
    )
    for name in NISSAN_RESPONSE_TARGET_NAMES:
        true = pd.to_numeric(predictions[f"{name}_true"], errors="coerce")
        pred = pd.to_numeric(predictions[f"{name}_pred"], errors="coerce")
        valid = true.notna() & pred.notna()
        metrics[name] = {
            "n": int(valid.sum()), "mae": float((pred[valid] - true[valid]).abs().mean())
        }
    return metrics


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(resolved_device)
    update(f"Loading trip bags and physics table on {device}")
    train_bags = EVTripBagDataset.from_npz(args.train_data)
    validation_bags = EVTripBagDataset.from_npz(args.validation_data)
    table = pd.read_parquet(args.physics_table) if args.physics_table.suffix == ".parquet" else pd.read_csv(args.physics_table)
    train_table = table.loc[table["split"].eq("train")].copy()
    validation_table = table.loc[table["split"].eq("validation")].copy()

    # First 80% of training trips fit the model; final 20% calibrate uncertainty.
    fit_count = int(math.floor(0.80 * len(train_bags)))
    fit_keys = {group[2] for group in train_bags.group_ids[:fit_count]}
    ambient_model = fit_ambient_adjustment(train_table.loc[train_table["trip_key"].isin(fit_keys)])
    train_table = apply_ambient_adjustment(train_table, ambient_model)
    validation_table = apply_ambient_adjustment(validation_table, ambient_model)
    joblib.dump(ambient_model, output / "ambient_temperature_adjustment.joblib")

    train_dataset = PhysicsTripDataset(train_bags, train_table)
    validation_dataset = PhysicsTripDataset(validation_bags, validation_table)
    fit_indices = list(range(fit_count))
    calibration_indices = list(range(fit_count, len(train_dataset)))
    validation_indices = list(range(len(validation_dataset)))
    physics_center, physics_scale = _finite_stats(train_dataset.physics[fit_indices].numpy())
    response_center, response_scale = _finite_stats(train_dataset.responses[fit_indices].numpy())
    bsi_fit = train_dataset.bsi[fit_indices].numpy()
    bsi_fit = bsi_fit[np.isfinite(bsi_fit)]
    thresholds = np.quantile(bsi_fit, [1 / 3, 2 / 3]).astype(np.float32)
    train_dataset.set_stress_thresholds(thresholds)
    validation_dataset.set_stress_thresholds(thresholds)

    backbone = _load_backbone(args.pretrained_checkpoint, device)
    model = PhysicsInformedDBPatchKAN(backbone, grid_size=args.kan_grid).to(device)
    model.set_physics_normalization(physics_center, physics_scale)
    model.set_response_normalization(response_center, response_scale)
    model.set_stress_thresholds(torch.tensor(thresholds))
    model.freeze_backbone(True)
    resume_state = None
    latest_path = output / "latest.pt"
    if args.resume and latest_path.exists():
        resume_state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model"], strict=True)
        update(
            f"Resuming {resume_state.get('phase')} epoch {resume_state.get('epoch')} "
            f"from {latest_path.name}"
        )
    fit_loader = DataLoader(
        Subset(train_dataset, fit_indices), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_trip_bags, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.head_lr,
                      weight_decay=0.01)
    warmup_start = 1
    if resume_state is not None:
        if resume_state.get("phase") == "warmup":
            warmup_start = int(resume_state.get("epoch", 0)) + 1
            if "optimizer" in resume_state:
                optimizer.load_state_dict(resume_state["optimizer"])
        elif resume_state.get("phase") == "end_to_end":
            warmup_start = args.warmup_epochs + 1
    history: list[dict[str, Any]] = []
    started = time.time()
    update(f"Warm-up: {args.warmup_epochs} epochs, observed concepts, frozen backbone")
    for epoch in range(warmup_start, args.warmup_epochs + 1):
        metrics = run_epoch(model, fit_loader, optimizer, device=device,
                            concept_source="observed", train=True)
        history.append({"phase": "warmup", "epoch": epoch, **metrics})
        if epoch == 1 or epoch % 5 == 0 or epoch == args.warmup_epochs:
            update(f"warm-up {epoch}/{args.warmup_epochs}: loss={metrics['total']:.5f}")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "backbone_config": model.backbone.config.to_dict(), "seed": args.seed,
            "epoch": epoch, "phase": "warmup", "test_used": False,
        }, latest_path)

    update(f"End-to-end: {args.end_to_end_epochs} epochs, predicted concepts, last blocks unfrozen")
    model.unfreeze_last_encoder_block()
    optimizer = AdamW([
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone")],
         "lr": args.encoder_lr},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone")],
         "lr": args.head_lr},
    ], weight_decay=0.01)
    end_start = 1
    if resume_state is not None and resume_state.get("phase") == "end_to_end":
        end_start = int(resume_state.get("epoch", 0)) + 1
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
    for epoch in range(end_start, args.end_to_end_epochs + 1):
        metrics = run_epoch(model, fit_loader, optimizer, device=device,
                            concept_source="predicted", train=True)
        history.append({"phase": "end_to_end", "epoch": epoch, **metrics})
        if epoch == 1 or epoch % 5 == 0 or epoch == args.end_to_end_epochs:
            update(f"end-to-end {epoch}/{args.end_to_end_epochs}: loss={metrics['total']:.5f}")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "backbone_config": model.backbone.config.to_dict(),
            "seed": args.seed, "epoch": epoch, "phase": "end_to_end",
            "physics_concept_names": NISSAN_PHYSICS_CONCEPT_NAMES,
            "response_target_names": NISSAN_RESPONSE_TARGET_NAMES,
            "fit_count": fit_count, "stress_thresholds": thresholds.tolist(),
            "test_used": False,
        }, latest_path)

    predictions = []
    for source in ("predicted", "observed"):
        predictions.append(predict(model, train_dataset, calibration_indices, device=device,
                                   concept_source=source, split_name="calibration",
                                   batch_size=args.batch_size))
        predictions.append(predict(model, validation_dataset, validation_indices, device=device,
                                   concept_source=source, split_name="validation",
                                   batch_size=args.batch_size))
    predictions_frame = pd.concat(predictions, ignore_index=True)
    predictions_frame.to_csv(output / "predictions.csv", index=False)
    validation_predicted = predictions_frame.loc[
        predictions_frame["split"].eq("validation")
        & predictions_frame["concept_source"].eq("predicted")
    ]
    metrics = metric_summary(validation_predicted)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    weight_rows = []
    for task, weights in model.monotonic_weights().items():
        for name, value, scale in zip(
            NISSAN_PHYSICS_CONCEPT_NAMES[: model.n_monotonic],
            weights.detach().cpu().numpy(), physics_scale[: model.n_monotonic], strict=True,
        ):
            weight_rows.append({
                "task": task, "concept": name, "positive_standardized_weight": float(value),
                "raw_unit_logit_slope": float(value / scale), "monotonic": True,
            })
    pd.DataFrame(weight_rows).to_csv(output / "monotonic_physics_weights.csv", index=False)
    torch.save({
        "model": model.state_dict(), "backbone_config": model.backbone.config.to_dict(),
        "seed": args.seed, "physics_concept_names": NISSAN_PHYSICS_CONCEPT_NAMES,
        "response_target_names": NISSAN_RESPONSE_TARGET_NAMES,
        "fit_count": fit_count, "calibration_count": len(calibration_indices),
        "stress_thresholds": thresholds.tolist(), "metrics": metrics,
        "test_used": False,
    }, output / "best.pt")
    run_summary = {
        "seed": args.seed, "device": str(device), "fit_trips": fit_count,
        "calibration_trips": len(calibration_indices),
        "validation_trips": len(validation_indices), "test_used": False,
        "runtime_minutes": (time.time() - started) / 60.0, "metrics": metrics,
        "architecture": "PI-DB-PatchKAN",
        "soh_latent_shortcut": False,
        "uncertainty_role": "calibration residuals are combined across five seeds by notebook",
    }
    (output / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    update(f"Complete seed {args.seed}; validation current-SOH MAE={metrics['current_soh']['mae_pp']:.4f} pp")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--physics-table", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--end-to-end-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--kan-grid", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
