"""Train one corrected Nissan Stage 4N-B physics-hybrid seed."""

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dbpatchkan.models import (
    CURRENT_INTERACTION_FEATURES,
    CURRENT_MONOTONIC_FEATURES,
    FORECAST_30D_FEATURES,
    FORECAST_500KM_FEATURES,
    CorrectedPhysicsHybrid,
    DBPatchKAN,
    DBPatchKANConfig,
    EVTripBagDataset,
    NISSAN_RESPONSE_TARGET_NAMES,
    PhysicsInformedDBPatchKAN,
    collate_trip_bags,
)


def update(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_v1_model(path: Path, device: torch.device) -> PhysicsInformedDBPatchKAN:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    backbone = DBPatchKAN(DBPatchKANConfig(**checkpoint["backbone_config"]))
    model = PhysicsInformedDBPatchKAN(backbone)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    for parameter in model.parameters(): parameter.requires_grad = False
    return model


def fit_ambient_adjustment(fit: pd.DataFrame) -> Any:
    features = ["ambient_mean_c", "physics_duration_h", "hvac_energy_kwh"]
    target = pd.to_numeric(fit["pack_temp_rise_c"], errors="coerce")
    valid = target.notna()
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(fit.loc[valid, features], target.loc[valid])
    return model


def apply_ambient_adjustment(table: pd.DataFrame, model: Any) -> pd.DataFrame:
    output = table.copy()
    features = ["ambient_mean_c", "physics_duration_h", "hvac_energy_kwh"]
    output["ambient_adjusted_pack_temp_rise_c"] = (
        pd.to_numeric(output["pack_temp_rise_c"], errors="coerce")
        - model.predict(output[features])
    )
    return output


def table_for_bags(bags: EVTripBagDataset, table: pd.DataFrame) -> pd.DataFrame:
    indexed = table.set_index("trip_key", drop=False)
    rows = []
    for _, _, key in bags.group_ids:
        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame): raise ValueError(f"Duplicate row for {key}")
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


@torch.no_grad()
def cache_temporal(
    model: PhysicsInformedDBPatchKAN,
    bags: EVTripBagDataset,
    *, device: torch.device, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(bags, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_trip_bags, num_workers=0)
    responses, stress = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        encoded = model.backbone.encode_trip_bag(
            batch["driver"], batch["battery"], segment_mask=batch["segment_mask"],
            driver_mask=batch["driver_mask"], battery_mask=batch["battery_mask"],
        )
        responses.append(model.response_head(encoded["embedding"]).float().cpu().numpy())
        stress.append(model.stress_class_head(encoded["embedding"]).float().cpu().numpy())
    return np.concatenate(responses), np.concatenate(stress)


class LogFeatureTransform:
    def __init__(self, columns: tuple[str, ...]) -> None:
        self.columns = columns
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, frame: pd.DataFrame) -> "LogFeatureTransform":
        values = np.log1p(np.clip(frame.loc[:, list(self.columns)].apply(pd.to_numeric, errors="coerce"), 0, None))
        self.scaler.fit(self.imputer.fit_transform(values))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.log1p(np.clip(frame.loc[:, list(self.columns)].apply(pd.to_numeric, errors="coerce"), 0, None))
        return np.clip(self.scaler.transform(self.imputer.transform(values)), -3.0, 3.0).astype(np.float32)


class ResidualizedLogFeatureTransform:
    """Represent excess cumulative stress after age/mileage-expected exposure."""

    def __init__(self, columns: tuple[str, ...]) -> None:
        self.columns = columns
        self.context_columns = ("calendar_days_since_start", "odometer_clean_km")
        self.value_imputer = SimpleImputer(strategy="median")
        self.context_model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
        )
        self.scaler = StandardScaler()

    def _values(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, list(self.columns)].apply(pd.to_numeric, errors="coerce")
        return np.log1p(np.clip(values, 0, None))

    def fit(self, frame: pd.DataFrame) -> "ResidualizedLogFeatureTransform":
        values = self.value_imputer.fit_transform(self._values(frame))
        context = frame.loc[:, list(self.context_columns)]
        self.context_model.fit(context, values)
        residual = values - self.context_model.predict(context)
        self.scaler.fit(residual)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.value_imputer.transform(self._values(frame))
        context = frame.loc[:, list(self.context_columns)]
        residual = values - self.context_model.predict(context)
        return np.clip(self.scaler.transform(residual), -3.0, 3.0).astype(np.float32)


def masked_huber(prediction: Tensor, target: Tensor) -> Tensor:
    valid = torch.isfinite(target)
    return F.huber_loss(prediction[valid], target[valid], delta=0.1) if torch.any(valid) else prediction.new_zeros(())


def metrics(true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(true) & np.isfinite(prediction)
    error = prediction[valid] - true[valid]
    return {
        "n": int(valid.sum()), "mae_pp": float(np.mean(np.abs(error))) if valid.any() else math.nan,
        "rmse_pp": float(np.sqrt(np.mean(error**2))) if valid.any() else math.nan,
        "bias_pp": float(np.mean(error)) if valid.any() else math.nan,
    }


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    started = time.time()
    update(f"Loading corrected seed {args.seed} on {device}")
    table = pd.read_parquet(args.physics_table)
    train_bags = EVTripBagDataset.from_npz(args.train_data)
    validation_bags = EVTripBagDataset.from_npz(args.validation_data)
    train_table = table_for_bags(train_bags, table[table.split.eq("train")])
    validation_table = table_for_bags(validation_bags, table[table.split.eq("validation")])
    fit_count = math.ceil(0.80 * len(train_table))
    fit = train_table.iloc[:fit_count].copy()
    calibration_indices = np.arange(fit_count, len(train_table))

    ambient_model = fit_ambient_adjustment(fit)
    train_table = apply_ambient_adjustment(train_table, ambient_model)
    validation_table = apply_ambient_adjustment(validation_table, ambient_model)
    joblib.dump(ambient_model, output / "ambient_adjustment.joblib")

    baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    age_features = ["calendar_days_since_start", "odometer_clean_km"]
    baseline.fit(fit[age_features], fit["current_soh_target"])
    baseline_train = baseline.predict(train_table[age_features]).astype(np.float32)
    baseline_validation = baseline.predict(validation_table[age_features]).astype(np.float32)
    joblib.dump(baseline, output / "age_mileage_baseline.joblib")

    transforms = {
        "current_monotonic": ResidualizedLogFeatureTransform(CURRENT_MONOTONIC_FEATURES).fit(fit),
        "current_interaction": LogFeatureTransform(CURRENT_INTERACTION_FEATURES).fit(fit),
        "forecast_30d": LogFeatureTransform(FORECAST_30D_FEATURES).fit(fit),
        "forecast_500km": LogFeatureTransform(FORECAST_500KM_FEATURES).fit(fit),
    }
    feature_train = {name: transform.transform(train_table) for name, transform in transforms.items()}
    feature_validation = {name: transform.transform(validation_table) for name, transform in transforms.items()}
    joblib.dump(transforms, output / "physics_feature_transforms.joblib")

    temporal = load_v1_model(args.v1_checkpoint, device)
    update("Caching frozen temporal response representations once")
    response_train, stress_train = cache_temporal(temporal, train_bags, device=device, batch_size=args.batch_size)
    response_validation, stress_validation = cache_temporal(temporal, validation_bags, device=device, batch_size=args.batch_size)
    median_30 = float(pd.to_numeric(fit.target_loss_30d_pp, errors="coerce").median())
    median_500 = float(pd.to_numeric(fit.target_loss_500km_pp, errors="coerce").median())
    model = CorrectedPhysicsHybrid(
        temporal, median_loss_30d_pp=median_30, median_loss_500km_pp=median_500
    ).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(parameters, lr=args.lr, weight_decay=0.01)

    def tensor(values: np.ndarray) -> Tensor: return torch.tensor(values, dtype=torch.float32, device=device)
    fit_slice = slice(0, fit_count)
    x_fit = {name: tensor(values[fit_slice]) for name, values in feature_train.items()}
    baseline_fit = tensor(baseline_train[fit_slice])
    response_fit = tensor(response_train[fit_slice])
    y_current = tensor(pd.to_numeric(fit.current_soh_target, errors="coerce").to_numpy(np.float32))
    y_30 = tensor(pd.to_numeric(fit.target_loss_30d_pp, errors="coerce").to_numpy(np.float32))
    y_500 = tensor(pd.to_numeric(fit.target_loss_500km_pp, errors="coerce").to_numpy(np.float32))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output_fit = model.forward_from_cached(
            baseline_soh=baseline_fit, current_monotonic=x_fit["current_monotonic"],
            current_interaction=x_fit["current_interaction"], forecast_30d=x_fit["forecast_30d"],
            forecast_500km=x_fit["forecast_500km"], response_normalized=response_fit,
        )
        current_loss = masked_huber(output_fit["current_soh"], y_current)
        loss_30 = masked_huber(output_fit["loss_30d"], y_30)
        loss_500 = masked_huber(output_fit["loss_500km"], y_500)
        regularization = model.regularization_loss()
        total = current_loss + loss_30 + loss_500 + 1.0e-3 * regularization
        total.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step()
        history.append({"epoch": epoch, "total": float(total.detach()),
                        "current": float(current_loss.detach()), "loss_30d": float(loss_30.detach()),
                        "loss_500km": float(loss_500.detach())})
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            update(f"epoch {epoch}/{args.epochs}: total={float(total.detach()):.6f}")

    def make_predictions(frame: pd.DataFrame, baseline_values: np.ndarray, features: dict[str, np.ndarray],
                         responses: np.ndarray, stress_logits: np.ndarray, indices: np.ndarray,
                         split: str, mode: str) -> pd.DataFrame:
        model.eval()
        with torch.no_grad():
            result = model.forward_from_cached(
                baseline_soh=tensor(baseline_values[indices]),
                current_monotonic=tensor(features["current_monotonic"][indices]),
                current_interaction=tensor(features["current_interaction"][indices]),
                forecast_30d=tensor(features["forecast_30d"][indices]),
                forecast_500km=tensor(features["forecast_500km"][indices]),
                response_normalized=tensor(responses[indices]), analytic_only=mode == "analytic_only",
            )
        records = frame.iloc[indices][["trip_key", "Trip", "current_soh_target", "target_loss_30d_pp", "target_loss_500km_pp"]].copy()
        records = records.rename(columns={"current_soh_target":"current_soh_true", "target_loss_30d_pp":"loss_30d_true", "target_loss_500km_pp":"loss_500km_true"})
        records["split"] = split; records["model_mode"] = mode
        for target in ("current_soh", "loss_30d", "loss_500km"):
            records[f"{target}_pred"] = result[target].cpu().numpy()
        records["baseline_soh_pred"] = result["baseline_soh"].cpu().numpy()
        records["current_correction_pp"] = result["current_correction_pp"].cpu().numpy()
        probabilities = torch.softmax(tensor(stress_logits[indices]), dim=-1).cpu().numpy()
        records["stress_class_true"] = np.digitize(
            pd.to_numeric(frame.iloc[indices].bsi_intensity, errors="coerce"),
            model.temporal_model.stress_thresholds.cpu().numpy(),
        )
        for klass, label in enumerate(("low","moderate","high")):
            records[f"stress_probability_{label}"] = probabilities[:, klass]
        response_physical = responses[indices] * model.temporal_model.response_scale.cpu().numpy() + model.temporal_model.response_center.cpu().numpy()
        for column, name in enumerate(NISSAN_RESPONSE_TARGET_NAMES):
            records[f"{name}_true"] = pd.to_numeric(frame.iloc[indices][name], errors="coerce").to_numpy()
            records[f"{name}_pred"] = response_physical[:, column]
        return records

    prediction_frames = []
    for mode in ("corrected_hybrid", "analytic_only"):
        prediction_frames.append(make_predictions(train_table, baseline_train, feature_train, response_train,
                                                   stress_train, calibration_indices, "calibration", mode))
        prediction_frames.append(make_predictions(validation_table, baseline_validation, feature_validation,
                                                   response_validation, stress_validation, np.arange(len(validation_table)),
                                                   "validation", mode))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    primary = predictions[(predictions.split == "validation") & (predictions.model_mode == "corrected_hybrid")]
    run_metrics = {
        target: metrics(primary[f"{target}_true"].to_numpy(float), primary[f"{target}_pred"].to_numpy(float))
        for target in ("current_soh", "loss_30d", "loss_500km")
    }
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
    checkpoint = {
        "model": model.state_dict(), "seed": args.seed, "metrics": run_metrics,
        "fit_count": fit_count, "calibration_count": len(calibration_indices),
        "median_loss_30d_pp": median_30, "median_loss_500km_pp": median_500,
        "architecture": "PI-DB-PatchKAN-v2-corrected", "test_used": False,
    }
    torch.save(checkpoint, output / "best.pt")
    summary = {**checkpoint, "model": "stored_in_best.pt", "runtime_minutes": (time.time()-started)/60}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    update(f"complete seed {args.seed}: current MAE={run_metrics['current_soh']['mae_pp']:.4f} pp")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--physics-table", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    train(parser.parse_args())


if __name__ == "__main__": main()
