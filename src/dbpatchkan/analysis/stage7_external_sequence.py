"""Stage 7: external dual-stream sequence transfer and gated health KAN.

The experiment has two deliberately separated levels:

1. A dual-stream Patch Transformer learns named trip-level battery-response
   concepts from raw driver and battery sequences.
2. A compact KAN (and parameter-matched MLP) consumes only interval aggregates
   of those named concepts.  Its non-negative degradation correction is
   multiplied by an inner-fold-selected gate.  A zero gate is an explicit,
   scientifically valid fallback to persistence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dbpatchkan.data.external_sequences import EXTERNAL_CONCEPT_NAMES, decode_utf8
from dbpatchkan.models.kan import ConceptKAN, ConceptMLP
from dbpatchkan.models.patchkan import DBPatchKAN, DBPatchKANConfig


HEALTH_CONCEPT_NAMES = (
    *tuple(f"mean_{name}" for name in EXTERNAL_CONCEPT_NAMES),
    "log1p_exposure_hours",
    "log1p_exposure_distance_km",
    "log1p_interval_days",
    "anchor_soh_pct",
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


class SequenceRows(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, ...]:
        index = self.indices[item]
        return (
            torch.from_numpy(self.arrays["driver"][index].astype(np.float32)),
            torch.from_numpy(self.arrays["battery"][index].astype(np.float32)),
            torch.from_numpy(self.arrays["driver_mask"][index]),
            torch.from_numpy(self.arrays["battery_mask"][index]),
            torch.from_numpy(self.arrays["concept_targets"][index].astype(np.float32)),
            torch.from_numpy(self.arrays["concept_mask"][index]),
            torch.tensor(index, dtype=torch.long),
        )


def load_sequence_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    required = {
        "driver", "battery", "driver_mask", "battery_mask", "concept_targets",
        "concept_mask", "vehicle_id_utf8", "trip_key_utf8", "trip_start_ns",
        "trip_end_ns", "duration_h", "distance_km",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"Sequence archive is missing {sorted(missing)}")
    arrays["vehicle_id"] = np.asarray(decode_utf8(arrays["vehicle_id_utf8"]))
    arrays["trip_key"] = np.asarray(decode_utf8(arrays["trip_key_utf8"]))
    return arrays


def _masked_channel_statistics(
    values: np.ndarray, mask: np.ndarray, indices: np.ndarray
) -> tuple[Tensor, Tensor]:
    center = np.zeros(values.shape[-1], dtype=np.float32)
    scale = np.ones(values.shape[-1], dtype=np.float32)
    for column in range(values.shape[-1]):
        current = values[indices, :, column].astype(np.float32)
        valid = mask[indices, :, column] & np.isfinite(current)
        if valid.any():
            selected = current[valid]
            center[column] = float(np.median(selected))
            q25, q75 = np.percentile(selected, [25, 75])
            robust = float((q75 - q25) / 1.349)
            ordinary = float(np.std(selected))
            scale[column] = robust if robust > 1e-6 else max(ordinary, 1.0)
    # Validity, dt, relative position, and missingness are explicit metadata.
    return torch.from_numpy(center), torch.from_numpy(scale)


def _concept_statistics(
    targets: np.ndarray, mask: np.ndarray, indices: np.ndarray
) -> tuple[Tensor, Tensor]:
    center = np.zeros(targets.shape[1], dtype=np.float32)
    scale = np.ones(targets.shape[1], dtype=np.float32)
    for column in range(targets.shape[1]):
        current = targets[indices, column].astype(np.float32)
        valid = mask[indices, column] & np.isfinite(current)
        if valid.any():
            selected = current[valid]
            center[column] = float(np.median(selected))
            q25, q75 = np.percentile(selected, [25, 75])
            robust = float((q75 - q25) / 1.349)
            scale[column] = robust if robust > 1e-6 else max(float(np.std(selected)), 1.0)
    return torch.from_numpy(center), torch.from_numpy(scale)


def _concept_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    center: Tensor,
    scale: Tensor,
) -> Tensor:
    normalized = (target - center) / scale
    finite = valid & torch.isfinite(normalized)
    if not torch.any(finite):
        return prediction.new_zeros(())
    return F.huber_loss(prediction[finite], normalized[finite], delta=1.0)


def _validation_mae(
    model: DBPatchKAN,
    loader: DataLoader,
    center: Tensor,
    scale: Tensor,
    device: torch.device,
) -> tuple[float, list[dict[str, float]]]:
    model.eval()
    truth, prediction, masks = [], [], []
    with torch.no_grad():
        for driver, battery, dmask, bmask, target, cmask, _ in loader:
            encoded = model.encode_segments(
                driver.to(device), battery.to(device),
                driver_mask=dmask.to(device), battery_mask=bmask.to(device),
            )
            normalized = model.concept_predictor(encoded["embedding"])
            prediction.append((normalized * scale + center).cpu().numpy())
            truth.append(target.numpy())
            masks.append(cmask.numpy())
    y = np.concatenate(truth)
    p = np.concatenate(prediction)
    valid = np.concatenate(masks) & np.isfinite(y) & np.isfinite(p)
    per_target = []
    for column, name in enumerate(EXTERNAL_CONCEPT_NAMES):
        selected = valid[:, column]
        mae = float(np.mean(np.abs(y[selected, column] - p[selected, column]))) if selected.any() else math.nan
        per_target.append({"concept": name, "mae": mae, "n": int(selected.sum())})
    all_mae = float(np.mean([
        row["mae"] / max(float(scale[i].cpu()), 1e-6)
        for i, row in enumerate(per_target) if np.isfinite(row["mae"])
    ]))
    return all_mae, per_target


def train_concept_encoder(
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    device: torch.device,
    output: Path,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[DBPatchKAN, pd.DataFrame]:
    """Train one fold-safe dual-stream encoder on named physical concepts."""

    if output.exists():
        update(f"resume: loading saved concept encoder {output.parent.name}")
        checkpoint = torch.load(
            output, map_location=device, weights_only=False
        )
        config_values = dict(checkpoint["config"])
        for name in (
            "concept_names", "kan_hidden", "mlp_hidden",
            "protected_driver_channels", "protected_battery_channels",
            "calibration_domains",
        ):
            if name in config_values:
                config_values[name] = tuple(config_values[name])
        model = DBPatchKAN(DBPatchKANConfig(**config_values)).to(device)
        model.load_state_dict(checkpoint["model_state"])
        history_path = output.with_name("concept_training_history.csv")
        history = (
            pd.read_csv(history_path)
            if history_path.exists()
            else pd.DataFrame([{
                "epoch": -1,
                "train_loss": math.nan,
                "validation_normalized_mae": checkpoint.get(
                    "best_validation_normalized_mae", math.nan
                ),
            }])
        )
        return model, history

    seed_everything(seed)  # must precede construction
    config = DBPatchKANConfig(
        segment_length=arrays["driver"].shape[1],
        patch_length=16,
        patch_stride=8,
        d_model=96,
        encoder_layers=2,
        n_heads=4,
        ff_dim=192,
        dropout=0.10,
        concept_names=EXTERNAL_CONCEPT_NAMES,
        fusion_head="kan",
        kan_hidden=(8,),
        kan_grid_size=3,
        calibration_domains=("capacity",),
    )
    model = DBPatchKAN(config).to(device)
    dcenter, dscale = _masked_channel_statistics(
        arrays["driver"], arrays["driver_mask"], train_indices
    )
    bcenter, bscale = _masked_channel_statistics(
        arrays["battery"], arrays["battery_mask"], train_indices
    )
    ccenter, cscale = _concept_statistics(
        arrays["concept_targets"], arrays["concept_mask"], train_indices
    )
    model.set_input_normalization(dcenter, dscale, bcenter, bscale)
    model.set_concept_normalization(ccenter, cscale)
    ccenter, cscale = ccenter.to(device), cscale.to(device)

    train_loader = DataLoader(
        SequenceRows(arrays, train_indices), batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    validation_loader = DataLoader(
        SequenceRows(arrays, validation_indices), batch_size=batch_size * 2,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    parameters = (
        list(model.driver_encoder.parameters())
        + list(model.battery_encoder.parameters())
        + list(model.driver_to_battery.parameters())
        + list(model.battery_to_driver.parameters())
        + list(model.driver_cross_norm.parameters())
        + list(model.battery_cross_norm.parameters())
        + list(model.driver_pool.parameters())
        + list(model.battery_pool.parameters())
        + list(model.fusion_projection.parameters())
        + list(model.concept_predictor.parameters())
    )
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=0.01)
    scaler_enabled = device.type == "cuda"
    best, best_state, patience_left = math.inf, None, 6
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for driver, battery, dmask, bmask, target, cmask, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            driver, battery = driver.to(device), battery.to(device)
            dmask, bmask = dmask.to(device), bmask.to(device)
            target, cmask = target.to(device), cmask.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=scaler_enabled,
            ):
                encoded = model.encode_segments(
                    driver, battery, driver_mask=dmask, battery_mask=bmask
                )
                normalized = model.concept_predictor(encoded["embedding"])
                loss = _concept_loss(normalized, target, cmask, ccenter, cscale)
                loss = loss + 1e-4 * model.channel_l0_penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation, per_target = _validation_mae(
            model, validation_loader, ccenter, cscale, device
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_normalized_mae": validation,
            **{f"mae_{row['concept']}": row["mae"] for row in per_target},
        })
        update(
            f"concept epoch {epoch}/{epochs}: train={np.mean(losses):.4f}, "
            f"validation normalized MAE={validation:.4f}"
        )
        if validation < best - 1e-5:
            best = validation
            best_state = copy.deepcopy(model.state_dict())
            patience_left = 6
        else:
            patience_left -= 1
            if patience_left == 0:
                update(f"concept early stopping at epoch {epoch}")
                break
    if best_state is None:
        raise RuntimeError("Concept encoder did not produce a checkpoint")
    model.load_state_dict(best_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "config": config.to_dict(),
        "concept_names": EXTERNAL_CONCEPT_NAMES,
        "best_validation_normalized_mae": best,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }, output)
    return model, pd.DataFrame(history)


def predict_concepts(
    model: DBPatchKAN,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        SequenceRows(arrays, indices), batch_size=batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    prediction = np.full_like(arrays["concept_targets"], np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for driver, battery, dmask, bmask, _, _, row_index in loader:
            encoded = model.encode_segments(
                driver.to(device), battery.to(device),
                driver_mask=dmask.to(device), battery_mask=bmask.to(device),
            )
            normalized = model.concept_predictor(encoded["embedding"])
            physical = normalized * model.concept_scale + model.concept_center
            prediction[row_index.numpy()] = physical.cpu().numpy()
    return prediction


def _interval_features(
    arrays: dict[str, np.ndarray],
    concept_prediction: np.ndarray,
    intervals: pd.DataFrame,
    *,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    trip_start = arrays["trip_start_ns"].astype(np.int64)
    trip_end = arrays["trip_end_ns"].astype(np.int64)
    duration = arrays["duration_h"].astype(float)
    distance = arrays["distance_km"].astype(float)
    rows, counts = [], []
    for interval in intervals.itertuples(index=False):
        if dataset == "RWTH":
            start = pd.Timestamp(interval.interval_start).value
            end = pd.Timestamp(interval.interval_end).value
            vehicle = str(interval.vehicle_id)
            chosen = (
                (arrays["vehicle_id"] == vehicle)
                & (trip_start >= start)
                & (trip_end <= end)
            )
            interval_days = float(interval.interval_days)
            anchor = float(interval.start_soh_pct)
        else:
            end = pd.Timestamp(interval.checkpoint_time).value
            previous = end - int(31 * 86400 * 1e9)
            chosen = (trip_start > previous) & (trip_end <= end)
            interval_days = float(interval.calendar_days)
            anchor = float(interval.previous_soh_trend_pct)
        indices = np.flatnonzero(chosen)
        counts.append(len(indices))
        if not len(indices):
            rows.append(np.full(len(HEALTH_CONCEPT_NAMES), np.nan))
            continue
        weights = np.maximum(duration[indices], 1e-6)
        concept_mean = np.asarray([
            np.average(concept_prediction[indices, column], weights=weights)
            for column in range(concept_prediction.shape[1])
        ])
        rows.append(np.concatenate((
            concept_mean,
            [
                np.log1p(np.sum(duration[indices])),
                np.log1p(np.sum(np.maximum(distance[indices], 0.0))),
                np.log1p(max(interval_days, 0.0)),
                anchor,
            ],
        )))
    return np.asarray(rows, dtype=np.float32), np.asarray(counts, dtype=np.int64)


def _robust_prepare(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.nanmedian(train, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    q25, q75 = np.nanpercentile(train, [25, 75], axis=0)
    scale = (q75 - q25) / 1.349
    ordinary = np.nanstd(train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, ordinary)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    train_filled = np.where(np.isfinite(train), train, center)
    test_filled = np.where(np.isfinite(test), test, center)
    return (
        ((train_filled - center) / scale).astype(np.float32),
        ((test_filled - center) / scale).astype(np.float32),
        center.astype(np.float32),
        scale.astype(np.float32),
    )


def _fit_health_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    kind: str,
    device: torch.device,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    seed_everything(seed)
    x_train, x_test, center, scale = _robust_prepare(train_x, test_x)
    if kind == "KAN":
        head: nn.Module = ConceptKAN(
            (x_train.shape[1], 8, 1), grid_size=3, spline_order=3
        )
    elif kind == "MLP":
        head = ConceptMLP((x_train.shape[1], 16, 8, 1), dropout=0.10)
    else:
        raise ValueError(kind)
    head = head.to(device)
    x_tensor = torch.from_numpy(x_train).to(device)
    y_tensor = torch.from_numpy(train_y.astype(np.float32)).to(device)
    target_scale = max(float(np.median(np.abs(train_y))), 1.0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=0.02)
    best_loss, best_state = math.inf, None
    for _ in range(epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = F.softplus(head(x_tensor).squeeze(-1)) * target_scale
        loss = F.huber_loss(prediction, y_tensor, delta=1.0)
        if kind == "KAN":
            loss = loss + 1e-3 * head.regularization_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        current = float(loss.detach().cpu())
        if current < best_loss:
            best_loss = current
            best_state = copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        prediction = (
            F.softplus(head(torch.from_numpy(x_test).to(device)).squeeze(-1))
            * target_scale
        ).cpu().numpy()
    state = {
        "kind": kind,
        "state_dict": best_state,
        "dimensions": getattr(head, "dimensions", None),
        "feature_names": HEALTH_CONCEPT_NAMES,
        "center": center,
        "scale": scale,
        "target_scale": target_scale,
        "train_loss": best_loss,
    }
    if kind == "KAN":
        state["edge_strengths"] = [value.numpy() for value in head.edge_strengths()]
    return prediction, state


def _inner_splits(
    rows: pd.DataFrame, *, dataset: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    if dataset == "RWTH":
        groups = rows.vehicle_id.astype(str).to_numpy()
        result = []
        for group in sorted(set(groups)):
            validation = np.flatnonzero(groups == group)
            train = np.flatnonzero(groups != group)
            if len(train) >= 3 and len(validation):
                result.append((train, validation))
        return result
    count = len(rows)
    if count < 8:
        return []
    minimum = max(6, count // 2)
    boundaries = np.linspace(minimum, count, 4, dtype=int)
    return [
        (np.arange(0, boundaries[i]), np.arange(boundaries[i], boundaries[i + 1]))
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] > boundaries[i]
    ]


def _select_gate(
    x: np.ndarray,
    y: np.ndarray,
    rows: pd.DataFrame,
    *,
    kind: str,
    dataset: str,
    device: torch.device,
    epochs: int,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    predictions = np.full(len(y), np.nan)
    for fold, (train, validation) in enumerate(_inner_splits(rows, dataset=dataset)):
        if len(train) < 3:
            continue
        predictions[validation], _ = _fit_health_head(
            x[train], y[train], x[validation], kind=kind, device=device,
            epochs=max(80, epochs // 2), seed=seed + fold,
        )
    valid = np.isfinite(predictions) & np.isfinite(y)
    candidates = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
    records = []
    for alpha in candidates:
        mae = float(np.mean(np.abs(y[valid] - alpha * predictions[valid]))) if valid.any() else math.nan
        records.append({"alpha": alpha, "inner_mae": mae, "n": int(valid.sum())})
    table = pd.DataFrame(records)
    persistence = float(np.mean(np.abs(y[valid]))) if valid.any() else math.nan
    if not valid.any() or not np.isfinite(persistence):
        return 0.0, table
    best = table.loc[table.inner_mae.idxmin()]
    alpha = float(best.alpha) if float(best.inner_mae) < 0.98 * persistence else 0.0
    table["selected"] = table.alpha.eq(alpha)
    table["persistence_inner_mae"] = persistence
    table["two_percent_safety_margin"] = 0.98 * persistence
    return alpha, table


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in predictions.groupby(["dataset", "model"]):
        y = group.true_loss_pp.to_numpy(float)
        p = group.predicted_loss_pp.to_numpy(float)
        rows.append({
            "dataset": dataset,
            "model": model,
            "n": len(group),
            "mae_pp": mean_absolute_error(y, p),
            "rmse_pp": mean_squared_error(y, p) ** 0.5,
            "bias_pp": float(np.mean(p - y)),
            "r2": r2_score(y, p) if len(y) >= 2 else math.nan,
            "mean_gate_alpha": float(group.gate_alpha.mean()),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "mae_pp"])


def _paired_cluster_bootstrap(
    predictions: pd.DataFrame, *, draws: int = 4000, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, group in predictions.groupby("dataset"):
        reference = group[group.model.eq("Persistence")].set_index("row_id")
        for model in sorted(set(group.model) - {"Persistence"}):
            challenger = group[group.model.eq(model)].set_index("row_id")
            joined = reference[["true_loss_pp", "predicted_loss_pp", "cluster"]].join(
                challenger[["predicted_loss_pp"]], rsuffix="_challenger", how="inner"
            )
            clusters = joined.cluster.astype(str).unique()
            if not len(clusters):
                continue
            values = []
            for _ in range(draws):
                sampled = rng.choice(clusters, size=len(clusters), replace=True)
                deltas = []
                for cluster in sampled:
                    current = joined[joined.cluster.astype(str).eq(cluster)]
                    truth = current.true_loss_pp.to_numpy(float)
                    delta = np.mean(np.abs(
                        truth - current.predicted_loss_pp_challenger.to_numpy(float)
                    )) - np.mean(np.abs(
                        truth - current.predicted_loss_pp.to_numpy(float)
                    ))
                    deltas.append(delta)
                values.append(float(np.mean(deltas)))
            array = np.asarray(values)
            rows.append({
                "dataset": dataset,
                "challenger": model,
                "reference": "Persistence",
                "clusters": len(clusters),
                "delta_macro_mae_pp": float(np.mean(array)),
                "ci_low": float(np.quantile(array, 0.025)),
                "ci_high": float(np.quantile(array, 0.975)),
                "challenger_significantly_better": bool(np.quantile(array, 0.975) < 0),
                "persistence_significantly_better": bool(np.quantile(array, 0.025) > 0),
            })
    return pd.DataFrame(rows)


def _validation_split(indices: np.ndarray, groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if groups is not None and len(set(groups[indices])) > 1:
        validation_group = sorted(set(groups[indices]))[-1]
        validation = indices[groups[indices] == validation_group]
        train = indices[groups[indices] != validation_group]
        return train, validation
    cut = max(1, int(round(0.85 * len(indices))))
    return indices[:cut], indices[cut:]


def run_rwth(
    arrays: dict[str, np.ndarray],
    intervals: pd.DataFrame,
    *,
    output: Path,
    device: torch.device,
    concept_epochs: int,
    health_epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    valid_intervals = intervals.loc[intervals.trip_count.gt(0)].copy()
    vehicles = sorted(valid_intervals.vehicle_id.unique())
    update(f"RWTH full dual-stream LOOV: {len(valid_intervals)} intervals, {len(vehicles)} vehicles")
    predictions: list[dict[str, object]] = []
    gates: list[pd.DataFrame] = []
    for fold, held_out in enumerate(vehicles):
        update(f"RWTH outer fold {fold + 1}/{len(vehicles)}: held out {held_out}")
        outer_train_seq = np.flatnonzero(arrays["vehicle_id"] != held_out)
        train_seq, validation_seq = _validation_split(
            outer_train_seq, arrays["vehicle_id"]
        )
        fold_dir = output / f"rwth_fold_{fold}_{held_out}"
        model, history = train_concept_encoder(
            arrays, train_seq, validation_seq, device=device,
            output=fold_dir / "best_concept_encoder.pt", epochs=concept_epochs,
            batch_size=batch_size, seed=seed + fold,
        )
        history.to_csv(fold_dir / "concept_training_history.csv", index=False)
        concept_prediction = predict_concepts(
            model, arrays, np.arange(len(arrays["driver"])), device=device,
            batch_size=batch_size,
        )
        trip_table = pd.DataFrame({
            "trip_key": arrays["trip_key"],
            "vehicle_id": arrays["vehicle_id"],
            **{
                f"true_{name}": arrays["concept_targets"][:, i]
                for i, name in enumerate(EXTERNAL_CONCEPT_NAMES)
            },
            **{
                f"predicted_{name}": concept_prediction[:, i]
                for i, name in enumerate(EXTERNAL_CONCEPT_NAMES)
            },
        })
        trip_table[trip_table.vehicle_id.eq(held_out)].to_csv(
            fold_dir / "heldout_trip_concept_predictions.csv", index=False
        )
        x, counts = _interval_features(
            arrays, concept_prediction, valid_intervals, dataset="RWTH"
        )
        fold_train = valid_intervals.vehicle_id.ne(held_out).to_numpy() & (counts > 0)
        fold_test = valid_intervals.vehicle_id.eq(held_out).to_numpy() & (counts > 0)
        train_rows = valid_intervals.loc[fold_train].reset_index(drop=True)
        train_x, test_x = x[fold_train], x[fold_test]
        train_y = train_rows.soh_loss_pp.to_numpy(float)
        test_y = valid_intervals.loc[fold_test, "soh_loss_pp"].to_numpy(float)
        if not len(test_y) or len(train_y) < 3:
            update(f"RWTH {held_out}: insufficient health rows; fold skipped")
            continue
        for kind in ("KAN", "MLP"):
            update(f"RWTH {held_out}: selecting {kind} persistence gate inside training vehicles")
            alpha, gate_table = _select_gate(
                train_x, train_y, train_rows, kind=kind, dataset="RWTH",
                device=device, epochs=health_epochs, seed=seed + 100 + fold,
            )
            gate_table.insert(0, "outer_fold", fold)
            gate_table.insert(1, "held_out_vehicle", held_out)
            gate_table.insert(2, "head", kind)
            gates.append(gate_table)
            raw, state = _fit_health_head(
                train_x, train_y, test_x, kind=kind, device=device,
                epochs=health_epochs, seed=seed + 200 + fold,
            )
            torch.save(state, fold_dir / f"health_{kind.lower()}_head.pt")
            update(f"RWTH {held_out}: {kind} selected alpha={alpha:.2f}")
            for local, (_, row) in enumerate(valid_intervals.loc[fold_test].iterrows()):
                for model_name, value, gate in (
                    (f"DBPatchTransformer-{kind}-alpha1", raw[local], 1.0),
                    (f"PersistenceGated-DBPatchTransformer-{kind}", alpha * raw[local], alpha),
                ):
                    predictions.append({
                        "dataset": "RWTH",
                        "model": model_name,
                        "outer_fold": fold,
                        "cluster": held_out,
                        "row_id": row.interval_id,
                        "true_loss_pp": float(row.soh_loss_pp),
                        "predicted_loss_pp": float(value),
                        "gate_alpha": float(gate),
                        "anchor_soh_pct": float(row.start_soh_pct),
                        "predicted_future_soh_pct": float(row.start_soh_pct - value),
                        "true_future_soh_pct": float(row.end_soh_pct),
                        "trip_sequences": int(counts[valid_intervals.index.get_loc(row.name)]),
                    })
        for _, row in valid_intervals.loc[fold_test].iterrows():
            predictions.append({
                "dataset": "RWTH", "model": "Persistence",
                "outer_fold": fold, "cluster": held_out,
                "row_id": row.interval_id, "true_loss_pp": float(row.soh_loss_pp),
                "predicted_loss_pp": 0.0, "gate_alpha": 0.0,
                "anchor_soh_pct": float(row.start_soh_pct),
                "predicted_future_soh_pct": float(row.start_soh_pct),
                "true_future_soh_pct": float(row.end_soh_pct),
                "trip_sequences": int(counts[valid_intervals.index.get_loc(row.name)]),
            })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return predictions, gates


def _eve_outer_splits(intervals: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    ordered = intervals.sort_values("checkpoint_time").reset_index(drop=True)
    minimum = max(12, int(0.50 * len(ordered)))
    boundaries = np.linspace(minimum, len(ordered), 5, dtype=int)
    return [
        (np.arange(0, boundaries[i]), np.arange(boundaries[i], boundaries[i + 1]))
        for i in range(4) if boundaries[i + 1] > boundaries[i]
    ]


def run_eve(
    arrays: dict[str, np.ndarray],
    intervals: pd.DataFrame,
    *,
    output: Path,
    device: torch.device,
    concept_epochs: int,
    health_epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    table = intervals.dropna(
        subset=["previous_soh_trend_pct", "signed_trend_loss_pp", "checkpoint_time"]
    ).sort_values("checkpoint_time").reset_index(drop=True)
    splits = _eve_outer_splits(table)
    update(f"EVE full dual-stream temporal validation: {len(table)} intervals, {len(splits)} folds")
    predictions: list[dict[str, object]] = []
    gates: list[pd.DataFrame] = []
    for fold, (train_rows_index, test_rows_index) in enumerate(splits):
        cutoff = pd.Timestamp(table.iloc[test_rows_index[0]].checkpoint_time).value
        outer_train_seq = np.flatnonzero(arrays["trip_end_ns"] < cutoff)
        train_seq, validation_seq = _validation_split(outer_train_seq)
        fold_dir = output / f"eve_fold_{fold}"
        update(
            f"EVE outer fold {fold + 1}/{len(splits)}: cutoff "
            f"{table.iloc[test_rows_index[0]].checkpoint_time}"
        )
        model, history = train_concept_encoder(
            arrays, train_seq, validation_seq, device=device,
            output=fold_dir / "best_concept_encoder.pt", epochs=concept_epochs,
            batch_size=batch_size, seed=seed + 20 + fold,
        )
        history.to_csv(fold_dir / "concept_training_history.csv", index=False)
        concept_prediction = predict_concepts(
            model, arrays, np.arange(len(arrays["driver"])), device=device,
            batch_size=batch_size,
        )
        x, counts = _interval_features(arrays, concept_prediction, table, dataset="EVE")
        usable_train = train_rows_index[counts[train_rows_index] > 0]
        usable_test = test_rows_index[counts[test_rows_index] > 0]
        train_rows = table.iloc[usable_train].reset_index(drop=True)
        train_x, test_x = x[usable_train], x[usable_test]
        train_y = train_rows.signed_trend_loss_pp.to_numpy(float)
        test_y = table.iloc[usable_test].signed_trend_loss_pp.to_numpy(float)
        for kind in ("KAN", "MLP"):
            update(f"EVE fold {fold}: selecting {kind} persistence gate inside past intervals")
            alpha, gate_table = _select_gate(
                train_x, train_y, train_rows, kind=kind, dataset="EVE",
                device=device, epochs=health_epochs, seed=seed + 300 + fold,
            )
            gate_table.insert(0, "outer_fold", fold)
            gate_table.insert(1, "held_out_vehicle", "chronological_future")
            gate_table.insert(2, "head", kind)
            gates.append(gate_table)
            raw, state = _fit_health_head(
                train_x, train_y, test_x, kind=kind, device=device,
                epochs=health_epochs, seed=seed + 400 + fold,
            )
            torch.save(state, fold_dir / f"health_{kind.lower()}_head.pt")
            update(f"EVE fold {fold}: {kind} selected alpha={alpha:.2f}")
            for local, position in enumerate(usable_test):
                row = table.iloc[position]
                for model_name, value, gate in (
                    (f"DBPatchTransformer-{kind}-alpha1", raw[local], 1.0),
                    (f"PersistenceGated-DBPatchTransformer-{kind}", alpha * raw[local], alpha),
                ):
                    predictions.append({
                        "dataset": "EVE", "model": model_name,
                        "outer_fold": fold, "cluster": f"time_fold_{fold}",
                        "row_id": row.checkpoint_id, "true_loss_pp": float(row.signed_trend_loss_pp),
                        "predicted_loss_pp": float(value), "gate_alpha": float(gate),
                        "anchor_soh_pct": float(row.previous_soh_trend_pct),
                        "predicted_future_soh_pct": float(row.previous_soh_trend_pct - value),
                        "true_future_soh_pct": float(row.bms_soh_trend_pct),
                        "trip_sequences": int(counts[position]),
                    })
        for position in usable_test:
            row = table.iloc[position]
            predictions.append({
                "dataset": "EVE", "model": "Persistence",
                "outer_fold": fold, "cluster": f"time_fold_{fold}",
                "row_id": row.checkpoint_id, "true_loss_pp": float(row.signed_trend_loss_pp),
                "predicted_loss_pp": 0.0, "gate_alpha": 0.0,
                "anchor_soh_pct": float(row.previous_soh_trend_pct),
                "predicted_future_soh_pct": float(row.previous_soh_trend_pct),
                "true_future_soh_pct": float(row.bms_soh_trend_pct),
                "trip_sequences": int(counts[position]),
            })
        trip_test = arrays["trip_end_ns"] >= cutoff
        pd.DataFrame({
            "trip_key": arrays["trip_key"][trip_test],
            **{
                f"true_{name}": arrays["concept_targets"][trip_test, i]
                for i, name in enumerate(EXTERNAL_CONCEPT_NAMES)
            },
            **{
                f"predicted_{name}": concept_prediction[trip_test, i]
                for i, name in enumerate(EXTERNAL_CONCEPT_NAMES)
            },
        }).to_csv(fold_dir / "future_trip_concept_predictions.csv", index=False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return predictions, gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rwth-sequences", type=Path, required=True)
    parser.add_argument("--rwth-intervals", type=Path, required=True)
    parser.add_argument("--eve-sequences", type=Path, required=True)
    parser.add_argument("--eve-intervals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concept-epochs", type=int, default=18)
    parser.add_argument("--health-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    update(f"Stage 7 starts on {device}; KAN splines remain float32")
    rwth = load_sequence_npz(args.rwth_sequences)
    eve = load_sequence_npz(args.eve_sequences)
    rwth_intervals = pd.read_csv(args.rwth_intervals)
    eve_intervals = pd.read_csv(args.eve_intervals)
    all_predictions: list[dict[str, object]] = []
    all_gates: list[pd.DataFrame] = []
    rwth_predictions, rwth_gates = run_rwth(
        rwth, rwth_intervals, output=args.output_dir / "rwth",
        device=device, concept_epochs=args.concept_epochs,
        health_epochs=args.health_epochs, batch_size=args.batch_size, seed=args.seed,
    )
    all_predictions.extend(rwth_predictions)
    all_gates.extend(rwth_gates)
    eve_predictions, eve_gates = run_eve(
        eve, eve_intervals, output=args.output_dir / "eve",
        device=device, concept_epochs=args.concept_epochs,
        health_epochs=args.health_epochs, batch_size=args.batch_size, seed=args.seed,
    )
    all_predictions.extend(eve_predictions)
    all_gates.extend(eve_gates)
    predictions = pd.DataFrame(all_predictions)
    metrics = _metric_rows(predictions)
    bootstrap = _paired_cluster_bootstrap(predictions, seed=args.seed)
    gates = pd.concat(all_gates, ignore_index=True) if all_gates else pd.DataFrame()
    predictions.to_csv(args.output_dir / "stage7_external_health_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "stage7_external_health_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "stage7_paired_cluster_bootstrap.csv", index=False)
    gates.to_csv(args.output_dir / "stage7_inner_gate_selection.csv", index=False)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "full_dual_stream_dbpatchkan_tested_externally": True,
        "health_head_uses_named_concepts_only": True,
        "persistence_gate_selected_inside_training_data": True,
        "nissan_locked_test_used": False,
        "rwth_sequence_trips": len(rwth["driver"]),
        "eve_sequence_trips": len(eve["driver"]),
        "concept_names": EXTERNAL_CONCEPT_NAMES,
        "health_input_names": HEALTH_CONCEPT_NAMES,
        "external_sequence_adapter": (
            "one 128-step time-normalized sequence per trip; RWTH is derived "
            "from 1-Hz source telemetry and EVE from the harmonised irregular source"
        ),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "limitations": [
            "RWTH degradation labels remain extremely sparse",
            "EVE contains one vehicle and uses chronological rather than cross-vehicle validation",
            "RWTH temperature response is a physics proxy; EVE temperature rise is measured",
            "A zero learned gate is a valid persistence fallback, not a failed run",
        ],
    }
    (args.output_dir / "STAGE7_EXTERNAL_SEQUENCE_AUDIT.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    update(f"Stage 7 complete in {audit['elapsed_minutes']:.1f} minutes")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
