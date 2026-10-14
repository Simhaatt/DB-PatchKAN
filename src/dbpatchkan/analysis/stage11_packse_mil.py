"""Stage 11: Pack-SE residual multiple-instance concept KAN.

RWTH and EVE are development datasets from this stage onward.  The sealed
Nissan test is never an input.  Variable numbers of trips between health
checkpoints are represented as bags of up to 64 chronological trip tokens.
A small Transformer assigns attention weights, but the KAN receives only the
attention-weighted *named* trip concepts plus eight named interval concepts.
Pack-SE supplies a physical baseline; KAN predicts its residual correction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from torch import Tensor, nn
from torch.nn import functional as F

from dbpatchkan.analysis.stage10_normalized_transfer import (
    DOMAIN_INDEX,
    DOMAIN_ORDER,
    build_normalized_table,
    inner_split,
    outer_folds,
    pooled_training_indices,
)
from dbpatchkan.models.kan import ConceptKAN, ConceptMLP


TRIP_CONCEPT_NAMES = (
    "speed_mean_kmh",
    "accel_rms_mps2",
    "decel_rms_mps2",
    "jerk_rms_mps3",
    "harsh_accel_fraction",
    "harsh_decel_fraction",
    "c_rate_peak",
    "high_c_rate_fraction",
    "pack_temp_mean_c",
    "soc_mean_pct",
    "dbsi",
    "stress_rate",
)
INTERVAL_CONCEPT_NAMES = (
    "log1p_calendar_days",
    "log1p_distance_km",
    "log1p_efc",
    "log1p_charge_depth_pct",
    "log1p_pack_damage_p50",
    "pack_damage_relative_90pct_width",
    "anchor_soh_pct",
    "log1p_trip_count",
)
KAN_CONCEPT_NAMES = (*TRIP_CONCEPT_NAMES, *INTERVAL_CONCEPT_NAMES)
TARGET = "relative_interval_loss_pct"


def update(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _num(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator.abs().gt(1e-9))


def canonical_trip_table(
    nissan_path: Path, rwth_path: Path, eve_path: Path
) -> pd.DataFrame:
    """Map three telemetry sources into twelve common named trip concepts."""

    tables: list[pd.DataFrame] = []
    n = pd.read_csv(nissan_path)
    duration_n = _num(n, "duration_s") / 3600.0
    tables.append(pd.DataFrame({
        "dataset": "Nissan-development", "vehicle_id": n["vehicle_id"].astype(str),
        "trip_key": n["trip_key"].astype(str), "trip_end": n["trip_end_utc"],
        "duration_h": duration_n, "distance_km": _num(n, "distance_km"),
        "speed_mean_kmh": _num(n, "speed_mean_kmh"),
        "accel_rms_mps2": _num(n, "accel_pos_rms"),
        "decel_rms_mps2": _num(n, "decel_abs_rms"),
        "jerk_rms_mps3": _num(n, "jerk_rms"),
        "harsh_accel_fraction": _num(n, "harsh_accel_per_min"),
        "harsh_decel_fraction": _num(n, "harsh_brake_per_min"),
        "c_rate_peak": _num(n, "current_dis_p95_a") / 111.11,
        "high_c_rate_fraction": (_num(n, "current_dis_p95_a") / 111.11 - 1.0).clip(lower=0.0),
        "pack_temp_mean_c": _num(n, "pack_temp_mean_c"),
        "soc_mean_pct": (_num(n, "soc_start_pct") + _num(n, "soc_end_pct")) / 2.0,
        "dbsi": _num(n, "bsi_intensity"),
        "stress_rate": _safe_ratio(_num(n, "stress_dose"), duration_n),
    }))

    r = pd.read_csv(rwth_path)
    duration_r = _num(r, "duration_observed_h")
    tables.append(pd.DataFrame({
        "dataset": "RWTH", "vehicle_id": r["vehicle_id"].astype(str),
        "trip_key": r["trip_key"].astype(str), "trip_end": r["trip_end"],
        "duration_h": duration_r, "distance_km": _num(r, "distance_km"),
        "speed_mean_kmh": _num(r, "speed_mean_kmh"),
        "accel_rms_mps2": _num(r, "accel_positive_rms_mps2"),
        "decel_rms_mps2": _num(r, "decel_rms_mps2"),
        "jerk_rms_mps3": _num(r, "jerk_rms_mps3"),
        "harsh_accel_fraction": _num(r, "harsh_accel_fraction"),
        "harsh_decel_fraction": _num(r, "harsh_decel_fraction"),
        "c_rate_peak": _num(r, "c_rate_discharge_peak"),
        "high_c_rate_fraction": _num(r, "high_c_rate_fraction"),
        # RWTH lacks pack temperature; ambient is an explicitly labelled proxy.
        "pack_temp_mean_c": _num(r, "ambient_temp_mean_c"),
        "soc_mean_pct": _num(r, "soc_mean_pct"),
        "dbsi": _num(r, "dbsi_partial_renormalized"),
        "stress_rate": _num(r, "dbsi_partial_renormalized") / 100.0,
    }))

    e = pd.read_csv(eve_path)
    duration_e = _num(e, "duration_h")
    tables.append(pd.DataFrame({
        "dataset": "EVE", "vehicle_id": e["vehicle_id"].astype(str),
        "trip_key": e["trip_key"].astype(str), "trip_end": e["trip_end"],
        "duration_h": duration_e, "distance_km": _num(e, "distance_km"),
        "speed_mean_kmh": _num(e, "speed_mean_kmh"),
        "accel_rms_mps2": _num(e, "accel_positive_rms_mps2"),
        "decel_rms_mps2": _num(e, "decel_rms_mps2"),
        "jerk_rms_mps3": _num(e, "jerk_rms_mps3"),
        "harsh_accel_fraction": _num(e, "harsh_accel_fraction"),
        "harsh_decel_fraction": _num(e, "harsh_decel_fraction"),
        "c_rate_peak": _num(e, "c_rate_abs_p95"),
        "high_c_rate_fraction": _num(e, "high_c_rate_fraction"),
        "pack_temp_mean_c": _num(e, "pack_temp_mean_c"),
        "soc_mean_pct": _num(e, "soc_mean_pct"),
        "dbsi": _num(e, "dbsi"),
        "stress_rate": _safe_ratio(_num(e, "stress_dose_h"), duration_e),
    }))
    output = pd.concat(tables, ignore_index=True)
    output["trip_end"] = pd.to_datetime(output["trip_end"], utc=True, format="mixed")
    return output


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[1], np.nan, dtype=np.float32)
    for column in range(values.shape[1]):
        valid = np.isfinite(values[:, column]) & np.isfinite(weights) & (weights > 0)
        if valid.any():
            result[column] = np.average(values[valid, column], weights=weights[valid])
    return result


def build_trip_bags(
    intervals: pd.DataFrame,
    trips: pd.DataFrame,
    *,
    max_tokens: int = 64,
) -> dict[str, np.ndarray]:
    tokens = np.full(
        (len(intervals), max_tokens, len(TRIP_CONCEPT_NAMES)), np.nan, dtype=np.float32
    )
    mask = np.zeros((len(intervals), max_tokens), dtype=bool)
    source_counts = np.zeros(len(intervals), dtype=np.int64)
    token_counts = np.zeros(len(intervals), dtype=np.int64)
    for row_position, interval in enumerate(intervals.itertuples(index=False)):
        start = pd.to_datetime(interval.interval_start, utc=True)
        end = pd.to_datetime(interval.interval_end, utc=True)
        chosen = trips.loc[
            trips["dataset"].eq(interval.dataset)
            & trips["vehicle_id"].astype(str).eq(str(interval.vehicle_id))
            & trips["trip_end"].gt(start)
            & trips["trip_end"].le(end)
        ].sort_values("trip_end")
        source_counts[row_position] = len(chosen)
        if chosen.empty:
            continue
        groups = np.array_split(np.arange(len(chosen)), min(max_tokens, len(chosen)))
        values = chosen.loc[:, TRIP_CONCEPT_NAMES].to_numpy(dtype=np.float32)
        duration = _num(chosen, "duration_h").fillna(0.0).clip(lower=1e-6).to_numpy()
        for token_position, group in enumerate(groups):
            tokens[row_position, token_position] = _weighted_mean(values[group], duration[group])
            mask[row_position, token_position] = True
        token_counts[row_position] = len(groups)
    if (source_counts == 0).any():
        missing = intervals.loc[source_counts == 0, ["dataset", "interval_id"]]
        raise ValueError(f"Intervals without trip bags:\n{missing}")
    return {
        "tokens": tokens,
        "token_mask": mask,
        "source_trip_count": source_counts,
        "compressed_token_count": token_counts,
    }


def save_bags(path: Path, bags: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bags)
    return path


def load_bags(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        result = {name: source[name] for name in source.files}
    required = {"tokens", "token_mask", "source_trip_count", "compressed_token_count"}
    if required - set(result):
        raise ValueError(f"Bag archive is missing {sorted(required - set(result))}")
    return result


def interval_context(table: pd.DataFrame, bags: dict[str, np.ndarray]) -> pd.DataFrame:
    result = pd.DataFrame(index=table.index)
    result["log1p_calendar_days"] = np.log1p(_num(table, "calendar_days").clip(lower=0.0))
    result["log1p_distance_km"] = np.log1p(_num(table, "distance_km").clip(lower=0.0))
    result["log1p_efc"] = np.log1p(_num(table, "efc").clip(lower=0.0))
    result["log1p_charge_depth_pct"] = np.log1p(_num(table, "charge_depth_pct").clip(lower=0.0))
    result["log1p_pack_damage_p50"] = np.log1p(_num(table, "pack_damage_p50").clip(lower=0.0))
    result["pack_damage_relative_90pct_width"] = _num(table, "relative_90pct_width")
    result["anchor_soh_pct"] = _num(table, "anchor_soh_pct")
    result["log1p_trip_count"] = np.log1p(bags["source_trip_count"].astype(float))
    return result.loc[:, INTERVAL_CONCEPT_NAMES]


class BagTransform:
    def __init__(self) -> None:
        self.token_center: np.ndarray | None = None
        self.token_scale: np.ndarray | None = None
        self.context_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.context_scaler = RobustScaler(quantile_range=(10.0, 90.0))

    def fit(
        self,
        bags: dict[str, np.ndarray],
        context: pd.DataFrame,
        indices: np.ndarray,
    ) -> "BagTransform":
        values = bags["tokens"][indices]
        valid_tokens = bags["token_mask"][indices]
        flat = values[valid_tokens]
        center = np.nanmedian(flat, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        q25, q75 = np.nanpercentile(flat, [25, 75], axis=0)
        scale = (q75 - q25) / 1.349
        ordinary = np.nanstd(flat, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, ordinary)
        self.token_center = center.astype(np.float32)
        self.token_scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
        imputed = self.context_imputer.fit_transform(context.loc[indices])
        self.context_scaler.fit(imputed)
        return self

    def transform_tokens(self, bags: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
        values = bags["tokens"][indices].copy()
        values = np.where(np.isfinite(values), values, self.token_center)
        values = (values - self.token_center) / self.token_scale
        values[~bags["token_mask"][indices]] = 0.0
        return values.astype(np.float32)

    def transform_context(self, context: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
        imputed = self.context_imputer.transform(context.loc[indices])
        return self.context_scaler.transform(imputed).astype(np.float32)


class PackSEResidualMIL(nn.Module):
    def __init__(self, kind: str, max_tokens: int = 64) -> None:
        super().__init__()
        width = 48
        self.kind = kind
        self.token_projection = nn.Linear(len(TRIP_CONCEPT_NAMES), width)
        self.position = nn.Parameter(torch.zeros(1, max_tokens, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=96, dropout=0.10,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.trip_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1, enable_nested_tensor=False
        )
        self.attention_score = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        if kind == "KAN":
            self.residual_head: nn.Module = ConceptKAN((20, 12, 6, 1), grid_size=5)
        elif kind == "MLP":
            self.residual_head = ConceptMLP((20, 24, 12, 1), dropout=0.10)
        else:
            raise ValueError(kind)
        self.pack_weight_raw = nn.Parameter(torch.zeros(len(DOMAIN_ORDER)))
        self.domain_scale_raw = nn.Parameter(torch.zeros(len(DOMAIN_ORDER)))
        self.domain_bias = nn.Parameter(torch.zeros(len(DOMAIN_ORDER)))

    def forward(
        self,
        tokens: Tensor,
        token_mask: Tensor,
        context: Tensor,
        pack_log_raw: Tensor,
        domain: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        encoded = self.token_projection(tokens) + self.position[:, : tokens.shape[1]]
        encoded = self.trip_encoder(encoded, src_key_padding_mask=~token_mask)
        logits = self.attention_score(encoded).squeeze(-1).masked_fill(~token_mask, -1e4)
        attention = torch.softmax(logits, dim=1)
        named_trip = torch.sum(attention.unsqueeze(-1) * tokens, dim=1)
        concepts = torch.cat((named_trip, context), dim=1)
        residual = self.residual_head(concepts).squeeze(-1)
        pack_term = F.softplus(self.pack_weight_raw[domain]) * pack_log_raw
        latent_physical = F.softplus(pack_term + residual)
        observed = self.domain_bias[domain] + F.softplus(self.domain_scale_raw[domain]) * latent_physical
        return observed, {
            "attention": attention,
            "named_trip": named_trip,
            "concepts": concepts,
            "residual": residual,
            "pack_term": pack_term,
            "latent_physical": latent_physical,
        }

    def regularization_loss(self) -> Tensor:
        return self.residual_head.regularization_loss() + 0.05 * self.domain_bias.square().mean()


def aggregate_features(
    bags: dict[str, np.ndarray], context: pd.DataFrame, indices: np.ndarray
) -> np.ndarray:
    tokens = bags["tokens"][indices]
    mask = bags["token_mask"][indices]
    means = np.full((len(indices), len(TRIP_CONCEPT_NAMES)), np.nan)
    for row in range(len(indices)):
        selected = tokens[row, mask[row]]
        means[row] = _weighted_mean(selected, np.ones(len(selected), dtype=float))
    domains = np.zeros((len(indices), len(DOMAIN_ORDER)), dtype=float)
    return np.concatenate((means, context.loc[indices].to_numpy(dtype=float), domains), axis=1)


def add_domain_onehot(matrix: np.ndarray, table: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    output = matrix.copy()
    output[:, -len(DOMAIN_ORDER):] = 0.0
    for row, domain in enumerate(table.loc[indices, "dataset"]):
        output[row, -len(DOMAIN_ORDER) + DOMAIN_INDEX[str(domain)]] = 1.0
    return output


def fit_rf_predict(
    table: pd.DataFrame,
    bags: dict[str, np.ndarray],
    context: pd.DataFrame,
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    x_train = add_domain_onehot(aggregate_features(bags, context, train_idx), table, train_idx)
    x_predict = add_domain_onehot(aggregate_features(bags, context, predict_idx), table, predict_idx)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x_train)
    model = RandomForestRegressor(
        n_estimators=500, max_features=0.65, min_samples_leaf=3,
        random_state=seed, n_jobs=-1,
    )
    model.fit(imputer.transform(x_train), _num(table.loc[train_idx], TARGET).to_numpy())
    return model.predict(imputer.transform(x_predict))


def rf_teacher_oof(
    table: pd.DataFrame,
    bags: dict[str, np.ndarray],
    context: pd.DataFrame,
    train_idx: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    x = add_domain_onehot(aggregate_features(bags, context, train_idx), table, train_idx)
    y = _num(table.loc[train_idx], TARGET).to_numpy()
    groups = (
        table.loc[train_idx, "dataset"].astype(str)
        + ":" + table.loc[train_idx, "vehicle_id"].astype(str)
    ).to_numpy()
    prediction = np.full(len(train_idx), np.nan)
    for fit, held in LeaveOneGroupOut().split(x, y, groups):
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[fit])
        model = RandomForestRegressor(
            n_estimators=150, max_features=0.65, min_samples_leaf=3,
            random_state=seed, n_jobs=-1,
        )
        model.fit(imputer.transform(x[fit]), y[fit])
        prediction[held] = model.predict(imputer.transform(x[held]))
    return prediction.astype(np.float32)


def fit_packse_predict(
    table: pd.DataFrame, train_idx: np.ndarray, predict_idx: np.ndarray
) -> np.ndarray:
    train = table.loc[train_idx]
    predict = table.loc[predict_idx]
    result = np.zeros(len(predict), dtype=float)
    for domain in predict["dataset"].unique():
        fit = train["dataset"].eq(domain)
        if fit.sum() < 4:
            fit = pd.Series(True, index=train.index)
        x_train = np.log1p(_num(train.loc[fit], "pack_damage_p50").clip(lower=0.0)).to_numpy().reshape(-1, 1)
        y_train = _num(train.loc[fit], TARGET).to_numpy()
        x_test = np.log1p(_num(predict, "pack_damage_p50").clip(lower=0.0)).to_numpy().reshape(-1, 1)
        selected = predict["dataset"].eq(domain).to_numpy()
        imputer = SimpleImputer(strategy="median").fit(x_train)
        model = LinearRegression(positive=True).fit(imputer.transform(x_train), y_train)
        result[selected] = model.predict(imputer.transform(x_test[selected]))
    return result


def _domain_scale(table: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    result = np.ones(len(DOMAIN_ORDER), dtype=np.float32)
    train = table.loc[indices]
    for domain, domain_id in DOMAIN_INDEX.items():
        values = _num(train[train["dataset"].eq(domain)], TARGET).dropna().to_numpy()
        if len(values):
            q25, q75 = np.percentile(values, [25, 75])
            result[domain_id] = max((q75 - q25) / 1.349, float(np.std(values)), 0.05)
    return result


def train_mil(
    table: pd.DataFrame,
    bags: dict[str, np.ndarray],
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    *,
    kind: str,
    epochs: int,
    seed: int,
    device: torch.device,
    teacher_values: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    seed_everything(seed)
    context = interval_context(table, bags)
    transform = BagTransform().fit(bags, context, train_idx)
    train_tokens = torch.from_numpy(transform.transform_tokens(bags, train_idx)).to(device)
    pred_tokens = torch.from_numpy(transform.transform_tokens(bags, predict_idx)).to(device)
    train_context = torch.from_numpy(transform.transform_context(context, train_idx)).to(device)
    pred_context = torch.from_numpy(transform.transform_context(context, predict_idx)).to(device)
    train_mask = torch.from_numpy(bags["token_mask"][train_idx]).to(device)
    pred_mask = torch.from_numpy(bags["token_mask"][predict_idx]).to(device)
    train_domain = torch.tensor(
        [DOMAIN_INDEX[x] for x in table.loc[train_idx, "dataset"]], dtype=torch.long, device=device
    )
    pred_domain = torch.tensor(
        [DOMAIN_INDEX[x] for x in table.loc[predict_idx, "dataset"]], dtype=torch.long, device=device
    )
    pack_train_values = np.log1p(
        _num(table.loc[train_idx], "pack_damage_p50").clip(lower=0.0).to_numpy()
    )
    pack_center = float(np.nanmedian(pack_train_values))
    pack_center = pack_center if np.isfinite(pack_center) else 0.0
    pack_train = torch.from_numpy(
        np.where(np.isfinite(pack_train_values), pack_train_values, pack_center).astype(np.float32)
    ).to(device)
    pack_pred_values = np.log1p(
        _num(table.loc[predict_idx], "pack_damage_p50").clip(lower=0.0).to_numpy()
    )
    pack_pred = torch.from_numpy(
        np.where(np.isfinite(pack_pred_values), pack_pred_values, pack_center).astype(np.float32)
    ).to(device)
    y = torch.from_numpy(_num(table.loc[train_idx], TARGET).to_numpy(dtype=np.float32)).to(device)
    if teacher_values is None:
        teacher_values = rf_teacher_oof(table, bags, context, train_idx, seed=42)
    if len(teacher_values) != len(train_idx):
        raise ValueError("Teacher predictions do not align with training indices")
    teacher = torch.from_numpy(np.asarray(teacher_values, dtype=np.float32)).to(device)
    scales = torch.from_numpy(_domain_scale(table, train_idx)).to(device)
    model = PackSEResidualMIL(kind, max_tokens=bags["tokens"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.03)
    best, best_state, patience = math.inf, None, 45
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, detail = model(train_tokens, train_mask, train_context, pack_train, train_domain)
        normalized = (prediction - y) / scales[train_domain]
        supervised = F.huber_loss(normalized, torch.zeros_like(normalized), delta=1.0)
        teacher_loss = F.huber_loss(
            (prediction - teacher) / scales[train_domain],
            torch.zeros_like(prediction), delta=1.0,
        )
        residual_shrink = detail["residual"].square().mean()
        entropy = -torch.sum(
            detail["attention"] * torch.log(detail["attention"].clamp_min(1e-8)), dim=1
        ).mean()
        loss = (
            supervised + 0.15 * teacher_loss + 5e-4 * model.regularization_loss()
            + 1e-3 * residual_shrink + 1e-4 * entropy
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        history.append({"epoch": epoch, "loss": value, "supervised": float(supervised.detach().cpu())})
        if value < best - 1e-6:
            best, best_state, patience = value, copy.deepcopy(model.state_dict()), 45
        else:
            patience -= 1
            if patience == 0:
                break
    if best_state is None:
        raise RuntimeError("MIL model did not produce a state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction, detail = model(pred_tokens, pred_mask, pred_context, pack_pred, pred_domain)
    arrays = {name: value.detach().cpu().numpy() for name, value in detail.items()}
    state: dict[str, object] = {
        "kind": kind, "state_dict": best_state, "trip_concepts": TRIP_CONCEPT_NAMES,
        "interval_concepts": INTERVAL_CONCEPT_NAMES, "kan_concepts": KAN_CONCEPT_NAMES,
        "token_center": transform.token_center, "token_scale": transform.token_scale,
        "context_imputer_statistics": transform.context_imputer.statistics_,
        "context_scaler_center": transform.context_scaler.center_,
        "context_scaler_scale": transform.context_scaler.scale_,
        "domain_pack_weight": F.softplus(model.pack_weight_raw).detach().cpu().numpy(),
        "domain_measurement_scale": F.softplus(model.domain_scale_raw).detach().cpu().numpy(),
        "domain_measurement_bias": model.domain_bias.detach().cpu().numpy(),
        "history": history, "best_loss": best,
    }
    if kind == "KAN":
        state["edge_strengths"] = [x.numpy() for x in model.residual_head.edge_strengths()]
    return prediction.cpu().numpy(), arrays, state


def choose_simplex_weights(
    truth: np.ndarray,
    expert_predictions: np.ndarray,
    *,
    step: float = 0.10,
) -> tuple[np.ndarray, float]:
    units = int(round(1.0 / step))
    best_weights = np.array([1.0, 0.0, 0.0, 0.0])
    best = math.inf
    for a in range(units + 1):
        for b in range(units - a + 1):
            for c in range(units - a - b + 1):
                d = units - a - b - c
                weights = np.array([a, b, c, d], dtype=float) / units
                prediction = expert_predictions @ weights
                error = float(np.mean(np.abs(truth - prediction)))
                if error < best - 1e-12:
                    best, best_weights = error, weights
    persistence = float(np.mean(np.abs(truth)))
    if best >= 0.98 * persistence:
        return np.array([1.0, 0.0, 0.0, 0.0]), persistence
    return best_weights, best


def finite_quantile(values: np.ndarray, coverage: float = 0.90) -> float:
    values = np.sort(values[np.isfinite(values)])
    if not len(values):
        return math.nan
    rank = min(int(math.ceil((len(values) + 1) * coverage)) - 1, len(values) - 1)
    return float(values[max(rank, 0)])


def prediction_row(
    row: pd.Series, fold: str, model: str, seed: int, predicted_relative: float,
    *, radius_pp: float = math.nan, weights: np.ndarray | None = None,
) -> dict[str, object]:
    anchor = float(row["anchor_soh_pct"])
    predicted_loss = anchor * predicted_relative / 100.0
    future = anchor - predicted_loss
    output = {
        "dataset": row["dataset"], "vehicle_id": row["vehicle_id"],
        "interval_id": row["interval_id"], "outer_fold": fold,
        "model": model, "seed": seed, "anchor_soh_pct": anchor,
        "true_loss_pp": float(row["signed_loss_pp"]), "predicted_loss_pp": predicted_loss,
        "true_future_soh_pct": float(row["future_soh_pct"]),
        "predicted_future_soh_pct": future,
        "interval_lower_soh_pct": future - radius_pp,
        "interval_upper_soh_pct": future + radius_pp,
    }
    labels = ("weight_persistence", "weight_mil", "weight_rf", "weight_packse")
    for name, value in zip(labels, weights if weights is not None else [math.nan] * 4):
        output[name] = float(value)
    return output


def run(
    table: pd.DataFrame,
    bags: dict[str, np.ndarray],
    *,
    output: Path,
    device: torch.device,
    epochs: int,
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    folds = outer_folds(table)
    update(f"Stage 11 has {len(folds)} outer folds, {len(seeds)} seeds, {len(table)} intervals")
    context = interval_context(table, bags)
    for fold_number, fold in enumerate(folds, 1):
        dataset, name = str(fold["dataset"]), str(fold["fold"])
        test_idx = np.asarray(fold["test"], dtype=int)
        train_idx = pooled_training_indices(table, fold)
        if np.intersect1d(train_idx, test_idx).size:
            raise AssertionError("Outer test leaked into training")
        subtrain, inner = inner_split(table, train_idx)
        target_inner = table.loc[inner, "dataset"].eq(dataset).to_numpy()
        selected_inner = inner[target_inner] if target_inner.any() else inner
        update(f"outer {fold_number}/{len(folds)} {dataset} {name}: train={len(train_idx)}, test={len(test_idx)}")

        # Seed-independent outer baselines are recorded once.
        rf_test = fit_rf_predict(table, bags, context, train_idx, test_idx, seed=42)
        pack_test = fit_packse_predict(table, train_idx, test_idx)
        for model, values in (("Persistence", np.zeros(len(test_idx))), ("RF-MIL-features", rf_test), ("PackSE-affine", pack_test)):
            for local, row_index in enumerate(test_idx):
                rows.append(prediction_row(table.loc[row_index], name, model, 0, float(values[local])))

        rf_inner = fit_rf_predict(table, bags, context, subtrain, inner, seed=42)
        pack_inner = fit_packse_predict(table, subtrain, inner)
        teacher_subtrain = rf_teacher_oof(table, bags, context, subtrain, seed=42)
        teacher_train = rf_teacher_oof(table, bags, context, train_idx, seed=42)
        truth_inner = _num(table.loc[selected_inner], TARGET).to_numpy()
        inner_positions = np.flatnonzero(np.isin(inner, selected_inner))
        for kind in ("KAN", "MLP"):
            for seed in seeds:
                update(f"{dataset} {name}: PackSE-residual MIL-{kind} seed={seed}")
                mil_inner, _, _ = train_mil(
                    table, bags, subtrain, inner, kind=kind,
                    epochs=max(20, epochs // 2), seed=seed, device=device,
                    teacher_values=teacher_subtrain,
                )
                experts_inner = np.column_stack((
                    np.zeros(len(inner)), mil_inner, rf_inner, pack_inner,
                ))[inner_positions]
                weights, inner_mae = choose_simplex_weights(truth_inner, experts_inner)
                inner_residual = np.abs(truth_inner - experts_inner @ weights)
                radius_rel = finite_quantile(inner_residual)
                radius_pp = radius_rel * float(np.nanmedian(_num(table.loc[selected_inner], "anchor_soh_pct"))) / 100.0
                mil_test, detail, state = train_mil(
                    table, bags, train_idx, test_idx, kind=kind,
                    epochs=epochs, seed=seed, device=device,
                    teacher_values=teacher_train,
                )
                experts_test = np.column_stack((
                    np.zeros(len(test_idx)), mil_test, rf_test, pack_test,
                ))
                hybrid = experts_test @ weights
                model_dir = output / "models" / dataset / name / kind / f"seed_{seed}"
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(state, model_dir / "model.pt")
                pd.DataFrame(state["history"]).to_csv(model_dir / "history.csv", index=False)
                for domain_id, domain_name in enumerate(DOMAIN_ORDER):
                    parameter_rows.append({
                        "outer_dataset": dataset, "outer_fold": name,
                        "head": kind, "seed": seed, "calibration_domain": domain_name,
                        "positive_packse_weight": float(state["domain_pack_weight"][domain_id]),
                        "positive_measurement_scale": float(state["domain_measurement_scale"][domain_id]),
                        "signed_measurement_bias": float(state["domain_measurement_bias"][domain_id]),
                    })
                weight_rows.append({
                    "dataset": dataset, "outer_fold": name, "head": kind, "seed": seed,
                    "inner_rows": len(truth_inner), "inner_mae_relative_pct": inner_mae,
                    "weight_persistence": weights[0], "weight_mil": weights[1],
                    "weight_rf": weights[2], "weight_packse": weights[3],
                    "conformal_radius_pp": radius_pp,
                })
                for local, row_index in enumerate(test_idx):
                    rows.append(prediction_row(
                        table.loc[row_index], name, f"PackSE-Residual-MIL-{kind}", seed,
                        float(mil_test[local]), radius_pp=radius_pp,
                    ))
                    rows.append(prediction_row(
                        table.loc[row_index], name, f"ConstrainedHybrid-MIL-{kind}", seed,
                        float(hybrid[local]), radius_pp=radius_pp, weights=weights,
                    ))
                    valid_attention = detail["attention"][local, bags["token_mask"][row_index]]
                    attention_rows.append({
                        "dataset": dataset, "outer_fold": name, "interval_id": table.loc[row_index, "interval_id"],
                        "head": kind, "seed": seed,
                        "source_trips": int(bags["source_trip_count"][row_index]),
                        "tokens": int(bags["compressed_token_count"][row_index]),
                        "attention_max": float(np.max(valid_attention)),
                        "attention_entropy": float(-np.sum(valid_attention * np.log(np.clip(valid_attention, 1e-8, None)))),
                        **{
                            f"weighted_{concept}": float(detail["named_trip"][local, i])
                            for i, concept in enumerate(TRIP_CONCEPT_NAMES)
                        },
                    })
    pd.DataFrame(attention_rows).to_csv(output / "mil_attention_audit.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output / "inner_constrained_ensemble_weights.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "domain_physics_calibration_parameters.csv", index=False)
    raw = pd.DataFrame(rows)
    return raw, ensemble(raw)


def ensemble(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "vehicle_id", "interval_id", "outer_fold", "model"]
    rows = []
    for key, group in raw.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        for column in ("anchor_soh_pct", "true_loss_pp", "true_future_soh_pct"):
            row[column] = float(group[column].iloc[0])
        for column in (
            "predicted_loss_pp", "predicted_future_soh_pct", "interval_lower_soh_pct",
            "interval_upper_soh_pct", "weight_persistence", "weight_mil", "weight_rf", "weight_packse",
        ):
            row[column] = float(group[column].median())
        row["seeds"] = int(group["seed"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in predictions.groupby(["dataset", "model"]):
        y = group["true_loss_pp"].to_numpy()
        p = group["predicted_loss_pp"].to_numpy()
        valid_interval = group["interval_lower_soh_pct"].notna()
        covered = (
            group["interval_lower_soh_pct"].le(group["true_future_soh_pct"])
            & group["interval_upper_soh_pct"].ge(group["true_future_soh_pct"])
        )
        rows.append({
            "dataset": dataset, "model": model, "n": len(group),
            "mae_pp": mean_absolute_error(y, p),
            "rmse_pp": math.sqrt(mean_squared_error(y, p)),
            "bias_loss_pp": float(np.mean(p - y)),
            "loss_r2": r2_score(y, p) if len(y) > 1 else math.nan,
            "interval_90_coverage": float(covered[valid_interval].mean()) if valid_interval.any() else math.nan,
        })
    return pd.DataFrame(rows).sort_values(["dataset", "mae_pp"])


def paired_bootstrap(
    predictions: pd.DataFrame,
    challengers: Sequence[str],
    *,
    iterations: int = 10_000,
    seed: int = 20260809,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, group in predictions.groupby("dataset"):
        wide = group.pivot(index=["vehicle_id", "interval_id"], columns="model", values="predicted_loss_pp")
        truth = group.drop_duplicates(["vehicle_id", "interval_id"]).set_index(
            ["vehicle_id", "interval_id"]
        )["true_loss_pp"].reindex(wide.index)
        vehicles = wide.index.get_level_values(0).to_numpy(dtype=str)
        for challenger in challengers:
            if challenger not in wide or "Persistence" not in wide:
                continue
            valid = truth.notna() & wide[challenger].notna() & wide["Persistence"].notna()
            y = truth[valid].to_numpy(); ref = wide.loc[valid, "Persistence"].to_numpy()
            chal = wide.loc[valid, challenger].to_numpy(); v = vehicles[valid.to_numpy()]
            delta = np.abs(y - chal) - np.abs(y - ref)
            samples = np.empty(iterations)
            unique = np.unique(v)
            if len(unique) > 1:
                clusters = {x: np.flatnonzero(v == x) for x in unique}
                for i in range(iterations):
                    chosen = rng.choice(unique, size=len(unique), replace=True)
                    idx = np.concatenate([clusters[x] for x in chosen])
                    samples[i] = np.mean(delta[idx])
            else:
                block = min(3, len(delta))
                for i in range(iterations):
                    pieces = []
                    while sum(map(len, pieces)) < len(delta):
                        start = int(rng.integers(0, max(len(delta) - block + 1, 1)))
                        pieces.append(np.arange(start, min(start + block, len(delta))))
                    samples[i] = np.mean(delta[np.concatenate(pieces)[: len(delta)]])
            rows.append({
                "dataset": dataset, "reference": "Persistence", "challenger": challenger,
                "n": len(delta), "delta_mae_pp": float(np.mean(delta)),
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
                "probability_challenger_better": float(np.mean(samples < 0)),
                "significant_win": bool(np.quantile(samples, 0.975) < 0),
            })
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-table", type=Path, required=True)
    parser.add_argument("--trip-bags", type=Path, required=True)
    parser.add_argument("--prior-stage10-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62, 72, 82])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.interval_table)
    bags = load_bags(args.trip_bags)
    if len(table) != len(bags["tokens"]):
        raise ValueError("Interval table and bag archive are not aligned")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    update(f"Stage 11 starts on {device}; KAN splines remain float32")
    raw, final = run(
        table, bags, output=args.output_dir, device=device,
        epochs=args.epochs, seeds=args.seeds,
    )
    score = metrics(final)
    bootstrap = paired_bootstrap(final, (
        "PackSE-Residual-MIL-KAN", "PackSE-Residual-MIL-MLP",
        "ConstrainedHybrid-MIL-KAN", "ConstrainedHybrid-MIL-MLP", "RF-MIL-features",
    ))
    raw.to_csv(args.output_dir / "all_seed_predictions.csv", index=False)
    final.to_csv(args.output_dir / "ensemble_predictions.csv", index=False)
    score.to_csv(args.output_dir / "stage11_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "stage11_paired_bootstrap.csv", index=False)
    if args.prior_stage10_metrics and args.prior_stage10_metrics.exists():
        shutil_frame = pd.read_csv(args.prior_stage10_metrics)
        shutil_frame.to_csv(args.output_dir / "prior_stage10_metrics.csv", index=False)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "11_packse_residual_mil_concept_kan",
        "nissan_locked_test_used": False, "nissan_locked_test_included": False,
        "rwth_and_eve_status": "development_from_stage11_onward",
        "new_external_confirmation_required": True,
        "trip_concepts": list(TRIP_CONCEPT_NAMES),
        "interval_concepts": list(INTERVAL_CONCEPT_NAMES),
        "kan_input_count": len(KAN_CONCEPT_NAMES),
        "max_chronological_trip_tokens": int(bags["tokens"].shape[1]),
        "packse_physical_baseline": True,
        "rf_out_of_group_distillation": True,
        "constrained_inner_fold_ensemble": True,
        "seeds": list(args.seeds), "epochs": args.epochs,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "limitations": [
            "RWTH still has eight eligible health intervals across four vehicles",
            "Nissan and EVE each contain one vehicle",
            "RWTH pack temperature is represented by a labelled ambient proxy",
            "This redesign cannot be called independently externally confirmed on RWTH/EVE",
        ],
    }
    (args.output_dir / "STAGE11_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    inventory = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_inventory.csv":
            inventory.append({
                "relative_path": path.relative_to(args.output_dir).as_posix(),
                "bytes": path.stat().st_size, "sha256": sha256(path),
            })
    pd.DataFrame(inventory).to_csv(args.output_dir / "artifact_inventory.csv", index=False)
    update(f"Stage 11 complete in {audit['elapsed_minutes']:.1f} minutes")
    print(score.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
