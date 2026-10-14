"""Stage 10: vehicle-relative degradation transfer with domain calibration.

This stage deliberately leaves the sealed Nissan test untouched.  It uses the
harmonised *development* intervals produced in Stage 9 and asks a narrow
question: does modelling vehicle-relative degradation, followed by a
dataset-specific measurement calibration, transfer better than modelling raw
SOH percentage-point loss?

The shared KAN/MLP consumes at most twenty named physical concepts.  It emits a
non-negative latent degradation severity.  A domain head maps that latent
severity to the signed measurement-space loss reported by Nissan/EVE BMS SOH
or RWTH capacity-derived SOH.  Hence apparent BMS SOH increases are represented
as measurement calibration/recalibration, never as negative physical damage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from torch import Tensor, nn
from torch.nn import functional as F

from dbpatchkan.models.kan import ConceptKAN, ConceptMLP


DOMAIN_ORDER = ("Nissan-development", "EVE", "RWTH")
DOMAIN_INDEX = {name: index for index, name in enumerate(DOMAIN_ORDER)}
TARGET_NAMES = (
    "relative_interval_loss_pct",
    "relative_loss_per_30d_pct",
    "relative_loss_per_500km_pct",
)
FEATURE_NAMES = (
    "log1p_calendar_days",
    "log1p_distance_km",
    "log1p_efc",
    "log1p_ah_throughput",
    "log1p_charge_event_count",
    "log1p_charge_depth_pct",
    "ambient_mean_c",
    "pack_temp_mean_c",
    "pack_temp_spread_c",
    "hot_exposure_fraction",
    "c_rate_p95",
    "high_c_rate_fraction",
    "log1p_c_rate_squared_dose_h",
    "log1p_electrothermal_dose_h",
    "log1p_transient_dose",
    "dbsi_mean",
    "log1p_stress_dose_h",
    "log1p_pack_damage_p50",
    "pack_damage_relative_90pct_width",
    "anchor_soh_pct",
)
LOG_SOURCE = {
    "log1p_calendar_days": "calendar_days",
    "log1p_distance_km": "distance_km",
    "log1p_efc": "efc",
    "log1p_ah_throughput": "ah_throughput",
    "log1p_charge_event_count": "charge_event_count",
    "log1p_charge_depth_pct": "charge_depth_pct",
    "log1p_c_rate_squared_dose_h": "c_rate_squared_dose_h",
    "log1p_electrothermal_dose_h": "electrothermal_dose_h",
    "log1p_transient_dose": "transient_dose",
    "log1p_stress_dose_h": "stress_dose_h",
    "log1p_pack_damage_p50": "pack_damage_p50",
}
DIRECT_SOURCE = {
    "ambient_mean_c": "ambient_mean_c",
    "pack_temp_mean_c": "pack_temp_mean_c",
    "pack_temp_spread_c": "pack_temp_spread_c",
    "hot_exposure_fraction": "hot_exposure_fraction",
    "c_rate_p95": "c_rate_p95",
    "high_c_rate_fraction": "high_c_rate_fraction",
    "dbsi_mean": "dbsi_mean",
    "pack_damage_relative_90pct_width": "relative_90pct_width",
    "anchor_soh_pct": "anchor_soh_pct",
}


def update(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _parse_interval_indices(value: str) -> tuple[int, int]:
    tail = str(value).rsplit(":", 1)[-1]
    left, right = tail.split("-")
    return int(left), int(right)


def _attach_nissan_anchors(intervals: pd.DataFrame, health_path: Path) -> pd.DataFrame:
    health = pd.read_csv(health_path)
    health["anchor_time_utc"] = pd.to_datetime(
        health["anchor_time_utc"], utc=True, format="mixed"
    )
    health = (
        health.sort_values("anchor_time_utc")
        .drop_duplicates("anchor_time_utc", keep="last")
        .reset_index(drop=True)
    )
    anchors = []
    for value in intervals["interval_id"]:
        start, _ = _parse_interval_indices(value)
        anchors.append(float(health.iloc[start]["soh_target_trend_pct"]))
    result = intervals.copy()
    result["anchor_soh_pct"] = anchors
    result["label_type"] = "BMS-reported SOH trend"
    return result


def _attach_rwth_anchors(intervals: pd.DataFrame, checkpoints_path: Path) -> pd.DataFrame:
    checkpoints = pd.read_csv(checkpoints_path)
    lookup: dict[tuple[str, int], float] = {}
    for vehicle, group in checkpoints.groupby("vehicle_id", sort=False):
        group = group.sort_values("checkpoint_start").reset_index(drop=True)
        for index, row in group.iterrows():
            lookup[(str(vehicle), int(index))] = float(row["capacity_soh_pct"])
    anchors = []
    for row in intervals.itertuples(index=False):
        start, _ = _parse_interval_indices(row.interval_id)
        anchors.append(lookup[(str(row.vehicle_id), start)])
    result = intervals.copy()
    result["anchor_soh_pct"] = anchors
    result["label_type"] = "capacity-derived SOH"
    return result


def _attach_eve_anchors(intervals: pd.DataFrame, health_path: Path) -> pd.DataFrame:
    health = pd.read_csv(health_path).sort_values("checkpoint_time").reset_index(drop=True)
    anchors = []
    for value in intervals["interval_id"]:
        start, _ = _parse_interval_indices(value)
        anchors.append(float(health.iloc[start]["bms_soh_trend_pct"]))
    result = intervals.copy()
    result["anchor_soh_pct"] = anchors
    result["label_type"] = "BMS-reported SOH trend"
    return result


def build_normalized_table(
    nissan_path: Path,
    rwth_path: Path,
    eve_path: Path,
    pack_path: Path,
    nissan_health_path: Path,
    rwth_checkpoints_path: Path,
    eve_health_path: Path,
) -> pd.DataFrame:
    """Join fold-independent measurements and construct reversible targets."""

    nissan = _attach_nissan_anchors(pd.read_csv(nissan_path), nissan_health_path)
    rwth = _attach_rwth_anchors(pd.read_csv(rwth_path), rwth_checkpoints_path)
    eve = _attach_eve_anchors(pd.read_csv(eve_path), eve_health_path)
    table = pd.concat((nissan, eve, rwth), ignore_index=True, sort=False)
    pack = pd.read_csv(pack_path)
    table = table.merge(
        pack[[
            "dataset", "vehicle_id", "interval_id", "pack_damage_p50",
            "relative_90pct_width", "pack_damage_p05", "pack_damage_p95",
        ]],
        on=["dataset", "vehicle_id", "interval_id"], how="left", validate="one_to_one",
    )
    anchor = _numeric(table, "anchor_soh_pct")
    signed = _numeric(table, "signed_loss_pp")
    days = _numeric(table, "calendar_days")
    distance = _numeric(table, "distance_km")
    table["future_soh_pct"] = anchor - signed
    table["vehicle_reference_soh_pct"] = table.groupby(
        ["dataset", "vehicle_id"], sort=False
    )["anchor_soh_pct"].transform("first")
    table["anchor_relative_to_vehicle_start_pct"] = (
        100.0 * anchor / table["vehicle_reference_soh_pct"]
    )
    table["future_relative_to_vehicle_start_pct"] = (
        100.0 * table["future_soh_pct"] / table["vehicle_reference_soh_pct"]
    )
    table[TARGET_NAMES[0]] = 100.0 * signed / anchor
    table[TARGET_NAMES[1]] = table[TARGET_NAMES[0]] * 30.0 / days
    table[TARGET_NAMES[2]] = table[TARGET_NAMES[0]] * 500.0 / distance
    table.loc[~distance.gt(1.0), TARGET_NAMES[2]] = np.nan
    table["physical_relative_loss_pct"] = table[TARGET_NAMES[0]].clip(lower=0.0)
    table["measurement_recalibration_component_pct"] = (
        table[TARGET_NAMES[0]] - table["physical_relative_loss_pct"]
    )
    reconstructed = anchor * (1.0 - table[TARGET_NAMES[0]] / 100.0)
    if not np.allclose(reconstructed, table["future_soh_pct"], equal_nan=True, atol=1e-8):
        raise AssertionError("Relative target is not exactly reversible to SOH")
    if table["anchor_soh_pct"].isna().any():
        raise AssertionError("An anchor SOH could not be recovered")
    table["row_id"] = np.arange(len(table), dtype=int)
    return table


def feature_matrix(table: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=table.index)
    for feature in FEATURE_NAMES:
        if feature in LOG_SOURCE:
            values = _numeric(table, LOG_SOURCE[feature]).clip(lower=0.0)
            result[feature] = np.log1p(values)
        else:
            result[feature] = _numeric(table, DIRECT_SOURCE[feature])
    return result


def expanding_splits(n_rows: int, folds: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_rows < 8:
        return []
    initial = max(6, int(math.floor(n_rows * 0.40)))
    chunks = [x for x in np.array_split(np.arange(initial, n_rows), folds) if len(x)]
    return [(np.arange(0, int(chunk[0])), chunk) for chunk in chunks]


def outer_folds(table: pd.DataFrame) -> list[dict[str, object]]:
    folds: list[dict[str, object]] = []
    for domain in DOMAIN_ORDER:
        indices = np.flatnonzero(table["dataset"].eq(domain).to_numpy())
        local = table.iloc[indices].sort_values("interval_end")
        ordered = local.index.to_numpy(dtype=int)
        if domain == "RWTH":
            vehicles = table.loc[ordered, "vehicle_id"].astype(str)
            for vehicle in sorted(vehicles.unique()):
                test = ordered[vehicles.to_numpy() == vehicle]
                target_train = ordered[vehicles.to_numpy() != vehicle]
                if len(test) and len(target_train) >= 4:
                    folds.append({
                        "dataset": domain,
                        "fold": f"LOOV_{vehicle}",
                        "target_train": target_train,
                        "test": test,
                    })
        else:
            for number, (train_local, test_local) in enumerate(expanding_splits(len(ordered)), 1):
                folds.append({
                    "dataset": domain,
                    "fold": f"time_fold_{number}",
                    "target_train": ordered[train_local],
                    "test": ordered[test_local],
                })
    return folds


def pooled_training_indices(table: pd.DataFrame, fold: dict[str, object]) -> np.ndarray:
    domain = str(fold["dataset"])
    auxiliary = table.index[~table["dataset"].eq(domain)].to_numpy(dtype=int)
    return np.unique(np.concatenate((np.asarray(fold["target_train"], dtype=int), auxiliary)))


def inner_split(table: pd.DataFrame, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic domain-stratified tail split fitted inside the outer fold."""

    train = table.loc[train_indices]
    validation: list[int] = []
    for _, group in train.groupby("dataset", sort=False):
        ordered = group.sort_values("interval_end").index.to_numpy(dtype=int)
        count = max(1, int(math.ceil(0.20 * len(ordered)))) if len(ordered) >= 5 else 1
        validation.extend(ordered[-count:].tolist())
    validation_idx = np.asarray(sorted(set(validation)), dtype=int)
    subtrain = np.setdiff1d(train_indices, validation_idx)
    if len(subtrain) < 10:
        raise ValueError("Inner split left too few training rows")
    return subtrain, validation_idx


class FoldTransform:
    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))

    def fit(self, values: pd.DataFrame) -> "FoldTransform":
        imputed = self.imputer.fit_transform(values)
        self.scaler.fit(imputed)
        return self

    def transform(self, values: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(self.imputer.transform(values)).astype(np.float32)


class DomainCalibratedConceptModel(nn.Module):
    """Shared non-negative degradation severity plus dataset measurement heads."""

    def __init__(self, kind: str, n_features: int, n_targets: int = 3) -> None:
        super().__init__()
        if kind == "KAN":
            self.backbone: nn.Module = ConceptKAN(
                (n_features, 16, 8, n_targets), grid_size=5, spline_order=3
            )
        elif kind == "MLP":
            self.backbone = ConceptMLP((n_features, 32, 16, n_targets), dropout=0.10)
        else:
            raise ValueError(kind)
        self.kind = kind
        self.domain_log_scale = nn.Parameter(torch.zeros(len(DOMAIN_ORDER), n_targets))
        self.domain_bias = nn.Parameter(torch.zeros(len(DOMAIN_ORDER), n_targets))

    def forward(self, features: Tensor, domain: Tensor) -> tuple[Tensor, Tensor]:
        latent = F.softplus(self.backbone(features.float()))
        scale = F.softplus(self.domain_log_scale[domain])
        observed = self.domain_bias[domain] + scale * latent
        return observed, latent

    def regularization_loss(self) -> Tensor:
        base = self.backbone.regularization_loss()
        return base + 0.05 * self.domain_bias.square().mean()


def _domain_target_scales(table: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    result = np.ones((len(DOMAIN_ORDER), len(TARGET_NAMES)), dtype=np.float32)
    for domain, domain_id in DOMAIN_INDEX.items():
        selected = table.loc[indices]
        selected = selected[selected["dataset"].eq(domain)]
        for target_id, target in enumerate(TARGET_NAMES):
            values = _numeric(selected, target).dropna().to_numpy(dtype=float)
            if len(values):
                q25, q75 = np.percentile(values, [25, 75])
                robust = (q75 - q25) / 1.349
                result[domain_id, target_id] = max(float(robust), float(np.std(values)), 0.05)
    return result


def train_model(
    table: pd.DataFrame,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    *,
    kind: str,
    device: torch.device,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    seed_everything(seed)
    matrix = feature_matrix(table)
    transform = FoldTransform().fit(matrix.loc[train_indices])
    x_train = torch.from_numpy(transform.transform(matrix.loc[train_indices])).to(device)
    x_predict = torch.from_numpy(transform.transform(matrix.loc[predict_indices])).to(device)
    domains_train = torch.tensor(
        [DOMAIN_INDEX[x] for x in table.loc[train_indices, "dataset"]],
        dtype=torch.long, device=device,
    )
    domains_predict = torch.tensor(
        [DOMAIN_INDEX[x] for x in table.loc[predict_indices, "dataset"]],
        dtype=torch.long, device=device,
    )
    y_array = table.loc[train_indices, TARGET_NAMES].to_numpy(dtype=np.float32)
    mask_array = np.isfinite(y_array)
    y = torch.from_numpy(np.nan_to_num(y_array, nan=0.0)).to(device)
    mask = torch.from_numpy(mask_array).to(device)
    target_scales = torch.from_numpy(_domain_target_scales(table, train_indices)).to(device)

    model = DomainCalibratedConceptModel(kind, len(FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.02)
    best, best_state, patience = math.inf, None, 80
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(x_train, domains_train)
        scale = target_scales[domains_train]
        residual = (prediction - y) / scale
        supervised = F.huber_loss(residual[mask], torch.zeros_like(residual[mask]), delta=1.0)
        regularization = (5e-4 if kind == "KAN" else 1e-5) * model.regularization_loss()
        loss = supervised + regularization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        current = float(loss.detach().cpu())
        history.append({"epoch": epoch, "loss": current})
        if current < best - 1e-7:
            best = current
            best_state = copy.deepcopy(model.state_dict())
            patience = 80
        else:
            patience -= 1
            if patience == 0:
                break
    if best_state is None:
        raise RuntimeError("No model state was produced")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction, latent = model(x_predict, domains_predict)
    state: dict[str, object] = {
        "kind": kind,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "domain_order": DOMAIN_ORDER,
        "state_dict": best_state,
        "imputer_statistics": transform.imputer.statistics_,
        "scaler_center": transform.scaler.center_,
        "scaler_scale": transform.scaler.scale_,
        "domain_scale": F.softplus(model.domain_log_scale).detach().cpu().numpy(),
        "domain_bias": model.domain_bias.detach().cpu().numpy(),
        "best_training_loss": best,
        "history": history,
    }
    if kind == "KAN":
        state["edge_strengths"] = [x.numpy() for x in model.backbone.edge_strengths()]
    return prediction.cpu().numpy(), latent.cpu().numpy(), state


def choose_gate(
    truth_pp: np.ndarray,
    model_loss_pp: np.ndarray,
) -> tuple[float, float, float]:
    valid = np.isfinite(truth_pp) & np.isfinite(model_loss_pp)
    if not valid.any():
        return 0.0, math.nan, math.nan
    candidates = np.linspace(0.0, 1.0, 21)
    errors = [np.mean(np.abs(truth_pp[valid] - alpha * model_loss_pp[valid])) for alpha in candidates]
    best = int(np.argmin(errors))
    persistence = float(np.mean(np.abs(truth_pp[valid])))
    # Two-percent safety margin: weak inner improvements fall back to persistence.
    alpha = float(candidates[best]) if errors[best] < 0.98 * persistence else 0.0
    gated_error = float(np.mean(np.abs(truth_pp[valid] - alpha * model_loss_pp[valid])))
    return alpha, gated_error, persistence


def _baseline_predictions(
    table: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    x = feature_matrix(table)
    train = table.loc[train_idx]
    test = table.loc[test_idx]
    y = _numeric(train, TARGET_NAMES[0]).to_numpy(dtype=float)
    anchor = _numeric(test, "anchor_soh_pct").to_numpy(dtype=float)
    output: dict[str, np.ndarray] = {"Persistence": np.zeros(len(test))}
    target_domain = str(test["dataset"].iloc[0])
    same = train["dataset"].eq(target_domain).to_numpy()
    output["DomainMedian"] = np.repeat(float(np.nanmedian(y[same] if same.any() else y)), len(test))
    for name, estimator in (
        ("Ridge", Ridge(alpha=10.0)),
        ("RandomForest", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.7, random_state=42, n_jobs=-1
        )),
        ("ExtraTrees", ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1
        )),
    ):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", RobustScaler(quantile_range=(10.0, 90.0))) if name == "Ridge" else ("identity", "passthrough"),
            ("model", estimator),
        ])
        pipe.fit(x.loc[train_idx], y)
        output[name] = pipe.predict(x.loc[test_idx])
    # D2B-style comparator: a non-negative affine mapping of pack physical damage.
    pack_train = np.log1p(_numeric(train, "pack_damage_p50").clip(lower=0.0)).to_numpy().reshape(-1, 1)
    pack_test = np.log1p(_numeric(test, "pack_damage_p50").clip(lower=0.0)).to_numpy().reshape(-1, 1)
    imputer = SimpleImputer(strategy="median").fit(pack_train)
    positive = LinearRegression(positive=True).fit(imputer.transform(pack_train), y)
    output["PackSE_NMC_Affine"] = positive.predict(imputer.transform(pack_test))
    # Convert relative loss to absolute percentage-point loss later.
    for key in output:
        output[key] = anchor * np.asarray(output[key], dtype=float) / 100.0
    return output


def _finite_sample_quantile(values: np.ndarray, coverage: float = 0.90) -> float:
    values = np.sort(values[np.isfinite(values)])
    if not len(values):
        return math.nan
    rank = min(int(math.ceil((len(values) + 1) * coverage)) - 1, len(values) - 1)
    return float(values[max(rank, 0)])


def run_experiment(
    table: pd.DataFrame,
    *,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    folds = outer_folds(table)
    update(f"Stage 10 has {len(folds)} outer folds and {len(seeds)} seeds")
    for fold_number, fold in enumerate(folds, 1):
        dataset, fold_name = str(fold["dataset"]), str(fold["fold"])
        test_idx = np.asarray(fold["test"], dtype=int)
        train_idx = pooled_training_indices(table, fold)
        if np.intersect1d(train_idx, test_idx).size:
            raise AssertionError("Outer test rows leaked into training")
        update(
            f"outer {fold_number}/{len(folds)}: {dataset} {fold_name}; "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )
        baseline = _baseline_predictions(table, train_idx, test_idx)
        for model_name, loss_pp in baseline.items():
            for local, row_index in enumerate(test_idx):
                row = table.loc[row_index]
                predictions.append(_prediction_row(
                    row, fold_name, model_name, 0, float(loss_pp[local]),
                    gate_alpha=0.0 if model_name == "Persistence" else 1.0,
                    conformal_radius=math.nan,
                ))
        subtrain_idx, inner_idx = inner_split(table, train_idx)
        for kind in ("KAN", "MLP"):
            for seed in seeds:
                update(f"{dataset} {fold_name}: {kind} seed={seed} inner gate")
                inner_pred_rel, _, _ = train_model(
                    table, subtrain_idx, inner_idx, kind=kind,
                    device=device, epochs=max(20, epochs // 2), seed=seed,
                )
                inner_anchor = _numeric(table.loc[inner_idx], "anchor_soh_pct").to_numpy(dtype=float)
                inner_pred_pp = inner_anchor * inner_pred_rel[:, 0] / 100.0
                inner_truth_pp = _numeric(table.loc[inner_idx], "signed_loss_pp").to_numpy(dtype=float)
                target_inner = table.loc[inner_idx, "dataset"].eq(dataset).to_numpy()
                if target_inner.any():
                    gate_truth, gate_prediction = inner_truth_pp[target_inner], inner_pred_pp[target_inner]
                else:
                    gate_truth, gate_prediction = inner_truth_pp, inner_pred_pp
                alpha, inner_mae, persistence_mae = choose_gate(gate_truth, gate_prediction)
                residual = np.abs(gate_truth - alpha * gate_prediction)
                conformal_radius = _finite_sample_quantile(residual, 0.90)
                gate_rows.append({
                    "dataset": dataset, "outer_fold": fold_name, "head": kind,
                    "seed": seed, "alpha": alpha, "inner_mae_pp": inner_mae,
                    "inner_persistence_mae_pp": persistence_mae,
                    "conformal_radius_pp": conformal_radius,
                    "inner_rows": int(len(gate_truth)),
                })
                outer_rel, latent, state = train_model(
                    table, train_idx, test_idx, kind=kind,
                    device=device, epochs=epochs, seed=seed,
                )
                update(
                    f"{dataset} {fold_name}: {kind} seed={seed} "
                    f"alpha={alpha:.2f}, conformal90={conformal_radius:.4f} pp"
                )
                fold_dir = output_dir / "models" / dataset / fold_name / kind / f"seed_{seed}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                torch.save(state, fold_dir / "model.pt")
                pd.DataFrame(state["history"]).to_csv(fold_dir / "history.csv", index=False)
                for domain_id, domain_name in enumerate(DOMAIN_ORDER):
                    for target_id, target_name in enumerate(TARGET_NAMES):
                        calibration_rows.append({
                            "outer_dataset": dataset, "outer_fold": fold_name,
                            "head": kind, "seed": seed, "calibration_domain": domain_name,
                            "target": target_name,
                            "positive_scale": float(state["domain_scale"][domain_id, target_id]),
                            "signed_bias": float(state["domain_bias"][domain_id, target_id]),
                        })
                anchor = _numeric(table.loc[test_idx], "anchor_soh_pct").to_numpy(dtype=float)
                raw_pp = anchor * outer_rel[:, 0] / 100.0
                for local, row_index in enumerate(test_idx):
                    row = table.loc[row_index]
                    predictions.append(_prediction_row(
                        row, fold_name, f"Normalized-{kind}", seed,
                        float(raw_pp[local]), gate_alpha=1.0,
                        conformal_radius=conformal_radius,
                        latent=float(latent[local, 0]),
                    ))
                    predictions.append(_prediction_row(
                        row, fold_name, f"PersistenceAnchored-Normalized-{kind}", seed,
                        float(alpha * raw_pp[local]), gate_alpha=alpha,
                        conformal_radius=conformal_radius,
                        latent=float(latent[local, 0]),
                    ))
    raw = pd.DataFrame(predictions)
    ensemble = ensemble_predictions(raw)
    return raw, ensemble, pd.DataFrame(gate_rows), pd.DataFrame(calibration_rows)


def _prediction_row(
    row: pd.Series,
    fold: str,
    model: str,
    seed: int,
    predicted_loss_pp: float,
    *,
    gate_alpha: float,
    conformal_radius: float,
    latent: float = math.nan,
) -> dict[str, object]:
    anchor = float(row["anchor_soh_pct"])
    true_loss = float(row["signed_loss_pp"])
    predicted_future = anchor - predicted_loss_pp
    return {
        "dataset": row["dataset"], "label_type": row["label_type"],
        "vehicle_id": row["vehicle_id"], "interval_id": row["interval_id"],
        "outer_fold": fold, "model": model, "seed": seed,
        "anchor_soh_pct": anchor, "true_loss_pp": true_loss,
        "predicted_loss_pp": predicted_loss_pp,
        "true_future_soh_pct": anchor - true_loss,
        "predicted_future_soh_pct": predicted_future,
        "relative_true_loss_pct": float(row[TARGET_NAMES[0]]),
        "relative_predicted_loss_pct": 100.0 * predicted_loss_pp / anchor,
        "gate_alpha": gate_alpha, "latent_physical_severity": latent,
        "interval_lower_soh_pct": predicted_future - conformal_radius,
        "interval_upper_soh_pct": predicted_future + conformal_radius,
    }


def ensemble_predictions(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "label_type", "vehicle_id", "interval_id", "outer_fold", "model"]
    rows: list[dict[str, object]] = []
    for key, group in raw.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key))
        for column in (
            "anchor_soh_pct", "true_loss_pp", "true_future_soh_pct", "relative_true_loss_pct"
        ):
            row[column] = float(group[column].iloc[0])
        for column in (
            "predicted_loss_pp", "predicted_future_soh_pct", "relative_predicted_loss_pct",
            "gate_alpha", "latent_physical_severity",
        ):
            row[column] = float(group[column].median())
        row["interval_lower_soh_pct"] = float(group["interval_lower_soh_pct"].median())
        row["interval_upper_soh_pct"] = float(group["interval_upper_soh_pct"].median())
        row["seeds"] = int(group["seed"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in predictions.groupby(["dataset", "model"], sort=False):
        truth = group["true_future_soh_pct"].to_numpy(dtype=float)
        pred = group["predicted_future_soh_pct"].to_numpy(dtype=float)
        true_loss = group["true_loss_pp"].to_numpy(dtype=float)
        predicted_loss = group["predicted_loss_pp"].to_numpy(dtype=float)
        covered = (
            group["interval_lower_soh_pct"].le(group["true_future_soh_pct"])
            & group["interval_upper_soh_pct"].ge(group["true_future_soh_pct"])
        )
        interval_valid = (
            group["interval_lower_soh_pct"].notna()
            & group["interval_upper_soh_pct"].notna()
        )
        rows.append({
            "dataset": dataset, "model": model, "n": len(group),
            "mae_pp": mean_absolute_error(truth, pred),
            "rmse_pp": math.sqrt(mean_squared_error(truth, pred)),
            "bias_pp": float(np.mean(pred - truth)),
            "loss_r2": r2_score(true_loss, predicted_loss) if len(group) > 1 else math.nan,
            "future_soh_r2": r2_score(truth, pred) if len(group) > 1 else math.nan,
            "interval_90_coverage": float(covered[interval_valid].mean())
            if interval_valid.any() else math.nan,
            "mean_gate_alpha": float(group["gate_alpha"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "mae_pp"])


def paired_bootstrap(
    predictions: pd.DataFrame,
    *,
    reference: str = "Persistence",
    iterations: int = 10_000,
    seed: int = 20260809,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for dataset, group in predictions.groupby("dataset"):
        wide = group.pivot(index=["vehicle_id", "interval_id"], columns="model", values="predicted_future_soh_pct")
        truth = group.drop_duplicates(["vehicle_id", "interval_id"]).set_index(
            ["vehicle_id", "interval_id"]
        )["true_future_soh_pct"].reindex(wide.index)
        vehicles = wide.index.get_level_values(0).to_numpy(dtype=str)
        for challenger in (
            "PersistenceAnchored-Normalized-KAN",
            "PersistenceAnchored-Normalized-MLP",
            "Normalized-KAN", "Normalized-MLP",
        ):
            if reference not in wide or challenger not in wide:
                continue
            valid = truth.notna() & wide[reference].notna() & wide[challenger].notna()
            y = truth[valid].to_numpy(dtype=float)
            ref = wide.loc[valid, reference].to_numpy(dtype=float)
            chal = wide.loc[valid, challenger].to_numpy(dtype=float)
            v = vehicles[np.flatnonzero(valid.to_numpy())]
            delta = np.abs(y - chal) - np.abs(y - ref)
            samples = np.empty(iterations, dtype=float)
            unique = np.unique(v)
            if len(unique) > 1:
                clusters = {name: np.flatnonzero(v == name) for name in unique}
                for i in range(iterations):
                    chosen = rng.choice(unique, size=len(unique), replace=True)
                    indices = np.concatenate([clusters[name] for name in chosen])
                    samples[i] = float(np.mean(delta[indices]))
            else:
                block = min(3, len(delta))
                for i in range(iterations):
                    pieces = []
                    while sum(len(x) for x in pieces) < len(delta):
                        start = int(rng.integers(0, max(len(delta) - block + 1, 1)))
                        pieces.append(np.arange(start, min(start + block, len(delta))))
                    samples[i] = float(np.mean(delta[np.concatenate(pieces)[: len(delta)]]))
            rows.append({
                "dataset": dataset, "reference": reference, "challenger": challenger,
                "n": len(delta), "delta_mae_pp": float(np.mean(delta)),
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
                "probability_challenger_better": float(np.mean(samples < 0.0)),
                "significant_win": bool(np.quantile(samples, 0.975) < 0.0),
            })
    return pd.DataFrame(rows)


def normalization_comparison(metrics: pd.DataFrame, prior_path: Path | None) -> pd.DataFrame:
    if prior_path is None or not prior_path.exists():
        return pd.DataFrame()
    prior = pd.read_csv(prior_path)
    rows = []
    mapping = {
        "KAN": "PersistenceGated-DBPatchTransformer-KAN",
        "MLP": "PersistenceGated-DBPatchTransformer-MLP",
    }
    for dataset in ("RWTH", "EVE"):
        for head, prior_model in mapping.items():
            old = prior[(prior["dataset"] == dataset) & (prior["model"] == prior_model)]
            new = metrics[
                (metrics["dataset"] == dataset)
                & (metrics["model"] == f"PersistenceAnchored-Normalized-{head}")
            ]
            if old.empty or new.empty:
                continue
            rows.append({
                "dataset": dataset, "head": head,
                "stage7_raw_loss_mae_pp": float(old.iloc[0]["mae_pp"]),
                "stage7_n": int(old.iloc[0]["n"]),
                "stage10_normalized_loss_mae_pp": float(new.iloc[0]["mae_pp"]),
                "stage10_n": int(new.iloc[0]["n"]),
                "mae_change_pp": float(new.iloc[0]["mae_pp"] - old.iloc[0]["mae_pp"]),
                "paired_comparison": False,
                "note": "Descriptive only because eligible rows/folds may differ between stages",
            })
    return pd.DataFrame(rows)


def write_report(
    output: Path,
    table: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    def plain_table(frame: pd.DataFrame) -> str:
        return "```text\n" + frame.to_string(index=False) + "\n```"

    counts = table.groupby(["dataset", "label_type"]).size().rename("intervals").reset_index()
    lines = [
        "# Stage 10 — Vehicle-relative degradation transfer",
        "",
        "The Nissan locked test was not included or accessed. Stage 10 is a development/external-transfer experiment.",
        "",
        "## Label harmonisation",
        "",
        "The shared target is `100 × (anchor SOH − future SOH) / anchor SOH`. "
        "Nissan/EVE retain BMS-specific calibration heads; RWTH retains a capacity-specific head.",
        "",
        "## Dataset counts",
        "",
        plain_table(counts),
        "",
        "## Held-out metrics",
        "",
        plain_table(metrics),
        "",
        "## Paired comparison with persistence",
        "",
        plain_table(bootstrap),
        "",
        "## Interpretation rule",
        "",
        "The proposed normalized KAN wins a dataset only when its held-out MAE is lower than persistence "
        "and the paired 95% bootstrap confidence interval is entirely below zero. Normalization is not "
        "allowed to erase the distinction between BMS-reported and capacity-derived SOH.",
    ]
    (output / "STAGE10_NORMALIZED_TRANSFER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nissan-intervals", type=Path, required=True)
    parser.add_argument("--rwth-intervals", type=Path, required=True)
    parser.add_argument("--eve-intervals", type=Path, required=True)
    parser.add_argument("--pack-uncertainty", type=Path, required=True)
    parser.add_argument("--nissan-health", type=Path, required=True)
    parser.add_argument("--rwth-checkpoints", type=Path, required=True)
    parser.add_argument("--eve-health", type=Path, required=True)
    parser.add_argument("--prior-stage7-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62, 72, 82])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    update(f"Stage 10 starts on {device}; KAN splines remain float32")
    table = build_normalized_table(
        args.nissan_intervals, args.rwth_intervals, args.eve_intervals,
        args.pack_uncertainty, args.nissan_health, args.rwth_checkpoints,
        args.eve_health,
    )
    table.to_csv(args.output_dir / "normalized_cross_dataset_intervals.csv", index=False)
    raw, ensemble, gates, calibration = run_experiment(
        table, output_dir=args.output_dir, device=device,
        epochs=args.epochs, seeds=args.seeds,
    )
    metrics = metric_table(ensemble)
    bootstrap = paired_bootstrap(ensemble)
    comparison = normalization_comparison(metrics, args.prior_stage7_metrics)
    raw.to_csv(args.output_dir / "all_seed_predictions.csv", index=False)
    ensemble.to_csv(args.output_dir / "ensemble_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "normalized_transfer_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_persistence_bootstrap.csv", index=False)
    gates.to_csv(args.output_dir / "inner_persistence_gate_selection.csv", index=False)
    calibration.to_csv(args.output_dir / "domain_calibration_parameters.csv", index=False)
    comparison.to_csv(args.output_dir / "stage7_vs_stage10_descriptive.csv", index=False)
    write_report(args.output_dir, table, metrics, bootstrap)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "10_vehicle_relative_normalized_degradation_transfer",
        "nissan_locked_test_used": False,
        "nissan_locked_test_included": False,
        "label_definitions_kept_separate": True,
        "shared_target": "100*(anchor_soh-future_soh)/anchor_soh",
        "shared_nonnegative_latent_physical_degradation": True,
        "domain_measurement_heads": list(DOMAIN_ORDER),
        "fold_safe_scaling_imputation_calibration_and_gate": True,
        "feature_names": list(FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "seeds": list(args.seeds),
        "interval_counts": table.groupby("dataset").size().astype(int).to_dict(),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "limitations": [
            "RWTH health labels remain extremely sparse",
            "Nissan and EVE each contain one vehicle",
            "Stage 10 reuses harmonised named concepts and does not retrain temporal encoders",
            "Stage 7 versus Stage 10 comparison is descriptive when eligible folds differ",
            "Normalization cannot make BMS-reported and capacity-derived SOH identical",
        ],
    }
    (args.output_dir / "STAGE10_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    inventory = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_inventory.csv":
            inventory.append({
                "relative_path": path.relative_to(args.output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    pd.DataFrame(inventory).to_csv(args.output_dir / "artifact_inventory.csv", index=False)
    update(f"Stage 10 complete in {audit['elapsed_minutes']:.1f} minutes")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
