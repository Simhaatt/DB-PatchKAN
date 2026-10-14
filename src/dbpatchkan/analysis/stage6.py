"""Leakage-safe Nissan development baselines and missing ablations.

Stage 6 deliberately operates only on the 101 development trips exported as
``segments_train.npz`` and ``segments_validation.npz``.  It never discovers,
opens, or accepts the locked 26-trip test file.  Completed Stage 1--5
experiments are registered as prior evidence and are not re-run here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TARGETS = ("current_soh", "loss_30d", "loss_500km")
TARGET_KEYS = {
    "current_soh": "current_soh_target",
    "loss_30d": "loss_30d_target",
    "loss_500km": "loss_500km_target",
}
DBSI_WEIGHTS = np.asarray((0.25, 0.25, 0.15, 0.15, 0.10, 0.10), dtype=np.float32)

COMPLETED_PRIOR_EVIDENCE = (
    "five_seed_KAN_vs_parameter_matched_MLP_selection",
    "Stage3_KAN_explainability",
    "corrected_Nissan_physics_vs_analytic_vs_persistence",
    "RWTH_and_EVE_external_driver_to_stress_validation",
    "RWTH_log_robust_normalization_sensitivity",
)


def update(message: str) -> None:
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{stamp} UTC] {message}", flush=True)


def _decode_utf8_rows(encoded: np.ndarray) -> np.ndarray:
    return np.asarray(
        [bytes(row[row != 0].tolist()).decode("utf-8") for row in encoded.astype(np.uint8)],
        dtype=object,
    )


def _encode_utf8_rows(values: Iterable[str]) -> np.ndarray:
    encoded = [str(value).encode("utf-8") for value in values]
    width = max(1, max(map(len, encoded), default=1))
    output = np.zeros((len(encoded), width), dtype=np.uint8)
    for index, value in enumerate(encoded):
        output[index, : len(value)] = np.frombuffer(value, dtype=np.uint8)
    return output


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def merge_development_npz(train_path: Path, validation_path: Path, destination: Path) -> Path:
    """Merge the two already-approved development splits without touching test data."""

    left = _load_npz(train_path)
    right = _load_npz(validation_path)
    if set(left) != set(right):
        raise ValueError("Development NPZ files do not have identical schemas")
    merged: dict[str, np.ndarray] = {}
    for name in left:
        if left[name].ndim == 0:
            if not np.array_equal(left[name], right[name], equal_nan=True):
                raise ValueError(f"Scalar metadata mismatch for {name}")
            merged[name] = left[name]
        else:
            merged[name] = np.concatenate((left[name], right[name]), axis=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **merged)
    return destination


def trip_order(npz_path: Path) -> tuple[list[str], np.ndarray]:
    arrays = _load_npz(npz_path)
    keys = _decode_utf8_rows(arrays["trip_key_utf8"])
    starts = np.asarray(arrays["window_start_unix_ms"], dtype=np.int64)
    ordered = sorted(set(keys.tolist()), key=lambda key: (int(starts[keys == key].min()), key))
    return ordered, keys


def write_trip_subset(source: Path, destination: Path, selected_trips: Iterable[str]) -> Path:
    arrays = _load_npz(source)
    keys = _decode_utf8_rows(arrays["trip_key_utf8"])
    selected = set(selected_trips)
    row_mask = np.asarray([key in selected for key in keys], dtype=bool)
    if not row_mask.any():
        raise ValueError("Trip subset is empty")
    output = {
        name: (value[row_mask] if value.ndim and len(value) == len(row_mask) else value)
        for name, value in arrays.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **output)
    return destination


def chronological_folds(ordered_trips: list[str]) -> list[dict[str, Any]]:
    """Four expanding folds: 45/14, 59/14, 73/14, and 87/14 trips."""

    if len(ordered_trips) != 101:
        raise ValueError(f"Expected 101 development trips, found {len(ordered_trips)}")
    folds: list[dict[str, Any]] = []
    for fold, train_end in enumerate((45, 59, 73, 87), start=1):
        outer_train = ordered_trips[:train_end]
        outer_test = ordered_trips[train_end : train_end + 14]
        inner_size = max(8, int(round(len(outer_train) * 0.20)))
        folds.append(
            {
                "fold": fold,
                "outer_train": outer_train,
                "inner_train": outer_train[:-inner_size],
                "inner_validation": outer_train[-inner_size:],
                "outer_test": outer_test,
                "purge_note": "trip-grouped chronological; forecast labels are precomputed past/future-safe",
            }
        )
    return folds


def prepare_stage6(train: Path, validation: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    merged = merge_development_npz(train, validation, output / "development_101_trips.npz")
    ordered, _ = trip_order(merged)
    folds = chronological_folds(ordered)
    fold_root = output / "fold_data"
    registry: list[dict[str, Any]] = []
    for spec in folds:
        folder = fold_root / f"fold_{spec['fold']}"
        paths = {}
        for role in ("inner_train", "inner_validation", "outer_test"):
            path = write_trip_subset(merged, folder / f"{role}.npz", spec[role])
            paths[role] = str(path.resolve())
        registry.append(
            {
                "fold": spec["fold"],
                "n_outer_train": len(spec["outer_train"]),
                "n_inner_train": len(spec["inner_train"]),
                "n_inner_validation": len(spec["inner_validation"]),
                "n_outer_test": len(spec["outer_test"]),
                "paths": paths,
                "outer_test_trip_keys": spec["outer_test"],
                "purge_note": spec["purge_note"],
            }
        )
    manifest = {
        "stage": 6,
        "development_trips": len(ordered),
        "locked_test_used": False,
        "completed_prior_evidence_not_repeated": list(COMPLETED_PRIOR_EVIDENCE),
        "folds": registry,
    }
    (output / "stage6_fold_manifest.json").write_text(json.dumps(manifest, indent=2))
    update("Prepared four expanding chronological folds; locked test remains absent")
    return manifest


def _trip_rows(npz_path: Path) -> pd.DataFrame:
    arrays = _load_npz(npz_path)
    keys = _decode_utf8_rows(arrays["trip_key_utf8"])
    starts = arrays["window_start_unix_ms"].astype(np.int64)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(keys.tolist()), key=lambda value: (int(starts[keys == value].min()), value)):
        idx = np.flatnonzero(keys == key)
        driver = np.asarray(arrays["driver"][idx], dtype=np.float32)
        battery = np.asarray(arrays["battery"][idx], dtype=np.float32)
        driver_mask = np.asarray(arrays["driver_mask"][idx], dtype=bool)
        battery_mask = np.asarray(arrays["battery_mask"][idx], dtype=bool)
        driver = np.where(driver_mask, driver, np.nan)
        battery = np.where(battery_mask, battery, np.nan)
        concept = np.asarray(arrays["concept_targets"][idx[0]], dtype=np.float32)
        row: dict[str, Any] = {
            "trip_key": key,
            "start_unix_ms": int(starts[idx].min()),
            "segments": len(idx),
        }
        for stream_name, values in (("driver", driver), ("battery", battery)):
            for channel in range(values.shape[-1]):
                flat = values[..., channel].reshape(-1)
                finite = flat[np.isfinite(flat)]
                for stat_name, stat in (
                    ("mean", np.mean), ("std", np.std), ("p05", lambda x: np.quantile(x, 0.05)),
                    ("p95", lambda x: np.quantile(x, 0.95)), ("max", np.max),
                ):
                    row[f"{stream_name}_{channel:02d}_{stat_name}"] = (
                        float(stat(finite)) if finite.size else np.nan
                    )
        for column, value in enumerate(concept):
            row[f"concept_{column:02d}"] = float(value)
        for target, key_name in TARGET_KEYS.items():
            values = np.asarray(arrays[key_name][idx], dtype=np.float32)
            finite = values[np.isfinite(values)]
            row[target] = float(finite[0]) if finite.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid].astype(float)
    predicted = predicted[valid].astype(float)
    if not len(observed):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    error = predicted - observed
    denominator = np.square(observed - observed.mean()).sum()
    return {
        "n": int(len(observed)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "r2": float(1 - np.square(error).sum() / denominator) if denominator > 0 else np.nan,
    }


def run_classical(manifest_path: Path, output: Path, seed: int = 42) -> None:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    manifest = json.loads(manifest_path.read_text())
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold_spec in manifest["folds"]:
        fold = int(fold_spec["fold"])
        train = _trip_rows(Path(fold_spec["paths"]["inner_train"]))
        validation = _trip_rows(Path(fold_spec["paths"]["inner_validation"]))
        outer = _trip_rows(Path(fold_spec["paths"]["outer_test"]))
        outer_train = pd.concat((train, validation), ignore_index=True)
        feature_columns = [
            name for name in outer_train
            if name.startswith(("driver_", "battery_", "concept_")) and name != "concept_12"
        ]
        aging_columns = ["concept_09", "concept_10"]
        for target in TARGETS:
            train_valid = np.isfinite(outer_train[target])
            test_valid = np.isfinite(outer[target])
            if train_valid.sum() < 8 or test_valid.sum() < 2:
                rows.append({"family": "classical", "fold": fold, "target": target,
                             "model": "ALL", "status": "insufficient_labels",
                             "n_train": int(train_valid.sum()), "n_test": int(test_valid.sum())})
                continue
            y_train = outer_train.loc[train_valid, target].to_numpy(float)
            y_test = outer.loc[test_valid, target].to_numpy(float)
            X_train = outer_train.loc[train_valid, feature_columns]
            X_test = outer.loc[test_valid, feature_columns]
            X_age_train = outer_train.loc[train_valid, aging_columns]
            X_age_test = outer.loc[test_valid, aging_columns]
            models: dict[str, tuple[Any, Any, Any]] = {
                "Aging_Ridge": (
                    Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()),
                              ("model", Ridge(alpha=1.0))]), X_age_train, X_age_test),
                "ElasticNet": (
                    Pipeline([("impute", SimpleImputer()), ("scale", RobustScaler()),
                              ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000,
                                                   random_state=seed))]), X_train, X_test),
                "RBF_SVR": (
                    Pipeline([("impute", SimpleImputer()), ("scale", RobustScaler()),
                              ("model", SVR(C=10.0, epsilon=0.05, gamma="scale"))]), X_train, X_test),
                "GPR": (
                    Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()),
                              ("model", GaussianProcessRegressor(
                                  kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
                                  normalize_y=True, random_state=seed, n_restarts_optimizer=1))]),
                    X_train, X_test),
                "RandomForest": (
                    Pipeline([("impute", SimpleImputer()), ("model", RandomForestRegressor(
                        n_estimators=400, min_samples_leaf=2, max_features=0.7,
                        random_state=seed, n_jobs=-1))]), X_train, X_test),
                "XGBoost": (
                    Pipeline([("impute", SimpleImputer()), ("model", XGBRegressor(
                        n_estimators=350, max_depth=3, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                        objective="reg:squarederror", random_state=seed, n_jobs=-1))]),
                    X_train, X_test),
            }
            try:
                from interpret.glassbox import ExplainableBoostingRegressor
                models["EBM"] = (
                    Pipeline([("impute", SimpleImputer()), ("model", ExplainableBoostingRegressor(
                        interactions=0, max_bins=32, max_rounds=5000,
                        learning_rate=0.03, random_state=seed))]), X_train, X_test)
            except Exception:
                pass
            baseline = float(y_train[-1]) if target == "current_soh" else 0.0
            baseline_pred = np.full_like(y_test, baseline)
            result = _metrics(y_test, baseline_pred)
            rows.append({"family": "classical", "fold": fold, "target": target,
                         "model": "Persistence", "status": "complete",
                         "n_train": len(y_train), "n_test": len(y_test), **result})
            for trip, observed, predicted in zip(outer.loc[test_valid, "trip_key"], y_test,
                                                  baseline_pred, strict=True):
                predictions.append({"family": "classical", "fold": fold, "target": target,
                                    "model": "Persistence", "trip_key": trip,
                                    "observed": observed, "predicted": predicted})
            for model_name, (model, fit_x, predict_x) in models.items():
                started = time.perf_counter()
                try:
                    model.fit(fit_x, y_train)
                    pred = np.asarray(model.predict(predict_x), dtype=float)
                    status = "complete"
                    error = ""
                except Exception as exc:
                    pred = np.full_like(y_test, np.nan)
                    status = "failed"
                    error = repr(exc)
                result = _metrics(y_test, pred)
                rows.append({"family": "classical", "fold": fold, "target": target,
                             "model": model_name, "status": status, "error": error,
                             "n_train": len(y_train), "n_test": len(y_test),
                             "elapsed_seconds": time.perf_counter() - started, **result})
                for trip, observed, predicted in zip(outer.loc[test_valid, "trip_key"], y_test,
                                                      pred, strict=True):
                    predictions.append({"family": "classical", "fold": fold, "target": target,
                                        "model": model_name, "trip_key": trip,
                                        "observed": observed, "predicted": predicted})
        update(f"Classical baselines complete for fold {fold}/4")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "classical_fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "classical_predictions.csv", index=False)


def _segment_summary(npz_path: Path, max_steps: int = 64) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    arrays = _load_npz(npz_path)
    keys = _decode_utf8_rows(arrays["trip_key_utf8"])
    starts = arrays["window_start_unix_ms"].astype(np.int64)
    ordered = sorted(set(keys.tolist()), key=lambda key: (int(starts[keys == key].min()), key))
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for key in ordered:
        idx = np.flatnonzero(keys == key)
        idx = idx[np.argsort(starts[idx])]
        if len(idx) > max_steps:
            idx = idx[np.linspace(0, len(idx) - 1, max_steps).round().astype(int)]
        streams = []
        for value_name, mask_name in (("driver", "driver_mask"), ("battery", "battery_mask")):
            values = arrays[value_name][idx].astype(np.float32)
            valid = arrays[mask_name][idx].astype(bool)
            values = np.where(valid, values, np.nan)
            mean = np.nanmean(values, axis=1)
            std = np.nanstd(values, axis=1)
            streams.extend((np.nan_to_num(mean), np.nan_to_num(std)))
        sequence = np.concatenate(streams, axis=1).astype(np.float32)
        padded = np.zeros((max_steps, sequence.shape[1]), dtype=np.float32)
        mask = np.zeros(max_steps, dtype=bool)
        padded[: len(sequence)] = sequence
        mask[: len(sequence)] = True
        sequences.append(padded)
        masks.append(mask)
        first = idx[0]
        rows.append({"trip_key": key, **{
            target: float(arrays[key_name][first]) for target, key_name in TARGET_KEYS.items()
        }})
    return np.stack(sequences), np.stack(masks), pd.DataFrame(rows)


def run_temporal_baselines(manifest_path: Path, output: Path, seed: int = 42,
                           epochs: int = 60) -> None:
    import torch
    from torch import nn
    from torch.nn import functional as F

    class TemporalNet(nn.Module):
        def __init__(self, input_dim: int, kind: str):
            super().__init__()
            self.kind = kind
            hidden = 64
            if kind == "TCN":
                self.encoder = nn.Sequential(
                    nn.Conv1d(input_dim, hidden, 5, padding=2), nn.GELU(),
                    nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
            elif kind in {"GRU", "LSTM"}:
                cls = nn.GRU if kind == "GRU" else nn.LSTM
                self.encoder = cls(input_dim, hidden, num_layers=2, dropout=0.1, batch_first=True)
            elif kind == "PatchTST_style":
                self.patcher = nn.Conv1d(input_dim, hidden, kernel_size=4, stride=2)
                layer = nn.TransformerEncoderLayer(hidden, 4, 128, 0.1, batch_first=True,
                                                   norm_first=True)
                self.encoder = nn.TransformerEncoder(layer, 2)
            elif kind == "iTransformer_style":
                self.inverted_projection = nn.Linear(64, hidden)
                layer = nn.TransformerEncoderLayer(hidden, 4, 128, 0.1, batch_first=True,
                                                   norm_first=True)
                self.encoder = nn.TransformerEncoder(layer, 2)
            elif kind == "TimesNet_style":
                self.multiscale = nn.ModuleList([
                    nn.Conv1d(input_dim, hidden, kernel_size=k, padding=k // 2)
                    for k in (3, 5, 7)
                ])
                self.fft_projection = nn.Linear(input_dim * 8, hidden)
                self.encoder = nn.Identity()
            else:
                self.projection = nn.Linear(input_dim, hidden)
                layer = nn.TransformerEncoderLayer(hidden, 4, 128, 0.1, batch_first=True,
                                                   norm_first=True)
                self.encoder = nn.TransformerEncoder(layer, 2)
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 32), nn.SiLU(),
                                      nn.Linear(32, 3))

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            if self.kind == "TCN":
                encoded = self.encoder(x.transpose(1, 2)).transpose(1, 2)
            elif self.kind in {"GRU", "LSTM"}:
                encoded = self.encoder(x)[0]
            elif self.kind == "PatchTST_style":
                encoded = self.encoder(self.patcher(x.transpose(1, 2)).transpose(1, 2))
                mask = torch.ones(encoded.shape[:2], dtype=torch.bool, device=x.device)
            elif self.kind == "iTransformer_style":
                # Variables are tokens and the 64-point trip history is embedded.
                encoded = self.encoder(self.inverted_projection(x.transpose(1, 2)))
                mask = torch.ones(encoded.shape[:2], dtype=torch.bool, device=x.device)
            elif self.kind == "TimesNet_style":
                time_features = torch.stack(
                    [F.gelu(conv(x.transpose(1, 2))).mean(-1) for conv in self.multiscale]
                ).mean(0)
                spectrum = torch.fft.rfft(x * mask.unsqueeze(-1), dim=1).abs()
                top = torch.topk(spectrum, k=min(8, spectrum.shape[1]), dim=1).values
                if top.shape[1] < 8:
                    top = torch.nn.functional.pad(top, (0, 0, 0, 8 - top.shape[1]))
                pooled = time_features + self.fft_projection(top.transpose(1, 2).reshape(x.shape[0], -1))
                return self.head(pooled)
            else:
                encoded = self.encoder(self.projection(x), src_key_padding_mask=~mask)
            pooled = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            return self.head(pooled)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(manifest_path.read_text())
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for fold_spec in manifest["folds"]:
        fold = int(fold_spec["fold"])
        tr_x, tr_m, tr_y = _segment_summary(Path(fold_spec["paths"]["inner_train"]))
        va_x, va_m, va_y = _segment_summary(Path(fold_spec["paths"]["inner_validation"]))
        te_x, te_m, te_y = _segment_summary(Path(fold_spec["paths"]["outer_test"]))
        center = np.nanmedian(tr_x.reshape(-1, tr_x.shape[-1]), axis=0)
        scale = np.nanquantile(tr_x.reshape(-1, tr_x.shape[-1]), .75, axis=0) - np.nanquantile(
            tr_x.reshape(-1, tr_x.shape[-1]), .25, axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        tr_x = (tr_x - center) / scale; va_x = (va_x - center) / scale; te_x = (te_x - center) / scale
        tr_x[~tr_m] = 0.0; va_x[~va_m] = 0.0; te_x[~te_m] = 0.0
        y_center = np.nanmedian(tr_y[list(TARGETS)].to_numpy(float), axis=0)
        y_scale = np.nanstd(tr_y[list(TARGETS)].to_numpy(float), axis=0)
        y_scale = np.where(y_scale > 1e-6, y_scale, 1.0)

        def tensors(x, m, y):
            target = y[list(TARGETS)].to_numpy(np.float32)
            return (torch.tensor(x, dtype=torch.float32), torch.tensor(m, dtype=torch.bool),
                    torch.tensor((target - y_center) / y_scale, dtype=torch.float32),
                    torch.tensor(np.isfinite(target), dtype=torch.bool))
        train_t = tensors(tr_x, tr_m, tr_y); val_t = tensors(va_x, va_m, va_y)
        for kind in (
            "TCN", "GRU", "LSTM", "VanillaTransformer", "PatchTST_style",
            "iTransformer_style", "TimesNet_style",
        ):
            torch.manual_seed(seed + fold)
            model = TemporalNet(tr_x.shape[-1], kind).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            best = math.inf; stale = 0; best_state = None; started = time.perf_counter()
            for epoch in range(epochs):
                model.train(); optimizer.zero_grad(set_to_none=True)
                pred = model(train_t[0].to(device), train_t[1].to(device))
                target = train_t[2].to(device); valid = train_t[3].to(device)
                loss = F.huber_loss(pred[valid], target[valid], delta=1.0)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                model.eval()
                with torch.no_grad():
                    vp = model(val_t[0].to(device), val_t[1].to(device))
                    vv = val_t[3].to(device); vt = val_t[2].to(device)
                    score = float(F.huber_loss(vp[vv], vt[vv], delta=1.0))
                if score < best - 1e-6:
                    best = score; stale = 0
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= 10: break
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(te_x, dtype=torch.float32, device=device),
                             torch.tensor(te_m, dtype=torch.bool, device=device)).cpu().numpy()
            pred = pred * y_scale + y_center
            pred[:, 1:] = np.maximum(pred[:, 1:], 0.0)
            for column, target in enumerate(TARGETS):
                observed = te_y[target].to_numpy(float)
                result = _metrics(observed, pred[:, column])
                metric_rows.append({"family": "temporal", "fold": fold, "target": target,
                                    "model": kind, "seed": seed, "status": "complete",
                                    "implementation_scope": (
                                        "compact adaptation for 101 trip-level sequences"
                                        if kind.endswith("_style") else "native compact baseline"
                                    ),
                                    "parameter_count": sum(p.numel() for p in model.parameters()),
                                    "elapsed_seconds": time.perf_counter() - started, **result})
                for trip, obs, value in zip(te_y.trip_key, observed, pred[:, column], strict=True):
                    prediction_rows.append({"family": "temporal", "fold": fold, "target": target,
                                            "model": kind, "trip_key": trip,
                                            "observed": obs, "predicted": value})
            model_dir = output / "temporal_models" / f"fold_{fold}" / kind
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": best_state, "input_center": center, "input_scale": scale,
                        "target_center": y_center, "target_scale": y_scale}, model_dir / "best.pt")
            update(f"Temporal {kind}: fold {fold}/4 complete")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(output / "temporal_fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "temporal_predictions.csv", index=False)


@dataclass(frozen=True)
class Ablation:
    name: str
    family: str
    modality: str = "both"
    coupling: str = "bidirectional"
    head: str = "kan"
    prediction_input: str = "concepts"
    concept_set: str = "components_dose"
    stage: str = "both"
    ssl_alignment: float = 0.1
    concept_weight: float = 0.2
    physics_weight: float = 0.1
    driver_id_weight: float = 0.1
    freeze_encoders: bool = True


def missing_ablations() -> list[Ablation]:
    """Only experiments absent from Stages 1--5; KAN-vs-MLP is not repeated."""

    return [
        Ablation("Reference", "reference"),
        Ablation("Driver_only", "input_modality", modality="driver"),
        Ablation("Battery_only", "input_modality", modality="battery"),
        Ablation("Concatenation", "driver_coupling", coupling="concatenation"),
        Ablation("One_way_attention", "driver_coupling", coupling="one_way"),
        Ablation("No_pretraining", "pretraining", stage="finetune", freeze_encoders=False),
        Ablation("Masked_reconstruction", "pretraining", ssl_alignment=0.0),
        Ablation("No_concept_bottleneck", "concepts", head="mlp", prediction_input="latent",
                 concept_weight=0.0),
        Ablation("DBSI_none", "dbsi", concept_set="none"),
        Ablation("DBSI_total", "dbsi", concept_set="total"),
        Ablation("DBSI_components", "dbsi", concept_set="components"),
        Ablation("Physics_off", "physics", physics_weight=0.0),
        Ablation("Encoders_unfrozen", "trainability", freeze_encoders=False),
        Ablation("DriverID_excluded", "personalisation", driver_id_weight=0.0),
    ]


class ConceptView:
    """Dataset view applying a named D-BSI representation without row leakage."""

    def __init__(self, dataset: Any, mode: str):
        from dbpatchkan.models import collate_trip_bags
        self.dataset = dataset
        self.mode = mode
        self.collate_fn = collate_trip_bags
        self.indices = {
            "components_dose": list(range(20)),
            "components": [i for i in range(20) if i != 11],
            "none": [i for i in range(20) if i not in {*range(6), 11}],
            "total": [*range(6, 20)],
        }[mode]

    @property
    def group_ids(self):
        return self.dataset.group_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch
        item = dict(self.dataset[index])
        if self.mode == "total":
            values = item["concept_targets"]
            mask = item["concept_mask"]
            weights = torch.tensor(DBSI_WEIGHTS, dtype=values.dtype)
            total = 100.0 * torch.sum(values[:6] * weights)
            total_mask = torch.all(mask[:6]).reshape(1)
            item["concept_targets"] = torch.cat((total.reshape(1), values[6:]))
            item["concept_mask"] = torch.cat((total_mask, mask[6:]))
        else:
            item["concept_targets"] = item["concept_targets"][self.indices]
            item["concept_mask"] = item["concept_mask"][self.indices]
        return item


def _concept_names(mode: str) -> tuple[str, ...]:
    from dbpatchkan.models.patchkan import DEFAULT_CONCEPT_NAMES
    names = list(DEFAULT_CONCEPT_NAMES)
    if mode == "components_dose": return tuple(names)
    if mode == "components": return tuple(name for i, name in enumerate(names) if i != 11)
    if mode == "none": return tuple(name for i, name in enumerate(names) if i not in {*range(6), 11})
    return ("dbsi_total", *names[6:])


def _outer_predictions(model: Any, dataset: Any, device: Any, precision: str) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader
    from dbpatchkan.training.train import autocast_context, move_batch
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)
    rows: list[dict[str, Any]] = []
    model.eval()
    offset = 0
    with torch.inference_mode():
        for raw in loader:
            batch = move_batch(raw, device)
            with autocast_context(device, precision):
                out = model.forward_trip_bag(
                    batch["driver"], batch["battery"], segment_mask=batch["segment_mask"],
                    driver_mask=batch.get("driver_mask"), battery_mask=batch.get("battery_mask"),
                    domain="nissan")
            count = len(out["current_soh"])
            for index in range(count):
                item = {"trip_index": offset + index}
                for target in TARGETS:
                    item[f"{target}_predicted"] = float(out[target][index].float().cpu())
                    observed = batch.get(TARGET_KEYS[target])
                    item[f"{target}_observed"] = (
                        float(observed[index].float().cpu()) if observed is not None else np.nan)
                rows.append(item)
            offset += count
    frame = pd.DataFrame(rows)
    frame["trip_key"] = [group[2] for group in dataset.group_ids]
    return frame


def run_patchkan_ablations(manifest_path: Path, output: Path, seed: int = 42,
                           pretrain_epochs: int = 3, finetune_epochs: int = 24) -> None:
    import torch
    from dbpatchkan.models import DBPatchKAN, DBPatchKANConfig, EVTripBagDataset
    from dbpatchkan.training.evaluate import load_checkpoint_model
    from dbpatchkan.training.train import TrainingConfig, train

    manifest = json.loads(manifest_path.read_text())
    metrics_path = output / "patchkan_ablation_fold_metrics.csv"
    predictions_path = output / "patchkan_ablation_predictions.csv"
    all_metrics: list[dict[str, Any]] = (
        pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    )
    all_predictions: list[pd.DataFrame] = (
        [pd.read_csv(predictions_path)] if predictions_path.exists() else []
    )
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "bf16" if device_name == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    variants = missing_ablations()
    for variant_index, variant in enumerate(variants, start=1):
        update(f"Ablation {variant_index}/{len(variants)} START: {variant.family}/{variant.name}")
        for fold_spec in manifest["folds"]:
            fold = int(fold_spec["fold"])
            already_done = any(
                row.get("variant") == variant.name and int(row.get("fold", -1)) == fold
                for row in all_metrics
            )
            if already_done:
                update(f"Ablation {variant.name}: fold {fold}/4 already complete; skipped")
                continue
            base_train = EVTripBagDataset.from_npz(fold_spec["paths"]["inner_train"])
            base_validation = EVTripBagDataset.from_npz(fold_spec["paths"]["inner_validation"])
            base_outer = EVTripBagDataset.from_npz(fold_spec["paths"]["outer_test"])
            train_data = ConceptView(base_train, variant.concept_set)
            validation_data = ConceptView(base_validation, variant.concept_set)
            outer_data = ConceptView(base_outer, variant.concept_set)
            config = DBPatchKANConfig(
                driver_channels=13, battery_channels=11, segment_length=256,
                patch_length=16, patch_stride=8, d_model=128, encoder_layers=3,
                n_heads=4, ff_dim=256, dropout=0.1, fusion_head=variant.head,
                input_modality=variant.modality, cross_attention_mode=variant.coupling,
                prediction_input=variant.prediction_input,
                concept_names=_concept_names(variant.concept_set),
            )
            run_dir = output / "patchkan_models" / variant.family / variant.name / f"fold_{fold}"
            completion_marker = run_dir / "stage6_complete.json"
            model = DBPatchKAN(config)
            effective_stage = variant.stage
            # Masked+alignment SSL is identical across all ordinary ablations:
            # it only touches the two encoders/pools/reconstruction layers. Reuse
            # the reference fold's pretrained state instead of repeating the
            # same deterministic computation. Masked-only and no-pretraining
            # remain genuinely separate experiments.
            if variant.name not in {"Reference", "Masked_reconstruction", "No_pretraining"}:
                reference_pretrain = (
                    output / "patchkan_models" / "reference" / "Reference"
                    / f"fold_{fold}" / "pretrain_latest.pt"
                )
                checkpoint = torch.load(reference_pretrain, map_location="cpu", weights_only=False)
                prefixes = (
                    "driver_encoder", "battery_encoder", "driver_pool", "battery_pool",
                    "driver_reconstruction", "battery_reconstruction",
                )
                reusable = {
                    name: value for name, value in checkpoint["model"].items()
                    if name.startswith(prefixes) and name in model.state_dict()
                    and model.state_dict()[name].shape == value.shape
                }
                model.load_state_dict(reusable, strict=False)
                effective_stage = "finetune"
            training = TrainingConfig(
                stage=effective_stage, device=device_name, precision=precision,
                pretrain_batch_size=256, finetune_batch_size=2,
                gradient_accumulation=1, pretrain_epochs=pretrain_epochs,
                finetune_epochs=finetune_epochs, patience=6, pretrain_lr=3e-4,
                encoder_lr=1e-5, head_lr=3e-4, num_workers=2, seed=seed,
                domain="nissan", freeze_encoders=variant.freeze_encoders, trip_bags=True,
                concept_weight=variant.concept_weight, physics_weight=variant.physics_weight,
                driver_id_weight=variant.driver_id_weight,
                ssl_alignment_weight=variant.ssl_alignment,
            )
            started = time.perf_counter()
            if completion_marker.exists() and (run_dir / "best.pt").exists():
                result = json.loads(completion_marker.read_text())
            else:
                resume = run_dir / "latest.pt"
                if not resume.exists():
                    resume = run_dir / "pretrain_latest.pt"
                if resume.exists():
                    training.resume_checkpoint = str(resume)
                    update(f"Resuming {variant.name} fold {fold} from {resume.name}")
                result = train(model, train_data, validation_data, training,
                               output_dir=run_dir)
                completion_marker.write_text(json.dumps(result, indent=2))
            checkpoint = run_dir / "best.pt"
            model, _ = load_checkpoint_model(checkpoint, device=torch.device(device_name))
            frame = _outer_predictions(model, outer_data, torch.device(device_name), precision)
            frame.insert(0, "seed", seed); frame.insert(0, "fold", fold)
            frame.insert(0, "variant", variant.name); frame.insert(0, "ablation_family", variant.family)
            all_predictions.append(frame)
            for target in TARGETS:
                metric = _metrics(frame[f"{target}_observed"].to_numpy(),
                                  frame[f"{target}_predicted"].to_numpy())
                all_metrics.append({"family": "patchkan_ablation", "ablation_family": variant.family,
                                    "variant": variant.name, "fold": fold, "seed": seed,
                                    "target": target, "status": "complete",
                                    "elapsed_seconds": time.perf_counter() - started,
                                    "parameter_count": result["parameter_count"],
                                    "peak_vram_gb": result["peak_vram_gb"], **metric})
            redundant_paths = [run_dir / "latest.pt"]
            if variant.name != "Reference":
                redundant_paths.append(run_dir / "pretrain_latest.pt")
            for redundant in redundant_paths:
                if redundant.exists(): redundant.unlink()
            output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
            pd.concat(all_predictions, ignore_index=True).drop_duplicates(
                ["ablation_family", "variant", "fold", "trip_key"], keep="last"
            ).to_csv(predictions_path, index=False)
            update(f"Ablation {variant.name}: fold {fold}/4 complete")
    output.mkdir(parents=True, exist_ok=True)


def _bootstrap_delta(reference: pd.DataFrame, challenger: pd.DataFrame, seed: int = 42,
                     repeats: int = 4000) -> dict[str, float]:
    merged = reference.merge(challenger, on=["fold", "trip_key", "target"],
                             suffixes=("_ref", "_alt"))
    merged = merged[np.isfinite(merged.observed_ref) & np.isfinite(merged.predicted_ref)
                    & np.isfinite(merged.predicted_alt)]
    if len(merged) < 3:
        return {"n": len(merged), "delta_mae": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    per_trip = np.abs(merged.predicted_alt - merged.observed_ref) - np.abs(
        merged.predicted_ref - merged.observed_ref)
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(per_trip, len(per_trip), replace=True).mean()
                        for _ in range(repeats)])
    return {"n": len(per_trip), "delta_mae": float(per_trip.mean()),
            "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975))}


def aggregate_stage6(output: Path, seed: int = 42) -> None:
    metric_files = [output / "classical_fold_metrics.csv", output / "temporal_fold_metrics.csv",
                    output / "patchkan_ablation_fold_metrics.csv"]
    frames = [pd.read_csv(path) for path in metric_files if path.exists()]
    if not frames:
        raise FileNotFoundError("No Stage 6 metrics exist")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(output / "all_fold_metrics.csv", index=False)
    name_column = np.where(combined["family"].eq("patchkan_ablation"),
                           combined.get("variant"), combined.get("model"))
    combined["model_or_variant"] = name_column
    summary = (combined[combined.status.eq("complete")]
               .groupby(["family", "target", "model_or_variant"], dropna=False)
               .agg(folds=("fold", "nunique"), n=("n", "sum"), mae_mean=("mae", "mean"),
                    mae_std=("mae", "std"), rmse_mean=("rmse", "mean"),
                    bias_mean=("bias", "mean"), r2_mean=("r2", "mean"),
                    parameters=("parameter_count", "median") if "parameter_count" in combined else ("n", "size"),
                    elapsed_seconds=("elapsed_seconds", "sum"))
               .reset_index())
    summary.to_csv(output / "model_comparison_summary.csv", index=False)

    ablation_path = output / "patchkan_ablation_predictions.csv"
    paired_rows: list[dict[str, Any]] = []
    if ablation_path.exists():
        wide = pd.read_csv(ablation_path)
        long_rows = []
        for target in TARGETS:
            part = wide[["ablation_family", "variant", "fold", "trip_key",
                         f"{target}_observed", f"{target}_predicted"]].copy()
            part.columns = ["ablation_family", "variant", "fold", "trip_key", "observed", "predicted"]
            part["target"] = target; long_rows.append(part)
        long = pd.concat(long_rows, ignore_index=True)
        reference = long[long.variant.eq("Reference")]
        for (family, variant, target), challenger in long[~long.variant.eq("Reference")].groupby(
            ["ablation_family", "variant", "target"]):
            delta = _bootstrap_delta(reference[reference.target.eq(target)], challenger, seed=seed)
            paired_rows.append({"ablation_family": family, "variant": variant, "target": target,
                                "reference": "Reference", **delta,
                                "interpretation": "negative delta favours ablation; positive favours reference"})
    pd.DataFrame(paired_rows).to_csv(output / "paired_bootstrap_ablation_deltas.csv", index=False)

    decision = {
        "stage": 6,
        "locked_test_used": False,
        "selection_authority": "development folds only; no final deployment claim",
        "completed_prior_evidence_not_repeated": list(COMPLETED_PRIOR_EVIDENCE),
        "result_tables": [path.name for path in output.glob("*.csv")],
        "next_action": "Review paired CIs, freeze one final configuration, then request explicit permission before opening the locked test.",
    }
    (output / "STAGE6_DECISION.json").write_text(json.dumps(decision, indent=2))
    update("Stage 6 aggregation complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "classical", "temporal", "ablations", "aggregate", "all"),
                        default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=24)
    parser.add_argument("--temporal-epochs", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest = output / "stage6_fold_manifest.json"
    if args.mode in {"prepare", "all"}:
        if args.train_data is None or args.validation_data is None:
            raise ValueError("prepare requires --train-data and --validation-data")
        prepare_stage6(args.train_data.resolve(), args.validation_data.resolve(), output)
    if args.mode in {"classical", "all"}: run_classical(manifest, output, args.seed)
    if args.mode in {"temporal", "all"}: run_temporal_baselines(manifest, output, args.seed, args.temporal_epochs)
    if args.mode in {"ablations", "all"}:
        run_patchkan_ablations(manifest, output, args.seed, args.pretrain_epochs, args.finetune_epochs)
    if args.mode in {"aggregate", "all"}: aggregate_stage6(output, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
