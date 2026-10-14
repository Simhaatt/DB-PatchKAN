"""Stage 13B recent-method comparison on frozen EVBattery transitions.

This module adds three architecture-faithful, persistence-anchored comparators:

* ModernTCN (ICLR 2024): large-kernel depthwise temporal convolution;
* TSLANet (ICML 2024): adaptive spectral filtering plus interactive convolution;
* KAL / KAN-Linformer (Neurocomputing 2025): low-rank attention with latent KAN FFN.

All methods receive the same current 128-step charge snippet and the same twenty
fold-scaled concepts.  They predict a signed residual added to current capacity.
Existing Stage 13 predictions are loaded unchanged and are never retrained.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dbpatchkan.analysis.stage13_persistence_forecast import (
    FORECAST_CONCEPTS,
    Scaling,
    TransitionDataset,
    forecast_matrix,
    load_transition_cache,
    metric,
    scale_fold,
    set_seed,
    stamp,
)
from dbpatchkan.models.kan import BSplineKANLayer


METHODS: tuple[str, ...] = ("moderntcn", "tslanet", "kal_linformer")
MODEL_NAMES = {
    "moderntcn": "PersistenceAnchored_ModernTCN_2024",
    "tslanet": "PersistenceAnchored_TSLANet_2024",
    "kal_linformer": "PersistenceAnchored_KALinformer_2025",
}


class ResidualMLP(nn.Module):
    def __init__(self, width: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values.float()).squeeze(-1)


class ModernTCNBlock(nn.Module):
    """Large-kernel depthwise temporal mixing plus inverted ConvFFN."""

    def __init__(self, width: int, *, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            width, width, kernel_size, padding=padding, groups=width
        )
        self.norm1 = nn.LayerNorm(width)
        self.expand = nn.Conv1d(width, 4 * width, 1)
        self.contract = nn.Conv1d(4 * width, width, 1)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _norm(values: torch.Tensor, layer: nn.LayerNorm) -> torch.Tensor:
        return layer(values.transpose(1, 2)).transpose(1, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.dropout(self.depthwise(values))
        values = self._norm(values, self.norm1)
        mixed = self.contract(self.dropout(F.gelu(self.expand(values))))
        return self._norm(values + self.dropout(mixed), self.norm2)


class PersistenceAnchoredModernTCN(nn.Module):
    def __init__(self, *, width: int = 96, blocks: int = 3):
        super().__init__()
        self.stem = nn.Conv1d(8, width, kernel_size=8, stride=4)
        self.blocks = nn.Sequential(
            *(ModernTCNBlock(width, kernel_size=31) for _ in range(blocks))
        )
        self.head = ResidualMLP(width + len(FORECAST_CONCEPTS))

    def forward(self, sequence: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
        encoded = self.blocks(self.stem(sequence.transpose(1, 2)))
        pooled = encoded.mean(dim=-1)
        return self.head(torch.cat((pooled, concepts.float()), dim=1))

    def regularization_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_zeros(())


class AdaptiveSpectralBlock(nn.Module):
    """TSLANet-style learnable spectral filter with adaptive shrinkage."""

    def __init__(self, width: int, tokens: int, *, dropout: float = 0.1):
        super().__init__()
        frequencies = tokens // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.randn(frequencies, width, 2) * 0.02
        )
        self.threshold_logit = nn.Parameter(torch.tensor(-3.0))
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        device_type = values.device.type if values.device.type in {"cpu", "cuda"} else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            spectrum = torch.fft.rfft(values.float(), dim=1, norm="ortho")
            weight = torch.view_as_complex(self.complex_weight.float().contiguous())
            magnitude = spectrum.abs()
            threshold = torch.sigmoid(self.threshold_logit.float()) * magnitude.mean(
                dim=(1, 2), keepdim=True
            )
            keep = F.relu(magnitude - threshold) / (magnitude + 1e-6)
            filtered = spectrum * keep * (1.0 + 0.1 * weight.unsqueeze(0))
            restored = torch.fft.irfft(
                filtered, n=values.shape[1], dim=1, norm="ortho"
            )
        return self.norm(residual.float() + self.dropout(restored))


class InteractiveConvBlock(nn.Module):
    """TSLANet-style fine/coarse interactive temporal convolution."""

    def __init__(self, width: int, *, dropout: float = 0.1):
        super().__init__()
        self.local = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.context = nn.Conv1d(width, width, 9, padding=4, groups=width)
        self.mix = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        channels = values.transpose(1, 2)
        local = self.local(channels)
        context = self.context(channels)
        interaction = torch.sigmoid(local) * context + torch.sigmoid(context) * local
        output = self.mix(interaction).transpose(1, 2)
        return self.norm(values + self.dropout(output))


class TSLANetBlock(nn.Module):
    def __init__(self, width: int, tokens: int):
        super().__init__()
        self.spectral = AdaptiveSpectralBlock(width, tokens)
        self.interactive = InteractiveConvBlock(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.interactive(self.spectral(values))


class PersistenceAnchoredTSLANet(nn.Module):
    def __init__(self, *, width: int = 96, blocks: int = 2):
        super().__init__()
        self.patch = nn.Conv1d(8, width, kernel_size=8, stride=4)
        tokens = 31
        self.blocks = nn.Sequential(*(TSLANetBlock(width, tokens) for _ in range(blocks)))
        self.head = ResidualMLP(width + len(FORECAST_CONCEPTS))

    def forward(self, sequence: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(sequence.transpose(1, 2)).transpose(1, 2)
        encoded = self.blocks(tokens)
        return self.head(torch.cat((encoded.mean(dim=1), concepts.float()), dim=1))

    def regularization_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_zeros(())


class LinformerKANBlock(nn.Module):
    """Low-rank self-attention followed by a KAN latent feed-forward block."""

    def __init__(
        self,
        width: int,
        tokens: int,
        *,
        heads: int = 4,
        rank: int = 8,
        grid_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("Linformer width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.rank = rank
        self.qkv = nn.Linear(width, 3 * width)
        self.key_projection = nn.Parameter(torch.randn(heads, rank, tokens) / math.sqrt(tokens))
        self.value_projection = nn.Parameter(torch.randn(heads, rank, tokens) / math.sqrt(tokens))
        self.output = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        # This intentionally mirrors the published KAL idea of replacing the
        # Linformer MLP with KAN.  Unlike DB-PatchKAN, these splines operate on
        # unnamed latent coordinates and are not concept-level explanations.
        self.kan_up = BSplineKANLayer(width, 2 * width, grid_size=grid_size)
        self.kan_down = BSplineKANLayer(2 * width, width, grid_size=grid_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = values.shape
        q, k, v = self.qkv(self.norm1(values)).chunk(3, dim=-1)
        shape = (batch, tokens, self.heads, self.head_width)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        k_low = torch.einsum("hrt,bhtd->bhrd", self.key_projection, k)
        v_low = torch.einsum("hrt,bhtd->bhrd", self.value_projection, v)
        scores = torch.einsum("bhtd,bhrd->bhtr", q, k_low) / math.sqrt(self.head_width)
        attended = torch.einsum("bhtr,bhrd->bhtd", scores.softmax(dim=-1), v_low)
        attended = attended.transpose(1, 2).reshape(batch, tokens, self.width)
        values = values + self.dropout(self.output(attended))
        latent = self.norm2(values)
        kan = self.kan_down(F.silu(self.kan_up(latent)))
        return values.float() + self.dropout(kan)

    def regularization_loss(self) -> torch.Tensor:
        return 0.5 * (
            self.kan_up.regularization_loss() + self.kan_down.regularization_loss()
        )


class PersistenceAnchoredKALinformer(nn.Module):
    def __init__(self, *, width: int = 48, blocks: int = 2):
        super().__init__()
        self.patch = nn.Conv1d(8, width, kernel_size=16, stride=8)
        tokens = 15
        self.blocks = nn.ModuleList(
            LinformerKANBlock(width, tokens, heads=4, rank=8) for _ in range(blocks)
        )
        self.norm = nn.LayerNorm(width)
        self.head = ResidualMLP(width + len(FORECAST_CONCEPTS))

    def forward(self, sequence: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(sequence.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        pooled = self.norm(tokens).mean(dim=1)
        return self.head(torch.cat((pooled, concepts.float()), dim=1))

    def regularization_loss(self) -> torch.Tensor:
        return torch.stack([block.regularization_loss() for block in self.blocks]).mean()


def make_model(method: str) -> nn.Module:
    if method == "moderntcn":
        return PersistenceAnchoredModernTCN()
    if method == "tslanet":
        return PersistenceAnchoredTSLANet()
    if method == "kal_linformer":
        return PersistenceAnchoredKALinformer()
    raise ValueError(f"Unknown recent method: {method}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def predict_recent(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaling: Scaling,
) -> list[tuple[int, float, float]]:
    model.eval()
    records: list[tuple[int, float, float]] = []
    with torch.no_grad():
        for sequence, physical, context, _, current, indices in loader:
            concepts = torch.cat((physical, context), dim=1)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                residual = model(sequence.to(device), concepts.to(device))
            delta = residual.float().cpu().numpy() * scaling.delta_scale + scaling.delta_center
            future = current.numpy() + delta
            records.extend(zip(indices.numpy().tolist(), delta.tolist(), future.tolist()))
    return records


def train_recent_fold(
    sequence: np.ndarray,
    concepts: np.ndarray,
    delta: np.ndarray,
    current: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    groups: np.ndarray,
    *,
    method: str,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
):
    set_seed(seed)
    split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_rel, val_rel = next(split.split(outer_train, groups=groups[outer_train]))
    train_idx, val_idx = outer_train[train_rel], outer_train[val_rel]
    seq, con, y, scaling = scale_fold(sequence, concepts, delta, train_idx)
    # TransitionDataset keeps the established physical/context split while the
    # recent models concatenate both parts back into the identical 20 features.
    physical_count = len(FORECAST_CONCEPTS) - 6
    physical = con[:, :physical_count]
    context = con[:, physical_count:]

    def loader(indices, shuffle, size):
        return DataLoader(
            TransitionDataset(seq, physical, context, y, current, indices),
            batch_size=size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )

    train_loader = loader(train_idx, True, batch_size)
    val_loader = loader(val_idx, False, 2 * batch_size)
    test_loader = loader(outer_test, False, 2 * batch_size)
    model = make_model(method).to(device)
    params = parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_state = None
    best_mae = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for seq_b, physical_b, context_b, y_b, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            feature_b = torch.cat((physical_b, context_b), dim=1)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                residual = model(seq_b.to(device), feature_b.to(device))
                beta = max(1e-3, 0.25 / scaling.delta_scale)
                regression = F.smooth_l1_loss(
                    residual.float(), y_b.to(device).float(), beta=beta
                )
                residual_ah = residual.float() * scaling.delta_scale + scaling.delta_center
                positive_penalty = torch.relu(residual_ah - 0.25).square().mean()
                loss = regression + 0.02 * positive_penalty + 1e-4 * model.regularization_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = predict_recent(model, val_loader, device, scaling)
        indices = np.asarray([item[0] for item in validation], dtype=int)
        predicted_delta = np.asarray([item[1] for item in validation])
        val_mae = float(np.mean(np.abs(delta[indices] - predicted_delta)))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_mae_ah": val_mae,
            }
        )
        if val_mae < best_mae - 1e-4:
            best_mae = val_mae
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            stamp(f"{method} seed={seed} epoch={epoch}: validation MAE={val_mae:.4f} Ah")
        if stale >= 12:
            break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {method}, seed {seed}")
    model.load_state_dict(best_state)
    validation = predict_recent(model, val_loader, device, scaling)
    val_indices = np.asarray([item[0] for item in validation], dtype=int)
    val_future = np.asarray([item[2] for item in validation])
    truth = current[val_indices] + delta[val_indices]
    q90 = float(np.quantile(np.abs(truth - val_future), 0.90, method="higher"))
    elapsed = (time.perf_counter() - started) / 60.0
    return (
        predict_recent(model, test_loader, device, scaling),
        pd.DataFrame(history),
        model,
        scaling,
        q90,
        params,
        elapsed,
    )


def _seed_ensemble(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "transition_id",
        "fold",
        "manufacturer_id",
        "vehicle_id",
        "vehicle_key",
        "forecast_horizon_km",
        "current_capacity_ah",
        "future_capacity_ah",
        "model",
    ]
    return predictions.groupby(keys, as_index=False).agg(
        prediction_capacity_ah=("prediction_capacity_ah", "mean"),
        interval_low_ah=("interval_low_ah", "mean"),
        interval_high_ah=("interval_high_ah", "mean"),
    )


def run_recent_validation(
    cache: str | Path,
    stage13_results: str | Path,
    output_dir: str | Path,
    *,
    folds: int,
    seeds: list[int],
    epochs: int,
    batch_size: int,
) -> None:
    loaded = load_transition_cache(cache)
    sequence = loaded["sequence"]
    table = loaded["table"].copy()
    concepts = forecast_matrix(table).to_numpy("float32")
    delta = pd.to_numeric(table.capacity_delta_ah).to_numpy("float32")
    current = pd.to_numeric(table.current_capacity_ah).to_numpy("float32")
    future = pd.to_numeric(table.future_capacity_ah).to_numpy("float32")
    groups = table.vehicle_key.astype(str).to_numpy()
    strata = table.manufacturer_id.astype(int).astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(concepts, strata, groups), 1
    ):
        stamp(
            f"Recent-method fold {fold}/{folds}: "
            f"train vehicles={len(set(groups[train_idx]))}, "
            f"test vehicles={len(set(groups[test_idx]))}"
        )
        for method in METHODS:
            for seed in seeds:
                stamp(f"START {method}, fold={fold}, seed={seed}")
                records, history, model, scaling, q90, params, elapsed = train_recent_fold(
                    sequence,
                    concepts,
                    delta,
                    current,
                    train_idx,
                    test_idx,
                    groups,
                    method=method,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    device=device,
                )
                indices = np.asarray([item[0] for item in records], dtype=int)
                prediction = np.asarray([item[2] for item in records])
                name = MODEL_NAMES[method]
                frame = pd.DataFrame(
                    {
                        "transition_id": table.iloc[indices].transition_id.to_numpy(),
                        "fold": fold,
                        "seed": seed,
                        "model": name,
                        "manufacturer_id": table.iloc[indices].manufacturer_id.to_numpy(),
                        "vehicle_id": table.iloc[indices].vehicle_id.to_numpy(),
                        "vehicle_key": groups[indices],
                        "forecast_horizon_km": table.iloc[indices].forecast_horizon_km.to_numpy(),
                        "current_capacity_ah": current[indices],
                        "future_capacity_ah": future[indices],
                        "prediction_capacity_ah": prediction,
                        "interval_low_ah": prediction - q90,
                        "interval_high_ah": prediction + q90,
                    }
                )
                predictions.append(frame)
                fold_metrics.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "model": name,
                        "parameter_count": params,
                        "train_time_min": elapsed,
                        **metric(frame),
                    }
                )
                run = output_dir / "models" / f"fold_{fold}" / f"{method}_seed_{seed}"
                run.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "scaling": asdict(scaling),
                        "q90_ah": q90,
                        "concepts": FORECAST_CONCEPTS,
                        "method": method,
                        "model_name": name,
                        "target": "next_distinct_mileage_partial_capacity_ah",
                    },
                    run / "model.pt",
                )
                history.to_csv(run / "history.csv", index=False)
                stamp(
                    f"DONE {method}, fold={fold}, seed={seed}, "
                    f"parameters={params:,}, elapsed={elapsed:.2f} min"
                )

    raw = pd.concat(predictions, ignore_index=True)
    raw.to_csv(output_dir / "stage13b_recent_method_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(
        output_dir / "stage13b_recent_method_fold_metrics.csv", index=False
    )
    recent_ensemble = _seed_ensemble(raw)
    frozen_path = Path(stage13_results) / "stage13_seed_ensemble_predictions.csv"
    frozen = pd.read_csv(frozen_path)
    required_frozen = {
        "Persistence",
        "PersistenceAnchored_KAN",
        "PersistenceAnchored_MLP",
        "Anchored_RF",
        "Anchored_Ridge",
    }
    if not required_frozen.issubset(set(frozen.model)):
        raise ValueError("Frozen Stage 13 ensemble predictions are incomplete")
    combined = pd.concat([frozen, recent_ensemble], ignore_index=True)
    combined.to_csv(output_dir / "stage13b_all_method_ensemble_predictions.csv", index=False)
    summary = pd.DataFrame(
        [{"model": name, **metric(frame)} for name, frame in combined.groupby("model")]
    ).sort_values("mae_ah")
    summary.to_csv(output_dir / "stage13b_all_method_metrics.csv", index=False)

    combined["ae"] = (
        combined.prediction_capacity_ah - combined.future_capacity_ah
    ).abs()
    vehicle = combined.groupby(["model", "vehicle_key"]).ae.mean().unstack("model")
    rng = np.random.default_rng(42)
    paired: list[dict[str, Any]] = []
    proposed = "PersistenceAnchored_KAN"
    challengers = [*MODEL_NAMES.values(), "Persistence", "PersistenceAnchored_MLP"]
    for challenger in challengers:
        difference = (vehicle[proposed] - vehicle[challenger]).dropna().to_numpy()
        bootstrap = np.mean(
            rng.choice(difference, (50000, len(difference)), replace=True), axis=1
        )
        paired.append(
            {
                "proposed": proposed,
                "challenger": challenger,
                "vehicles": len(difference),
                "mean_delta_mae_ah": float(difference.mean()),
                "proposed_win_fraction": float(np.mean(difference < 0)),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
                "negative_favors_proposed": True,
            }
        )
    pd.DataFrame(paired).to_csv(output_dir / "stage13b_proposed_paired_bootstrap.csv", index=False)

    bands = [
        (0, 500, "<=500km"),
        (500, 1000, "500-1000km"),
        (1000, 2000, "1000-2000km"),
        (2000, np.inf, ">2000km"),
    ]
    horizon: list[dict[str, Any]] = []
    manufacturer: list[dict[str, Any]] = []
    for name, frame in combined.groupby("model"):
        for low, high, label in bands:
            part = frame[
                (frame.forecast_horizon_km > low)
                & (frame.forecast_horizon_km <= high)
            ]
            if len(part):
                horizon.append({"model": name, "horizon_band": label, **metric(part)})
        for manufacturer_id, part in frame.groupby("manufacturer_id"):
            manufacturer.append(
                {
                    "model": name,
                    "manufacturer_id": int(manufacturer_id),
                    **metric(part),
                }
            )
    pd.DataFrame(horizon).to_csv(output_dir / "stage13b_horizon_metrics.csv", index=False)
    pd.DataFrame(manufacturer).to_csv(
        output_dir / "stage13b_manufacturer_metrics.csv", index=False
    )
    audit = {
        "stage": "13B",
        "task": "recent_method_persistence_anchored_comparison",
        "frozen_stage13_predictions_reused": True,
        "new_methods": list(MODEL_NAMES.values()),
        "architecture_status": {
            "ModernTCN_2024": "architecture-faithful task adaptation",
            "TSLANet_2024": "architecture-faithful task adaptation",
            "KALinformer_2025": "architecture-faithful adaptation; no task-ready public code used",
        },
        "identical_information": True,
        "vehicle_disjoint": True,
        "folds": folds,
        "seeds": seeds,
        "transitions": len(table),
        "vehicles": int(table.vehicle_key.nunique()),
        "target_is_absolute_soh": False,
        "driver_telemetry_available": False,
        "source_cache_sha256": loaded["source_cache_sha256"],
        "references": {
            "ModernTCN": "https://openreview.net/forum?id=vpJMJerXHU",
            "TSLANet": "https://proceedings.mlr.press/v235/eldele24a.html",
            "KALinformer": "https://doi.org/10.1016/j.neucom.2025.130181",
        },
    }
    (output_dir / "STAGE13B_AUDIT.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    decision = {
        "lowest_mae_model": str(summary.iloc[0].model),
        "lowest_mae_ah": float(summary.iloc[0].mae_ah),
        "proposed_model": proposed,
        "proposed_mae_ah": float(summary.loc[summary.model.eq(proposed), "mae_ah"].iloc[0]),
        "claim_rule": "superiority requires paired vehicle ci_high below zero",
    }
    (output_dir / "STAGE13B_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    stamp("Stage 13B recent-method comparison complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--stage13-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    run_recent_validation(
        args.cache,
        args.stage13_results,
        args.output_dir,
        folds=args.folds,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
