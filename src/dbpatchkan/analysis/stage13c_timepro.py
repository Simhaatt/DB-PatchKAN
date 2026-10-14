"""Stage 13C post-hoc TimePro comparison on frozen EVBattery transitions.

TimePro (ICML 2025) is adapted to the established persistence-anchored task.
The official implementation uses custom CUDA selective-scan and DCNv4 kernels;
this reproducible adaptation preserves its defining ideas using pure PyTorch:

* independent temporal patching of each measured variable;
* bidirectional selective state-space processing over variable tokens;
* variable- and time-aware hyper-state mixing over the patch/variable grid;
* a signed residual added to the current-capacity persistence anchor.

This experiment was chosen after Stage 13B was inspected and must always be
reported as a post-hoc supplementary comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
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
from dbpatchkan.analysis.stage13b_recent_methods import ResidualMLP, parameter_count


MODEL_NAME = "PersistenceAnchored_TimePro_2025_POSTHOC"


class SelectiveStateSpace(nn.Module):
    """Pure-PyTorch selective SSM used in the TimePro adaptation."""

    def __init__(
        self,
        width: int,
        *,
        state_size: int = 8,
        conv_kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.width = width
        self.state_size = state_size
        self.norm = nn.LayerNorm(width)
        self.in_projection = nn.Linear(width, 2 * width, bias=False)
        self.depthwise = nn.Conv1d(
            width,
            width,
            conv_kernel,
            padding=conv_kernel // 2,
            groups=width,
        )
        self.delta_projection = nn.Linear(width, width)
        self.bc_projection = nn.Linear(width, 2 * state_size, bias=False)
        initial_a = torch.arange(1, state_size + 1, dtype=torch.float32)
        self.a_log = nn.Parameter(initial_a.log().repeat(width, 1))
        self.skip = nn.Parameter(torch.ones(width))
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        values, gate = self.in_projection(self.norm(tokens)).chunk(2, dim=-1)
        values = F.silu(self.depthwise(values.transpose(1, 2)).transpose(1, 2))
        delta = F.softplus(self.delta_projection(values).float())
        b_value, c_value = self.bc_projection(values).float().chunk(2, dim=-1)
        a_value = -torch.exp(self.a_log.float())
        state = torch.zeros(
            values.shape[0], self.width, self.state_size,
            device=values.device, dtype=torch.float32,
        )
        outputs: list[torch.Tensor] = []
        values32 = values.float()
        for index in range(values.shape[1]):
            step = delta[:, index]
            transition = torch.exp(step.unsqueeze(-1) * a_value.unsqueeze(0))
            drive = (
                step.unsqueeze(-1)
                * b_value[:, index].unsqueeze(1)
                * values32[:, index].unsqueeze(-1)
            )
            state = transition * state + drive
            observed = (state * c_value[:, index].unsqueeze(1)).sum(dim=-1)
            observed = observed + self.skip.float() * values32[:, index]
            outputs.append(observed)
        output = torch.stack(outputs, dim=1)
        output = self.output_norm(output) * F.silu(gate.float())
        return self.dropout(self.output_projection(output))


class VariableTimeHyperState(nn.Module):
    """Mix the flattened state on its original patch × variable geometry."""

    def __init__(
        self,
        patch_count: int,
        patch_width: int,
        variables: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_count = patch_count
        self.patch_width = patch_width
        self.variables = variables
        self.mixer = nn.Conv2d(
            patch_width,
            patch_width,
            kernel_size=3,
            padding=1,
            groups=patch_width,
        )
        self.channel_mix = nn.Conv2d(patch_width, patch_width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, variables, width = tokens.shape
        expected = self.patch_count * self.patch_width
        if variables != self.variables or width != expected:
            raise ValueError(
                f"Expected [B,{self.variables},{expected}], got {tuple(tokens.shape)}"
            )
        grid = tokens.reshape(
            batch, variables, self.patch_count, self.patch_width
        ).permute(0, 3, 2, 1)
        mixed = self.channel_mix(F.silu(self.mixer(grid)))
        restored = mixed.permute(0, 3, 2, 1).reshape(batch, variables, width)
        return tokens + self.dropout(restored)


class TimeProBlock(nn.Module):
    """Bidirectional selective hyper-state block plus feed-forward residual."""

    def __init__(
        self,
        width: int,
        patch_count: int,
        patch_width: int,
        variables: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.forward_ssm = SelectiveStateSpace(width, dropout=dropout)
        self.reverse_ssm = SelectiveStateSpace(width, dropout=dropout)
        self.hyper = VariableTimeHyperState(
            patch_count, patch_width, variables, dropout=dropout
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        forward = self.forward_ssm(tokens)
        reverse = self.reverse_ssm(tokens.flip(dims=(1,))).flip(dims=(1,))
        tokens = tokens + self.hyper(forward + reverse)
        return tokens + self.ffn(tokens)


class PersistenceAnchoredTimePro(nn.Module):
    """TimePro-style variable/time hyper-state encoder with residual head."""

    def __init__(
        self,
        *,
        variables: int = 8,
        sequence_length: int = 128,
        patch_length: int = 16,
        stride: int = 8,
        patch_width: int = 8,
        blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.variables = variables
        self.patch_length = patch_length
        self.stride = stride
        self.patch_width = patch_width
        self.patch_count = (sequence_length - patch_length) // stride + 2
        self.patch_projection = nn.Linear(patch_length, patch_width)
        width = self.patch_count * patch_width
        self.blocks = nn.ModuleList(
            TimeProBlock(
                width,
                self.patch_count,
                patch_width,
                variables,
                dropout=dropout,
            )
            for _ in range(blocks)
        )
        self.final_norm = nn.LayerNorm(width)
        self.variable_score = nn.Linear(width, 1)
        self.head = ResidualMLP(width + len(FORECAST_CONCEPTS), dropout=dropout)

    def forward(self, sequence: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
        # The official TimePro path uses per-sample non-stationary normalization.
        mean = sequence.mean(dim=1, keepdim=True).detach()
        scale = sequence.var(dim=1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        normalized = (sequence - mean) / scale
        by_variable = normalized.transpose(1, 2)
        padded = F.pad(by_variable, (0, self.stride))
        patches = padded.unfold(-1, self.patch_length, self.stride)
        embedded = self.patch_projection(patches)
        tokens = embedded.reshape(
            sequence.shape[0], self.variables, self.patch_count * self.patch_width
        )
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        weights = self.variable_score(tokens).squeeze(-1).softmax(dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return self.head(torch.cat((pooled, concepts.float()), dim=1))

    def regularization_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_zeros(())


def predict_timepro(
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


def train_timepro_fold(
    sequence: np.ndarray,
    concepts: np.ndarray,
    delta: np.ndarray,
    current: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    groups: np.ndarray,
    *,
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
    model = PersistenceAnchoredTimePro().to(device)
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
            features = torch.cat((physical_b, context_b), dim=1)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                residual = model(seq_b.to(device), features.to(device))
                beta = max(1e-3, 0.25 / scaling.delta_scale)
                regression = F.smooth_l1_loss(
                    residual.float(), y_b.to(device).float(), beta=beta
                )
                residual_ah = residual.float() * scaling.delta_scale + scaling.delta_center
                positive_penalty = torch.relu(residual_ah - 0.25).square().mean()
                loss = regression + 0.02 * positive_penalty
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = predict_timepro(model, val_loader, device, scaling)
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
            stamp(f"timepro seed={seed} epoch={epoch}: validation MAE={val_mae:.4f} Ah")
        if stale >= 12:
            break
    if best_state is None:
        raise RuntimeError(f"No TimePro checkpoint selected for seed {seed}")
    model.load_state_dict(best_state)
    validation = predict_timepro(model, val_loader, device, scaling)
    val_indices = np.asarray([item[0] for item in validation], dtype=int)
    val_future = np.asarray([item[2] for item in validation])
    truth = current[val_indices] + delta[val_indices]
    q90 = float(np.quantile(np.abs(truth - val_future), 0.90, method="higher"))
    elapsed = (time.perf_counter() - started) / 60.0
    return (
        predict_timepro(model, test_loader, device, scaling),
        pd.DataFrame(history),
        model,
        scaling,
        q90,
        params,
        elapsed,
    )


def run_timepro_validation(
    cache: str | Path,
    frozen_stage13b: str | Path,
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
            f"TimePro fold {fold}/{folds}: train vehicles={len(set(groups[train_idx]))}, "
            f"test vehicles={len(set(groups[test_idx]))}"
        )
        for seed in seeds:
            stamp(f"START TimePro, fold={fold}, seed={seed}")
            records, history, model, scaling, q90, params, elapsed = train_timepro_fold(
                sequence,
                concepts,
                delta,
                current,
                train_idx,
                test_idx,
                groups,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
            )
            indices = np.asarray([item[0] for item in records], dtype=int)
            prediction = np.asarray([item[2] for item in records])
            frame = pd.DataFrame(
                {
                    "transition_id": table.iloc[indices].transition_id.to_numpy(),
                    "fold": fold,
                    "seed": seed,
                    "model": MODEL_NAME,
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
                    "model": MODEL_NAME,
                    "parameter_count": params,
                    "train_time_min": elapsed,
                    **metric(frame),
                }
            )
            run = output_dir / "models" / f"fold_{fold}" / f"timepro_seed_{seed}"
            run.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "scaling": asdict(scaling),
                    "q90_ah": q90,
                    "concepts": FORECAST_CONCEPTS,
                    "model_name": MODEL_NAME,
                    "target": "next_distinct_mileage_partial_capacity_ah",
                    "posthoc": True,
                },
                run / "model.pt",
            )
            history.to_csv(run / "history.csv", index=False)
            stamp(
                f"DONE TimePro, fold={fold}, seed={seed}, parameters={params:,}, "
                f"elapsed={elapsed:.2f} min"
            )

    raw = pd.concat(predictions, ignore_index=True)
    raw.to_csv(output_dir / "stage13c_timepro_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(
        output_dir / "stage13c_timepro_fold_metrics.csv", index=False
    )
    keys = [
        "transition_id", "fold", "manufacturer_id", "vehicle_id", "vehicle_key",
        "forecast_horizon_km", "current_capacity_ah", "future_capacity_ah", "model",
    ]
    ensemble = raw.groupby(keys, as_index=False).agg(
        prediction_capacity_ah=("prediction_capacity_ah", "mean"),
        interval_low_ah=("interval_low_ah", "mean"),
        interval_high_ah=("interval_high_ah", "mean"),
    )
    ensemble.to_csv(output_dir / "stage13c_timepro_ensemble_predictions.csv", index=False)

    frozen_path = Path(frozen_stage13b) / "stage13b_all_method_ensemble_predictions.csv"
    frozen = pd.read_csv(frozen_path)
    required = {
        "Persistence", "PersistenceAnchored_KAN", "PersistenceAnchored_TSLANet_2024",
        "PersistenceAnchored_ModernTCN_2024", "PersistenceAnchored_KALinformer_2025",
    }
    if not required.issubset(set(frozen.model)):
        raise ValueError("Frozen Stage 13B predictions are incomplete")
    combined = pd.concat([frozen, ensemble], ignore_index=True)
    combined.to_csv(output_dir / "stage13c_all_method_ensemble_predictions.csv", index=False)
    summary = pd.DataFrame(
        [{"model": name, **metric(frame)} for name, frame in combined.groupby("model")]
    ).sort_values("mae_ah")
    summary.to_csv(output_dir / "stage13c_all_method_metrics.csv", index=False)

    combined["ae"] = (
        combined.prediction_capacity_ah - combined.future_capacity_ah
    ).abs()
    vehicle = combined.groupby(["model", "vehicle_key"]).ae.mean().unstack("model")
    pairs = [
        ("PersistenceAnchored_KAN", MODEL_NAME),
        ("PersistenceAnchored_TSLANet_2024", MODEL_NAME),
        (MODEL_NAME, "Persistence"),
    ]
    rng = np.random.default_rng(42)
    paired: list[dict[str, Any]] = []
    for reference, challenger in pairs:
        difference = (vehicle[reference] - vehicle[challenger]).dropna().to_numpy()
        bootstrap = np.mean(
            rng.choice(difference, (50000, len(difference)), replace=True), axis=1
        )
        paired.append(
            {
                "reference": reference,
                "challenger": challenger,
                "difference_definition": "reference_mae_minus_challenger_mae",
                "vehicles": len(difference),
                "mean_delta_mae_ah": float(difference.mean()),
                "reference_win_fraction": float(np.mean(difference < 0)),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
            }
        )
    pd.DataFrame(paired).to_csv(output_dir / "stage13c_paired_bootstrap.csv", index=False)

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
                {"model": name, "manufacturer_id": int(manufacturer_id), **metric(part)}
            )
    pd.DataFrame(horizon).to_csv(output_dir / "stage13c_horizon_metrics.csv", index=False)
    pd.DataFrame(manufacturer).to_csv(
        output_dir / "stage13c_manufacturer_metrics.csv", index=False
    )

    audit = {
        "stage": "13C",
        "task": "posthoc_timepro_state_space_comparison",
        "posthoc_after_stage13b": True,
        "frozen_stage13b_predictions_reused": True,
        "method": MODEL_NAME,
        "architecture_status": "pure-PyTorch architecture-faithful task adaptation",
        "official_custom_cuda_or_dcnv4_used": False,
        "identical_information": True,
        "vehicle_disjoint": True,
        "folds": folds,
        "seeds": seeds,
        "transitions": len(table),
        "vehicles": int(table.vehicle_key.nunique()),
        "target_is_absolute_soh": False,
        "driver_telemetry_available": False,
        "source_cache_sha256": loaded["source_cache_sha256"],
        "reference": "https://openreview.net/forum?id=s69Ei2VrIW",
        "official_code": "https://github.com/xwmaxwma/TimePro",
    }
    (output_dir / "STAGE13C_AUDIT.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    timepro_mae = float(summary.loc[summary.model.eq(MODEL_NAME), "mae_ah"].iloc[0])
    decision = {
        "posthoc": True,
        "lowest_mae_model": str(summary.iloc[0].model),
        "lowest_mae_ah": float(summary.iloc[0].mae_ah),
        "timepro_mae_ah": timepro_mae,
        "claim_rule": "paired interval required; result remains supplementary post-hoc",
    }
    (output_dir / "STAGE13C_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    stamp("Stage 13C post-hoc TimePro comparison complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--frozen-stage13b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    run_timepro_validation(
        args.cache,
        args.frozen_stage13b,
        args.output_dir,
        folds=args.folds,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
