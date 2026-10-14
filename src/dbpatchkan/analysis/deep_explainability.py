"""Concept-KAN and sequence-occlusion explanations for frozen DB-PatchKAN models.

The helpers in this module never fit a model.  They operate on restored,
validation-selected checkpoints and keep attention weights separate from feature
importance.  Multilayer KAN explanations include both graph-edge/path strength
and full-head one-concept response curves; the latter capture downstream
nonlinearities that a first-layer spline plot alone would miss.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dbpatchkan.models import BATTERY_CHANNELS, DRIVER_CHANNELS, collate_trip_bags


OUTPUT_NAMES = ("current_soh", "loss_30d", "loss_500km")


def _kan_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    layers = getattr(getattr(model, "fusion_head", None), "layers", None)
    if layers is None:
        raise TypeError("The model does not have a spline ConceptKAN fusion head")
    return list(layers)


def _edge_strength(layer: torch.nn.Module) -> tuple[Tensor, Tensor, Tensor]:
    mask = layer.edge_mask.detach().float().cpu()
    base = layer.base_weight.detach().abs().float().cpu() * mask
    spline = layer.spline_weight.detach().abs().float().mean(dim=-1).cpu() * mask
    return base, spline, base + spline


def kan_edge_strength_table(
    model: torch.nn.Module,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return every active/inactive KAN graph edge with auditable strengths."""

    layers = _kan_layers(model)
    concept_names = tuple(model.config.concept_names)
    rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        base, spline, total = _edge_strength(layer)
        input_names = (
            concept_names
            if layer_index == 0
            else tuple(f"hidden_{layer_index}_{i}" for i in range(layer.in_features))
        )
        output_names = (
            OUTPUT_NAMES
            if layer_index == len(layers) - 1 and layer.out_features == len(OUTPUT_NAMES)
            else tuple(f"hidden_{layer_index + 1}_{i}" for i in range(layer.out_features))
        )
        for output_index, output_name in enumerate(output_names):
            for input_index, input_name in enumerate(input_names):
                rows.append(
                    {
                        "Seed": seed,
                        "Layer": layer_index,
                        "Input_Index": input_index,
                        "Input_Name": input_name,
                        "Output_Index": output_index,
                        "Output_Name": output_name,
                        "Base_Strength": float(base[output_index, input_index]),
                        "Spline_Strength": float(spline[output_index, input_index]),
                        "Total_Strength": float(total[output_index, input_index]),
                        "Active": bool(layer.edge_mask[output_index, input_index].item()),
                    }
                )
    return pd.DataFrame(rows)


def kan_path_importance_table(
    model: torch.nn.Module,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Propagate non-negative edge strength to each named concept/output pair."""

    matrices = [_edge_strength(layer)[2].double() for layer in _kan_layers(model)]
    propagated = matrices[0]
    for matrix in matrices[1:]:
        propagated = matrix @ propagated
    if propagated.shape[0] != len(OUTPUT_NAMES):
        raise ValueError("KAN final layer does not match the three regression outputs")
    rows: list[dict[str, Any]] = []
    concept_names = tuple(model.config.concept_names)
    for output_index, output_name in enumerate(OUTPUT_NAMES):
        denominator = float(propagated[output_index].sum().clamp_min(1.0e-12))
        for concept_index, concept_name in enumerate(concept_names):
            strength = float(propagated[output_index, concept_index])
            rows.append(
                {
                    "Seed": seed,
                    "Output_Name": output_name,
                    "Concept_Index": concept_index,
                    "Concept_Name": concept_name,
                    "Path_Strength": strength,
                    "Normalized_Path_Importance": strength / denominator,
                }
            )
    return pd.DataFrame(rows)


def _head_outputs_from_normalized(
    model: torch.nn.Module,
    normalized_concepts: Tensor,
    *,
    domain: str,
) -> dict[str, Tensor]:
    raw = model.fusion_head(normalized_concepts.float())
    if "last_known_soh" in model.config.concept_names:
        index = model.config.concept_names.index("last_known_soh")
        current_input = normalized_concepts.clone()
        current_input[:, index] = 0.0
        current_raw = model.fusion_head(current_input.float())[:, 0]
        raw = torch.cat((current_raw.unsqueeze(-1), raw[:, 1:]), dim=-1)
    current, loss_30d, loss_500km = model.calibration[domain](raw)
    return {
        "current_soh": current,
        "loss_30d": loss_30d,
        "loss_500km": loss_500km,
    }


@torch.no_grad()
def concept_response_curves(
    model: torch.nn.Module,
    concept_ranges: Mapping[str, tuple[float, float]],
    *,
    points: int = 101,
    domain: str = "nissan",
    seed: int | None = None,
) -> pd.DataFrame:
    """Evaluate the full KAN head while varying one physical concept at a time."""

    if points < 3:
        raise ValueError("points must be at least three")
    device = next(model.parameters()).device
    center = model.concept_center.detach().to(device=device, dtype=torch.float32)
    scale = model.concept_scale.detach().to(device=device, dtype=torch.float32).clamp_min(1e-6)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(model.config.concept_names):
        if name not in concept_ranges:
            raise KeyError(f"Missing response range for concept {name!r}")
        lower, upper = map(float, concept_ranges[name])
        values = torch.linspace(lower, upper, points, device=device)
        normalized = torch.zeros(points, model.config.n_concepts, device=device)
        normalized[:, index] = (values - center[index]) / scale[index]
        outputs = _head_outputs_from_normalized(model, normalized, domain=domain)
        for output_name, output_values in outputs.items():
            for value, response in zip(values.cpu(), output_values.detach().cpu()):
                rows.append(
                    {
                        "Seed": seed,
                        "Concept_Index": index,
                        "Concept_Name": name,
                        "Concept_Value": float(value),
                        "Output_Name": output_name,
                        "Response": float(response),
                        "Held_Constant": "all_other_concepts_at_training_center",
                    }
                )
    return pd.DataFrame(rows)


def aggregate_response_curves(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate cross-seed curves and audit response direction stability."""

    required = {"Seed", "Concept_Name", "Concept_Value", "Output_Name", "Response"}
    missing = required - set(curves)
    if missing:
        raise ValueError(f"Response curves are missing {sorted(missing)}")
    aggregate = (
        curves.groupby(["Concept_Name", "Concept_Value", "Output_Name"], as_index=False)
        .agg(
            Response_Mean=("Response", "mean"),
            Response_Lower=("Response", lambda x: float(np.quantile(x, 0.025))),
            Response_Upper=("Response", lambda x: float(np.quantile(x, 0.975))),
            Seed_Count=("Seed", "nunique"),
        )
    )
    seed_rows: list[dict[str, Any]] = []
    for (seed, concept, output), subset in curves.groupby(
        ["Seed", "Concept_Name", "Output_Name"], sort=False
    ):
        ordered = subset.sort_values("Concept_Value")
        x_rank = ordered["Concept_Value"].rank(method="average").to_numpy(float)
        y_rank = ordered["Response"].rank(method="average").to_numpy(float)
        correlation = float(np.corrcoef(x_rank, y_rank)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
        amplitude = float(ordered["Response"].max() - ordered["Response"].min())
        direction = "increasing" if correlation > 0.05 else "decreasing" if correlation < -0.05 else "flat"
        seed_rows.append(
            {
                "Seed": seed,
                "Concept_Name": concept,
                "Output_Name": output,
                "Spearman": correlation,
                "Absolute_Amplitude": amplitude,
                "Direction": direction,
            }
        )
    per_seed = pd.DataFrame(seed_rows)
    stability_rows: list[dict[str, Any]] = []
    for (concept, output), subset in per_seed.groupby(["Concept_Name", "Output_Name"]):
        counts = subset["Direction"].value_counts(normalize=True)
        majority = str(counts.index[0])
        agreement = float(counts.iloc[0])
        median_spearman = float(subset["Spearman"].median())
        stability_rows.append(
            {
                "Concept_Name": concept,
                "Output_Name": output,
                "Majority_Direction": majority,
                "Direction_Agreement": agreement,
                "Median_Spearman": median_spearman,
                "Median_Absolute_Amplitude": float(subset["Absolute_Amplitude"].median()),
                "Stable_Response": bool(agreement >= 0.80 and abs(median_spearman) >= 0.50),
            }
        )
    return aggregate, pd.DataFrame(stability_rows)


def concept_integrated_gradients(
    model: torch.nn.Module,
    concept_values: np.ndarray | Tensor,
    concept_mask: np.ndarray | Tensor | None = None,
    *,
    output_name: str = "current_soh",
    steps: int = 32,
    domain: str = "nissan",
    seed: int | None = None,
) -> pd.DataFrame:
    """Integrated gradients from the training-center concept baseline."""

    if output_name not in OUTPUT_NAMES:
        raise ValueError(f"Unknown output {output_name!r}")
    if steps < 2:
        raise ValueError("steps must be at least two")
    device = next(model.parameters()).device
    values = torch.as_tensor(concept_values, dtype=torch.float32, device=device)
    values = values[:, : model.config.n_concepts]
    center = model.concept_center.detach().to(device)
    scale = model.concept_scale.detach().to(device).clamp_min(1e-6)
    finite = torch.isfinite(values)
    if concept_mask is not None:
        finite &= torch.as_tensor(concept_mask, dtype=torch.bool, device=device)[
            :, : model.config.n_concepts
        ]
    values = torch.where(finite, values, center.unsqueeze(0))
    normalized_input = (values - center) / scale
    gradient_sum = torch.zeros_like(normalized_input)
    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    with torch.enable_grad():
        for alpha in alphas:
            point = (normalized_input * alpha).detach().requires_grad_(True)
            output = _head_outputs_from_normalized(model, point, domain=domain)[output_name]
            gradient = torch.autograd.grad(output.sum(), point, retain_graph=False)[0]
            gradient_sum += gradient.detach()
    attribution = normalized_input * gradient_sum / steps
    rows = []
    for index, name in enumerate(model.config.concept_names):
        column = attribution[:, index].detach().cpu().numpy()
        rows.append(
            {
                "Seed": seed,
                "Output_Name": output_name,
                "Concept_Index": index,
                "Concept_Name": name,
                "IG_Mean_Absolute": float(np.mean(np.abs(column))),
                "IG_Mean_Signed": float(np.mean(column)),
                "IG_Positive_Fraction": float(np.mean(column > 0)),
            }
        )
    return pd.DataFrame(rows)


def channel_gate_table(model: torch.nn.Module, *, seed: int | None = None) -> pd.DataFrame:
    """Return deterministic channel gates and stochastic inclusion probabilities."""

    rows: list[dict[str, Any]] = []
    streams = (
        ("driver", model.driver_encoder.gate, DRIVER_CHANNELS),
        ("battery", model.battery_encoder.gate, BATTERY_CHANNELS),
    )
    for stream, gate, canonical_names in streams:
        names = tuple(canonical_names[: gate.channels]) + tuple(
            f"{stream}_channel_{index}"
            for index in range(len(canonical_names), gate.channels)
        )
        probability = gate.probabilities().detach().cpu()
        deterministic = gate.deterministic_gates().detach().cpu()
        for index, name in enumerate(names):
            rows.append(
                {
                    "Seed": seed,
                    "Stream": stream,
                    "Channel_Index": index,
                    "Channel_Name": name,
                    "Gate_Probability": float(probability[index]),
                    "Deterministic_Gate": float(deterministic[index]),
                    "Protected": bool(gate.protected_mask[index].item()),
                    "Pruned": bool(gate.pruned_mask[index].item()),
                }
            )
    return pd.DataFrame(rows)


def _move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


@torch.no_grad()
def trip_predictions(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    device: torch.device | None = None,
    batch_size: int = 2,
    num_workers: int = 0,
    intervention: Mapping[str, Any] | None = None,
    use_observed_concepts: bool = True,
    domain: str = "nissan",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict trip-level SOH after one explicit validation-only intervention."""

    device = device or next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_trip_bags,
    )
    targets: list[Tensor] = []
    predictions: list[Tensor] = []
    intervention = dict(intervention or {})
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        driver = batch["driver"].clone()
        battery = batch["battery"].clone()
        concepts = batch.get("observed_concepts")
        concept_mask = batch.get("observed_concept_mask")
        if concepts is not None:
            concepts = concepts[:, : model.config.n_concepts].clone()
        if concept_mask is not None:
            concept_mask = concept_mask[:, : model.config.n_concepts].clone()
        elif concepts is not None:
            concept_mask = torch.isfinite(concepts)

        kind = intervention.get("kind")
        if kind == "concept":
            pass
        elif kind == "driver_channel":
            index = int(intervention["index"])
            driver[..., index] = model.driver_input_center[index]
        elif kind == "battery_channel":
            index = int(intervention["index"])
            battery[..., index] = model.battery_input_center[index]
        elif kind == "time_block":
            start = int(intervention["start"])
            stop = int(intervention["stop"])
            driver[..., start:stop, :9] = model.driver_input_center[:9]
            battery[..., start:stop, :7] = model.battery_input_center[:7]
        elif kind is not None:
            raise ValueError(f"Unknown intervention kind {kind!r}")

        outputs = model.forward_trip_bag(
            driver,
            battery,
            segment_mask=batch["segment_mask"],
            driver_mask=batch.get("driver_mask"),
            battery_mask=batch.get("battery_mask"),
            observed_concepts=concepts if use_observed_concepts else None,
            observed_concept_mask=concept_mask if use_observed_concepts else None,
            domain=domain,
        )
        if kind == "concept":
            index = int(intervention["index"])
            if concepts is None:
                concepts = torch.full_like(outputs["concepts"], float("nan"))
                concept_mask = torch.zeros_like(outputs["concepts"], dtype=torch.bool)
            concepts[:, index] = model.concept_center[index]
            concept_mask[:, index] = True
            outputs = model._predict_from_embedding(
                outputs["embedding"],
                outputs["driver_embedding"],
                observed_concepts=concepts,
                observed_concept_mask=concept_mask,
                domain=domain,
            )
        target = batch["current_soh_target"]
        valid = torch.isfinite(target) & torch.isfinite(outputs["current_soh"])
        targets.append(target[valid].detach().cpu())
        predictions.append(outputs["current_soh"][valid].detach().cpu())
    return torch.cat(targets).numpy(), torch.cat(predictions).numpy()


def validation_occlusion_tables(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    seed: int | None = None,
    device: torch.device | None = None,
    batch_size: int = 2,
    patch_length: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Concept ablation plus encoder-only channel and time-block occlusion."""

    device = device or next(model.parameters()).device
    target, prediction = trip_predictions(
        model, dataset, device=device, batch_size=batch_size, use_observed_concepts=True
    )
    baseline_mae = float(np.mean(np.abs(prediction - target)))
    concept_rows = []
    for index, name in enumerate(model.config.concept_names):
        if progress is not None:
            progress(f"concept ablation {index + 1}/{model.config.n_concepts}: {name}")
        actual, estimate = trip_predictions(
            model,
            dataset,
            device=device,
            batch_size=batch_size,
            intervention={"kind": "concept", "index": index},
            use_observed_concepts=True,
        )
        mae = float(np.mean(np.abs(estimate - actual)))
        concept_rows.append(
            {
                "Seed": seed,
                "Concept_Index": index,
                "Concept_Name": name,
                "Baseline_MAE_pp": baseline_mae,
                "Occluded_MAE_pp": mae,
                "Ablation_Delta_pp": mae - baseline_mae,
                "Pathway": "predicted_concept_overridden_at_training_center",
            }
        )

    target, prediction = trip_predictions(
        model, dataset, device=device, batch_size=batch_size, use_observed_concepts=False
    )
    encoder_baseline_mae = float(np.mean(np.abs(prediction - target)))
    channel_rows = []
    stream_specs = (
        ("driver", tuple(DRIVER_CHANNELS[: model.config.driver_channels])),
        ("battery", tuple(BATTERY_CHANNELS[: model.config.battery_channels])),
    )
    for stream, names in stream_specs:
        for index, name in enumerate(names):
            if progress is not None:
                progress(f"{stream} channel occlusion {index + 1}/{len(names)}: {name}")
            actual, estimate = trip_predictions(
                model,
                dataset,
                device=device,
                batch_size=batch_size,
                intervention={"kind": f"{stream}_channel", "index": index},
                use_observed_concepts=False,
            )
            mae = float(np.mean(np.abs(estimate - actual)))
            channel_rows.append(
                {
                    "Seed": seed,
                    "Stream": stream,
                    "Channel_Index": index,
                    "Channel_Name": name,
                    "Baseline_MAE_pp": encoder_baseline_mae,
                    "Occluded_MAE_pp": mae,
                    "Channel_Occlusion_Delta_pp": mae - encoder_baseline_mae,
                    "Pathway": "end_to_end_predicted_concept_pathway",
                }
            )

    width = int(patch_length or model.config.patch_length)
    patch_rows = []
    starts = list(range(0, model.config.segment_length, width))
    for patch_index, start in enumerate(starts):
        stop = min(model.config.segment_length, start + width)
        if progress is not None:
            progress(
                f"temporal block occlusion {patch_index + 1}/{len(starts)}: {start}-{stop}s"
            )
        actual, estimate = trip_predictions(
            model,
            dataset,
            device=device,
            batch_size=batch_size,
            intervention={"kind": "time_block", "start": start, "stop": stop},
            use_observed_concepts=False,
        )
        mae = float(np.mean(np.abs(estimate - actual)))
        patch_rows.append(
            {
                "Seed": seed,
                "Start_s": start,
                "Stop_s": stop,
                "Baseline_MAE_pp": encoder_baseline_mae,
                "Occluded_MAE_pp": mae,
                "Patch_Occlusion_Delta_pp": mae - encoder_baseline_mae,
                "Pathway": "end_to_end_predicted_concept_pathway",
            }
        )
    return {
        "concept": pd.DataFrame(concept_rows),
        "channel": pd.DataFrame(channel_rows),
        "patch": pd.DataFrame(patch_rows),
    }


__all__ = [
    "OUTPUT_NAMES",
    "aggregate_response_curves",
    "channel_gate_table",
    "concept_integrated_gradients",
    "concept_response_curves",
    "kan_edge_strength_table",
    "kan_path_importance_table",
    "trip_predictions",
    "validation_occlusion_tables",
]
