"""Post-locked development-only hyperparameter robustness analysis.

This stage does not select or replace the frozen Stage 8 model.  It evaluates
local perturbations of three scientifically important settings on the original
87/14 chronological development split:

* available temporal context: 128, 192, and 256 seconds;
* concept-supervision weight: 0.10, 0.20, and 0.40;
* physics-consistency weight: 0.00, 0.05, 0.10, and 0.20.

The locked 26-trip test is neither accepted nor packaged.  Every result is
labelled post-hoc robustness evidence and cannot be used to refit the final
locked-test model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dbpatchkan.analysis.stage6 import (
    ConceptView,
    TARGETS,
    _concept_names,
    _metrics,
    _outer_predictions,
    trip_order,
    write_trip_subset,
)


REFERENCE_NAME = "FrozenReference_ctx256_concept0.20_physics0.10"


def update(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SensitivitySetting:
    name: str
    family: str
    segment_length: int = 256
    concept_weight: float = 0.20
    physics_weight: float = 0.10


SETTINGS = (
    SensitivitySetting(REFERENCE_NAME, "reference"),
    SensitivitySetting("Context_128s", "context", segment_length=128),
    SensitivitySetting("Context_192s", "context", segment_length=192),
    SensitivitySetting("ConceptWeight_0.10", "concept_weight", concept_weight=0.10),
    SensitivitySetting("ConceptWeight_0.40", "concept_weight", concept_weight=0.40),
    SensitivitySetting("PhysicsWeight_0.00", "physics_weight", physics_weight=0.00),
    SensitivitySetting("PhysicsWeight_0.05", "physics_weight", physics_weight=0.05),
    SensitivitySetting("PhysicsWeight_0.20", "physics_weight", physics_weight=0.20),
)


TIME_ARRAYS = {
    "driver",
    "battery",
    "driver_mask",
    "battery_mask",
    "driver_missing",
    "battery_missing",
    "validity_mask",
    "time_delta_s",
    "dt_s",
    "relative_position",
}


def _crop_npz(source: Path, destination: Path, length: int) -> Path:
    """Create a deterministic centred view of native 256-second windows."""

    if length not in {128, 192, 256}:
        raise ValueError("Only native or shorter contexts 128/192/256 are valid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    native = int(arrays["driver"].shape[1])
    if native != 256:
        raise ValueError(f"Expected native 256-second windows, found {native}")
    start = (native - length) // 2
    stop = start + length
    output: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        if name in TIME_ARRAYS and values.ndim >= 2 and values.shape[1] == native:
            output[name] = values[:, start:stop].copy()
        else:
            output[name] = values
    if "relative_position" in output:
        positions = np.linspace(0.0, 1.0, length, dtype=np.float32)
        output["relative_position"] = np.broadcast_to(
            positions, output["relative_position"].shape
        ).copy()
    np.savez_compressed(destination, **output)
    return destination


def _prepare_data(development: Path, output: Path) -> dict[int, dict[str, Path]]:
    ordered, _ = trip_order(development)
    if len(ordered) != 101:
        raise ValueError(f"Expected 101 development trips, found {len(ordered)}")
    train_trips, validation_trips = ordered[:87], ordered[87:]
    prepared: dict[int, dict[str, Path]] = {}
    for length in (128, 192, 256):
        context = _crop_npz(
            development,
            output / "context_data" / f"development_ctx{length}.npz",
            length,
        )
        folder = output / "split_data" / f"ctx{length}"
        prepared[length] = {
            "train": write_trip_subset(context, folder / "train_87.npz", train_trips),
            "validation": write_trip_subset(
                context, folder / "validation_14.npz", validation_trips
            ),
        }
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_sha256": sha256(development),
        "development_trips": 101,
        "training_trips": 87,
        "validation_trips": 14,
        "locked_test_used": False,
        "post_locked_sensitivity": True,
        "selection_permitted": False,
        "contexts_seconds": [128, 192, 256],
        "context_method": "deterministic centred crop of native 256-second windows",
        "why_no_512": (
            "The immutable export contains native 256-second windows. Joining overlapping "
            "windows would duplicate observations and alter the sampling distribution."
        ),
        "settings": [asdict(setting) for setting in SETTINGS],
    }
    (output / "STAGE14_PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    return prepared


def _seed(seed: int) -> None:
    from dbpatchkan.training.train import seed_everything

    seed_everything(seed)


def _model_config(segment_length: int):
    from dbpatchkan.models import DBPatchKANConfig

    return DBPatchKANConfig(
        driver_channels=13,
        battery_channels=11,
        segment_length=segment_length,
        patch_length=16,
        patch_stride=8,
        d_model=128,
        encoder_layers=3,
        n_heads=4,
        ff_dim=256,
        dropout=0.10,
        fusion_head="kan",
        input_modality="both",
        cross_attention_mode="bidirectional",
        prediction_input="concepts",
        concept_names=_concept_names("total"),
        kan_grid_size=5,
        kan_hidden=(16, 8),
        mlp_hidden=(32, 16),
    )


def _training_config(
    *,
    seed: int,
    stage: str,
    pretrain_epochs: int,
    finetune_epochs: int,
    concept_weight: float,
    physics_weight: float,
):
    from dbpatchkan.training.train import TrainingConfig

    return TrainingConfig(
        stage=stage,
        device="cuda",
        precision="bf16",
        pretrain_batch_size=256,
        finetune_batch_size=2,
        gradient_accumulation=1,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        patience=8,
        pretrain_lr=3e-4,
        encoder_lr=1e-5,
        head_lr=3e-4,
        weight_decay=0.01,
        warmup_fraction=0.05,
        gradient_clip=1.0,
        mask_ratio=0.4,
        num_workers=2,
        seed=seed,
        domain="nissan",
        freeze_encoders=True,
        trip_bags=True,
        concept_weight=concept_weight,
        physics_weight=physics_weight,
        driver_id_weight=0.1,
        ssl_alignment_weight=0.1,
    )


def _datasets(train_path: Path, validation_path: Path):
    from dbpatchkan.models import EVTripBagDataset

    return (
        ConceptView(EVTripBagDataset.from_npz(train_path), "total"),
        ConceptView(EVTripBagDataset.from_npz(validation_path), "total"),
    )


def _load_shared(model, checkpoint: Path) -> None:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    current = model.state_dict()
    prefixes = (
        "driver_encoder",
        "battery_encoder",
        "driver_pool",
        "battery_pool",
        "driver_reconstruction",
        "battery_reconstruction",
    )
    reusable = {
        name: value
        for name, value in state.items()
        if name.startswith(prefixes)
        and name in current
        and current[name].shape == value.shape
    }
    model.load_state_dict(reusable, strict=False)


def _pretrain(
    train_path: Path,
    validation_path: Path,
    *,
    output: Path,
    segment_length: int,
    seed: int,
    epochs: int,
) -> Path:
    from dbpatchkan.models import DBPatchKAN
    from dbpatchkan.training.train import train

    checkpoint = output / "pretrain_latest.pt"
    if checkpoint.exists():
        update(f"Shared pretraining already present: context={segment_length}, seed={seed}")
        return checkpoint
    output.mkdir(parents=True, exist_ok=True)
    train_data, validation_data = _datasets(train_path, validation_path)
    _seed(seed)
    model = DBPatchKAN(_model_config(segment_length))
    config = _training_config(
        seed=seed,
        stage="pretrain",
        pretrain_epochs=epochs,
        finetune_epochs=1,
        concept_weight=0.2,
        physics_weight=0.1,
    )
    update(
        f"START shared pretraining: context={segment_length}s, seed={seed}, epochs={epochs}"
    )
    train(model, train_data, validation_data, config, output_dir=output)
    if not checkpoint.exists():
        raise RuntimeError("Pretraining checkpoint was not created")
    return checkpoint


def _fit(
    train_path: Path,
    validation_path: Path,
    *,
    setting: SensitivitySetting,
    seed: int,
    shared: Path,
    output: Path,
    epochs: int,
) -> tuple[Path, dict[str, Any]]:
    import torch
    from dbpatchkan.models import DBPatchKAN
    from dbpatchkan.training.train import train

    lightweight = output / "model_lightweight.pt"
    complete = output / "complete.json"
    if lightweight.exists() and complete.exists():
        return lightweight, json.loads(complete.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    train_data, validation_data = _datasets(train_path, validation_path)
    _seed(seed)
    model = DBPatchKAN(_model_config(setting.segment_length))
    _load_shared(model, shared)
    config = _training_config(
        seed=seed,
        stage="finetune",
        pretrain_epochs=1,
        finetune_epochs=epochs,
        concept_weight=setting.concept_weight,
        physics_weight=setting.physics_weight,
    )
    result = train(model, train_data, validation_data, config, output_dir=output)
    state = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    payload = {
        "model_state": state["model"],
        "model_config": state["model_config"],
        "training_config": state["training_config"],
        "best_epoch": int(state["epoch"]),
        "best_metrics": state["metrics"],
        "setting": asdict(setting),
        "seed": seed,
        "development_only": True,
        "post_locked_sensitivity": True,
    }
    torch.save(payload, lightweight)
    result["best_epoch"] = int(state["epoch"])
    result["checkpoint"] = str(lightweight.resolve())
    complete.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for redundant in (output / "best.pt", output / "latest.pt"):
        if redundant.exists():
            redundant.unlink()
    return lightweight, result


def _tuple_config(values: dict[str, Any]) -> dict[str, Any]:
    result = dict(values)
    for key in (
        "concept_names",
        "kan_hidden",
        "mlp_hidden",
        "protected_driver_channels",
        "protected_battery_channels",
        "calibration_domains",
    ):
        if key in result:
            result[key] = tuple(result[key])
    return result


def _evaluate(checkpoint: Path, data_path: Path) -> pd.DataFrame:
    import torch
    from dbpatchkan.models import DBPatchKAN, DBPatchKANConfig, EVTripBagDataset

    device = torch.device("cuda")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DBPatchKAN(DBPatchKANConfig(**_tuple_config(state["model_config"])))
    model.load_state_dict(state["model_state"])
    model.to(device)
    dataset = ConceptView(EVTripBagDataset.from_npz(data_path), "total")
    return _outer_predictions(model, dataset, device, "bf16")


def _metric_rows(
    frame: pd.DataFrame,
    *,
    setting: SensitivitySetting,
    seed: int,
    best_epoch: int,
    parameters: int,
    elapsed_minutes: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        metric = _metrics(
            frame[f"{target}_observed"].to_numpy(float),
            frame[f"{target}_predicted"].to_numpy(float),
        )
        rows.append(
            {
                **asdict(setting),
                "seed": seed,
                "target": target,
                "best_epoch": best_epoch,
                "parameter_count": parameters,
                "elapsed_minutes": elapsed_minutes,
                **metric,
            }
        )
    return rows


def _prediction_rows(
    frame: pd.DataFrame, *, setting: SensitivitySetting, seed: int
) -> list[pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    for target in TARGETS:
        part = frame[
            ["trip_key", f"{target}_observed", f"{target}_predicted"]
        ].copy()
        part.columns = ["trip_key", "observed", "predicted"]
        part.insert(0, "target", target)
        part.insert(0, "seed", seed)
        part.insert(0, "family", setting.family)
        part.insert(0, "setting", setting.name)
        outputs.append(part)
    return outputs


def _paired_bootstrap(
    predictions: pd.DataFrame, *, repeats: int = 4000, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    reference = predictions[predictions.setting.eq(REFERENCE_NAME)]
    for target in TARGETS:
        ref = reference[reference.target.eq(target)].set_index(["seed", "trip_key"])
        for setting in sorted(set(predictions.setting) - {REFERENCE_NAME}):
            alt = predictions[
                predictions.setting.eq(setting) & predictions.target.eq(target)
            ].set_index(["seed", "trip_key"])
            joined = ref[["observed", "predicted"]].join(
                alt[["predicted"]], rsuffix="_alt", how="inner"
            ).dropna()
            if len(joined) < 3:
                rows.append(
                    {
                        "target": target,
                        "reference": REFERENCE_NAME,
                        "challenger": setting,
                        "n_seed_trip_pairs": len(joined),
                        "delta_mae": math.nan,
                        "ci_low": math.nan,
                        "ci_high": math.nan,
                        "status": "not_estimable",
                    }
                )
                continue
            joined = joined.reset_index()
            joined["delta"] = (
                np.abs(joined.predicted_alt - joined.observed)
                - np.abs(joined.predicted - joined.observed)
            )
            per_trip = joined.groupby("trip_key", as_index=False).delta.mean()
            values = per_trip.delta.to_numpy(float)
            draws = np.asarray(
                [rng.choice(values, len(values), replace=True).mean() for _ in range(repeats)]
            )
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "target": target,
                    "reference": REFERENCE_NAME,
                    "challenger": setting,
                    "n_seed_trip_pairs": len(joined),
                    "n_unique_trips": len(values),
                    "delta_mae": float(values.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "challenger_significantly_better": bool(high < 0),
                    "reference_significantly_better": bool(low > 0),
                    "statistically_equivocal": bool(low <= 0 <= high),
                    "status": "complete",
                }
            )
    return pd.DataFrame(rows)


def _summaries(metrics: pd.DataFrame, predictions: pd.DataFrame, output: Path) -> None:
    summary = (
        metrics.groupby(
            ["name", "family", "segment_length", "concept_weight", "physics_weight", "target"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            n_mean=("n", "mean"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            bias_mean=("bias", "mean"),
            r2_mean=("r2", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            elapsed_minutes_mean=("elapsed_minutes", "mean"),
        )
        .sort_values(["target", "mae_mean"])
    )
    summary.to_csv(output / "stage14_sensitivity_summary.csv", index=False)
    paired = _paired_bootstrap(predictions)
    paired.to_csv(output / "stage14_paired_bootstrap.csv", index=False)

    reference = summary[summary.name.eq(REFERENCE_NAME)][["target", "mae_mean"]].rename(
        columns={"mae_mean": "reference_mae"}
    )
    robustness = summary.merge(reference, on="target", how="left")
    robustness["relative_mae_change_pct"] = 100.0 * (
        robustness.mae_mean - robustness.reference_mae
    ) / robustness.reference_mae.replace(0, np.nan)
    robustness["within_5pct_of_reference"] = (
        robustness.relative_mae_change_pct.abs() <= 5.0
    )
    robustness.to_csv(output / "stage14_robustness_table.csv", index=False)

    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post_locked_development_only_sensitivity_complete",
        "frozen_model_changed": False,
        "locked_test_used": False,
        "selection_permitted": False,
        "reference": REFERENCE_NAME,
        "interpretation_rule": (
            "Use this stage only to quantify local robustness. It cannot replace the "
            "pre-registered frozen model or modify locked-test conclusions."
        ),
        "settings_within_5pct_by_target": {
            target: sorted(
                robustness[
                    robustness.target.eq(target) & robustness.within_5pct_of_reference
                ].name.tolist()
            )
            for target in TARGETS
        },
    }
    (output / "STAGE14_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )


def _figures(output: Path) -> None:
    import matplotlib.pyplot as plt

    summary = pd.read_csv(output / "stage14_sensitivity_summary.csv")
    display_targets = {
        "current_soh": "Current SOH",
        "loss_30d": "30-day SOH loss",
        "loss_500km": "500-km SOH loss",
    }
    figure_root = output / "figures"
    figure_root.mkdir(exist_ok=True)
    for family, filename, title in (
        ("context", "context_length_sensitivity.png", "Available-context sensitivity"),
        ("concept_weight", "concept_weight_sensitivity.png", "Concept-loss sensitivity"),
        ("physics_weight", "physics_weight_sensitivity.png", "Physics-loss sensitivity"),
    ):
        subset = summary[summary.family.isin(("reference", family))].copy()
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
        for axis, target in zip(axes, TARGETS, strict=True):
            part = subset[subset.target.eq(target)].copy()
            if family == "context":
                part = part.sort_values("segment_length")
                x = part.segment_length.astype(str)
                xlabel = "Context (s)"
            elif family == "concept_weight":
                part = part.sort_values("concept_weight")
                x = part.concept_weight.map(lambda value: f"{value:.2f}")
                xlabel = r"$\lambda_{concept}$"
            else:
                part = part.sort_values("physics_weight")
                x = part.physics_weight.map(lambda value: f"{value:.2f}")
                xlabel = r"$\lambda_{physics}$"
            axis.errorbar(
                x,
                part.mae_mean,
                yerr=part.mae_std.fillna(0),
                marker="o",
                capsize=4,
                color="#3559a6",
            )
            axis.set_title(display_targets[target])
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Validation MAE")
            axis.grid(alpha=0.25)
        fig.suptitle(title + " (three seeds; development validation only)")
        fig.savefig(figure_root / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)


def run(
    development: Path,
    output: Path,
    *,
    seeds: list[int],
    pretrain_epochs: int,
    finetune_epochs: int,
) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 14 requires a CUDA GPU")
    if "test" in development.name.casefold() or "untouched" in development.name.casefold():
        raise ValueError("Stage 14 refuses test/untouched inputs")
    output.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_data(development, output)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    registry: list[dict[str, Any]] = []
    total = len(seeds) * len(SETTINGS)
    completed = 0
    started_all = time.time()
    for seed_index, seed in enumerate(seeds, start=1):
        update(f"SEED {seed_index}/{len(seeds)} START: {seed}")
        shared_by_context: dict[int, Path] = {}
        for length in (128, 192, 256):
            paths = prepared[length]
            shared_by_context[length] = _pretrain(
                paths["train"],
                paths["validation"],
                output=output / "working_pretraining" / f"seed_{seed}" / f"ctx{length}",
                segment_length=length,
                seed=seed,
                epochs=pretrain_epochs,
            )
        for setting in SETTINGS:
            completed += 1
            run_started = time.time()
            update(
                f"FIT {completed}/{total}: {setting.name}, seed={seed}, "
                f"context={setting.segment_length}, concept={setting.concept_weight:.2f}, "
                f"physics={setting.physics_weight:.2f}"
            )
            paths = prepared[setting.segment_length]
            model_root = output / "models" / setting.name / f"seed_{seed}"
            checkpoint, result = _fit(
                paths["train"],
                paths["validation"],
                setting=setting,
                seed=seed,
                shared=shared_by_context[setting.segment_length],
                output=model_root,
                epochs=finetune_epochs,
            )
            frame = _evaluate(checkpoint, paths["validation"])
            elapsed = (time.time() - run_started) / 60.0
            metric_rows.extend(
                _metric_rows(
                    frame,
                    setting=setting,
                    seed=seed,
                    best_epoch=int(result["best_epoch"]),
                    parameters=int(result["parameter_count"]),
                    elapsed_minutes=elapsed,
                )
            )
            prediction_frames.extend(_prediction_rows(frame, setting=setting, seed=seed))
            registry.append(
                {
                    "setting": asdict(setting),
                    "seed": seed,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256(checkpoint),
                    "best_epoch": int(result["best_epoch"]),
                    "parameter_count": int(result["parameter_count"]),
                    "elapsed_minutes": elapsed,
                }
            )
            update(f"DONE {completed}/{total}: {setting.name}, seed={seed}, {elapsed:.2f} min")
        pretrain_root = output / "working_pretraining" / f"seed_{seed}"
        if pretrain_root.exists():
            shutil.rmtree(pretrain_root)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(output / "stage14_seed_metrics.csv", index=False)
    predictions.to_csv(output / "stage14_validation_predictions.csv", index=False)
    (output / "STAGE14_MODEL_REGISTRY.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "models": registry,
                "expected_models": total,
                "locked_test_used": False,
                "post_locked_sensitivity": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _summaries(metrics, predictions, output)
    _figures(output)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "seeds": seeds,
        "fits_completed": len(registry),
        "expected_fits": total,
        "pretrain_epochs": pretrain_epochs,
        "finetune_epochs": finetune_epochs,
        "elapsed_minutes": (time.time() - started_all) / 60.0,
        "locked_test_used": False,
        "frozen_configuration_changed": False,
    }
    (output / "STAGE14_AUDIT.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    update(
        f"Stage 14 complete: {len(registry)}/{total} fits in "
        f"{audit['elapsed_minutes']:.1f} min"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.development_data,
        args.output_dir,
        seeds=list(args.seeds),
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
    )


if __name__ == "__main__":
    main()
