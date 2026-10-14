"""Final seed-controlled Nissan confirmation and one-time locked-test audit.

Development phase
-----------------
* corrected initialization-matched essential ablations on four chronological
  folds;
* five-seed frozen KAN/MLP training using 87 training and 14 validation trips;
* no loading of the locked test arrays.

Locked-test phase
-----------------
* requires the frozen-configuration artifact from the development phase;
* opens the 26-trip test exactly once for final inference;
* evaluates strict persistence, fixed XGBoost, KAN, and matched MLP;
* performs no hyperparameter selection or architecture changes.
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
    TARGET_KEYS,
    TARGETS,
    _concept_names,
    _metrics,
    _outer_predictions,
    _trip_rows,
    chronological_folds,
    trip_order,
    write_trip_subset,
)


SEEDS = (42, 52, 62, 72, 82)


def update(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


@dataclass(frozen=True)
class FinalVariant:
    name: str
    head: str = "kan"
    modality: str = "both"
    concept_set: str = "total"


FINAL_KAN = FinalVariant("Final_DBSI_Total_KAN")
FINAL_MLP = FinalVariant("Matched_DBSI_Total_MLP", head="mlp")
ESSENTIAL_ABLATIONS = (
    FINAL_KAN,
    FINAL_MLP,
    FinalVariant("Battery_Only_DBSI_Total_KAN", modality="battery"),
    FinalVariant("Driver_Only_DBSI_Total_KAN", modality="driver"),
    FinalVariant("No_DBSI_KAN", concept_set="none"),
    FinalVariant("DBSI_Components_Dose_KAN", concept_set="components_dose"),
)


def _seed(seed: int) -> None:
    from dbpatchkan.training.train import seed_everything

    seed_everything(seed)


def _model_config(variant: FinalVariant):
    from dbpatchkan.models import DBPatchKANConfig

    return DBPatchKANConfig(
        driver_channels=13,
        battery_channels=11,
        segment_length=256,
        patch_length=16,
        patch_stride=8,
        d_model=128,
        encoder_layers=3,
        n_heads=4,
        ff_dim=256,
        dropout=0.10,
        fusion_head=variant.head,
        input_modality=variant.modality,
        cross_attention_mode="bidirectional",
        prediction_input="concepts",
        concept_names=_concept_names(variant.concept_set),
        kan_grid_size=5,
        kan_hidden=(16, 8),
        mlp_hidden=(32, 16),
    )


def _tuple_config(values: dict[str, Any]) -> dict[str, Any]:
    result = dict(values)
    for name in (
        "concept_names", "kan_hidden", "mlp_hidden", "protected_driver_channels",
        "protected_battery_channels", "calibration_domains",
    ):
        if name in result:
            result[name] = tuple(result[name])
    return result


def _load_lightweight(path: Path, device):
    import torch
    from dbpatchkan.models import DBPatchKAN, DBPatchKANConfig

    state = torch.load(path, map_location=device, weights_only=False)
    model = DBPatchKAN(DBPatchKANConfig(**_tuple_config(state["model_config"])))
    model.load_state_dict(state["model_state"])
    return model.to(device), state


def _shared_prefixes() -> tuple[str, ...]:
    return (
        "driver_encoder",
        "battery_encoder",
        "driver_pool",
        "battery_pool",
        "driver_reconstruction",
        "battery_reconstruction",
    )


def _load_shared_pretraining(model, checkpoint: Path) -> None:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    current = model.state_dict()
    reusable = {
        name: value
        for name, value in state.items()
        if name.startswith(_shared_prefixes())
        and name in current
        and current[name].shape == value.shape
    }
    model.load_state_dict(reusable, strict=False)


def _training_config(
    *,
    seed: int,
    stage: str,
    pretrain_epochs: int,
    finetune_epochs: int,
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
        concept_weight=0.2,
        physics_weight=0.1,
        driver_id_weight=0.1,
        ssl_alignment_weight=0.1,
    )


def _prepare_development(dev: Path, output: Path) -> dict[str, Any]:
    ordered, _ = trip_order(dev)
    if len(ordered) != 101:
        raise ValueError(f"Expected 101 development trips, found {len(ordered)}")
    fold_specs = chronological_folds(ordered)
    registry = []
    for fold in fold_specs:
        folder = output / "fold_data" / f"fold_{fold['fold']}"
        paths = {}
        for role in ("inner_train", "inner_validation", "outer_test"):
            paths[role] = str(
                write_trip_subset(dev, folder / f"{role}.npz", fold[role]).resolve()
            )
        registry.append({
            "fold": fold["fold"],
            "paths": paths,
            "outer_test_trip_keys": fold["outer_test"],
        })
    final_folder = output / "final_split"
    final_train = ordered[:87]
    final_validation = ordered[87:]
    final_paths = {
        "train": str(write_trip_subset(
            dev, final_folder / "train_87.npz", final_train
        ).resolve()),
        "validation": str(write_trip_subset(
            dev, final_folder / "validation_14.npz", final_validation
        ).resolve()),
    }
    manifest = {
        "development_path": str(dev.resolve()),
        "development_sha256": file_sha256(dev),
        "development_trips": len(ordered),
        "folds": registry,
        "final_split": {
            **final_paths,
            "n_train": len(final_train),
            "n_validation": len(final_validation),
        },
        "locked_test_loaded": False,
    }
    (output / "DEVELOPMENT_PROTOCOL.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _datasets(train_path: Path, validation_path: Path, variant: FinalVariant):
    from dbpatchkan.models import EVTripBagDataset

    train = ConceptView(EVTripBagDataset.from_npz(train_path), variant.concept_set)
    validation = ConceptView(
        EVTripBagDataset.from_npz(validation_path), variant.concept_set
    )
    return train, validation


def _pretrain_shared(
    train_path: Path,
    validation_path: Path,
    *,
    output: Path,
    seed: int,
    epochs: int,
) -> Path:
    from dbpatchkan.models import DBPatchKAN
    from dbpatchkan.training.train import train

    checkpoint = output / "pretrain_latest.pt"
    if checkpoint.exists():
        update(f"Shared pretraining already complete: {output.name}")
        return checkpoint
    train_data, validation_data = _datasets(
        train_path, validation_path, FINAL_KAN
    )
    _seed(seed)  # initialization is now explicitly controlled
    model = DBPatchKAN(_model_config(FINAL_KAN))
    config = _training_config(
        seed=seed, stage="pretrain",
        pretrain_epochs=epochs, finetune_epochs=1,
    )
    update(f"Shared masked+alignment pretraining: seed={seed}, epochs={epochs}")
    train(model, train_data, validation_data, config, output_dir=output)
    if not checkpoint.exists():
        raise RuntimeError("Shared pretraining checkpoint was not created")
    return checkpoint


def _fit_variant(
    train_path: Path,
    validation_path: Path,
    *,
    variant: FinalVariant,
    shared_checkpoint: Path,
    output: Path,
    seed: int,
    epochs: int,
) -> tuple[Path, dict[str, Any]]:
    import torch
    from dbpatchkan.models import DBPatchKAN
    from dbpatchkan.training.train import train

    lightweight = output / "final_lightweight.pt"
    result_path = output / "complete.json"
    if lightweight.exists() and result_path.exists():
        return lightweight, json.loads(result_path.read_text())
    train_data, validation_data = _datasets(train_path, validation_path, variant)
    _seed(seed)  # fixes the Stage 6 construction-before-seeding confound
    model = DBPatchKAN(_model_config(variant))
    _load_shared_pretraining(model, shared_checkpoint)
    config = _training_config(
        seed=seed, stage="finetune",
        pretrain_epochs=1, finetune_epochs=epochs,
    )
    update(f"Fine-tuning {variant.name}: seed={seed}")
    result = train(model, train_data, validation_data, config, output_dir=output)
    best = output / "best.pt"
    state = torch.load(best, map_location="cpu", weights_only=False)
    torch.save({
        "model_state": state["model"],
        "model_config": state["model_config"],
        "training_config": state["training_config"],
        "best_epoch": state["epoch"],
        "best_metrics": state["metrics"],
        "variant": asdict(variant),
        "seed": seed,
        "development_only": True,
    }, lightweight)
    result["best_epoch"] = int(state["epoch"])
    result["lightweight_checkpoint"] = str(lightweight.resolve())
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for redundant in (
        output / "best.pt", output / "latest.pt", output / "pretrain_latest.pt"
    ):
        if redundant.exists():
            redundant.unlink()
    return lightweight, result


def _evaluate_checkpoint(
    checkpoint: Path,
    data_path: Path,
    variant: FinalVariant,
    *,
    device,
    precision: str,
) -> pd.DataFrame:
    from dbpatchkan.models import EVTripBagDataset

    model, _ = _load_lightweight(checkpoint, device)
    dataset = ConceptView(EVTripBagDataset.from_npz(data_path), variant.concept_set)
    return _outer_predictions(model, dataset, device, precision)


def _append_metrics(
    rows: list[dict[str, Any]],
    predictions: pd.DataFrame,
    *,
    phase: str,
    variant: str,
    fold: int | str,
    seed: int,
) -> None:
    for target in TARGETS:
        metric = _metrics(
            predictions[f"{target}_observed"].to_numpy(float),
            predictions[f"{target}_predicted"].to_numpy(float),
        )
        rows.append({
            "phase": phase,
            "variant": variant,
            "fold": fold,
            "seed": seed,
            "target": target,
            **metric,
        })


def _bootstrap_ablation(
    predictions: pd.DataFrame,
    *,
    repeats: int = 4000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    reference = predictions[predictions.variant.eq(FINAL_KAN.name)]
    for target in TARGETS:
        ref = reference[reference.target.eq(target)].set_index(["fold", "trip_key"])
        for variant in sorted(set(predictions.variant) - {FINAL_KAN.name}):
            alt = predictions[
                predictions.variant.eq(variant) & predictions.target.eq(target)
            ].set_index(["fold", "trip_key"])
            joined = ref[["observed", "predicted"]].join(
                alt[["predicted"]], rsuffix="_alt", how="inner"
            ).dropna()
            if len(joined) < 3:
                rows.append({
                    "target": target, "reference": FINAL_KAN.name,
                    "challenger": variant, "n": len(joined),
                    "delta_mae": math.nan, "ci_low": math.nan, "ci_high": math.nan,
                })
                continue
            delta = (
                np.abs(joined.predicted_alt - joined.observed)
                - np.abs(joined.predicted - joined.observed)
            ).to_numpy(float)
            draws = np.asarray([
                rng.choice(delta, size=len(delta), replace=True).mean()
                for _ in range(repeats)
            ])
            rows.append({
                "target": target,
                "reference": FINAL_KAN.name,
                "challenger": variant,
                "n": len(delta),
                "delta_mae": float(delta.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "challenger_significantly_better": bool(np.quantile(draws, 0.975) < 0),
                "reference_significantly_better": bool(np.quantile(draws, 0.025) > 0),
            })
    return pd.DataFrame(rows)


def run_corrected_ablations(
    manifest: dict[str, Any],
    output: Path,
    *,
    pretrain_epochs: int,
    finetune_epochs: int,
) -> None:
    import torch

    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "corrected_ablation_metrics.csv"
    predictions_path = output / "corrected_ablation_predictions.csv"
    if metrics_path.exists() and predictions_path.exists():
        update("Corrected essential ablations already complete")
        return
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    device = torch.device("cuda")
    for fold_spec in manifest["folds"]:
        fold = int(fold_spec["fold"])
        train_path = Path(fold_spec["paths"]["inner_train"])
        validation_path = Path(fold_spec["paths"]["inner_validation"])
        outer_path = Path(fold_spec["paths"]["outer_test"])
        shared = _pretrain_shared(
            train_path, validation_path,
            output=output / "working_models" / f"fold_{fold}" / "shared_pretraining",
            seed=42, epochs=pretrain_epochs,
        )
        for index, variant in enumerate(ESSENTIAL_ABLATIONS, start=1):
            update(
                f"Corrected ablation fold {fold}/4, "
                f"variant {index}/{len(ESSENTIAL_ABLATIONS)}: {variant.name}"
            )
            checkpoint, _ = _fit_variant(
                train_path, validation_path, variant=variant,
                shared_checkpoint=shared,
                output=output / "working_models" / f"fold_{fold}" / variant.name,
                seed=42, epochs=finetune_epochs,
            )
            frame = _evaluate_checkpoint(
                checkpoint, outer_path, variant, device=device, precision="bf16"
            )
            _append_metrics(
                metric_rows, frame, phase="corrected_ablation",
                variant=variant.name, fold=fold, seed=42,
            )
            for target in TARGETS:
                part = frame[
                    ["trip_key", f"{target}_observed", f"{target}_predicted"]
                ].copy()
                part.columns = ["trip_key", "observed", "predicted"]
                part.insert(0, "target", target)
                part.insert(0, "fold", fold)
                part.insert(0, "variant", variant.name)
                prediction_frames.append(part)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(predictions_path, index=False)
    _bootstrap_ablation(predictions).to_csv(
        output / "corrected_ablation_bootstrap.csv", index=False
    )
    shutil.rmtree(output / "working_models")
    update("Corrected essential ablations complete; temporary fold checkpoints removed")


def run_final_multiseed(
    manifest: dict[str, Any],
    output: Path,
    *,
    pretrain_epochs: int,
    finetune_epochs: int,
) -> list[dict[str, Any]]:
    import torch

    output.mkdir(parents=True, exist_ok=True)
    registry_path = output / "FROZEN_MODEL_REGISTRY.json"
    if registry_path.exists():
        update("Five-seed frozen models already complete")
        return json.loads(registry_path.read_text())["models"]
    train_path = Path(manifest["final_split"]["train"])
    validation_path = Path(manifest["final_split"]["validation"])
    device = torch.device("cuda")
    registry: list[dict[str, Any]] = []
    development_metrics: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS, start=1):
        update(f"Final seed {seed_index}/{len(SEEDS)}: {seed}")
        seed_root = output / f"seed_{seed}"
        shared = _pretrain_shared(
            train_path, validation_path,
            output=seed_root / "shared_pretraining",
            seed=seed, epochs=pretrain_epochs,
        )
        for variant in (FINAL_KAN, FINAL_MLP):
            checkpoint, result = _fit_variant(
                train_path, validation_path, variant=variant,
                shared_checkpoint=shared, output=seed_root / variant.name,
                seed=seed, epochs=finetune_epochs,
            )
            frame = _evaluate_checkpoint(
                checkpoint, validation_path, variant,
                device=device, precision="bf16",
            )
            _append_metrics(
                development_metrics, frame, phase="final_validation",
                variant=variant.name, fold="final_14", seed=seed,
            )
            registry.append({
                "seed": seed,
                "variant": asdict(variant),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "best_epoch": result["best_epoch"],
            })
        shutil.rmtree(seed_root / "shared_pretraining")
    pd.DataFrame(development_metrics).to_csv(
        output / "final_multiseed_validation_metrics.csv", index=False
    )
    summary = (
        pd.DataFrame(development_metrics)
        .groupby(["variant", "target"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            n_mean=("n", "mean"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            bias_mean=("bias", "mean"),
        )
    )
    summary.to_csv(output / "final_multiseed_validation_summary.csv", index=False)
    registry_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": registry,
        "architecture_frozen_before_locked_test": True,
        "seeds": SEEDS,
    }
    registry_path.write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")
    return registry


def freeze_configuration(
    manifest: dict[str, Any],
    registry: list[dict[str, Any]],
    output: Path,
) -> Path:
    path = output / "FROZEN_CONFIGURATION.json"
    payload = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "development_sha256": manifest["development_sha256"],
        "architecture": {
            "name": "Persistence-Anchored DB-PatchKAN",
            "current_soh_output": "strict pre-test persistence",
            "forecast_output": "anchor SOH minus nonnegative KAN loss",
            "driver_battery_encoder": "dual-stream 1D Patch Transformer",
            "health_head": "named D-BSI-total concept KAN",
            "comparison_head": "parameter-matched named-concept MLP",
            "configuration": _model_config(FINAL_KAN).to_dict(),
        },
        "seeds": SEEDS,
        "registered_models": len(registry),
        "selection_complete": True,
        "locked_test_seen": False,
        "prohibited_after_test": [
            "architecture changes", "hyperparameter tuning",
            "feature selection", "threshold selection",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    update("Frozen configuration written; development phase is complete")
    return path


def run_development(
    dev: Path,
    output: Path,
    *,
    pretrain_epochs: int,
    finetune_epochs: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    freeze = output / "FROZEN_CONFIGURATION.json"
    if freeze.exists():
        update("Development phase already frozen; skipping")
        return
    manifest = _prepare_development(dev, output)
    run_corrected_ablations(
        manifest, output / "corrected_ablations",
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
    )
    registry = run_final_multiseed(
        manifest, output / "final_models",
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
    )
    freeze_configuration(manifest, registry, output)


def _locked_baselines(dev: Path, test: Path, output: Path) -> pd.DataFrame:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor

    dev_rows = _trip_rows(dev).sort_values("start_unix_ms").reset_index(drop=True)
    test_rows = _trip_rows(test).sort_values("start_unix_ms").reset_index(drop=True)
    predictions: list[dict[str, Any]] = []
    last_current = float(dev_rows.current_soh.dropna().iloc[-1])
    for target in TARGETS:
        observed = test_rows[target].to_numpy(float)
        persistence = (
            np.full(len(test_rows), last_current)
            if target == "current_soh"
            else np.zeros(len(test_rows))
        )
        for trip, truth, prediction in zip(
            test_rows.trip_key, observed, persistence, strict=True
        ):
            predictions.append({
                "model": "Strict_Pretest_Persistence",
                "seed": -1,
                "target": target,
                "trip_key": trip,
                "observed": truth,
                "predicted": float(prediction),
            })
    feature_columns = [
        name
        for name in dev_rows
        if name.startswith(("driver_", "battery_", "concept_"))
        and name != "concept_12"
    ]
    for target in TARGETS:
        train_valid = np.isfinite(dev_rows[target])
        test_valid = np.isfinite(test_rows[target])
        if train_valid.sum() < 8 or not test_valid.any():
            continue
        model = Pipeline([
            ("impute", SimpleImputer()),
            ("model", XGBRegressor(
                n_estimators=350,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        model.fit(
            dev_rows.loc[train_valid, feature_columns],
            dev_rows.loc[train_valid, target],
        )
        prediction = model.predict(test_rows.loc[test_valid, feature_columns])
        for trip, truth, value in zip(
            test_rows.loc[test_valid, "trip_key"],
            test_rows.loc[test_valid, target],
            prediction,
            strict=True,
        ):
            predictions.append({
                "model": "Fixed_XGBoost",
                "seed": 42,
                "target": target,
                "trip_key": trip,
                "observed": float(truth),
                "predicted": float(value),
            })
    frame = pd.DataFrame(predictions)
    frame.to_csv(output / "locked_baseline_predictions.csv", index=False)
    return frame


def _deep_locked_predictions(
    registry: list[dict[str, Any]],
    test: Path,
    *,
    output: Path,
) -> pd.DataFrame:
    import torch

    device = torch.device("cuda")
    frames = []
    for index, item in enumerate(registry, start=1):
        variant = FinalVariant(**item["variant"])
        update(
            f"Locked inference {index}/{len(registry)}: "
            f"{variant.name}, seed={item['seed']}"
        )
        frame = _evaluate_checkpoint(
            Path(item["checkpoint"]), test, variant,
            device=device, precision="bf16",
        )
        rows = []
        for target in TARGETS:
            for trip, observed, predicted in zip(
                frame.trip_key,
                frame[f"{target}_observed"],
                frame[f"{target}_predicted"],
                strict=True,
            ):
                rows.append({
                    "model": variant.name,
                    "seed": int(item["seed"]),
                    "target": target,
                    "trip_key": trip,
                    "observed": observed,
                    "predicted": predicted,
                })
        frames.append(pd.DataFrame(rows))
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output / "locked_deep_raw_predictions.csv", index=False)
    return result


def _final_system_predictions(
    deep: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    persistence = baseline[
        baseline.model.eq("Strict_Pretest_Persistence")
    ][["target", "trip_key", "predicted"]].rename(
        columns={"predicted": "persistence"}
    )
    rows = []
    for (model, seed), group in deep.groupby(["model", "seed"]):
        system_name = (
            "Final_System_KAN"
            if model == FINAL_KAN.name
            else "Matched_System_MLP"
        )
        merged = group.merge(persistence, on=["target", "trip_key"], how="left")
        merged["system_prediction"] = np.where(
            merged.target.eq("current_soh"),
            merged.persistence,
            merged.predicted,
        )
        for row in merged.itertuples(index=False):
            rows.append({
                "model": system_name,
                "seed": seed,
                "target": row.target,
                "trip_key": row.trip_key,
                "observed": row.observed,
                "predicted": row.system_prediction,
            })
    return pd.DataFrame(rows)


def _test_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed, target), group in predictions.groupby(
        ["model", "seed", "target"]
    ):
        metric = _metrics(
            group.observed.to_numpy(float),
            group.predicted.to_numpy(float),
        )
        rows.append({
            "model": model,
            "seed": seed,
            "target": target,
            **metric,
        })
    return pd.DataFrame(rows).sort_values(["target", "mae"])


def _ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    deep = predictions[
        predictions.model.isin(("Final_System_KAN", "Matched_System_MLP"))
    ]
    rows = []
    for (model, target, trip), group in deep.groupby(
        ["model", "target", "trip_key"]
    ):
        observed = group.observed.dropna()
        rows.append({
            "model": model + "_Ensemble",
            "seed": -1,
            "target": target,
            "trip_key": trip,
            "observed": float(observed.iloc[0]) if len(observed) else math.nan,
            "predicted": float(group.predicted.mean()),
            "seed_prediction_std": float(group.predicted.std(ddof=1)),
        })
    return pd.DataFrame(rows)


def _locked_bootstrap(
    predictions: pd.DataFrame,
    *,
    repeats: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    reference = predictions[
        predictions.model.eq("Strict_Pretest_Persistence")
    ].set_index(["target", "trip_key"])
    rows = []
    for model in (
        "Fixed_XGBoost",
        "Final_System_KAN_Ensemble",
        "Matched_System_MLP_Ensemble",
    ):
        challenger = predictions[predictions.model.eq(model)].set_index(
            ["target", "trip_key"]
        )
        for target in TARGETS:
            ref = reference.loc[
                reference.index.get_level_values("target") == target
            ]
            alt = challenger.loc[
                challenger.index.get_level_values("target") == target
            ]
            joined = ref[["observed", "predicted"]].join(
                alt[["predicted"]], rsuffix="_alt", how="inner"
            ).dropna()
            if len(joined) < 3:
                rows.append({
                    "target": target, "reference": "Strict_Pretest_Persistence",
                    "challenger": model, "n": len(joined),
                    "delta_mae": math.nan, "ci_low": math.nan, "ci_high": math.nan,
                    "status": "not_estimable",
                })
                continue
            delta = (
                np.abs(joined.predicted_alt - joined.observed)
                - np.abs(joined.predicted - joined.observed)
            ).to_numpy(float)
            draws = np.asarray([
                rng.choice(delta, len(delta), replace=True).mean()
                for _ in range(repeats)
            ])
            rows.append({
                "target": target,
                "reference": "Strict_Pretest_Persistence",
                "challenger": model,
                "n": len(delta),
                "delta_mae": float(delta.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "challenger_significantly_better": bool(np.quantile(draws, 0.975) < 0),
                "persistence_significantly_better": bool(np.quantile(draws, 0.025) > 0),
                "status": "complete",
            })
    return pd.DataFrame(rows)


def run_locked_test(dev: Path, test: Path, output: Path) -> None:
    freeze_path = output / "FROZEN_CONFIGURATION.json"
    registry_path = output / "final_models" / "FROZEN_MODEL_REGISTRY.json"
    if not freeze_path.exists() or not registry_path.exists():
        raise RuntimeError("Development models must be frozen before opening the test")
    final_dir = output / "locked_test_once"
    final_dir.mkdir(parents=True, exist_ok=True)
    completed = final_dir / "LOCKED_TEST_COMPLETE.json"
    if completed.exists():
        update("Locked test was already completed; refusing to evaluate it again")
        return
    freeze = json.loads(freeze_path.read_text())
    if not freeze.get("selection_complete") or freeze.get("locked_test_seen"):
        raise RuntimeError("Frozen-configuration state is invalid")
    registry = json.loads(registry_path.read_text())["models"]
    if len(registry) != 10:
        raise RuntimeError(f"Expected 10 frozen models, found {len(registry)}")
    test_hash = file_sha256(test)
    access = {
        "opened_utc": datetime.now(timezone.utc).isoformat(),
        "test_path": str(test.resolve()),
        "test_sha256": test_hash,
        "configuration_frozen_utc": freeze["frozen_utc"],
        "models_frozen_before_open": len(registry),
        "purpose": "one-time final inference only",
        "tuning_after_open_prohibited": True,
        "completed": False,
    }
    (final_dir / "LOCKED_TEST_ACCESS_AUDIT.json").write_text(
        json.dumps(access, indent=2), encoding="utf-8"
    )
    update("LOCKED TEST OPENED: final inference only; no tuning is permitted")
    baseline = _locked_baselines(dev, test, final_dir)
    deep = _deep_locked_predictions(registry, test, output=final_dir)
    system = _final_system_predictions(deep, baseline)
    ensemble = _ensemble_predictions(system)
    combined = pd.concat((baseline, system, ensemble), ignore_index=True, sort=False)
    combined.to_csv(final_dir / "locked_test_all_predictions.csv", index=False)
    metrics = _test_metrics(combined)
    metrics.to_csv(final_dir / "locked_test_metrics.csv", index=False)
    seed_summary = (
        metrics[metrics.model.isin(("Final_System_KAN", "Matched_System_MLP"))]
        .groupby(["model", "target"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            n_mean=("n", "mean"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            bias_mean=("bias", "mean"),
        )
    )
    seed_summary.to_csv(final_dir / "locked_test_multiseed_summary.csv", index=False)
    _locked_bootstrap(combined).to_csv(
        final_dir / "locked_test_paired_bootstrap.csv", index=False
    )
    label_counts = (
        combined[combined.model.eq("Strict_Pretest_Persistence")]
        .groupby("target")["observed"].apply(
            lambda values: int(values.notna().sum())
        )
        .to_dict()
    )
    access["completed"] = True
    access["completed_utc"] = datetime.now(timezone.utc).isoformat()
    access["label_counts"] = label_counts
    access["thirty_day_test_status"] = (
        "not estimable: locked test contains no valid 30-day labels"
    )
    completed.write_text(json.dumps(access, indent=2), encoding="utf-8")
    freeze["locked_test_seen"] = True
    freeze["locked_test_sha256"] = test_hash
    freeze["locked_test_completed_utc"] = access["completed_utc"]
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    update("Locked test complete and sealed; do not tune models after this point")


def build_final_decision(output: Path) -> None:
    metrics_path = output / "locked_test_once" / "locked_test_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Locked-test metrics are absent")
    metrics = pd.read_csv(metrics_path)
    ensemble_models = metrics[
        metrics.model.isin((
            "Strict_Pretest_Persistence",
            "Fixed_XGBoost",
            "Final_System_KAN_Ensemble",
            "Matched_System_MLP_Ensemble",
        ))
    ]
    best = (
        ensemble_models[np.isfinite(ensemble_models.mae)]
        .sort_values(["target", "mae"])
        .groupby("target", as_index=False)
        .first()[["target", "model", "mae"]]
    )
    best.to_csv(output / "FINAL_LOCKED_TEST_WINNERS.csv", index=False)
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiments_complete": True,
        "architecture_tuning_closed": True,
        "locked_test_may_not_be_reused_for_selection": True,
        "best_by_target": best.to_dict("records"),
        "claim_rule": (
            "DB-PatchKAN is superior only where its locked-test MAE is lower "
            "and the paired bootstrap interval excludes zero."
        ),
        "thirty_day_test_limitation": (
            "No valid 30-day target exists in the locked 26-trip split; "
            "report development-CV evidence only."
        ),
    }
    (output / "FINAL_STAGE8_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "locked_test", "decision"), required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--locked-test-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "development":
        run_development(
            args.development_data, args.output_dir,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
        )
    elif args.phase == "locked_test":
        if args.locked_test_data is None:
            raise ValueError("--locked-test-data is required for locked_test phase")
        run_locked_test(
            args.development_data, args.locked_test_data, args.output_dir
        )
    else:
        build_final_decision(args.output_dir)
    update(f"Stage 8 phase {args.phase} complete in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
