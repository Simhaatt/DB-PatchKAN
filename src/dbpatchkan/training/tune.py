"""Optuna Hyperband tuning for DB-PatchKAN on train/validation data only.

Optuna is intentionally imported inside :func:`run_tuning`, so the core
package and its training APIs remain importable with CPU-only dependencies.
No test/holdout data argument exists in this CLI by design.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from dbpatchkan.models import (
    DEFAULT_CONCEPT_NAMES,
    DBPatchKAN,
    DBPatchKANConfig,
    DBPatchKANMultiTaskLoss,
    EVSegmentDataset,
    EVTripBagDataset,
    MaskedSelfSupervisedLoss,
    assert_disjoint_trip_bags,
)
from dbpatchkan.training.train import (
    evaluate_finetuning,
    fit_input_normalization,
    resolve_device,
    resolve_precision,
    seed_everything,
    train_finetuning_epoch,
    train_pretraining_epoch,
)


SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    "patch_length": (8, 16, 32),
    "d_model": (96, 128),
    "encoder_layers": (2, 3, 4),
    "dropout": (0.1, 0.2),
    "mask_ratio": (0.3, 0.4, 0.5),
    "n_concepts": (12, 16, 20),
    "kan_grid_size": (3, 5, 7),
    "kan_hidden_width": (8, 16, 24),
}

REDUCED_SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    "patch_length": (16,),
    "d_model": (128,),
    "encoder_layers": (3,),
    "dropout": (0.1, 0.2),
    "mask_ratio": (0.4,),
    "n_concepts": (12, 16, 20),
    "kan_grid_size": (3, 5),
    "kan_hidden_width": (8, 16),
}

SEARCH_PROFILES: dict[str, dict[str, tuple[Any, ...]]] = {
    "full": SEARCH_SPACE,
    "reduced": REDUCED_SEARCH_SPACE,
}


def sample_search_space(
    trial: Any,
    search_space: Mapping[str, tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    """Sample the fixed, thesis-declared search space from an Optuna-like trial."""

    choices_by_name = SEARCH_SPACE if search_space is None else search_space
    return {
        name: trial.suggest_categorical(name, list(values))
        for name, values in choices_by_name.items()
    }


def validate_search_parameters(
    parameters: Mapping[str, Any],
    *,
    segment_length: int,
    n_heads: int = 4,
    available_concepts: int = 20,
) -> None:
    """Reject configurations outside the pre-registered categorical space."""

    missing = set(SEARCH_SPACE) - set(parameters)
    if missing:
        raise ValueError(f"Missing search parameters: {sorted(missing)}")
    for name, choices in SEARCH_SPACE.items():
        if parameters[name] not in choices:
            raise ValueError(f"{name}={parameters[name]!r} is outside {choices}")
    if parameters["patch_length"] > segment_length:
        raise ValueError("patch_length exceeds the available segment length")
    if parameters["d_model"] % n_heads:
        raise ValueError("d_model must be divisible by the number of attention heads")
    if parameters["n_concepts"] > available_concepts:
        raise ValueError(
            f"Trial requires {parameters['n_concepts']} concepts, but only "
            f"{available_concepts} are available"
        )


def build_trial_model_config(
    parameters: Mapping[str, Any],
    *,
    driver_channels: int,
    battery_channels: int,
    segment_length: int,
    available_concepts: int = 20,
    n_heads: int = 4,
    ff_dim: int = 256,
) -> DBPatchKANConfig:
    """Convert a sampled point into a dimensionally compatible model config."""

    validate_search_parameters(
        parameters,
        segment_length=segment_length,
        n_heads=n_heads,
        available_concepts=available_concepts,
    )
    n_concepts = int(parameters["n_concepts"])
    concept_names = DEFAULT_CONCEPT_NAMES[:n_concepts]
    if len(concept_names) < n_concepts:  # pragma: no cover - defaults define 20
        concept_names = tuple(f"concept_{index}" for index in range(n_concepts))
    default_driver_protected = (0, 9, 10, 11, 12)
    default_battery_protected = (0, 1, 2, 3, 7, 8, 9, 10)
    patch_length = int(parameters["patch_length"])
    return DBPatchKANConfig(
        driver_channels=driver_channels,
        battery_channels=battery_channels,
        segment_length=segment_length,
        patch_length=patch_length,
        patch_stride=patch_length // 2,
        d_model=int(parameters["d_model"]),
        encoder_layers=int(parameters["encoder_layers"]),
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout=float(parameters["dropout"]),
        concept_names=tuple(concept_names),
        fusion_head="kan",
        kan_hidden=(int(parameters["kan_hidden_width"]), 8),
        kan_grid_size=int(parameters["kan_grid_size"]),
        protected_driver_channels=tuple(
            index for index in default_driver_protected if index < driver_channels
        ),
        protected_battery_channels=tuple(
            index for index in default_battery_protected if index < battery_channels
        ),
    )


class ConceptSubsetDataset(Dataset[dict[str, Tensor]]):
    """Training-fold view that exposes only the trial's first named concepts."""

    _CONCEPT_KEYS = (
        "concept_targets",
        "concept_mask",
        "observed_concepts",
        "observed_concept_mask",
    )

    def __init__(self, dataset: Dataset[dict[str, Tensor]], n_concepts: int) -> None:
        if n_concepts < 1 or n_concepts > 20:
            raise ValueError("n_concepts must be between 1 and 20")
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        sample = dataset[0]
        available = next(
            (
                int(sample[key].shape[-1])
                for key in self._CONCEPT_KEYS
                if key in sample
            ),
            0,
        )
        if available < n_concepts:
            raise ValueError(
                f"Requested {n_concepts} concepts, but dataset exposes {available}"
            )
        self.dataset = dataset
        self.n_concepts = n_concepts

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = dict(self.dataset[index])
        for key in self._CONCEPT_KEYS:
            if key in item:
                item[key] = item[key][..., : self.n_concepts]
        return item

    @property
    def collate_fn(self):
        """Preserve variable-size trip collation through the concept view."""

        return getattr(self.dataset, "collate_fn", None)


def sqlite_storage_url(path: str | Path) -> str:
    """Return a cross-platform absolute SQLite URL for a persistent study."""

    resolved = Path(path).expanduser().resolve()
    return f"sqlite:///{resolved.as_posix()}"


def _concept_statistics(dataset: Dataset, n_concepts: int) -> tuple[Tensor, Tensor]:
    rows: list[Tensor] = []
    for index in range(len(dataset)):
        item = dataset[index]
        value = item.get("concept_targets", item.get("observed_concepts"))
        if value is not None:
            rows.append(value.float())
    if not rows:
        return torch.zeros(n_concepts), torch.ones(n_concepts)
    values = torch.stack(rows)
    center = torch.zeros(n_concepts)
    scale = torch.ones(n_concepts)
    for column in range(n_concepts):
        valid = torch.isfinite(values[:, column])
        if torch.any(valid):
            center[column] = values[valid, column].mean()
            standard_deviation = values[valid, column].std(unbiased=False)
            if standard_deviation > 1.0e-6:
                scale[column] = standard_deviation
    return center, scale


def _loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    collate = getattr(dataset, "collate_fn", None)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
        collate_fn=collate,
    )


def _constant_scheduler(optimizer: torch.optim.Optimizer) -> LambdaLR:
    return LambdaLR(optimizer, lambda _: 1.0)


def _finetune_optimizer(
    model: DBPatchKAN,
    criterion: DBPatchKANMultiTaskLoss,
    *,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> AdamW:
    encoder_prefixes = (
        "driver_encoder",
        "battery_encoder",
        "driver_to_battery",
        "battery_to_driver",
        "driver_cross_norm",
        "battery_cross_norm",
    )
    encoder_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(encoder_prefixes):
            encoder_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    head_parameters.extend(
        parameter for parameter in criterion.parameters() if parameter.requires_grad
    )
    groups: list[dict[str, Any]] = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": encoder_lr})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": head_lr})
    return AdamW(groups, weight_decay=weight_decay)


def _available_concepts(dataset: Dataset[dict[str, Tensor]]) -> int:
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty")
    sample = dataset[0]
    values = sample.get("concept_targets", sample.get("observed_concepts"))
    if values is None:
        raise ValueError("Tuning requires named concept targets or observations")
    return int(values.shape[-1])


def make_objective(
    *,
    optuna_module: Any,
    train_dataset: Dataset[dict[str, Tensor]],
    validation_dataset: Dataset[dict[str, Tensor]],
    device: torch.device,
    precision: str,
    pretrain_epochs: int,
    finetune_epochs: int,
    pretrain_batch_size: int,
    finetune_batch_size: int,
    gradient_accumulation: int,
    pretrain_lr: float,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
    gradient_clip: float,
    num_workers: int,
    base_seed: int,
    domain: str,
    freeze_encoders: bool,
    search_space: Mapping[str, tuple[Any, ...]] | None = None,
) -> Any:
    """Build an objective whose score is computed exclusively on validation data."""

    train_is_trip = isinstance(train_dataset, EVTripBagDataset)
    validation_is_trip = isinstance(validation_dataset, EVTripBagDataset)
    if train_is_trip != validation_is_trip:
        raise ValueError("Train and validation must use the same segment/trip dataset kind")
    if train_is_trip:
        assert_disjoint_trip_bags(train_dataset, validation_dataset)
    train_sample = train_dataset[0]
    validation_sample = validation_dataset[0]
    if train_sample["driver"].ndim != validation_sample["driver"].ndim:
        raise ValueError("Train/validation dataset ranks must match")
    driver_channels = int(train_sample["driver"].shape[-1])
    battery_channels = int(train_sample["battery"].shape[-1])
    segment_length = int(train_sample["driver"].shape[-2])
    if validation_sample["driver"].shape[-2:] != train_sample["driver"].shape[-2:]:
        raise ValueError("Train/validation driver segment shapes must match")
    if validation_sample["battery"].shape[-2:] != train_sample["battery"].shape[-2:]:
        raise ValueError("Train/validation battery segment shapes must match")
    concepts_available = min(
        _available_concepts(train_dataset), _available_concepts(validation_dataset)
    )
    active_search_space = SEARCH_SPACE if search_space is None else search_space
    if concepts_available < max(active_search_space["n_concepts"]):
        raise ValueError(
            "The registered search requires at least 20 named concepts in both datasets"
        )

    def objective(trial: Any) -> float:
        parameters = sample_search_space(trial, active_search_space)
        model_config = build_trial_model_config(
            parameters,
            driver_channels=driver_channels,
            battery_channels=battery_channels,
            segment_length=segment_length,
            available_concepts=concepts_available,
        )
        seed = base_seed + int(trial.number)
        seed_everything(seed)
        trial_train = ConceptSubsetDataset(train_dataset, model_config.n_concepts)
        trial_validation = ConceptSubsetDataset(
            validation_dataset, model_config.n_concepts
        )
        ssl_dataset: Dataset = (
            train_dataset.segments if train_is_trip else trial_train
        )
        train_loader = _loader(
            ssl_dataset,
            pretrain_batch_size,
            shuffle=True,
            workers=num_workers,
            device=device,
            seed=seed,
        )
        model = DBPatchKAN(model_config).to(device)
        center, scale = _concept_statistics(trial_train, model_config.n_concepts)
        model.set_concept_normalization(center.to(device), scale.to(device))
        input_statistics = fit_input_normalization(
            trial_train,
            driver_channels=model_config.driver_channels,
            battery_channels=model_config.battery_channels,
        )
        model.set_input_normalization(
            *(statistic.to(device) for statistic in input_statistics)
        )
        scaler = torch.amp.GradScaler(
            device.type, enabled=device.type == "cuda" and precision == "fp16"
        )
        try:
            if pretrain_epochs:
                ssl_criterion = MaskedSelfSupervisedLoss().to(device)
                pretrain_optimizer = AdamW(
                    model.parameters(), lr=pretrain_lr, weight_decay=weight_decay
                )
                pretrain_scheduler = _constant_scheduler(pretrain_optimizer)
                for _ in range(pretrain_epochs):
                    train_pretraining_epoch(
                        model,
                        train_loader,
                        ssl_criterion,
                        pretrain_optimizer,
                        pretrain_scheduler,
                        scaler,
                        device=device,
                        precision=precision,
                        mask_ratio=float(parameters["mask_ratio"]),
                        gradient_accumulation=gradient_accumulation,
                        gradient_clip=gradient_clip,
                    )

            if freeze_encoders:
                model.freeze_patch_encoders(True)
            finetune_train_loader = _loader(
                trial_train,
                finetune_batch_size,
                shuffle=True,
                workers=num_workers,
                device=device,
                seed=seed,
            )
            validation_loader = _loader(
                trial_validation,
                finetune_batch_size,
                shuffle=False,
                workers=num_workers,
                device=device,
                seed=seed,
            )
            criterion = DBPatchKANMultiTaskLoss().to(device)
            optimizer = _finetune_optimizer(
                model,
                criterion,
                encoder_lr=encoder_lr,
                head_lr=head_lr,
                weight_decay=weight_decay,
            )
            scheduler = _constant_scheduler(optimizer)
            best_score = math.inf
            best_epoch = 0
            for epoch in range(finetune_epochs):
                train_finetuning_epoch(
                    model,
                    finetune_train_loader,
                    criterion,
                    optimizer,
                    scheduler,
                    scaler,
                    device=device,
                    precision=precision,
                    domain=domain,
                    gradient_accumulation=gradient_accumulation,
                    gradient_clip=gradient_clip,
                )
                validation_metrics = evaluate_finetuning(
                    model,
                    validation_loader,
                    criterion,
                    device=device,
                    precision=precision,
                    domain=domain,
                )
                score = float(
                    validation_metrics.get(
                        "current_soh_mae_pp", validation_metrics["total"]
                    )
                )
                if not math.isfinite(score):
                    raise optuna_module.TrialPruned("Non-finite validation objective")
                if score < best_score:
                    best_score = score
                    best_epoch = epoch + 1
                print(
                    json.dumps(
                        {
                            "stage": "tuning",
                            "trial": int(trial.number),
                            "epoch": epoch + 1,
                            "validation_current_soh_mae_pp": score,
                            "best_so_far": best_score,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                trial.report(score, step=epoch + 1)
                if trial.should_prune():
                    raise optuna_module.TrialPruned(
                        f"Hyperband pruned after validation epoch {epoch + 1}"
                    )
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("parameter_count", sum(p.numel() for p in model.parameters()))
            trial.set_user_attr("model_config", model_config.to_dict())
            trial.set_user_attr("trip_bags", train_is_trip)
            return best_score
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise optuna_module.TrialPruned("CUDA out of memory") from error
            raise
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return objective


def run_tuning(args: argparse.Namespace) -> dict[str, Any]:
    """Run/resume the persistent study and save its auditable outputs."""

    try:
        import optuna
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Optuna is optional. Install it with `pip install -e .[tuning]`."
        ) from error

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    study_db = Path(args.study_db or output_dir / "optuna_study.db").resolve()
    study_db.parent.mkdir(parents=True, exist_ok=True)
    dataset_type = EVTripBagDataset if args.trip_bags else EVSegmentDataset
    train_dataset = dataset_type.from_npz(args.train_data)
    validation_dataset = dataset_type.from_npz(args.validation_data)
    if args.trip_bags:
        assert_disjoint_trip_bags(train_dataset, validation_dataset)
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    search_profile = getattr(args, "search_profile", "full")
    search_space = SEARCH_PROFILES[search_profile]
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=args.finetune_epochs,
        reduction_factor=3,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=sqlite_storage_url(study_db),
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
    )
    objective = make_objective(
        optuna_module=optuna,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        device=device,
        precision=precision,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        finetune_batch_size=args.finetune_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        pretrain_lr=args.pretrain_lr,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        num_workers=args.num_workers,
        base_seed=args.seed,
        domain=args.domain,
        freeze_encoders=args.freeze_encoders,
        search_space=search_space,
    )
    # ``--trials`` is a total study budget, not an additional-trials count, so
    # reconnecting an interrupted Colab session does not silently exceed 24.
    remaining_trials = max(0, args.trials - len(study.trials))
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, gc_after_trial=True)
    trials_path = output_dir / "trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    try:
        best = study.best_trial
    except ValueError as error:
        raise RuntimeError("No tuning trial completed successfully") from error
    best_payload = {
        "study_name": study.study_name,
        "objective": "validation_current_soh_mae_pp",
        "direction": "minimize",
        "best_trial": best.number,
        "best_value": best.value,
        "parameters": best.params,
        "model_config": best.user_attrs.get("model_config"),
        "best_epoch": best.user_attrs.get("best_epoch"),
        "search_profile": search_profile,
        "search_space": {name: list(values) for name, values in search_space.items()},
        "train_data": str(Path(args.train_data).resolve()),
        "validation_data": str(Path(args.validation_data).resolve()),
        "held_out_test_used": False,
        "trip_bags": bool(args.trip_bags),
        "study_db": str(study_db),
    }
    best_path = output_dir / "best_config.json"
    best_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    return {
        "study_name": study.study_name,
        "completed_trials": len(study.trials),
        "best_value": best.value,
        "trials_csv": str(trials_path),
        "best_config": str(best_path),
        "study_db": str(study_db),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument(
        "--trip-bags",
        action="store_true",
        help="Tune on composite trip bags so each validation label is scored once",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study-db", type=Path)
    parser.add_argument("--study-name", default="dbpatchkan_a100")
    parser.add_argument(
        "--search-profile",
        choices=tuple(SEARCH_PROFILES),
        default="full",
        help="Use the full registered space or the reduced fixed-encoder screening space",
    )
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto"
    )
    parser.add_argument("--domain", choices=("nissan", "capacity"), default="nissan")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--finetune-batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--pretrain-lr", type=float, default=3.0e-4)
    parser.add_argument("--encoder-lr", type=float, default=1.0e-5)
    parser.add_argument("--head-lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--freeze-encoders", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if args.finetune_epochs < 1:
        raise ValueError("--finetune-epochs must be positive")
    result = run_tuning(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SEARCH_SPACE",
    "ConceptSubsetDataset",
    "sample_search_space",
    "validate_search_parameters",
    "build_trial_model_config",
    "sqlite_storage_url",
    "make_objective",
    "run_tuning",
    "build_parser",
    "main",
]
