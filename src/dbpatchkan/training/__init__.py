"""Training APIs for DB-PatchKAN, imported lazily for clean ``-m`` execution."""

from importlib import import_module
from typing import Any

_TRAIN_EXPORTS = {
    "TrainingConfig",
    "build_parser",
    "evaluate_finetuning",
    "fit_input_normalization",
    "main",
    "resolve_device",
    "resolve_precision",
    "run_synthetic_smoke",
    "seed_everything",
    "train",
    "train_finetuning_epoch",
    "train_pretraining_epoch",
}

_TUNE_EXPORTS = {
    "SEARCH_SPACE",
    "ConceptSubsetDataset",
    "sample_search_space",
    "validate_search_parameters",
    "build_trial_model_config",
    "sqlite_storage_url",
    "make_objective",
    "run_tuning",
}

_EVALUATION_EXPORTS = {
    "state_fingerprint",
    "load_checkpoint_model",
    "detect_npz_dataset_kind",
    "load_lineage_metadata",
    "evaluate_checkpoint",
    "evaluation_main",
}

__all__ = sorted(_TRAIN_EXPORTS | _TUNE_EXPORTS | _EVALUATION_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _TRAIN_EXPORTS:
        return getattr(import_module(".train", __name__), name)
    if name in _TUNE_EXPORTS:
        return getattr(import_module(".tune", __name__), name)
    if name in _EVALUATION_EXPORTS:
        return getattr(import_module(".evaluate", __name__), name)
    raise AttributeError(name)
