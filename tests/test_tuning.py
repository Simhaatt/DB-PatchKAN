from __future__ import annotations

from pathlib import Path

import pytest
import torch

import dbpatchkan.training.tune as tuning
from dbpatchkan.models import EVSegmentDataset, EVTripBagDataset


EXPECTED_SPACE = {
    "patch_length": (8, 16, 32),
    "d_model": (96, 128),
    "encoder_layers": (2, 3, 4),
    "dropout": (0.1, 0.2),
    "mask_ratio": (0.3, 0.4, 0.5),
    "n_concepts": (12, 16, 20),
    "kan_grid_size": (3, 5, 7),
    "kan_hidden_width": (8, 16, 24),
}

EXPECTED_REDUCED_SPACE = {
    "patch_length": (16,),
    "d_model": (128,),
    "encoder_layers": (3,),
    "dropout": (0.1, 0.2),
    "mask_ratio": (0.4,),
    "n_concepts": (12, 16, 20),
    "kan_grid_size": (3, 5),
    "kan_hidden_width": (8, 16),
}


class FakeTrial:
    def __init__(self, choose_last: bool = False) -> None:
        self.choose_last = choose_last
        self.seen: dict[str, tuple[object, ...]] = {}

    def suggest_categorical(self, name, choices):
        self.seen[name] = tuple(choices)
        return choices[-1] if self.choose_last else choices[0]


class TinyConceptDataset:
    def __init__(self, samples: int = 2) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int):
        return {
            "driver": torch.zeros(256, 13),
            "battery": torch.zeros(256, 11),
            "concept_targets": torch.arange(20, dtype=torch.float32),
            "concept_mask": torch.ones(20, dtype=torch.bool),
            "observed_concepts": torch.arange(20, dtype=torch.float32),
            "observed_concept_mask": torch.ones(20, dtype=torch.bool),
            "current_soh_target": torch.tensor(96.0 - index),
        }


def make_trip_dataset(prefix: str, first_trip_segments: int) -> EVTripBagDataset:
    counts = (first_trip_segments, 1)
    n_segments = sum(counts)
    driver = torch.zeros(n_segments, 64, 13)
    battery = torch.zeros(n_segments, 64, 11)
    concepts = torch.cat(
        (
            torch.zeros(counts[0], 20),
            torch.ones(counts[1], 20),
        )
    )
    current_soh = torch.cat(
        (torch.full((counts[0],), 96.0), torch.full((counts[1],), 95.0))
    )
    segments = EVSegmentDataset(
        driver,
        battery,
        driver_mask=torch.ones_like(driver, dtype=torch.bool),
        battery_mask=torch.ones_like(battery, dtype=torch.bool),
        concept_targets=concepts,
        concept_mask=torch.ones_like(concepts, dtype=torch.bool),
        targets={"current_soh_target": current_soh},
    )
    return EVTripBagDataset(
        segments,
        [f"{prefix}-a"] * counts[0] + [f"{prefix}-b"] * counts[1],
        dataset_ids=["nissan"] * n_segments,
        vehicle_ids=[prefix] * n_segments,
    )


def test_module_import_does_not_import_optuna_symbol() -> None:
    assert "optuna" not in vars(tuning)


def test_registered_search_space_is_exact() -> None:
    assert tuning.SEARCH_SPACE == EXPECTED_SPACE
    trial = FakeTrial(choose_last=True)
    sampled = tuning.sample_search_space(trial)
    assert trial.seen == EXPECTED_SPACE
    assert sampled == {name: choices[-1] for name, choices in EXPECTED_SPACE.items()}


def test_reduced_search_profile_fixes_expensive_encoder_dimensions() -> None:
    assert tuning.REDUCED_SEARCH_SPACE == EXPECTED_REDUCED_SPACE
    trial = FakeTrial(choose_last=True)
    sampled = tuning.sample_search_space(trial, tuning.REDUCED_SEARCH_SPACE)
    assert trial.seen == EXPECTED_REDUCED_SPACE
    assert sampled["patch_length"] == 16
    assert sampled["d_model"] == 128
    assert sampled["encoder_layers"] == 3


def test_trial_config_has_compatible_patch_heads_concepts_and_kan() -> None:
    parameters = {name: choices[-1] for name, choices in EXPECTED_SPACE.items()}
    config = tuning.build_trial_model_config(
        parameters,
        driver_channels=13,
        battery_channels=11,
        segment_length=256,
    )
    assert config.patch_length == 32
    assert config.patch_stride == 16
    assert config.n_patches == 15
    assert config.d_model == 128
    assert config.d_model % config.n_heads == 0
    assert config.encoder_layers == 4
    assert config.n_concepts == 20
    assert config.kan_grid_size == 7
    assert config.kan_hidden == (24, 8)


def test_trial_config_rejects_insufficient_concepts() -> None:
    parameters = {name: choices[0] for name, choices in EXPECTED_SPACE.items()}
    with pytest.raises(ValueError, match="only 8 are available"):
        tuning.build_trial_model_config(
            parameters,
            driver_channels=13,
            battery_channels=11,
            segment_length=256,
            available_concepts=8,
        )


def test_concept_subset_view_slices_all_concept_arrays() -> None:
    view = tuning.ConceptSubsetDataset(TinyConceptDataset(), 12)
    item = view[0]
    assert item["driver"].shape == (256, 13)
    for key in (
        "concept_targets",
        "concept_mask",
        "observed_concepts",
        "observed_concept_mask",
    ):
        assert item[key].shape == (12,)


def test_parser_has_separate_train_validation_and_no_test_argument(tmp_path) -> None:
    parser = tuning.build_parser()
    args = parser.parse_args(
        [
            "--train-data",
            str(tmp_path / "train.npz"),
            "--validation-data",
            str(tmp_path / "validation.npz"),
            "--output-dir",
            str(tmp_path / "output"),
            "--trip-bags",
        ]
    )
    assert args.trials == 24
    assert args.search_profile == "full"
    assert args.train_data.name == "train.npz"
    assert args.validation_data.name == "validation.npz"
    assert args.trip_bags is True
    assert not hasattr(args, "test_data")


def test_sqlite_storage_url_is_absolute(tmp_path) -> None:
    url = tuning.sqlite_storage_url(Path(tmp_path) / "study.db")
    assert url.startswith("sqlite:///")
    assert url.endswith("study.db")


def test_trip_concept_view_preserves_variable_bag_collation() -> None:
    dataset = make_trip_dataset("train", first_trip_segments=3)
    view = tuning.ConceptSubsetDataset(dataset, 12)
    loader = tuning._loader(
        view,
        2,
        shuffle=False,
        workers=0,
        device=torch.device("cpu"),
        seed=42,
    )
    batch = next(iter(loader))
    assert batch["driver"].shape == (2, 3, 64, 13)
    assert batch["segment_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert batch["concept_targets"].shape == (2, 12)
    assert batch["current_soh_target"].shape == (2,)


def test_objective_accepts_different_segment_counts_and_keeps_trip_rank() -> None:
    training = make_trip_dataset("train", first_trip_segments=2)
    validation = make_trip_dataset("validation", first_trip_segments=3)

    class DummyOptuna:
        class TrialPruned(RuntimeError):
            pass

    objective = tuning.make_objective(
        optuna_module=DummyOptuna,
        train_dataset=training,
        validation_dataset=validation,
        device=torch.device("cpu"),
        precision="fp32",
        pretrain_epochs=0,
        finetune_epochs=1,
        pretrain_batch_size=4,
        finetune_batch_size=1,
        gradient_accumulation=1,
        pretrain_lr=3.0e-4,
        encoder_lr=1.0e-5,
        head_lr=3.0e-4,
        weight_decay=0.01,
        gradient_clip=1.0,
        num_workers=0,
        base_seed=42,
        domain="nissan",
        freeze_encoders=True,
    )
    assert callable(objective)
