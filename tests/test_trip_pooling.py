from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dbpatchkan.models import (
    DBPatchKAN,
    DBPatchKANConfig,
    DBPatchKANMultiTaskLoss,
    EVSegmentDataset,
    EVTripBagDataset,
    assert_disjoint_trip_bags,
    collate_trip_bags,
)
from dbpatchkan.training.train import TrainingConfig, evaluate_finetuning, train


def _config() -> DBPatchKANConfig:
    return DBPatchKANConfig(
        driver_channels=6,
        battery_channels=5,
        segment_length=32,
        patch_length=8,
        patch_stride=4,
        d_model=32,
        encoder_layers=1,
        n_heads=4,
        ff_dim=64,
        dropout=0.0,
        concept_names=tuple(f"concept_{index}" for index in range(8)),
        protected_driver_channels=(0,),
        protected_battery_channels=(0, 1, 2, 3),
    )


def _segments() -> EVSegmentDataset:
    generator = torch.Generator().manual_seed(9)
    # Within each trip the windows are deliberately identical.  That makes it
    # possible to verify that duplication does not duplicate trip-level loss.
    driver_trip_a = torch.randn(1, 32, 6, generator=generator)
    battery_trip_a = torch.randn(1, 32, 5, generator=generator)
    driver_trip_b = torch.randn(1, 32, 6, generator=generator)
    battery_trip_b = torch.randn(1, 32, 5, generator=generator)
    driver = torch.cat((driver_trip_a.repeat(3, 1, 1), driver_trip_b.repeat(2, 1, 1)))
    battery = torch.cat((battery_trip_a.repeat(3, 1, 1), battery_trip_b.repeat(2, 1, 1)))
    concepts = torch.cat((torch.ones(3, 8), torch.full((2, 8), 2.0)))
    return EVSegmentDataset(
        driver,
        battery,
        driver_mask=torch.ones_like(driver, dtype=torch.bool),
        battery_mask=torch.ones_like(battery, dtype=torch.bool),
        concept_targets=concepts,
        concept_mask=torch.ones_like(concepts, dtype=torch.bool),
        observed_concepts=concepts,
        observed_concept_mask=torch.ones_like(concepts, dtype=torch.bool),
        targets={
            "current_soh_target": torch.tensor([96.0, 96.0, 96.0, 91.0, 91.0]),
            "loss_30d_target": torch.tensor([0.1, 0.1, 0.1, 0.2, 0.2]),
            "loss_500km_target": torch.tensor([0.1, 0.1, 0.1, 0.2, 0.2]),
            "behavior_target": torch.tensor([0, 0, 0, 2, 2]),
            "driver_target": torch.tensor([0, 0, 0, 4, 4]),
        },
    )


def _bags() -> EVTripBagDataset:
    return EVTripBagDataset(
        _segments(),
        ["trip-a", "trip-a", "trip-a", "trip-b", "trip-b"],
        dataset_ids=["nissan"] * 5,
        vehicle_ids=["leaf-1"] * 5,
        window_starts=[0, 64, 128, 200, 264],
    )


def _utf8(values: list[str]) -> np.ndarray:
    width = max(map(len, values))
    result = np.zeros((len(values), width), dtype=np.uint8)
    for row, value in enumerate(values):
        encoded = value.encode()
        result[row, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return result


def test_trip_bag_groups_segments_and_collates_one_target_row_per_trip() -> None:
    dataset = _bags()
    assert len(dataset) == 2
    assert dataset[0]["driver"].shape == (3, 32, 6)
    assert dataset[0]["current_soh_target"].ndim == 0
    batch = collate_trip_bags([dataset[0], dataset[1]])
    assert batch["driver"].shape == (2, 3, 32, 6)
    assert batch["segment_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert batch["current_soh_target"].tolist() == [96.0, 91.0]
    assert batch["concept_targets"].shape == (2, 8)


def test_trip_pooling_outputs_and_losses_are_once_per_trip() -> None:
    torch.manual_seed(5)
    dataset = _bags()
    batch = collate_trip_bags([dataset[0], dataset[1]])
    model = DBPatchKAN(_config()).eval()
    outputs = model.forward_trip_bag(
        batch["driver"],
        batch["battery"],
        segment_mask=batch["segment_mask"],
        driver_mask=batch["driver_mask"],
        battery_mask=batch["battery_mask"],
        observed_concepts=batch["observed_concepts"],
        observed_concept_mask=batch["observed_concept_mask"],
    )
    assert outputs["current_soh"].shape == (2,)
    assert outputs["concepts"].shape == (2, 8)
    assert outputs["trip_segment_attention"].shape == (2, 3)
    assert outputs["trip_segment_attention"][1, 2].item() == 0.0
    torch.testing.assert_close(
        outputs["trip_segment_attention"].sum(dim=1), torch.ones(2)
    )
    losses = DBPatchKANMultiTaskLoss()(outputs, batch)
    assert torch.isfinite(losses["total"])

    # Repeating an identical segment inside one trip does not multiply either
    # its prediction row or its supervised/concept loss contribution.
    single = EVTripBagDataset(
        EVSegmentDataset(
            dataset.segments.driver[:1],
            dataset.segments.battery[:1],
            driver_mask=dataset.segments.driver_mask[:1],
            battery_mask=dataset.segments.battery_mask[:1],
            concept_targets=dataset.segments.concept_targets[:1],
            concept_mask=dataset.segments.concept_mask[:1],
            observed_concepts=dataset.segments.observed_concepts[:1],
            observed_concept_mask=dataset.segments.observed_concept_mask[:1],
            targets={name: value[:1] for name, value in dataset.segments.targets.items()},
        ),
        ["trip-a"],
    )
    one = collate_trip_bags([single[0]])
    one_output = model.forward_trip_bag(
        one["driver"],
        one["battery"],
        segment_mask=one["segment_mask"],
        driver_mask=one["driver_mask"],
        battery_mask=one["battery_mask"],
        observed_concepts=one["observed_concepts"],
        observed_concept_mask=one["observed_concept_mask"],
    )
    torch.testing.assert_close(outputs["current_soh"][:1], one_output["current_soh"])


def test_evaluation_metrics_use_two_trips_not_five_windows() -> None:
    dataset = _bags()
    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    model = DBPatchKAN(_config()).eval()
    batch = next(iter(loader))
    with torch.no_grad():
        expected = model.forward_trip_bag(
            batch["driver"],
            batch["battery"],
            segment_mask=batch["segment_mask"],
            driver_mask=batch["driver_mask"],
            battery_mask=batch["battery_mask"],
            observed_concepts=batch["observed_concepts"],
            observed_concept_mask=batch["observed_concept_mask"],
        )
    expected_mae = (
        expected["current_soh"] - batch["current_soh_target"]
    ).abs().mean()
    metrics = evaluate_finetuning(
        model,
        loader,
        DBPatchKANMultiTaskLoss(),
        device=torch.device("cpu"),
        precision="fp32",
        domain="nissan",
    )
    assert metrics["current_soh_mae_pp"] == pytest.approx(float(expected_mae), rel=1e-6)


def test_one_batch_hierarchical_training_writes_trip_checkpoint_metadata(tmp_path) -> None:
    training = _bags()
    validation = EVTripBagDataset(
        _segments(),
        ["trip-c", "trip-c", "trip-c", "trip-d", "trip-d"],
        dataset_ids=["nissan"] * 5,
        vehicle_ids=["leaf-2"] * 5,
    )
    config = TrainingConfig(
        stage="finetune",
        device="cpu",
        precision="fp32",
        finetune_batch_size=2,
        finetune_epochs=1,
        patience=1,
        max_batches=1,
    )
    result = train(
        DBPatchKAN(_config()),
        training,
        validation,
        config,
        output_dir=tmp_path,
    )
    assert result["epochs_completed"] == 1
    assert config.trip_bags is True
    checkpoint = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["training_config"]["trip_bags"] is True


def test_npz_trip_loading_and_split_overlap_guard(tmp_path) -> None:
    segments = _segments()
    source = tmp_path / "trip_segments.npz"
    arrays = {
        "driver": segments.driver.numpy(),
        "battery": segments.battery.numpy(),
        "driver_mask": segments.driver_mask.numpy(),
        "battery_mask": segments.battery_mask.numpy(),
        "concept_targets": segments.concept_targets.numpy(),
        "concept_mask": segments.concept_mask.numpy(),
        "current_soh_target": segments.targets["current_soh_target"].numpy(),
        "trip_key_utf8": _utf8(["trip-a"] * 3 + ["trip-b"] * 2),
        "dataset_id_utf8": _utf8(["nissan"] * 5),
        "vehicle_id_utf8": _utf8(["leaf-1"] * 5),
        "window_start_unix_ms": np.arange(5, dtype=np.int64),
    }
    np.savez_compressed(source, **arrays)
    loaded = EVTripBagDataset.from_npz(source)
    assert len(loaded) == 2
    with pytest.raises(ValueError, match="share trip groups"):
        assert_disjoint_trip_bags(loaded, loaded)


def test_conflicting_window_labels_are_never_averaged() -> None:
    segments = _segments()
    segments.targets["current_soh_target"][1] = 95.0
    with pytest.raises(ValueError, match="Inconsistent repeated 'current_soh_target'"):
        EVTripBagDataset(
            segments,
            ["trip-a", "trip-a", "trip-a", "trip-b", "trip-b"],
        )
