from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dbpatchkan.models import DBPatchKAN, DBPatchKANConfig, make_synthetic_dataset
from dbpatchkan.training.evaluate import (
    build_parser,
    detect_npz_dataset_kind,
    evaluate_checkpoint,
    load_checkpoint_model,
    state_fingerprint,
)


def evaluation_config() -> DBPatchKANConfig:
    return DBPatchKANConfig(
        segment_length=32,
        patch_length=8,
        patch_stride=4,
        d_model=32,
        encoder_layers=1,
        n_heads=4,
        ff_dim=64,
        dropout=0.0,
        concept_names=(
            "bsi_rate",
            "bsi_thermal",
            "bsi_soc",
            "bsi_cycle",
            "bsi_transient",
            "bsi_interaction",
            "behavior_share_eco",
            "behavior_share_balanced",
            "behavior_share_aggressive",
            "calendar_age_days",
            "odometer_km",
            "cumulative_stress_dose",
        ),
    )


def save_selected_checkpoint(path: Path, config: DBPatchKANConfig, *, trip: bool) -> str:
    torch.manual_seed(11)
    model = DBPatchKAN(config)
    model.set_input_normalization(
        torch.linspace(0.0, 1.0, config.driver_channels),
        torch.linspace(1.0, 2.0, config.driver_channels),
        torch.linspace(0.0, 1.0, config.battery_channels),
        torch.linspace(1.0, 2.0, config.battery_channels),
    )
    fingerprint = state_fingerprint(model)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.config.to_dict(),
            "stage": "finetune",
            "epoch": 3,
            "metrics": {"validation_current_soh_mae_pp": 0.42},
            "training_config": {"domain": "nissan", "trip_bags": trip},
        },
        path,
    )
    return fingerprint


def encode_utf8(values: list[str]) -> np.ndarray:
    width = max(len(value.encode("utf-8")) for value in values)
    output = np.zeros((len(values), width), dtype=np.uint8)
    for index, value in enumerate(values):
        encoded = value.encode("utf-8")
        output[index, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return output


def save_synthetic_npz(
    path: Path, config: DBPatchKANConfig, *, samples: int = 4, trip_metadata: bool = False
) -> None:
    dataset = make_synthetic_dataset(config, n_samples=samples, seed=22)
    dataset.to_npz(path)
    if not trip_metadata:
        return
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays.update(
        {
            "trip_key_utf8": encode_utf8([f"trip:{index}" for index in range(samples)]),
            "dataset_id_utf8": encode_utf8(["synthetic"] * samples),
            "vehicle_id_utf8": encode_utf8(["vehicle:1"] * samples),
            "window_start_unix_ms": np.arange(samples, dtype=np.int64) * 1000,
        }
    )
    np.savez_compressed(path, **arrays)


def test_checkpoint_loader_restores_buffers_and_freezes_every_parameter(tmp_path) -> None:
    config = evaluation_config()
    checkpoint_path = tmp_path / "best.pt"
    original_fingerprint = save_selected_checkpoint(checkpoint_path, config, trip=False)
    model, checkpoint = load_checkpoint_model(checkpoint_path, device=torch.device("cpu"))
    assert checkpoint["stage"] == "finetune"
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert state_fingerprint(model) == original_fingerprint
    torch.testing.assert_close(
        model.driver_input_center,
        torch.linspace(0.0, 1.0, config.driver_channels),
    )


def test_segment_checkpoint_evaluation_writes_frozen_metrics_and_predictions(
    tmp_path,
) -> None:
    config = evaluation_config()
    checkpoint_path = tmp_path / "best.pt"
    data_path = tmp_path / "synthetic_holdout.npz"
    output = tmp_path / "frozen_evaluation"
    save_selected_checkpoint(checkpoint_path, config, trip=False)
    save_synthetic_npz(data_path, config)
    assert detect_npz_dataset_kind(data_path) == "segment"

    result = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        test_data_path=data_path,
        output_dir=output,
        device_name="cpu",
        precision_name="fp32",
        batch_size=2,
        acknowledge_final_evaluation=True,
    )
    report = json.loads(Path(result["metrics"]).read_text(encoding="utf-8"))
    predictions = Path(result["predictions"]).read_text(encoding="utf-8")
    assert result["sample_count"] == 4
    assert report["evaluation_status"] == "frozen_untouched_test_complete"
    assert report["evaluation_grain"] == "segment"
    assert report["fit_operations"] == []
    assert report["optimizer_created"] is False
    assert report["weights_updated"] is False
    assert report["state_fingerprint_before"] == report["state_fingerprint_after"]
    assert report["regression"]["current_soh"]["count"] == 4
    assert "split_role,current_soh_pred_pct" in predictions

    with pytest.raises(FileExistsError, match="already exist"):
        evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            test_data_path=data_path,
            output_dir=output,
            device_name="cpu",
            acknowledge_final_evaluation=True,
        )


def test_trip_checkpoint_auto_dispatches_to_one_prediction_per_trip(tmp_path) -> None:
    config = evaluation_config()
    checkpoint_path = tmp_path / "best.pt"
    data_path = tmp_path / "synthetic_trip_holdout.npz"
    output = tmp_path / "trip_evaluation"
    save_selected_checkpoint(checkpoint_path, config, trip=True)
    save_synthetic_npz(data_path, config, samples=3, trip_metadata=True)
    # The exchange tensor remains rank 3; checkpoint metadata selects trip bags.
    assert detect_npz_dataset_kind(data_path) == "segment"
    result = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        test_data_path=data_path,
        output_dir=output,
        device_name="cpu",
        batch_size=2,
        acknowledge_final_evaluation=True,
    )
    report = json.loads(Path(result["metrics"]).read_text(encoding="utf-8"))
    predictions = Path(result["predictions"]).read_text(encoding="utf-8")
    assert result["sample_count"] == 3
    assert report["evaluation_grain"] == "trip"
    assert "trip_key" in predictions
    assert "trip_attention_entropy" in predictions


def test_programmatic_and_cli_paths_require_final_freeze_acknowledgement(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--test-data",
            str(tmp_path / "holdout.npz"),
            "--output-dir",
            str(tmp_path / "result"),
        ]
    )
    assert args.final_evaluation is False
    with pytest.raises(ValueError, match="must be frozen"):
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            test_data_path=args.test_data,
            output_dir=args.output_dir,
        )
