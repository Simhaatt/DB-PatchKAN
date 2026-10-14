from __future__ import annotations

import torch
import numpy as np

from dbpatchkan.models import (
    BSplineKANLayer,
    DBPatchKAN,
    DBPatchKANConfig,
    DBPatchKANMultiTaskLoss,
    EVSegmentDataset,
    MaskedSelfSupervisedLoss,
    StochasticChannelGate,
    make_synthetic_dataset,
)


def tiny_config(*, fusion_head: str = "kan") -> DBPatchKANConfig:
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
        concept_names=(
            "bsi_rate",
            "bsi_thermal",
            "bsi_soc",
            "bsi_cycle",
            "behavior_eco",
            "behavior_balanced",
            "calendar_age",
            "odometer",
        ),
        protected_driver_channels=(0,),
        protected_battery_channels=(0, 1, 2, 3),
        fusion_head=fusion_head,
    )


def stack_batch(dataset, count: int = 2) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([dataset[index][key] for index in range(count)])
        for key in dataset[0]
    }


def test_default_patch_layout_is_exactly_31_patches() -> None:
    config = DBPatchKANConfig()
    assert config.segment_length == 256
    assert config.patch_length == 16
    assert config.patch_stride == 8
    assert config.n_patches == 31
    assert config.d_model == 128
    assert config.encoder_layers == 3
    assert config.n_heads == 4
    assert config.ff_dim == 256


def test_cubic_kan_is_fp32_and_differentiable() -> None:
    layer = BSplineKANLayer(4, 3, grid_size=5, spline_order=3)
    values = torch.randn(2, 4, dtype=torch.float16, requires_grad=True)
    output = layer(values)
    assert output.shape == (2, 3)
    assert output.dtype == torch.float32
    assert layer.b_splines(values.float()).shape == (2, 4, 8)
    output.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_stochastic_gate_never_prunes_protected_channels() -> None:
    gate = StochasticChannelGate(4, protected_indices=(0, 2), noise_std=0.5)
    gate.train()
    _, sampled = gate(torch.ones(2, 5, 4))
    assert sampled[0].item() == 1.0
    assert sampled[2].item() == 1.0
    gate.eval()
    deterministic = gate.deterministic_gates()
    assert deterministic[0].item() == 1.0
    assert deterministic[2].item() == 1.0


def test_supervised_shapes_softplus_forecasts_and_all_missing_masks() -> None:
    config = tiny_config()
    model = DBPatchKAN(config).eval()
    driver = torch.randn(2, config.segment_length, config.driver_channels)
    battery = torch.randn(2, config.segment_length, config.battery_channels)
    driver_mask = torch.ones_like(driver, dtype=torch.bool)
    battery_mask = torch.ones_like(battery, dtype=torch.bool)
    # An all-missing sample must be finite and must not trigger all-key-masked NaNs.
    driver_mask[1] = False
    battery_mask[1] = False
    driver[0, 2, 1] = float("nan")
    outputs = model(
        driver,
        battery,
        driver_mask=driver_mask,
        battery_mask=battery_mask,
    )
    assert outputs["current_soh"].shape == (2,)
    assert outputs["behavior_logits"].shape == (2, 3)
    assert outputs["driver_logits"].shape == (2, 7)
    assert outputs["concepts"].shape == (2, 8)
    assert outputs["embedding"].shape == (2, 32)
    assert outputs["driver_patch_attention"].shape == (2, 7)
    assert outputs["battery_patch_attention"].shape == (2, 7)
    assert torch.isfinite(outputs["current_soh"]).all()
    assert torch.all(outputs["loss_30d"] >= 0)
    assert torch.all(outputs["loss_500km"] >= 0)
    torch.testing.assert_close(
        outputs["future_soh_30d"], outputs["current_soh"] - outputs["loss_30d"]
    )


def test_ssl_shapes_masked_loss_and_one_optimizer_step() -> None:
    torch.manual_seed(7)
    config = tiny_config()
    dataset = make_synthetic_dataset(config, n_samples=2, seed=7)
    batch = stack_batch(dataset)
    model = DBPatchKAN(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    ssl_outputs = model.forward_ssl(
        batch["driver"],
        batch["battery"],
        driver_mask=batch["driver_mask"],
        battery_mask=batch["battery_mask"],
        mask_ratio=0.4,
    )
    assert ssl_outputs["driver_reconstruction"].shape == (2, 7, 8, 6)
    assert ssl_outputs["battery_reconstruction"].shape == (2, 7, 8, 5)
    assert ssl_outputs["driver_masked_patches"].any(dim=1).all()
    ssl_losses = MaskedSelfSupervisedLoss()(ssl_outputs)
    assert torch.isfinite(ssl_losses["total"])

    optimizer.zero_grad(set_to_none=True)
    supervised = model(
        batch["driver"],
        batch["battery"],
        driver_mask=batch["driver_mask"],
        battery_mask=batch["battery_mask"],
        observed_concepts=batch["observed_concepts"],
        observed_concept_mask=batch["observed_concept_mask"],
    )
    criterion = DBPatchKANMultiTaskLoss()
    loss = criterion(supervised, batch)["total"]
    before = model.fusion_projection[0].weight.detach().clone()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    after = model.fusion_projection[0].weight.detach()
    assert torch.isfinite(loss)
    assert not torch.equal(before, after)


def test_mlp_challenger_has_identical_public_outputs() -> None:
    config = tiny_config(fusion_head="mlp")
    model = DBPatchKAN(config).eval()
    driver = torch.zeros(1, config.segment_length, config.driver_channels)
    battery = torch.zeros(1, config.segment_length, config.battery_channels)
    outputs = model(driver, battery, domain="capacity")
    assert outputs["current_soh"].shape == (1,)
    assert outputs["kan_regularization"].item() == 0.0


def test_npz_loader_augments_cpu_exported_streams(tmp_path) -> None:
    n, time = 2, 32
    driver = np.ones((n, time, 9), dtype=np.float32)
    battery = np.ones((n, time, 7), dtype=np.float32)
    driver_missing = np.zeros_like(driver)
    battery_missing = np.zeros_like(battery)
    driver_missing[0, 0, 2] = 1.0
    source = tmp_path / "segments.npz"
    np.savez_compressed(
        source,
        driver=driver,
        battery=battery,
        driver_missing=driver_missing,
        battery_missing=battery_missing,
        validity_mask=np.ones((n, time), dtype=np.float32),
        time_delta_s=np.ones((n, time), dtype=np.float32),
        relative_position=np.broadcast_to(
            np.linspace(0, 1, time, dtype=np.float32), (n, time)
        ),
        current_soh_target=np.array([96.0, 95.9], dtype=np.float32),
    )
    dataset = EVSegmentDataset.from_npz(source)
    assert dataset.driver.shape == (n, time, 13)
    assert dataset.battery.shape == (n, time, 11)
    assert not dataset.driver_mask[0, 0, 2]
    assert dataset.driver[0, 0, -1].item() > 0
