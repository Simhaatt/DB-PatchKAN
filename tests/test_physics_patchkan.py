import torch

from dbpatchkan.models.patchkan import DBPatchKAN, DBPatchKANConfig
from dbpatchkan.models.physics_patchkan import PhysicsInformedDBPatchKAN


def test_hybrid_has_no_latent_soh_shortcut_and_shapes_are_correct() -> None:
    config = DBPatchKANConfig(
        driver_channels=13, battery_channels=11, segment_length=32,
        patch_length=8, patch_stride=8, d_model=16, encoder_layers=1,
        n_heads=4, ff_dim=32, dropout=0.0,
    )
    model = PhysicsInformedDBPatchKAN(DBPatchKAN(config))
    driver = torch.randn(2, 2, 32, 13)
    battery = torch.randn(2, 2, 32, 11)
    output = model.forward_trip_bag(
        driver, battery, segment_mask=torch.ones(2, 2, dtype=torch.bool)
    )
    assert output["current_soh"].shape == (2,)
    assert output["loss_30d"].shape == (2,)
    assert output["physics_concepts"].shape == (2, 20)
    assert output["response_predictions_normalized"].shape == (2, 7)
    assert output["stress_class_logits"].shape == (2, 3)
    assert torch.all(output["loss_500km"] >= 0)


def test_observed_physics_monotonicity() -> None:
    config = DBPatchKANConfig(
        driver_channels=13, battery_channels=11, segment_length=16,
        patch_length=8, patch_stride=8, d_model=16, encoder_layers=1,
        n_heads=4, ff_dim=32, dropout=0.0,
    )
    model = PhysicsInformedDBPatchKAN(DBPatchKAN(config)).eval()
    driver = torch.zeros(2, 1, 16, 13)
    battery = torch.zeros(2, 1, 16, 11)
    physics = torch.zeros(2, 20)
    physics[1, 2] = 1.0
    output = model.forward_trip_bag(
        driver, battery, segment_mask=torch.ones(2, 1, dtype=torch.bool),
        observed_physics=physics, concept_source="observed",
    )
    assert output["current_degradation_pp"][1] >= output["current_degradation_pp"][0]
