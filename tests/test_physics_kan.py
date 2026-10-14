import torch

from dbpatchkan.models.physics_kan import PhysicsMediatedKAN


def test_physics_path_is_monotone_and_output_nonnegative() -> None:
    torch.manual_seed(7)
    model = PhysicsMediatedKAN(3, 4, residual_limit_pp=0.25)
    behaviour = torch.zeros(2, 4)
    low = torch.zeros(2, 3)
    high = low.clone()
    high[:, 1] = 1.0
    low_output = model(low, behaviour)["loss_pp"]
    high_output = model(high, behaviour)["loss_pp"]
    assert torch.all(high_output >= low_output)
    assert torch.all(low_output >= 0)


def test_behaviour_residual_is_bounded() -> None:
    model = PhysicsMediatedKAN(2, 3, residual_limit_pp=0.4)
    result = model(torch.randn(8, 2), 100 * torch.randn(8, 3))
    assert torch.all(result["behaviour_residual"].abs() <= 0.400001)
