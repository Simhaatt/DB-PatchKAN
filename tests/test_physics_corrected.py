import torch

from dbpatchkan.models.physics_corrected import (
    BoundedMonotonicResidualHead,
    ScaledForecastHead,
)


def test_current_residual_is_bounded_and_monotone() -> None:
    head = BoundedMonotonicResidualHead(4, 5, limit_pp=1.0)
    interaction = torch.zeros(2, 5)
    low = torch.zeros(2, 4)
    high = low.clone(); high[:, 0] = 1.0
    low_y = head(low, interaction)["correction_pp"]
    high_y = head(high, interaction)["correction_pp"]
    assert torch.all(high_y >= low_y)
    assert torch.all(high_y.abs() <= 1.0)


def test_forecast_is_nonnegative_and_initially_on_label_scale() -> None:
    head = ScaledForecastHead(
        4, 7, output_scale_pp=0.1, initial_median_pp=0.04
    )
    output = head(torch.zeros(3, 4), torch.zeros(3, 7))["loss_pp"]
    assert torch.all(output >= 0)
    assert torch.all(output < 0.2)
