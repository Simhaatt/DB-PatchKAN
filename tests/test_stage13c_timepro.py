from __future__ import annotations

import torch

from dbpatchkan.analysis.stage13_persistence_forecast import FORECAST_CONCEPTS
from dbpatchkan.analysis.stage13c_timepro import (
    MODEL_NAME,
    PersistenceAnchoredTimePro,
    SelectiveStateSpace,
    VariableTimeHyperState,
)


def test_timepro_output_shape_and_gradient():
    model = PersistenceAnchoredTimePro(blocks=1)
    sequence = torch.randn(3, 128, 8)
    concepts = torch.randn(3, len(FORECAST_CONCEPTS))
    output = model(sequence, concepts)
    assert output.shape == (3,)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_selective_state_space_shape():
    layer = SelectiveStateSpace(32, state_size=4)
    output = layer(torch.randn(2, 8, 32))
    assert output.shape == (2, 8, 32)
    assert torch.isfinite(output).all()


def test_hyper_state_grid_shape_and_posthoc_name():
    layer = VariableTimeHyperState(4, 8, 3)
    output = layer(torch.randn(2, 3, 32))
    assert output.shape == (2, 3, 32)
    assert "TimePro_2025_POSTHOC" in MODEL_NAME
