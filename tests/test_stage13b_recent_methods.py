from __future__ import annotations

import torch

from dbpatchkan.analysis.stage13_persistence_forecast import FORECAST_CONCEPTS
from dbpatchkan.analysis.stage13b_recent_methods import (
    METHODS,
    MODEL_NAMES,
    PersistenceAnchoredKALinformer,
    PersistenceAnchoredModernTCN,
    PersistenceAnchoredTSLANet,
    make_model,
    parameter_count,
)


def test_recent_method_output_shapes_and_finite_values():
    sequence = torch.randn(3, 128, 8)
    concepts = torch.randn(3, len(FORECAST_CONCEPTS))
    for method in METHODS:
        model = make_model(method)
        output = model(sequence, concepts)
        assert output.shape == (3,)
        assert torch.isfinite(output).all()
        assert parameter_count(model) > 0
        assert torch.isfinite(model.regularization_loss())


def test_recent_method_names_are_unique_and_dated():
    assert len(MODEL_NAMES) == len(set(MODEL_NAMES.values())) == 3
    assert all("2024" in name or "2025" in name for name in MODEL_NAMES.values())


def test_architecture_classes_construct():
    assert isinstance(make_model("moderntcn"), PersistenceAnchoredModernTCN)
    assert isinstance(make_model("tslanet"), PersistenceAnchoredTSLANet)
    assert isinstance(make_model("kal_linformer"), PersistenceAnchoredKALinformer)
