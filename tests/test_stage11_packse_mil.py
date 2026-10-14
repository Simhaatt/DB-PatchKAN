from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dbpatchkan.analysis.stage11_packse_mil import (
    INTERVAL_CONCEPT_NAMES,
    KAN_CONCEPT_NAMES,
    TRIP_CONCEPT_NAMES,
    PackSEResidualMIL,
    build_trip_bags,
    choose_simplex_weights,
)


def test_mil_kan_has_twenty_named_inputs() -> None:
    assert len(TRIP_CONCEPT_NAMES) == 12
    assert len(INTERVAL_CONCEPT_NAMES) == 8
    assert len(KAN_CONCEPT_NAMES) == 20
    assert len(set(KAN_CONCEPT_NAMES)) == 20


def test_trip_bags_compress_chronologically() -> None:
    intervals = pd.DataFrame({
        "dataset": ["RWTH"], "vehicle_id": ["v1"], "interval_id": ["v1:0-1"],
        "interval_start": ["2020-01-01"], "interval_end": ["2020-01-11"],
    })
    trips = pd.DataFrame({
        "dataset": ["RWTH"] * 10, "vehicle_id": ["v1"] * 10,
        "trip_end": pd.date_range("2020-01-02", periods=10, freq="D", tz="UTC"),
        "duration_h": np.ones(10), "distance_km": np.arange(10),
        **{name: np.arange(10, dtype=float) for name in TRIP_CONCEPT_NAMES},
    })
    bags = build_trip_bags(intervals, trips, max_tokens=4)
    assert bags["source_trip_count"].tolist() == [10]
    assert bags["compressed_token_count"].tolist() == [4]
    assert bags["token_mask"].sum() == 4
    # Chronological bins must retain increasing means.
    speed = bags["tokens"][0, :4, 0]
    assert np.all(np.diff(speed) > 0)


def test_packse_mil_outputs_valid_attention_and_positive_physical_latent() -> None:
    model = PackSEResidualMIL("KAN", max_tokens=6)
    tokens = torch.zeros((3, 6, len(TRIP_CONCEPT_NAMES)))
    mask = torch.tensor([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
    ], dtype=torch.bool)
    context = torch.zeros((3, len(INTERVAL_CONCEPT_NAMES)))
    prediction, detail = model(tokens, mask, context, torch.ones(3), torch.arange(3))
    assert prediction.shape == (3,)
    assert detail["concepts"].shape == (3, 20)
    assert torch.all(detail["latent_physical"] > 0)
    assert torch.allclose(detail["attention"].sum(dim=1), torch.ones(3))
    assert torch.all(detail["attention"][~mask] < 1e-6)


def test_simplex_selector_can_choose_rf_expert() -> None:
    truth = np.array([1.0, 2.0, 3.0])
    experts = np.column_stack((
        np.zeros(3),
        np.array([5.0, 5.0, 5.0]),
        truth,
        np.array([4.0, 4.0, 4.0]),
    ))
    weights, error = choose_simplex_weights(truth, experts)
    assert np.isclose(error, 0.0)
    assert np.argmax(weights) == 2
    assert np.isclose(weights.sum(), 1.0)
