from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dbpatchkan.analysis.stage12_evbattery import PHYSICAL_CONCEPTS
from dbpatchkan.analysis.stage13_persistence_forecast import (
    FORECAST_CONTEXT,
    FORECAST_PHYSICAL,
    PersistenceAnchoredPatchConcept,
    build_transition_cache,
    load_transition_cache,
)
from dbpatchkan.data.evbattery import EVBATTERY_CHANNELS, save_evbattery_sample


def test_transition_cache_collapses_same_mileage(tmp_path):
    rows = []
    for vehicle in (1, 2):
        for mileage, capacity in ((100.0, 40.0), (100.0, 40.01), (600.0, 39.8), (1500.0, 39.5)):
            row = {
                "manufacturer_id": 1,
                "vehicle_id": vehicle,
                "snippet_id": str(len(rows)),
                "charge_segment": str(len(rows)),
                "capacity_ah": capacity,
                "anomaly_label": 0,
                "source_label": "00",
                "mileage": mileage,
            }
            row.update({name: 1.0 for name in PHYSICAL_CONCEPTS})
            rows.append(row)
    sample = {
        "sequence": np.zeros((len(rows), 128, 8), dtype="float32"),
        "channels": np.asarray(EVBATTERY_CHANNELS),
        "concepts": pd.DataFrame(rows),
    }
    source = save_evbattery_sample(tmp_path / "source.npz", sample)
    target = build_transition_cache(source, tmp_path / "transitions.npz")
    loaded = load_transition_cache(target)
    assert loaded["sequence"].shape == (4, 128, 8)
    assert len(loaded["table"]) == 4
    assert sorted(loaded["table"]["forecast_horizon_km"].tolist()) == [500.0, 500.0, 900.0, 900.0]


def test_persistence_anchored_model_shapes():
    model = PersistenceAnchoredPatchConcept(head="kan", d_model=32)
    residual, physical, attention = model(
        torch.randn(3, 128, 8),
        torch.randn(3, len(FORECAST_PHYSICAL)),
        torch.randn(3, len(FORECAST_CONTEXT)),
    )
    assert residual.shape == (3,)
    assert physical.shape == (3, len(FORECAST_PHYSICAL))
    assert attention.shape[0] == 3
