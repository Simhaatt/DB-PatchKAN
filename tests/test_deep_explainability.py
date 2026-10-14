from __future__ import annotations

import numpy as np
import pandas as pd

from dbpatchkan.analysis.deep_explainability import (
    aggregate_response_curves,
    channel_gate_table,
    concept_integrated_gradients,
    concept_response_curves,
    kan_edge_strength_table,
    kan_path_importance_table,
)
from dbpatchkan.models import DBPatchKAN, DBPatchKANConfig


def _small_kan() -> DBPatchKAN:
    config = DBPatchKANConfig(
        driver_channels=4,
        battery_channels=4,
        segment_length=16,
        patch_length=4,
        patch_stride=4,
        d_model=16,
        encoder_layers=1,
        n_heads=4,
        ff_dim=32,
        concept_names=("bsi_rate", "calendar_age_days", "odometer_km"),
        kan_hidden=(4,),
        protected_driver_channels=(0,),
        protected_battery_channels=(0,),
    )
    model = DBPatchKAN(config).eval()
    model.set_concept_normalization(
        np.asarray([0.2, 100.0, 5_000.0]),
        np.asarray([0.1, 50.0, 1_000.0]),
    )
    return model


def test_kan_graph_and_path_tables_are_complete_and_normalized() -> None:
    model = _small_kan()
    edges = kan_edge_strength_table(model, seed=42)
    assert len(edges) == 3 * 4 + 4 * 3
    assert set(edges["Seed"]) == {42}
    paths = kan_path_importance_table(model, seed=42)
    assert len(paths) == 3 * 3
    totals = paths.groupby("Output_Name")["Normalized_Path_Importance"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_response_curves_aggregation_and_integrated_gradients_are_finite() -> None:
    model = _small_kan()
    ranges = {
        "bsi_rate": (0.0, 1.0),
        "calendar_age_days": (0.0, 200.0),
        "odometer_km": (3_000.0, 8_000.0),
    }
    first = concept_response_curves(model, ranges, points=11, seed=42)
    second = concept_response_curves(model, ranges, points=11, seed=52)
    assert len(first) == 3 * 3 * 11
    aggregate, stability = aggregate_response_curves(
        pd.concat([first, second], ignore_index=True)
    )
    assert len(aggregate) == len(first)
    assert len(stability) == 3 * 3
    assert np.isfinite(aggregate["Response_Mean"]).all()

    concepts = np.asarray([[0.1, 80.0, 4_500.0], [0.4, 140.0, 6_000.0]])
    gradients = concept_integrated_gradients(model, concepts, steps=4, seed=42)
    assert len(gradients) == 3
    assert np.isfinite(gradients["IG_Mean_Absolute"]).all()


def test_channel_gate_table_covers_both_streams() -> None:
    model = _small_kan()
    gates = channel_gate_table(model, seed=42)
    assert len(gates) == 8
    assert set(gates["Stream"]) == {"driver", "battery"}
