from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dbpatchkan.analysis.stage10_normalized_transfer import (
    DOMAIN_ORDER,
    FEATURE_NAMES,
    DomainCalibratedConceptModel,
    choose_gate,
    expanding_splits,
    feature_matrix,
)


def test_concept_budget_and_feature_transform() -> None:
    assert len(FEATURE_NAMES) == 20
    frame = pd.DataFrame(
        {
            "calendar_days": [30.0],
            "distance_km": [500.0],
            "efc": [2.0],
            "ah_throughput": [200.0],
            "charge_event_count": [3.0],
            "charge_depth_pct": [120.0],
            "ambient_mean_c": [20.0],
            "pack_temp_mean_c": [25.0],
            "pack_temp_spread_c": [2.0],
            "hot_exposure_fraction": [0.1],
            "c_rate_p95": [1.2],
            "high_c_rate_fraction": [0.05],
            "c_rate_squared_dose_h": [4.0],
            "electrothermal_dose_h": [5.0],
            "transient_dose": [1.0],
            "dbsi_mean": [35.0],
            "stress_dose_h": [2.0],
            "pack_damage_p50": [1.5],
            "relative_90pct_width": [0.2],
            "anchor_soh_pct": [95.0],
        }
    )
    matrix = feature_matrix(frame)
    assert tuple(matrix.columns) == FEATURE_NAMES
    assert np.isclose(matrix.iloc[0]["log1p_distance_km"], np.log1p(500.0))


def test_domain_model_has_nonnegative_latent_and_positive_scale() -> None:
    model = DomainCalibratedConceptModel("KAN", len(FEATURE_NAMES))
    x = torch.zeros((len(DOMAIN_ORDER), len(FEATURE_NAMES)))
    domains = torch.arange(len(DOMAIN_ORDER))
    observed, latent = model(x, domains)
    assert observed.shape == (len(DOMAIN_ORDER), 3)
    assert torch.all(latent >= 0)
    assert torch.all(torch.nn.functional.softplus(model.domain_log_scale) > 0)


def test_expanding_splits_are_past_only_and_disjoint() -> None:
    for train, test in expanding_splits(43):
        assert not np.intersect1d(train, test).size
        assert train.max() < test.min()


def test_gate_falls_back_when_model_does_not_improve_two_percent() -> None:
    truth = np.array([1.0, 2.0, 1.5])
    alpha, _, persistence = choose_gate(truth, np.zeros(3))
    assert alpha == 0.0
    assert np.isclose(persistence, 1.5)


def test_relative_loss_is_reversible() -> None:
    anchor = np.array([96.0, 80.0, 100.0])
    future = np.array([95.5, 80.5, 97.0])
    relative = 100.0 * (anchor - future) / anchor
    reconstructed = anchor * (1.0 - relative / 100.0)
    assert np.allclose(reconstructed, future)
