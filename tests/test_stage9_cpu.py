from __future__ import annotations

import numpy as np
import pandas as pd

from dbpatchkan.analysis.stage9_cpu import (
    add_pack_se_nmc_stress,
    comparison_matrix,
    evaluate_nested_attribution,
)


def _synthetic_intervals(dataset: str = "Nissan-development", n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "vehicle_id": "vehicle-1",
            "interval_id": [f"i-{idx}" for idx in range(n)],
            "interval_end": pd.date_range("2020-01-01", periods=n, freq="30D"),
            "calendar_days": 30.0,
            "distance_km": np.linspace(100.0, 500.0, n),
            "efc": np.linspace(0.5, 2.0, n),
            "ah_throughput": np.linspace(20.0, 80.0, n),
            "charge_event_count": np.linspace(2.0, 8.0, n),
            "charge_depth_pct": np.linspace(50.0, 200.0, n),
            "charge_high_soc_count": np.linspace(1.0, 5.0, n),
            "charge_plug_in_soc_mean_pct": 45.0,
            "ambient_mean_c": 20.0,
            "pack_temp_mean_c": 25.0,
            "pack_temp_max_c": 30.0,
            "pack_temp_spread_c": 2.0,
            "hot_exposure_fraction": 0.0,
            "speed_mean_kmh": np.linspace(20.0, 50.0, n),
            "speed_p95_kmh": np.linspace(40.0, 90.0, n),
            "accel_rms_mps2": np.linspace(0.1, 0.5, n),
            "decel_rms_mps2": np.linspace(0.1, 0.4, n),
            "jerk_rms_mps3": np.linspace(0.02, 0.2, n),
            "harsh_accel_fraction": 0.01,
            "harsh_decel_fraction": 0.01,
            "c_rate_p95": np.linspace(0.4, 1.2, n),
            "high_c_rate_fraction": np.linspace(0.0, 0.1, n),
            "c_rate_squared_dose_h": np.linspace(0.1, 1.0, n),
            "electrothermal_dose_h": np.linspace(0.1, 1.2, n),
            "transient_dose": np.linspace(0.01, 0.2, n),
            "dbsi_mean": np.linspace(2.0, 10.0, n),
            "stress_dose_h": np.linspace(0.01, 0.2, n),
            "dod_mean_pct": 40.0,
            "soc_mean_pct": 60.0,
            "exposure_duration_h": 10.0,
        }
    )
    frame["signed_loss_pp"] = (
        0.01 * frame["calendar_days"]
        + 0.02 * frame["efc"]
        + 0.04 * frame["c_rate_p95"]
        + rng.normal(0.0, 0.01, n)
    )
    frame["positive_loss_pp"] = frame["signed_loss_pp"].clip(lower=0.0)
    return frame


def test_pack_se_nmc_stress_is_nonnegative_and_pack_aware() -> None:
    frame = _synthetic_intervals()
    transformed = add_pack_se_nmc_stress(frame)
    assert transformed["packse_pack_corrected_damage_index"].ge(0.0).all()
    assert (
        transformed["packse_pack_corrected_damage_index"]
        >= transformed["packse_homogeneous_damage_index"]
    ).all()


def test_nested_attribution_returns_all_planned_models() -> None:
    frame = _synthetic_intervals()
    predictions, metrics, increments = evaluate_nested_attribution([frame])
    assert not predictions.empty
    assert not metrics.empty
    assert not increments.empty
    assert "Persistence_zero_loss" in set(metrics["model"])
    assert "Full+DriverResidual" in set(metrics["model"])


def test_comparison_matrix_marks_surrogates_as_not_exact_reproductions() -> None:
    matrix = comparison_matrix()
    reproduction = matrix.loc[matrix["criterion"].eq("Exact reproduction status")].iloc[0]
    assert "Not reproduced" in reproduction["D2B"]
    assert "Not reproduced" in reproduction["EV_DOS"]
