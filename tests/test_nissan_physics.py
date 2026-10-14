import numpy as np
import pandas as pd

from dbpatchkan.data.nissan_physics import (
    append_past_only_exposures,
    estimate_trip_resistance,
    nissan_trip_physics_features,
    relative_arrhenius_factor,
)


def _telemetry() -> pd.DataFrame:
    current = np.array([0.0, 20.0, 40.0, 10.0])
    return pd.DataFrame({
        "dataset_id": "d", "vehicle_id": "v", "trip_key": "d:v:1", "Trip": 1,
        "datetime_utc": pd.date_range("2024-01-01", periods=4, freq="s", tz="UTC"),
        "segment_id": 0, "dt_s": 1.0, "current_std_a": current,
        "voltage_v": 360.0 - 0.1 * current, "soc_pct": [90, 89, 88, 87],
        "pack_temp_avg_clean_c": [25, 26, 27, 28], "ambient_temp_c": 20.0,
    })


def test_resistance_and_physics_doses_are_auditable() -> None:
    frame = _telemetry()
    resistance, count, _ = estimate_trip_resistance(frame)
    assert count == 3
    assert np.isclose(resistance, 0.1)
    result = nissan_trip_physics_features(frame)
    assert result["joule_heat_estimate_kwh"] > 0
    assert result["c_rate_squared_dose_h"] > 0
    assert result["electrothermal_dose_h"] > 0
    assert result["soc_high_dose_h"] > 0
    assert result["soc_high_hot_time_h"] == 0
    assert result["past_only_feature"] is True


def test_arrhenius_factor_is_one_at_reference_and_increases() -> None:
    values = relative_arrhenius_factor(np.array([15.0, 25.0, 35.0]))
    assert np.isclose(values[1], 1.0)
    assert values[0] < values[1] < values[2]


def test_history_never_uses_future_trip() -> None:
    trips = pd.DataFrame({
        "dataset_id": "d", "vehicle_id": "v", "trip_key": ["a", "b"],
        "trip_end_utc": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        "distance_km": [10.0, 30.0], "ah_throughput": [1.0, 3.0],
    })
    result = append_past_only_exposures(trips)
    assert result.loc[0, "cum_distance_km"] == 10.0
    assert result.loc[1, "cum_distance_km"] == 40.0
    assert result.loc[0, "hist_7d_distance_km"] == 10.0
