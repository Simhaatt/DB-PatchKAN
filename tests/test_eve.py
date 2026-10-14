from __future__ import annotations

import numpy as np
import pandas as pd

from dbpatchkan.data.eve import eve_monthly_checkpoints, eve_trip_features


def test_eve_trip_features_exposes_measured_thermal_physics() -> None:
    timestamp = pd.date_range("2022-01-01", periods=180, freq="5s")
    speed = 35.0 + 20.0 * np.sin(np.linspace(0, 8 * np.pi, len(timestamp)))
    current = 25.0 + speed
    telemetry = pd.DataFrame({
        "timestamp": timestamp,
        "speed_kmh": speed,
        "current_a": current,
        "voltage_v": 350.0 - 0.05 * current,
        "power_kw": current * 350.0 / 1000.0,
        "soc_pct": np.linspace(85, 75, len(timestamp)),
        "odometer_km": 1000 + np.linspace(0, 6, len(timestamp)),
        "ambient_c": 20.0,
        "pack_temp_mean_c": np.linspace(24, 29, len(timestamp)),
        "pack_temp_spread_c": 1.5,
        "bms_soh_pct_raw": 92.0,
    })
    trips = eve_trip_features(telemetry, reporter=lambda _: None)
    assert len(trips) == 1
    row = trips.iloc[0]
    assert row["pack_temp_rise_c"] > 4.9
    assert row["electrothermal_dose_h"] > 0
    assert row["c_rate_squared_dose_h"] > 0
    assert 0 <= row["dbsi"] <= 100
    assert row["energy_kwh_per_100km"] > 0


def test_eve_monthly_checkpoints_are_past_only(tmp_path) -> None:
    trip_times = pd.to_datetime(["2022-01-10", "2022-02-10", "2022-03-10"])
    trips = pd.DataFrame({
        "trip_end": trip_times,
        "duration_h": [1.0, 1.0, 1.0],
        "distance_km": [20.0, 20.0, 20.0],
        "efc": [0.1, 0.2, 0.3],
        "energy_discharge_kwh": [3.0, 3.0, 3.0],
        "c_rate_squared_dose_h": [0.2, 0.2, 0.2],
        "electrothermal_dose_h": [0.3, 0.3, 0.3],
        "transient_dose": [1.0, 1.0, 1.0],
        "high_soc_hot_time_h": [0.0, 0.0, 0.0],
        "stress_dose_h": [0.1, 0.1, 0.1],
        "speed_mean_kmh": [20.0, 30.0, 40.0],
        "accel_positive_rms_mps2": [0.2, 0.3, 0.4],
        "decel_rms_mps2": [0.2, 0.3, 0.4],
        "jerk_rms_mps3": [0.1, 0.1, 0.1],
        "harsh_accel_fraction": [0.0, 0.0, 0.0],
        "harsh_decel_fraction": [0.0, 0.0, 0.0],
        "c_rate_abs_p95": [0.5, 0.6, 0.7],
        "high_c_rate_fraction": [0.0, 0.0, 0.0],
        "pack_temp_mean_c": [20.0, 21.0, 22.0],
        "pack_temp_max_c": [22.0, 23.0, 24.0],
        "pack_temp_rise_c": [1.0, 1.0, 1.0],
        "dbsi": [10.0, 20.0, 30.0],
        "ambient_mean_c": [10.0, 11.0, 12.0],
    })
    health = tmp_path / "soh.tab"
    health.write_text(
        "Datetime\tValue\tUnit\n"
        "2022-01-31 12:00:00\t95\t%\n"
        "2022-02-28 12:00:00\t94\t%\n"
        "2022-03-31 12:00:00\t93\t%\n",
        encoding="utf-8",
    )
    checkpoints, intervals = eve_monthly_checkpoints(trips, health)
    assert checkpoints.loc[0, "trip_count_cumulative"] == 1
    assert checkpoints.loc[1, "trip_count_cumulative"] == 2
    assert np.isclose(checkpoints.loc[1, "cum_efc"], 0.3)
    assert len(intervals) == 2
    assert intervals.loc[0, "signed_soh_loss_pp"] == 1.0
