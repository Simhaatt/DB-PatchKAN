from __future__ import annotations

import numpy as np
import pandas as pd

from dbpatchkan.data.rwth import (
    SMART_SPEC,
    build_driver_soh_intervals,
    build_loov_folds,
    capacity_checkpoints_from_frame,
    matlab_datenum_to_datetime,
    trip_physics_features,
)


def _matlab_datenum(timestamp: str) -> float:
    value = pd.Timestamp(timestamp)
    return value.to_julian_date() - 1721058.5


def test_matlab_datenum_matches_rwth_wall_clock() -> None:
    parsed = matlab_datenum_to_datetime([735654.680961])[0]
    assert parsed.strftime("%Y-%m-%d %H:%M:%S") == "2014-02-24 16:20:35"


def test_capacity_checkpoint_uses_fixed_voltage_window() -> None:
    seconds = np.arange(0.0, 10.1, 0.1)
    base = _matlab_datenum("2014-01-01")
    voltage = np.full(len(seconds), 340.0)
    voltage[:10] = SMART_SPEC.window_pack_voltage_high_v + 1.0
    frame = pd.DataFrame({
        "time_num": base + seconds / 86400.0,
        "Odometer": 1000.0,
        "Current": -50.0,
        "Voltage": voltage,
        "SoC_Real": np.linspace(100.0, 0.0, len(seconds)),
        "Temp_Ambient": 25.0,
        "test_number": 1.0,
        "test_direction": -1.0,
    })
    checkpoint = capacity_checkpoints_from_frame(frame, vehicle_id="Smart-X").iloc[0]
    assert checkpoint["capacity_usable_ah"] > checkpoint["capacity_window_ah"] > 0
    assert checkpoint["label_type"] == "1C_discharge_fixed_voltage_window"
    assert not checkpoint["pack_temperature_observed"]


def test_trip_features_preserve_physics_and_missing_thermal_semantics() -> None:
    time = pd.date_range("2014-01-01", periods=20, freq="s")
    speed = np.r_[np.arange(10), np.arange(10, 0, -1)] * 3.6
    telemetry = pd.DataFrame({
        "datetime": time,
        "Speed": speed,
        "Current": -50.0,
        "Voltage": 350.0,
        "SoC_Real": np.linspace(90.0, 80.0, len(time)),
        "Temp_Ambient": 25.0,
    })
    result = trip_physics_features(telemetry, vehicle_id="Smart-1")
    assert result["c_rate_discharge_mean"] > 0.9
    assert result["joule_c2_dose_h"] > 0
    assert result["electrothermal_proxy_dose_h"] > 0
    assert result["dbsi_observed_weight_coverage"] == 0.65
    assert not result["pack_temperature_observed"]
    assert "latent_proxy" in result["thermal_feature_status"]


def test_interval_builder_is_past_only_and_loov_is_vehicle_grouped() -> None:
    checkpoints = pd.DataFrame({
        "vehicle_id": ["Smart-1", "Smart-1", "Smart-2", "Smart-2"],
        "vehicle_family": ["Smart"] * 4,
        "test_number": [1, 2, 1, 2],
        "checkpoint_start": pd.to_datetime([
            "2014-01-01", "2014-02-01", "2014-01-01", "2014-02-01"
        ]),
        "checkpoint_end": pd.to_datetime([
            "2014-01-01 01:00", "2014-02-01 01:00",
            "2014-01-01 01:00", "2014-02-01 01:00",
        ]),
        "odometer_km": [0.0, 1000.0, 0.0, 800.0],
        "capacity_soh_pct": [100.0, 98.0, 100.0, 99.0],
    })
    rows = []
    for vehicle in ("Smart-1", "Smart-2"):
        base = {
            "vehicle_id": vehicle,
            "trip_start": pd.Timestamp("2014-01-10"),
            "trip_end": pd.Timestamp("2014-01-10 01:00"),
            "distance_km": 10.0,
            "duration_observed_h": 1.0,
        }
        for column in (
            "ah_throughput", "efc", "energy_discharge_kwh", "energy_regen_kwh",
            "joule_c2_dose_h", "electrothermal_proxy_dose_h",
        ):
            base[column] = 1.0
        for column in (
            "speed_mean_kmh", "speed_max_kmh", "speed_std_kmh",
            "accel_positive_rms_mps2", "decel_rms_mps2", "jerk_rms_mps3",
            "harsh_accel_fraction", "harsh_decel_fraction", "high_speed_fraction",
            "stop_fraction", "current_discharge_mean_a", "current_discharge_peak_a",
            "c_rate_discharge_mean", "c_rate_discharge_peak", "c_rate_regen_mean",
            "c_rate_regen_peak", "high_c_rate_fraction", "soc_mean_pct",
            "soc_high_fraction", "soc_low_fraction", "dod_pct", "ambient_temp_mean_c",
            "arrhenius_ambient_factor_mean", "thermal_memory_proxy_mean",
            "thermal_memory_proxy_peak", "bsi_rate", "bsi_soc", "bsi_cycle",
            "bsi_transient", "dbsi_observed_lower_bound", "dbsi_partial_renormalized",
        ):
            base[column] = 1.0
        rows.append(base)
    intervals = build_driver_soh_intervals(checkpoints, pd.DataFrame(rows))
    assert intervals["past_only_exposure"].all()
    assert intervals.set_index("vehicle_id").loc["Smart-1", "soh_loss_pp"] == 2.0
    folds = build_loov_folds(intervals)
    for held_out, group in folds.groupby("held_out_vehicle"):
        assert set(group.loc[group["split"].eq("test"), "vehicle_id"]) == {held_out}
