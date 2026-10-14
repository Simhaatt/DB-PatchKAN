from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dbpatchkan.data import (
    CANONICAL_COLUMNS,
    DBSIConfig,
    NatureEVAdapter,
    NissanAdapter,
    PreprocessConfig,
    PreprocessState,
    RWTHAdapter,
    aggregate_trip_features,
    build_health_windows,
    compute_dbsi,
    data_dictionary_frame,
    iter_fixed_windows,
    preprocess_nissan_frame,
    resample_telemetry_1hz,
    run_nissan_pipeline,
    validate_canonical_frame,
    window_to_arrays,
)


def synthetic_nissan(n: int = 40, *, trip: int = 1, start_us: int = 1_600_000_000_000_000) -> pd.DataFrame:
    time = np.arange(n, dtype=np.int64)
    return pd.DataFrame(
        {
            "index": time,
            "Timestamp": start_us + time * 250_000,
            "Speed": np.linspace(5.0, 55.0, n),
            "Odometer": 3_578.0 + time / 10_000,
            "RPM": 1_000.0 + time,
            "SOC": np.linspace(90.0, 84.0, n),
            "SOH": 95.0,
            "Torque": 25.0,
            "Motor_Temp": 25.0,
            "TP_FR": 230.0,
            "TP_FL": 231.0,
            "TP_RR": 232.0,
            "TP_RL": 233.0,
            "Aux_Pwr_100W": 200.0,
            "AC_Pwr_250W": 250.0,
            "EstPwr_AC_50W": 250.0,
            "HV_Bat_Voltage": 360.0,
            "Brake": 0.0,
            "Acc_Pedal": 50.0,
            "ECO": 1.0,
            "ePedal": 0.0,
            "HV_Bat_Current_2": -50.0,
            "Ambient_C_Temp": 20.0,
            "Pack_1_C_Temp": 30.0,
            "Pack_2_C_Temp": 31.0,
            "Pack_4_C_Temp": 32.0,
            "Pack_Avg_C_Temp": 31.0,
            "kWh_Remaining": 20.0,
            "Horsepower": 24.0,
            "Instant_kW": -18.0,
            "Driver_ID": 0,
            "Trip": trip,
        }
    )


def test_qc_sign_timestamp_segmentation_and_native_integrals() -> None:
    raw = synthetic_nissan(8)
    raw.loc[1, "Motor_Temp"] = -40
    raw.loc[2, "RPM"] = 16383
    raw.loc[3, "HV_Bat_Voltage"] = 655.35
    raw.loc[4, ["TP_FR", "TP_FL"]] = 0
    # A 3.25-second delta creates a new segment and must not be integrated.
    raw.loc[5:, "Timestamp"] += 3_000_000

    clean = preprocess_nissan_frame(raw)
    assert str(clean["datetime_utc"].dtype) == "datetime64[us, UTC]"
    assert clean.loc[0, "current_std_a"] == pytest.approx(50.0)
    assert clean.loc[0, "p_batt_kw_clean"] == pytest.approx(18.0)
    assert pd.isna(clean.loc[1, "Motor_Temp"])
    assert pd.isna(clean.loc[2, "RPM"])
    assert pd.isna(clean.loc[3, "HV_Bat_Voltage"])
    assert pd.isna(clean.loc[4, "TP_FR"])
    assert clean.loc[5, "segment_id"] == clean.loc[4, "segment_id"] + 1
    assert clean.loc[5, "dt_s"] == 0.0
    assert clean.loc[1, "energy_dis_step_kwh"] == pytest.approx(18.0 * 0.25 / 3600.0)
    assert clean.loc[5, "energy_dis_step_kwh"] == 0.0
    assert clean["qc_bitmask"].ne(0).sum() == 4


def test_state_preserves_segments_across_csv_chunks() -> None:
    raw = synthetic_nissan(12)
    state = PreprocessState()
    first = preprocess_nissan_frame(raw.iloc[:6], state=state)
    second = preprocess_nissan_frame(raw.iloc[6:], state=state)
    combined = pd.concat([first, second], ignore_index=True)
    whole = preprocess_nissan_frame(raw)
    np.testing.assert_array_equal(combined["segment_id"], whole["segment_id"])
    np.testing.assert_allclose(combined["dt_s"], whole["dt_s"])


def test_one_hz_windows_are_fixed_length_and_do_not_cross_segments() -> None:
    clean = preprocess_nissan_frame(synthetic_nissan(1_200))  # 300 seconds at 4 Hz
    one = resample_telemetry_1hz(clean)
    assert 299 <= len(one) <= 300
    assert one["dt_s"].iloc[1] == pytest.approx(1.0)
    assert {"accel_mps2", "jerk_mps3", "speed_kmh", "hv_battery_voltage_v"}.issubset(one)

    windows = list(iter_fixed_windows(one, length_s=256, stride_s=128, min_coverage=0.8))
    assert len(windows) == 2
    assert all(len(window.data) == 256 for window in windows)
    assert all(window.data.loc[window.data["observed_1hz"].eq(1), "segment_id"].nunique() == 1 for window in windows)
    arrays = window_to_arrays(
        windows[0],
        ["speed_kmh", "accel_mps2", "valid_fraction"],
        ["hv_battery_voltage_v", "current_std_a", "soc_pct"],
    )
    assert arrays["driver"].shape == (256, 3)
    assert arrays["battery_missing"].shape == (256, 3)
    assert np.isfinite(arrays["driver"]).all()


def test_exact_dbsi_weighting_and_stress_dose() -> None:
    cfg = DBSIConfig()
    # Every high-stress severity saturates at one.  Duration is one hour and the
    # DoD/EFC terms are explicitly supplied to make all six components equal 1.
    frame = pd.DataFrame(
        {
            "dt_s": [1800.0, 1800.0],
            "current_std_a": [4 * cfg.nominal_capacity_ah, -2 * cfg.nominal_capacity_ah],
            "c_dis": [4.0, 0.0],
            "c_regen": [0.0, 2.0],
            "pack_temp_avg_clean_c": [50.0, -5.0],
            "pack_temp_spread_c": [12.0, 12.0],
            "SOC": [100.0, 100.0],
            "c_rate_slew": [2.0, 2.0],
            "dod_pct": [100.0, 100.0],
            "efc": [2.0, 2.0],
            "duration_s": [3600.0, 3600.0],
        }
    )
    result = compute_dbsi(frame, cfg)
    # Time weighting includes zero-severity intervals, so manually verify the
    # transparent formula rather than assuming every component saturates.
    expected_rate = 0.75 * 0.5 + 0.25 * 0.5
    expected_thermal = 0.70 * 0.5 + 0.20 * 0.5 + 0.10 * 1.0
    expected_soc = 0.70 * 1.0 + 0.30 * 0.0
    expected_cycle = 1.0
    expected_transient = 1.0
    expected_interaction = 0.5 * 0.5 + 0.5 * 0.5
    expected = 100 * (
        0.25 * expected_rate + 0.25 * expected_thermal + 0.15 * expected_soc
        + 0.15 * expected_cycle + 0.10 * expected_transient
        + 0.10 * expected_interaction
    )
    assert result["bsi_rate"] == pytest.approx(expected_rate)
    assert result["bsi_intensity"] == pytest.approx(expected)
    assert result["stress_dose"] == pytest.approx(expected / 100.0)
    contribution_sum = sum(v for k, v in result.items() if k.startswith("bsi_contribution_"))
    assert contribution_sum == pytest.approx(result["bsi_intensity"])


def test_trip_and_health_features_are_chronological_and_past_only() -> None:
    trip_frames = []
    for i, soh in enumerate([96.0, 95.5, 95.0, 94.5]):
        raw = synthetic_nissan(80, trip=i + 1, start_us=1_600_000_000_000_000 + i * 30 * 86400 * 1_000_000)
        raw["SOH"] = soh
        raw["Odometer"] = 1_000 + i * 500
        trip_frames.append(preprocess_nissan_frame(raw))
    clean = pd.concat(trip_frames, ignore_index=True)
    trips = aggregate_trip_features(clean)
    # Synthetic speed integration is short, so set controlled 500-km exposures.
    trips["distance_km"] = 500.0
    health = build_health_windows(trips)
    assert len(trips) == 4
    assert len(health) == 4
    assert health["soh_target_trend_pct"].is_monotonic_decreasing
    # The forecast target uses the declared past-only trailing-three trend.  With
    # two checkpoints its median is (96.0 + 95.5) / 2 = 95.75.
    assert health.loc[0, "target_soh_30d_pct"] == pytest.approx(95.75)
    assert health.loc[0, "target_soh_500km_pct"] == pytest.approx(95.75)
    assert health.loc[1, "hist_30d_bsi_dose"] >= trips.loc[1, "stress_dose"]


def test_dataset_adapters_emit_one_canonical_contract() -> None:
    nissan = NissanAdapter().adapt(synthetic_nissan(8))
    assert set(CANONICAL_COLUMNS).issubset(nissan)
    assert nissan["current_a"].iloc[0] > 0
    assert validate_canonical_frame(nissan, strict=True)["Present"].all()

    external = pd.DataFrame(
        {
            "vehicle": ["v1", "v1"],
            "time": [1_600_000_000, 1_600_000_001],
            "trip": ["a", "a"],
            "speed": [10.0, 20.0],
            "soc": [80.0, 79.9],
            "voltage": [350.0, 349.0],
            "current": [-20.0, -30.0],
            "battery_temp": [25.0, 25.2],
            "capacity_ah": [90.0, 90.0],
            "rated_capacity_ah": [100.0, 100.0],
        }
    )
    rwth = RWTHAdapter(timestamp_unit="s", current_positive_discharge=False).adapt(external)
    nature = NatureEVAdapter(timestamp_unit="s", current_positive_discharge=False).adapt(external)
    for adapted in (rwth, nature):
        assert set(CANONICAL_COLUMNS).issubset(adapted)
        assert adapted["current_a"].iloc[0] == 20.0
        assert adapted["capacity_soh_pct"].iloc[0] == pytest.approx(90.0)
        assert adapted["power_kw"].iloc[0] == pytest.approx(7.0)


def test_data_dictionary_covers_all_source_and_dbsi_fields() -> None:
    dictionary = data_dictionary_frame()
    names = set(dictionary["Column_Name"])
    assert set(synthetic_nissan(1).columns).issubset(names)
    assert {
        "bsi_rate", "bsi_thermal", "bsi_soc", "bsi_cycle",
        "bsi_transient", "bsi_interaction", "bsi_intensity", "stress_dose",
    }.issubset(names)


def test_quick_pipeline_writes_all_contract_outputs(tmp_path: Path) -> None:
    source = tmp_path / "leaf.csv"
    synthetic_nissan(1_200).to_csv(source, index=False)
    outputs = run_nissan_pipeline(
        source,
        tmp_path / "output",
        config=PreprocessConfig(write_parquet=False, segment_length_s=256),
        quick=True,
    )
    expected = {
        "profile.csv", "qc_summary.csv", "raw_sample.csv", "window_features.csv",
        "trip_features.csv", "health_windows.csv", "bsi_results.csv",
        "data_dictionary.csv", "manifest.json",
    }
    assert expected == {path.name for path in outputs.values()}
    assert all(path.exists() for path in outputs.values())
    assert len(pd.read_csv(outputs["raw_sample.csv"])) <= 30_000
    manifest = json.loads(outputs["manifest.json"].read_text(encoding="utf-8"))
    assert manifest["rows_processed"] == 1_200
    assert manifest["quick_mode"] is True
    assert manifest["source_sha256"]
