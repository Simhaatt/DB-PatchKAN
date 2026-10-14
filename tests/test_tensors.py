from __future__ import annotations

import numpy as np
import pandas as pd

from dbpatchkan.data import (
    DEFAULT_CONCEPT_NAMES,
    NatureEVAdapter,
    decode_utf8_rows,
    export_segment_npz,
)
from dbpatchkan.models import EVSegmentDataset


def _synthetic_one_hz(*, include_soh: bool = True) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for trip_index, (trip, driver, soh) in enumerate(((10, 0, 96.0), (11, 4, 95.5))):
        length = 40
        time = pd.date_range(
            pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(days=trip_index),
            periods=length,
            freq="1s",
        )
        values = np.arange(length, dtype=float)
        frame = pd.DataFrame(
            {
                "datetime_utc": time,
                # Deliberately reused between trips; the exporter must make the
                # compound segment identity safe before windowing.
                "segment_id": 0,
                "trip_key": f"nissan:v1:{trip}",
                "Trip": trip,
                "dataset_id": "nissan",
                "vehicle_id": "v1",
                "Driver_ID": driver,
                "observed_1hz": 1,
                "key_sensor_valid": True,
                "Speed": 36.0 + values * 0.1,
                "accel_mps2": 0.1,
                "jerk_mps3": 0.01,
                "Torque": 12.0,
                "Acc_Pedal": 40.0,
                "Brake": 0.0,
                "RPM": -1200.0,
                "ECO": 1 if trip == 10 else 0,
                "ePedal": 0 if trip == 10 else 1,
                "HV_Bat_Voltage": 380.0,
                # Nissan raw sign: negative is discharge.
                "HV_Bat_Current_2": -20.0,
                "SOC": 80.0 - values * 0.05,
                "Pack_1_C_Temp": 25.0,
                "Pack_2_C_Temp": 26.0,
                "Pack_4_C_Temp": 27.0,
                "Ambient_C_Temp": 20.0,
                "AC_Pwr_250W": 500.0,
            }
        )
        if include_soh:
            frame["SOH"] = soh
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def _trip_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_key": ["nissan:v1:10", "nissan:v1:11"],
            "Trip": [10, 11],
            "Driver_ID": [0, 4],
            "soh_target_raw_pct": [95.9, 95.4],
            "target_loss_30d_pp": [0.10, 0.20],
            "target_loss_rate_pp_per_1000km": [0.30, 0.40],
            "bsi_rate": [0.10, 0.50],
            "bsi_thermal": [0.20, 0.60],
            "bsi_soc": [0.30, 0.70],
            "bsi_cycle": [0.40, 0.80],
            "bsi_transient": [0.15, 0.55],
            "bsi_interaction": [0.25, 0.65],
            "odometer_clean_km": [10_000.0, 10_040.0],
            "cumulative_stress_dose": [1.2, 1.5],
            "accel_pos_rms": [0.3, 0.8],
            "jerk_rms": [0.1, 0.4],
            "energy_kwh_per_100km": [14.0, 19.0],
            "pack_temp_max_c": [27.0, 31.0],
            "ah_throughput": [4.0, 7.0],
            "regen_fraction": [0.30, 0.15],
            "stop_fraction": [0.20, 0.10],
        }
    )


def test_export_segment_npz_loads_with_ev_dataset(tmp_path) -> None:
    behavior = pd.DataFrame(
        {
            "trip_key": ["nissan:v1:10", "nissan:v1:11"],
            "behavior_level": ["Eco", "Aggressive"],
        }
    )
    output = export_segment_npz(
        _synthetic_one_hz(),
        tmp_path / "segments.npz",
        trip_features=_trip_features(),
        behavior_clusters=behavior,
        segment_length=32,
        stride=16,
        min_coverage=0.8,
    )

    with np.load(output, allow_pickle=False) as arrays:
        assert arrays["driver"].shape == (4, 32, 9)
        assert arrays["battery"].shape == (4, 32, 7)
        assert arrays["driver_mask"].dtype == np.bool_
        assert arrays["battery_mask"].dtype == np.bool_
        np.testing.assert_allclose(arrays["driver"][0, 0, 0], 10.0)
        np.testing.assert_allclose(arrays["driver"][0, 0, 6], 1200.0)
        np.testing.assert_allclose(arrays["driver"][0, 0, 7:9], [1.0, 0.0])
        np.testing.assert_allclose(arrays["battery"][0, 0, 1], 20.0)
        np.testing.assert_allclose(arrays["battery"][0, 0, 3:5], [26.0, 2.0])
        assert arrays["concept_targets"].shape == (4, 20)
        assert arrays["concept_mask"].shape == (4, 20)
        # No contemporaneous SOH is substituted for this past-only concept.
        last_soh_index = DEFAULT_CONCEPT_NAMES.index("last_known_soh")
        assert not arrays["concept_mask"][:, last_soh_index].any()
        eco = DEFAULT_CONCEPT_NAMES.index("behavior_share_eco")
        aggressive = DEFAULT_CONCEPT_NAMES.index("behavior_share_aggressive")
        np.testing.assert_array_equal(arrays["concept_targets"][:2, eco], 1.0)
        np.testing.assert_array_equal(arrays["concept_targets"][2:, aggressive], 1.0)
        np.testing.assert_array_equal(arrays["behavior_target"], [0, 0, 2, 2])
        np.testing.assert_array_equal(arrays["driver_target"], [0, 0, 4, 4])
        assert decode_utf8_rows(arrays["window_id_utf8"]) == [
            "nissan:v1:10:seg0:1609459200000000",
            "nissan:v1:10:seg0:1609459208000000",
            "nissan:v1:11:seg1:1609545600000000",
            "nissan:v1:11:seg1:1609545608000000",
        ]
        assert decode_utf8_rows(arrays["trip_key_utf8"]) == [
            "nissan:v1:10",
            "nissan:v1:10",
            "nissan:v1:11",
            "nissan:v1:11",
        ]
        assert "concept_00_bsi_rate" in arrays.files

    dataset = EVSegmentDataset.from_npz(output)
    assert len(dataset) == 4
    assert dataset.driver.shape == (4, 32, 13)
    assert dataset.battery.shape == (4, 32, 11)
    assert dataset.concept_targets is not None
    assert dataset.concept_targets.shape == (4, 20)
    assert dataset.targets["behavior_target"].dtype.is_floating_point is False
    assert dataset.targets["driver_target"].tolist() == [0, 0, 4, 4]


def test_conflicting_duplicate_trip_labels_are_not_exported(tmp_path) -> None:
    one_hz = _synthetic_one_hz(include_soh=False).iloc[:40].copy()
    conflicting = pd.DataFrame(
        {
            "trip_key": ["nissan:v1:10", "nissan:v1:10"],
            "soh_target_raw_pct": [96.0, 92.0],
        }
    )
    behavior = pd.DataFrame(
        {
            "trip_key": ["nissan:v1:10", "nissan:v1:10"],
            "behavior_level": ["Eco", "Aggressive"],
        }
    )
    output = export_segment_npz(
        one_hz,
        tmp_path / "unsafe_labels.npz",
        trip_features=conflicting,
        behavior_clusters=behavior,
        segment_length=32,
        stride=32,
    )
    with np.load(output, allow_pickle=False) as arrays:
        assert "current_soh_target" not in arrays.files
        assert "behavior_target" not in arrays.files


def test_parquet_input_uses_the_same_exchange_contract(tmp_path) -> None:
    source = tmp_path / "one_hz.parquet"
    _synthetic_one_hz().iloc[:40].to_parquet(source, index=False)
    output = export_segment_npz(
        source,
        tmp_path / "from_parquet",
        trip_features=_trip_features().iloc[:1],
        segment_length=16,
        stride=16,
    )
    assert output.suffix == ".npz"
    dataset = EVSegmentDataset.from_npz(output)
    assert len(dataset) == 3
    assert dataset.driver.shape[1:] == (16, 13)


def test_low_cadence_adapter_preserves_native_time_delta(tmp_path) -> None:
    rows = 20
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="10s", tz="UTC"),
            "vehicle_id": "fleet-1",
            "trip_id": "drive-1",
            "speed_kmh": np.linspace(20.0, 40.0, rows),
            "soc_pct": np.linspace(80.0, 75.0, rows),
            "voltage_v": 360.0,
            "current_a": 15.0,
            "pack_temp_c": 25.0,
            "ambient_temp_c": 18.0,
        }
    )
    canonical = NatureEVAdapter().adapt(raw)
    assert "segment_id" in canonical
    output = export_segment_npz(
        canonical,
        tmp_path / "native_01hz.npz",
        segment_length=8,
        stride=4,
        min_coverage=0.8,
    )
    with np.load(output, allow_pickle=False) as arrays:
        assert arrays["driver"].shape[1] == 8
        np.testing.assert_allclose(arrays["time_delta_s"][:, 1:], 10.0)
        assert arrays["window_id_utf8"].shape[0] == arrays["driver"].shape[0]
