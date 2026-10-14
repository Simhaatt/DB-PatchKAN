"""Fixed-length external telemetry sequences for Stage 7.

The adapter preserves measured sampling order and creates one 128-step
representation per trip.  Missing vehicle-specific channels remain masked;
they are never filled with invented physical measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .eve import EVE_NOMINAL_CAPACITY_AH, _trip_record
from .rwth import (
    _load_trip_metadata,
    matlab_datenum_to_datetime,
    rwth_vehicle_spec,
)


EXTERNAL_CONCEPT_NAMES = (
    "c_rate_response",
    "high_current_fraction",
    "electrothermal_intensity",
    "energy_kwh_per_100km",
    "temperature_response",
    "dbsi_response",
)


def encode_utf8(values: Iterable[object]) -> np.ndarray:
    text = [str(value) for value in values]
    width = max((len(value.encode("utf-8")) for value in text), default=1)
    result = np.zeros((len(text), width), dtype=np.uint8)
    for row, value in enumerate(text):
        encoded = value.encode("utf-8")
        result[row, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return result


def decode_utf8(values: np.ndarray) -> list[str]:
    array = np.asarray(values, dtype=np.uint8)
    return [bytes(row[row != 0]).decode("utf-8") for row in array]


def _interpolate(
    elapsed_s: np.ndarray,
    values: np.ndarray,
    grid_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(elapsed_s) & np.isfinite(values)
    if valid.sum() < 2:
        return np.full(len(grid_s), np.nan, dtype=np.float32), np.zeros(len(grid_s), dtype=bool)
    x = elapsed_s[valid]
    y = values[valid]
    unique = np.concatenate(([True], np.diff(x) > 0))
    x, y = x[unique], y[unique]
    if len(x) < 2:
        return np.full(len(grid_s), np.nan, dtype=np.float32), np.zeros(len(grid_s), dtype=bool)
    interpolated = np.interp(grid_s, x, y).astype(np.float32)
    in_range = (grid_s >= x[0]) & (grid_s <= x[-1])
    interpolated[~in_range] = np.nan
    return interpolated, in_range


def _fixed_sequence(
    frame: pd.DataFrame,
    *,
    timestamp: str,
    speed_kmh: str,
    current_discharge_a: str,
    voltage_v: str,
    soc_pct: str,
    ambient_c: str,
    pack_temp_c: str | None,
    pack_temp_spread_c: str | None,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values(timestamp, kind="mergesort").copy()
    times = pd.to_datetime(ordered[timestamp], errors="coerce")
    elapsed = (times - times.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
    finite_t = elapsed[np.isfinite(elapsed)]
    duration = float(finite_t.max()) if len(finite_t) else 0.0
    if duration <= 0:
        raise ValueError("Trip has no positive duration")
    grid = np.linspace(0.0, duration, length, dtype=np.float64)

    def source(column: str | None) -> np.ndarray:
        if column is None or column not in ordered:
            return np.full(len(ordered), np.nan)
        return pd.to_numeric(ordered[column], errors="coerce").to_numpy(dtype=np.float64)

    speed, speed_ok = _interpolate(elapsed, source(speed_kmh), grid)
    speed_mps = speed / 3.6
    dt = np.full(length, duration / max(length - 1, 1), dtype=np.float32)
    dt[0] = 0.0
    acceleration = np.gradient(speed_mps, grid, edge_order=1).astype(np.float32)
    jerk = np.gradient(acceleration, grid, edge_order=1).astype(np.float32)
    acceleration[np.abs(acceleration) > 8.0] = np.nan
    jerk[np.abs(jerk) > 20.0] = np.nan

    current, current_ok = _interpolate(elapsed, source(current_discharge_a), grid)
    voltage, voltage_ok = _interpolate(elapsed, source(voltage_v), grid)
    soc, soc_ok = _interpolate(elapsed, source(soc_pct), grid)
    ambient, ambient_ok = _interpolate(elapsed, source(ambient_c), grid)
    pack_temp, pack_temp_ok = _interpolate(elapsed, source(pack_temp_c), grid)
    spread, spread_ok = _interpolate(elapsed, source(pack_temp_spread_c), grid)

    driver = np.full((length, 13), np.nan, dtype=np.float32)
    battery = np.full((length, 11), np.nan, dtype=np.float32)
    driver[:, 0] = speed_mps
    driver[:, 1] = acceleration
    driver[:, 2] = jerk
    battery[:, 0] = voltage
    battery[:, 1] = current
    battery[:, 2] = soc
    battery[:, 3] = pack_temp
    battery[:, 4] = spread
    battery[:, 5] = ambient

    core_valid = np.stack((speed_ok, current_ok, voltage_ok, soc_ok, ambient_ok), axis=1)
    valid_fraction = core_valid.mean(axis=1).astype(np.float32)
    relative = np.linspace(0.0, 1.0, length, dtype=np.float32)
    driver[:, 9] = valid_fraction
    driver[:, 10] = dt
    driver[:, 11] = relative
    battery[:, 7] = valid_fraction
    battery[:, 8] = dt
    battery[:, 9] = relative

    driver_mask = np.isfinite(driver)
    battery_mask = np.isfinite(battery)
    driver[:, 12] = 1.0 - driver_mask[:, :9].mean(axis=1)
    battery[:, 10] = 1.0 - battery_mask[:, :7].mean(axis=1)
    driver_mask[:, 9:13] = True
    battery_mask[:, 7:11] = True
    driver = np.nan_to_num(driver, nan=0.0)
    battery = np.nan_to_num(battery, nan=0.0)
    return driver, battery, driver_mask, battery_mask


def build_rwth_sequences(
    field_dir: str | Path,
    trip_dir: str | Path,
    trip_features: pd.DataFrame,
    *,
    length: int = 128,
    reporter=print,
) -> dict[str, np.ndarray]:
    """Map RWTH 1-Hz telemetry to published trips and fixed sequences."""

    field_dir, trip_dir = Path(field_dir), Path(trip_dir)
    metadata_all = _load_trip_metadata(trip_dir)
    feature_lookup = trip_features.set_index("trip_key", drop=False)
    records: list[dict[str, object]] = []
    streams: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    concepts: list[np.ndarray] = []
    source_columns = [
        "time_num", "Speed", "Odometer", "Temp_Ambient", "SoC_Real",
        "Current", "Voltage", "Power",
    ]
    for path in sorted(field_dir.glob("*.parquet")):
        vehicle = path.stem
        reporter(f"RWTH sequences: loading {vehicle}")
        raw = pq.read_table(path, columns=source_columns).to_pandas()
        raw["datetime"] = matlab_datenum_to_datetime(raw["time_num"])
        raw["current_discharge_a"] = -pd.to_numeric(raw["Current"], errors="coerce")
        vehicle_meta = metadata_all.loc[metadata_all.vehicle.eq(vehicle)].reset_index(drop=True)
        starts = vehicle_meta.starttime.to_numpy(dtype="datetime64[ns]").astype("int64")
        ends = vehicle_meta.endtime.to_numpy(dtype="datetime64[ns]").astype("int64")
        times = raw.datetime.to_numpy(dtype="datetime64[ns]").astype("int64")
        location = np.searchsorted(starts, times, side="right") - 1
        valid = (location >= 0) & (location < len(vehicle_meta))
        safe = np.clip(location, 0, max(len(vehicle_meta) - 1, 0))
        valid &= times <= ends[safe]
        mapped = raw.loc[valid].copy()
        mapped["trip_location"] = location[valid]
        spec = rwth_vehicle_spec(vehicle)
        kept = 0
        for position, group in mapped.groupby("trip_location", sort=True):
            row = vehicle_meta.iloc[int(position)]
            key = str(row.trip_key)
            if key not in feature_lookup.index or len(group) < 20:
                continue
            feature = feature_lookup.loc[key]
            try:
                streams.append(_fixed_sequence(
                    group,
                    timestamp="datetime",
                    speed_kmh="Speed",
                    current_discharge_a="current_discharge_a",
                    voltage_v="Voltage",
                    soc_pct="SoC_Real",
                    ambient_c="Temp_Ambient",
                    pack_temp_c=None,
                    pack_temp_spread_c=None,
                    length=length,
                ))
            except ValueError:
                continue
            duration = max(float(feature.duration_observed_h), 1e-4)
            distance = max(float(feature.distance_km), 0.1)
            concepts.append(np.asarray([
                float(feature.c_rate_discharge_peak),
                float(feature.high_c_rate_fraction),
                float(feature.electrothermal_proxy_dose_h) / duration,
                100.0 * float(feature.energy_discharge_kwh) / distance,
                float(feature.thermal_memory_proxy_peak),
                float(feature.dbsi_partial_renormalized),
            ], dtype=np.float32))
            records.append({
                "dataset_id": "RWTH",
                "vehicle_id": vehicle,
                "trip_key": key,
                "trip_start_ns": pd.Timestamp(row.starttime).value,
                "trip_end_ns": pd.Timestamp(row.endtime).value,
                "duration_h": duration,
                "distance_km": float(feature.distance_km),
                "nominal_capacity_ah": spec.nominal_capacity_ah,
            })
            kept += 1
        reporter(f"RWTH sequences: {vehicle} complete ({kept} trips)")
    if not records:
        raise ValueError("No RWTH sequences were produced")
    driver, battery, driver_mask, battery_mask = map(
        np.stack, zip(*streams, strict=True)
    )
    table = pd.DataFrame(records)
    targets = np.stack(concepts)
    return {
        "driver": driver.astype(np.float16),
        "battery": battery.astype(np.float16),
        "driver_mask": driver_mask,
        "battery_mask": battery_mask,
        "concept_targets": targets,
        "concept_mask": np.isfinite(targets),
        "dataset_id_utf8": encode_utf8(table.dataset_id),
        "vehicle_id_utf8": encode_utf8(table.vehicle_id),
        "trip_key_utf8": encode_utf8(table.trip_key),
        "trip_start_ns": table.trip_start_ns.to_numpy(dtype=np.int64),
        "trip_end_ns": table.trip_end_ns.to_numpy(dtype=np.int64),
        "duration_h": table.duration_h.to_numpy(dtype=np.float32),
        "distance_km": table.distance_km.to_numpy(dtype=np.float32),
    }


def build_eve_sequences(
    telemetry: pd.DataFrame,
    *,
    length: int = 128,
    reporter=print,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Create fixed sequences and trip concepts from harmonised EVE telemetry."""

    frame = telemetry.sort_values("timestamp").copy()
    frame["session"] = frame.timestamp.diff().dt.total_seconds().gt(300).cumsum()
    streams: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    features: list[dict[str, object]] = []
    concepts: list[np.ndarray] = []
    for _, group in frame.groupby("session", sort=True):
        duration_s = (group.timestamp.max() - group.timestamp.min()).total_seconds()
        if len(group) < 12 or duration_s < 60 or group.speed_kmh.max() < 5:
            continue
        record = _trip_record(group, len(features) + 1)
        if record["distance_km"] < 0.2 or record["duration_h"] <= 0:
            continue
        streams.append(_fixed_sequence(
            group,
            timestamp="timestamp",
            speed_kmh="speed_kmh",
            current_discharge_a="current_a",
            voltage_v="voltage_v",
            soc_pct="soc_pct",
            ambient_c="ambient_c",
            pack_temp_c="pack_temp_mean_c",
            pack_temp_spread_c="pack_temp_spread_c",
            length=length,
        ))
        features.append(record)
        duration = max(float(record["duration_h"]), 1e-4)
        concepts.append(np.asarray([
            float(record["c_rate_abs_p95"]),
            float(record["high_c_rate_fraction"]),
            float(record["electrothermal_dose_h"]) / duration,
            float(record["energy_kwh_per_100km"]),
            float(record["pack_temp_rise_c"]),
            float(record["dbsi"]),
        ], dtype=np.float32))
        if len(features) % 250 == 0:
            reporter(f"EVE sequences: {len(features)} trips complete")
    if not features:
        raise ValueError("No EVE sequences were produced")
    driver, battery, driver_mask, battery_mask = map(
        np.stack, zip(*streams, strict=True)
    )
    table = pd.DataFrame(features)
    targets = np.stack(concepts)
    arrays = {
        "driver": driver.astype(np.float16),
        "battery": battery.astype(np.float16),
        "driver_mask": driver_mask,
        "battery_mask": battery_mask,
        "concept_targets": targets,
        "concept_mask": np.isfinite(targets),
        "dataset_id_utf8": encode_utf8(["EVE"] * len(table)),
        "vehicle_id_utf8": encode_utf8(table.vehicle_id),
        "trip_key_utf8": encode_utf8(table.trip_key),
        "trip_start_ns": pd.to_datetime(table.trip_start).astype("int64").to_numpy(),
        "trip_end_ns": pd.to_datetime(table.trip_end).astype("int64").to_numpy(),
        "duration_h": table.duration_h.to_numpy(dtype=np.float32),
        "distance_km": table.distance_km.to_numpy(dtype=np.float32),
    }
    reporter(f"EVE sequences complete: {len(table)} trips")
    return arrays, table


def save_sequence_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


__all__ = [
    "EXTERNAL_CONCEPT_NAMES", "build_eve_sequences", "build_rwth_sequences",
    "decode_utf8", "encode_utf8", "save_sequence_npz",
]
