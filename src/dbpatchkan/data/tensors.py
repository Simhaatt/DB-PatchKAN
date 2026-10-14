"""Production export of leakage-safe fixed EV telemetry windows.

The exchange format deliberately keeps the nine driver and seven battery
physical channels separate from shared sequence metadata.  The GPU-side
``EVSegmentDataset.from_npz`` loader appends validity, time delta, relative
position, and missing-fraction channels to create the canonical 13/11 stream
layout.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import SequenceWindow, iter_fixed_windows, window_to_arrays


DEFAULT_DRIVER_CHANNELS: tuple[str, ...] = (
    "speed_mps",
    "accel_mps2",
    "jerk_mps3",
    "torque_nm",
    "accel_pedal_norm",
    "brake_active",
    "rpm_abs",
    "eco",
    "epedal",
)

DEFAULT_BATTERY_CHANNELS: tuple[str, ...] = (
    "voltage_v",
    "current_discharge_a",
    "soc_pct",
    "pack_temp_avg_c",
    "pack_temp_spread_c",
    "ambient_temp_c",
    "hvac_power_w",
)

# This order is the public concept contract used by DBPatchKANConfig.  Missing
# concepts remain NaN with a false mask; columns never shift between datasets.
DEFAULT_CONCEPT_NAMES: tuple[str, ...] = (
    "bsi_rate",
    "bsi_thermal",
    "bsi_soc",
    "bsi_cycle",
    "bsi_transient",
    "bsi_interaction",
    "behavior_share_eco",
    "behavior_share_balanced",
    "behavior_share_aggressive",
    "calendar_age_days",
    "odometer_km",
    "cumulative_stress_dose",
    "last_known_soh",
    "accel_pos_rms",
    "jerk_rms",
    "energy_kwh_per_100km",
    "pack_temp_max_c",
    "ah_throughput",
    "regen_fraction",
    "stop_fraction",
)


_READ_ALIASES: Mapping[str, tuple[str, ...]] = {
    "speed_mps": ("speed_mps", "Speed", "speed_kmh"),
    "accel_mps2": ("accel_mps2", "acceleration_mps2", "acceleration"),
    "jerk_mps3": ("jerk_mps3", "jerk"),
    "torque_nm": ("torque_nm", "Torque", "torque"),
    "accel_pedal_norm": ("accel_pedal_norm", "Acc_Pedal", "accelerator_position"),
    "brake_active": ("brake_active", "Brake", "brake"),
    "rpm_abs": ("rpm_abs", "RPM", "rpm"),
    "eco": ("eco", "ECO", "Eco", "eco_status"),
    "epedal": ("epedal", "ePedal", "E_PEDAL", "e_pedal", "epedal_status"),
    "voltage_v": ("voltage_v", "hv_battery_voltage_v", "HV_Bat_Voltage"),
    "current_discharge_a": (
        "current_discharge_a",
        "current_std_a",
        "HV_Bat_Current_2",
    ),
    "soc_pct": ("soc_pct", "SOC", "soc"),
    "pack_temp_avg_c": (
        "pack_temp_avg_c",
        "pack_temp_avg_clean_c",
        "Pack_Avg_C_Temp",
    ),
    "pack_temp_spread_c": ("pack_temp_spread_c", "pack_temperature_spread_c"),
    "ambient_temp_c": ("ambient_temp_c", "Ambient_C_Temp", "ambient_temperature_c"),
    "hvac_power_w": ("hvac_power_w", "AC_Pwr_250W", "hvac_power_kw"),
}

_CONCEPT_ALIASES: Mapping[str, tuple[str, ...]] = {
    "bsi_rate": ("bsi_rate", "S_rate", "dbsi_rate"),
    "bsi_thermal": ("bsi_thermal", "S_thermal", "dbsi_thermal"),
    "bsi_soc": ("bsi_soc", "S_soc", "dbsi_soc"),
    "bsi_cycle": ("bsi_cycle", "S_cycle", "dbsi_cycle"),
    "bsi_transient": ("bsi_transient", "S_transient", "dbsi_transient"),
    "bsi_interaction": ("bsi_interaction", "S_interaction", "dbsi_interaction"),
    "behavior_share_eco": (
        "behavior_share_eco",
        "hist_30d_behavior_share_eco",
        "eco_trip_proportion",
        "eco_proportion",
    ),
    "behavior_share_balanced": (
        "behavior_share_balanced",
        "hist_30d_behavior_share_balanced",
        "balanced_trip_proportion",
        "balanced_proportion",
    ),
    "behavior_share_aggressive": (
        "behavior_share_aggressive",
        "hist_30d_behavior_share_aggressive",
        "aggressive_trip_proportion",
        "aggressive_proportion",
    ),
    "calendar_age_days": ("calendar_age_days", "vehicle_age_days", "age_days"),
    "odometer_km": ("odometer_km", "odometer_clean_km", "Odometer"),
    "cumulative_stress_dose": ("cumulative_stress_dose", "cumulative_dbsi_dose"),
    # Never substitute contemporaneous SOH here.  Only an explicitly lagged,
    # past-known value is eligible for the forecasting concept.
    "last_known_soh": (
        "last_known_soh",
        "last_known_soh_pct",
        "past_soh_pct",
        "soh_lag1_pct",
    ),
    "accel_pos_rms": ("accel_pos_rms", "positive_acceleration_rms"),
    "jerk_rms": ("jerk_rms", "absolute_jerk_rms"),
    "energy_kwh_per_100km": ("energy_kwh_per_100km", "energy_per_100km"),
    "pack_temp_max_c": ("pack_temp_max_c", "maximum_pack_temperature_c"),
    "ah_throughput": ("ah_throughput", "absolute_ah_throughput"),
    "regen_fraction": ("regen_fraction", "regenerative_energy_fraction"),
    "stop_fraction": ("stop_fraction", "stopped_time_fraction"),
}

_CURRENT_SOH_ALIASES = ("soh_target_raw_pct", "SOH", "soh_target_trend_pct")
_LOSS_30D_ALIASES = ("target_loss_30d_pp", "loss_30d_target")
_LOSS_500KM_ALIASES = (
    "target_loss_500km_pp",
    "loss_500km_target",
)
_BEHAVIOR_ALIASES = ("behavior_target", "behavior_level", "Behavior_Level")


def _path_table(value: str | Path | pd.DataFrame | None) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=False)
    path = Path(value)
    suffix = path.suffix.casefold()
    if suffix in {".parquet", ".pq"} or path.is_dir():
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _projected_one_hz(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read only potentially required Parquet fields when schema access permits."""

    if isinstance(value, pd.DataFrame):
        return value.copy(deep=False)
    path = Path(value)
    if path.suffix.casefold() not in {".parquet", ".pq"} and not path.is_dir():
        raise ValueError("one_hz_parquet_or_df must be a DataFrame or Parquet path")

    required = {
        "segment_id",
        "datetime_utc",
        "trip_key",
        "Trip",
        "dataset_id",
        "vehicle_id",
        "Driver_ID",
        "SOH",
        "key_sensor_valid",
        "observed_1hz",
        "source_row_id",
        "sampling_interval_s",
        "Pack_1_C_Temp",
        "Pack_2_C_Temp",
        "Pack_4_C_Temp",
    }
    for aliases in _READ_ALIASES.values():
        required.update(aliases)
    try:
        import pyarrow.dataset as ds

        available = set(ds.dataset(path, format="parquet").schema.names)
        columns = sorted(required & available)
        return pd.read_parquet(path, columns=columns)
    except (ImportError, OSError, ValueError, TypeError):
        # The unprojected fallback supports alternative pandas Parquet engines.
        return pd.read_parquet(path)


def _numeric_alias(frame: pd.DataFrame, canonical: str) -> pd.Series:
    aliases = _READ_ALIASES.get(canonical, (canonical,))
    source = next((name for name in aliases if name in frame), None)
    if source is None:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    values = pd.to_numeric(frame[source], errors="coerce")
    if canonical == "speed_mps" and source in {"Speed", "speed_kmh"}:
        values = values / 3.6
    elif canonical == "accel_pedal_norm" and source == "Acc_Pedal":
        values = (values / 200.0).clip(0.0, 1.0)
    elif canonical == "brake_active" and source in {"Brake", "brake"}:
        values = values.gt(0).where(values.notna()).astype("float64")
    elif canonical == "rpm_abs" and source in {"RPM", "rpm"}:
        values = values.abs()
    elif canonical == "current_discharge_a" and source == "HV_Bat_Current_2":
        values = -values
    elif canonical == "hvac_power_w" and source == "hvac_power_kw":
        values = values * 1000.0
    return values


def _canonicalise_channels(
    frame: pd.DataFrame,
    driver_channels: Sequence[str],
    battery_channels: Sequence[str],
) -> pd.DataFrame:
    out = frame.copy()
    out["datetime_utc"] = pd.to_datetime(out.get("datetime_utc"), utc=True, errors="coerce")
    if "segment_id" not in out:
        raise ValueError("one-Hz telemetry must contain segment_id")
    out["segment_id"] = pd.to_numeric(out["segment_id"], errors="coerce")
    out = out.loc[out["datetime_utc"].notna() & out["segment_id"].notna()].copy()
    if out.empty:
        return out

    if "trip_key" not in out:
        if "Trip" not in out:
            raise ValueError("one-Hz telemetry must contain trip_key or Trip")
        out["trip_key"] = "trip:" + out["Trip"].astype("string").fillna("unknown")
    out["trip_key"] = out["trip_key"].astype("string").fillna("trip:unknown")

    # Some external adapters number segments from zero within each trip.  The
    # shared iterator groups only on segment_id, so make (trip, segment) pairs
    # globally unique without depending on hash randomisation.
    pair = pd.MultiIndex.from_arrays(
        [out["trip_key"].astype(str), out["segment_id"].astype("int64")]
    )
    if out.groupby("segment_id", dropna=False)["trip_key"].nunique().gt(1).any():
        out["segment_id"] = pd.factorize(pair, sort=True)[0].astype("int64")
    else:
        out["segment_id"] = out["segment_id"].astype("int64")

    requested = tuple(dict.fromkeys([*driver_channels, *battery_channels]))
    for canonical in requested:
        if canonical in out:
            out[canonical] = pd.to_numeric(out[canonical], errors="coerce")
        else:
            out[canonical] = _numeric_alias(out, canonical)

    # Reconstruct the audited temperature statistics when only sensor channels
    # are present.  The source-computed average is otherwise merely a fallback.
    sensor_columns = [name for name in ("Pack_1_C_Temp", "Pack_2_C_Temp", "Pack_4_C_Temp") if name in out]
    if sensor_columns:
        sensors = out[sensor_columns].apply(pd.to_numeric, errors="coerce")
        enough = sensors.notna().sum(axis=1).ge(2)
        if "pack_temp_avg_c" in requested:
            audited_mean = sensors.mean(axis=1).where(enough)
            out["pack_temp_avg_c"] = audited_mean.combine_first(out["pack_temp_avg_c"])
        if "pack_temp_spread_c" in requested:
            audited_spread = (sensors.max(axis=1) - sensors.min(axis=1)).where(enough)
            out["pack_temp_spread_c"] = audited_spread.combine_first(out["pack_temp_spread_c"])

    if "observed_1hz" not in out:
        out["observed_1hz"] = 1
    else:
        out["observed_1hz"] = pd.to_numeric(out["observed_1hz"], errors="coerce").fillna(0).gt(0).astype("uint8")
    if "key_sensor_valid" not in out:
        core = [name for name in ("speed_mps", "current_discharge_a", "voltage_v", "soc_pct", "pack_temp_avg_c") if name in out]
        out["key_sensor_valid"] = out[core].notna().all(axis=1) if core else False
    else:
        out["key_sensor_valid"] = out["key_sensor_valid"].fillna(False).astype(bool)

    sort_columns = ["segment_id", "datetime_utc"]
    if "source_row_id" in out:
        sort_columns.append("source_row_id")
    return out.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _normalise_key(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _consistent_value(values: pd.Series) -> Any | None:
    valid = values.dropna()
    if valid.empty:
        return None
    numeric = pd.to_numeric(valid, errors="coerce")
    if numeric.notna().all():
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            return None
        first = float(finite.iloc[0])
        if np.allclose(finite.to_numpy(dtype=float), first, rtol=1e-6, atol=1e-8):
            return first
        return None
    text = valid.astype(str).str.strip()
    folded = text.str.casefold().unique()
    return text.iloc[0] if len(folded) == 1 else None


class _SafeLookup:
    """Per-trip lookup that refuses ambiguous duplicate labels."""

    def __init__(self, table: pd.DataFrame | None, source: pd.DataFrame) -> None:
        self.table = table
        self.key_column: str | None = None
        self._maps: dict[str, dict[str, Any]] = {}
        if table is not None:
            self.key_column = next(
                (name for name in ("trip_key", "Trip") if name in table and name in source),
                None,
            )

    def _map(self, column: str) -> dict[str, Any]:
        if column in self._maps:
            return self._maps[column]
        mapping: dict[str, Any] = {}
        if self.table is not None and self.key_column is not None and column in self.table:
            subset = self.table[[self.key_column, column]].copy()
            subset["__key"] = subset[self.key_column].map(_normalise_key)
            subset = subset.loc[subset["__key"].notna()]
            for key, group in subset.groupby("__key", sort=True, dropna=False):
                value = _consistent_value(group[column])
                if value is not None:
                    mapping[str(key)] = value
        self._maps[column] = mapping
        return mapping

    def get(self, window: SequenceWindow, aliases: Sequence[str]) -> Any | None:
        if self.key_column is None:
            return None
        if self.key_column == "trip_key":
            key = _normalise_key(window.trip_key)
        else:
            trip = window.data.get("Trip", pd.Series(dtype=object)).dropna()
            key = _normalise_key(trip.iloc[0]) if len(trip) else None
        if key is None:
            return None
        for column in aliases:
            value = self._map(column).get(key)
            if value is not None:
                return value
        return None


def _first_window_value(window: SequenceWindow, column: str, default: Any = "") -> Any:
    if column not in window.data:
        return default
    values = window.data[column].dropna()
    return values.iloc[0] if len(values) else default


def _as_finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _behavior_code(value: Any) -> int:
    if value is None:
        return -100
    numeric = _as_finite_float(value)
    if np.isfinite(numeric) and numeric.is_integer() and 0 <= int(numeric) <= 2:
        return int(numeric)
    return {"eco": 0, "balanced": 1, "aggressive": 2}.get(str(value).strip().casefold(), -100)


def _driver_code(value: Any, *, already_encoded: bool = False) -> int:
    numeric = _as_finite_float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        return -100
    identifier = int(numeric)
    if already_encoded and identifier >= 0:
        return identifier
    # The registered Nissan analysis excludes rare IDs 3, 5 and 6.  Preserve
    # the four eligible source IDs because the diagnostic head uses the
    # original seven-class index space.
    return identifier if identifier in {0, 1, 2, 4} else -100


def _utf8_matrix(values: Sequence[Any]) -> np.ndarray:
    encoded = [str(value).encode("utf-8") for value in values]
    width = max((len(value) for value in encoded), default=1)
    output = np.zeros((len(encoded), max(width, 1)), dtype=np.uint8)
    for row, value in enumerate(encoded):
        if value:
            output[row, : len(value)] = np.frombuffer(value, dtype=np.uint8)
    return output


def decode_utf8_rows(values: np.ndarray) -> list[str]:
    """Decode the numeric UTF-8 row format used for NPZ metadata."""

    array = np.asarray(values, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("encoded UTF-8 metadata must have shape [N, bytes]")
    return [bytes(row[row != 0]).decode("utf-8") for row in array]


def _resolve_from_lookups(
    window: SequenceWindow,
    lookups: Sequence[_SafeLookup],
    aliases: Sequence[str],
) -> Any | None:
    for lookup in lookups:
        value = lookup.get(window, aliases)
        if value is not None:
            return value
    return None


def _concept_value(
    name: str,
    window: SequenceWindow,
    trip_lookup: _SafeLookup,
    behavior_lookup: _SafeLookup,
) -> float:
    value = _resolve_from_lookups(
        window, (trip_lookup, behavior_lookup), _CONCEPT_ALIASES[name]
    )
    result = _as_finite_float(value)
    if np.isfinite(result):
        return result

    if name.startswith("behavior_share_"):
        label = behavior_lookup.get(window, _BEHAVIOR_ALIASES)
        code = _behavior_code(label)
        expected = {
            "behavior_share_eco": 0,
            "behavior_share_balanced": 1,
            "behavior_share_aggressive": 2,
        }[name]
        if code >= 0:
            return float(code == expected)
    return np.nan


def _native_sample_windows(
    frame: pd.DataFrame,
    *,
    length: int,
    stride: int,
    min_coverage: float,
) -> Iterator[SequenceWindow]:
    """Yield fixed sample-count windows without fabricating higher cadence.

    This is the low-cadence adapter used for sources such as 0.1-Hz fleet
    telemetry.  Actual time deltas are retained as a model channel, while
    padding remains explicitly missing and masked.
    """

    for segment_id, source in frame.groupby("segment_id", sort=True):
        group = source.sort_values("datetime_utc", kind="mergesort").drop_duplicates(
            "datetime_utc", keep="last"
        )
        available = len(group)
        if available == 0:
            continue
        max_offset = max(available - length, 0)
        offsets = list(range(0, max_offset + 1, stride)) or [0]
        if available > length and offsets[-1] != max_offset:
            offsets.append(max_offset)
        positive_dt = group["datetime_utc"].diff().dt.total_seconds()
        cadence = float(positive_dt.loc[positive_dt.gt(0)].median())
        if not np.isfinite(cadence) or cadence <= 0:
            supplied = (
                pd.to_numeric(group["sampling_interval_s"], errors="coerce")
                if "sampling_interval_s" in group
                else pd.Series(np.nan, index=group.index, dtype="float64")
            )
            cadence = float(supplied.loc[supplied.gt(0)].median()) if supplied.notna().any() else 1.0
        cadence = cadence if np.isfinite(cadence) and cadence > 0 else 1.0

        for offset in offsets:
            observed_part = group.iloc[offset : offset + length].copy()
            observed_count = len(observed_part)
            coverage = observed_count / float(length)
            if coverage < min_coverage:
                continue
            start = pd.Timestamp(observed_part["datetime_utc"].iloc[0])
            full_time = pd.date_range(start, periods=length, freq=pd.to_timedelta(cadence, unit="s"))
            window = pd.DataFrame(index=np.arange(length), columns=group.columns)
            window.iloc[:observed_count] = observed_part.to_numpy()
            window["datetime_utc"] = full_time
            observed = np.zeros(length, dtype="uint8")
            observed[:observed_count] = 1
            window["observed_1hz"] = observed
            key_valid = pd.Series(False, index=window.index)
            if "key_sensor_valid" in observed_part:
                key_valid.iloc[:observed_count] = observed_part["key_sensor_valid"].fillna(False).to_numpy(dtype=bool)
            window["validity_mask"] = key_valid.astype("uint8")
            actual_dt = pd.to_datetime(window["datetime_utc"], utc=True).diff().dt.total_seconds()
            window["time_delta_s"] = actual_dt.fillna(0.0)
            window["relative_position"] = np.arange(length, dtype="float32") / max(length - 1, 1)
            trip_key = str(group["trip_key"].iloc[0])
            end = pd.Timestamp(window["datetime_utc"].iloc[-1])
            yield SequenceWindow(
                window_id=f"{trip_key}:seg{int(segment_id)}:native:{start.value // 1000}",
                trip_key=trip_key,
                segment_id=int(segment_id),
                start_utc=start,
                end_utc=end,
                coverage_pct=coverage * 100.0,
                data=window,
            )


def _export_windows(
    frame: pd.DataFrame,
    *,
    length: int,
    stride: int,
    min_coverage: float,
) -> Iterator[SequenceWindow]:
    deltas = frame.groupby("segment_id", sort=False)["datetime_utc"].diff().dt.total_seconds()
    median_cadence = float(deltas.loc[deltas.gt(0)].median()) if deltas.gt(0).any() else 1.0
    if np.isfinite(median_cadence) and median_cadence > 1.5:
        yield from _native_sample_windows(
            frame, length=length, stride=stride, min_coverage=min_coverage
        )
    else:
        yield from iter_fixed_windows(
            frame,
            length_s=length,
            stride_s=stride,
            min_coverage=min_coverage,
            include_partial=True,
        )


def export_segment_npz(
    one_hz_parquet_or_df: str | Path | pd.DataFrame,
    output_path: str | Path,
    trip_features: str | Path | pd.DataFrame | None = None,
    behavior_clusters: str | Path | pd.DataFrame | None = None,
    segment_length: int = 256,
    stride: int = 64,
    min_coverage: float = 0.8,
    driver_channels: Sequence[str] = DEFAULT_DRIVER_CHANNELS,
    battery_channels: Sequence[str] = DEFAULT_BATTERY_CHANNELS,
) -> Path:
    """Export deterministic fixed windows to the GPU-ready compressed NPZ contract.

    All sequence values are raw audited engineering quantities; this function
    intentionally performs no scaling or feature fitting.  Trip labels are
    attached only when every non-missing value for that trip agrees.  Conflicting
    duplicates are masked instead of being averaged or selected by row order.

    Text metadata is represented as padded numeric UTF-8 matrices so the NPZ can
    be loaded with ``allow_pickle=False`` and remains compatible with
    :meth:`dbpatchkan.models.EVSegmentDataset.from_npz`.
    """

    if segment_length <= 0 or stride <= 0:
        raise ValueError("segment_length and stride must be positive")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must lie in [0, 1]")
    driver_names = tuple(str(name) for name in driver_channels)
    battery_names = tuple(str(name) for name in battery_channels)
    if not driver_names or not battery_names:
        raise ValueError("driver_channels and battery_channels must be non-empty")
    if len(set(driver_names)) != len(driver_names) or len(set(battery_names)) != len(battery_names):
        raise ValueError("channel names must be unique within each stream")

    one_hz = _canonicalise_channels(
        _projected_one_hz(one_hz_parquet_or_df), driver_names, battery_names
    )
    destination = Path(output_path)
    if destination.suffix.casefold() != ".npz":
        destination = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)

    window_arguments = {
        "length": int(segment_length),
        "stride": int(stride),
        "min_coverage": float(min_coverage),
    }
    count = sum(1 for _ in _export_windows(one_hz, **window_arguments))

    driver = np.empty((count, segment_length, len(driver_names)), dtype=np.float32)
    battery = np.empty((count, segment_length, len(battery_names)), dtype=np.float32)
    driver_mask = np.empty(driver.shape, dtype=bool)
    battery_mask = np.empty(battery.shape, dtype=bool)
    validity = np.empty((count, segment_length), dtype=np.float32)
    time_delta = np.empty((count, segment_length), dtype=np.float32)
    relative_position = np.empty((count, segment_length), dtype=np.float32)
    concept_targets = np.full((count, len(DEFAULT_CONCEPT_NAMES)), np.nan, dtype=np.float32)
    concept_mask = np.zeros(concept_targets.shape, dtype=bool)

    trip_table = _path_table(trip_features)
    behavior_table = _path_table(behavior_clusters)
    trip_lookup = _SafeLookup(trip_table, one_hz)
    behavior_lookup = _SafeLookup(behavior_table, one_hz)
    source_lookup = _SafeLookup(one_hz, one_hz)

    current_soh = np.full(count, np.nan, dtype=np.float32)
    loss_30d = np.full(count, np.nan, dtype=np.float32)
    loss_500km = np.full(count, np.nan, dtype=np.float32)
    behavior_target = np.full(count, -100, dtype=np.int64)
    driver_target = np.full(count, -100, dtype=np.int64)
    window_ids: list[str] = []
    trip_keys: list[str] = []
    dataset_ids: list[str] = []
    vehicle_ids: list[str] = []
    starts = np.empty(count, dtype=np.int64)
    ends = np.empty(count, dtype=np.int64)
    segment_ids = np.empty(count, dtype=np.int64)
    coverage = np.empty(count, dtype=np.float32)

    for index, window in enumerate(_export_windows(one_hz, **window_arguments)):
        arrays = window_to_arrays(window, driver_names, battery_names)
        driver[index] = arrays["driver"]
        battery[index] = arrays["battery"]
        driver_mask[index] = ~arrays["driver_missing"].astype(bool)
        battery_mask[index] = ~arrays["battery_missing"].astype(bool)
        validity[index] = arrays["validity_mask"]
        time_delta[index] = arrays["time_delta_s"]
        relative_position[index] = arrays["relative_position"]

        window_ids.append(window.window_id)
        trip_keys.append(window.trip_key)
        dataset_ids.append(str(_first_window_value(window, "dataset_id", "")))
        vehicle_ids.append(str(_first_window_value(window, "vehicle_id", "")))
        starts[index] = int(window.start_utc.value // 1_000_000)
        ends[index] = int(window.end_utc.value // 1_000_000)
        segment_ids[index] = int(window.segment_id)
        coverage[index] = float(window.coverage_pct)

        current_soh[index] = _as_finite_float(
            _resolve_from_lookups(window, (trip_lookup, source_lookup), _CURRENT_SOH_ALIASES)
        )
        loss_30d[index] = _as_finite_float(
            _resolve_from_lookups(window, (trip_lookup,), _LOSS_30D_ALIASES)
        )
        loss_500km[index] = _as_finite_float(
            _resolve_from_lookups(window, (trip_lookup,), _LOSS_500KM_ALIASES)
        )
        behavior_target[index] = _behavior_code(
            _resolve_from_lookups(window, (behavior_lookup, trip_lookup), _BEHAVIOR_ALIASES)
        )
        encoded_driver = trip_lookup.get(window, ("driver_target",))
        if encoded_driver is not None:
            driver_target[index] = _driver_code(encoded_driver, already_encoded=True)
        else:
            raw_driver = _resolve_from_lookups(
                window, (trip_lookup, source_lookup), ("Driver_ID", "driver_id")
            )
            driver_target[index] = _driver_code(raw_driver)

        for concept_index, concept_name in enumerate(DEFAULT_CONCEPT_NAMES):
            concept_targets[index, concept_index] = _concept_value(
                concept_name, window, trip_lookup, behavior_lookup
            )
        concept_mask[index] = np.isfinite(concept_targets[index])

    arrays_to_write: dict[str, np.ndarray] = {
        "driver": driver,
        "battery": battery,
        "driver_mask": driver_mask,
        "battery_mask": battery_mask,
        "validity_mask": validity,
        "time_delta_s": time_delta,
        "relative_position": relative_position,
        "concept_targets": concept_targets,
        "concept_mask": concept_mask,
        "window_index": np.arange(count, dtype=np.int64),
        "window_start_unix_ms": starts,
        "window_end_unix_ms": ends,
        "segment_id": segment_ids,
        "coverage_pct": coverage,
        "window_id_utf8": _utf8_matrix(window_ids),
        "trip_key_utf8": _utf8_matrix(trip_keys),
        "dataset_id_utf8": _utf8_matrix(dataset_ids),
        "vehicle_id_utf8": _utf8_matrix(vehicle_ids),
    }

    # The key embeds the fixed concept order while retaining numeric dtypes and
    # a leading sample dimension for compatibility with the current loader.
    for index, name in enumerate(DEFAULT_CONCEPT_NAMES):
        arrays_to_write[f"concept_{index:02d}_{name}"] = concept_targets[:, index]
        arrays_to_write[f"concept_mask_{index:02d}_{name}"] = concept_mask[:, index]

    regression_targets = {
        "current_soh_target": current_soh,
        "loss_30d_target": loss_30d,
        "loss_500km_target": loss_500km,
    }
    for name, values in regression_targets.items():
        mask = np.isfinite(values)
        if name == "current_soh_target":
            mask &= (values >= 0.0) & (values <= 110.0)
        else:
            mask &= values >= 0.0
        if mask.any():
            arrays_to_write[name] = values
            arrays_to_write[name.replace("_target", "_mask")] = mask
    if np.any(behavior_target != -100):
        arrays_to_write["behavior_target"] = behavior_target
        arrays_to_write["behavior_target_mask"] = behavior_target != -100
    if np.any(driver_target != -100):
        arrays_to_write["driver_target"] = driver_target
        arrays_to_write["driver_target_mask"] = driver_target != -100

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}.", suffix=".tmp.npz", dir=destination.parent, delete=False
        ) as handle:
            temporary_name = handle.name
        np.savez_compressed(temporary_name, **arrays_to_write)
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return destination


__all__ = [
    "DEFAULT_BATTERY_CHANNELS",
    "DEFAULT_CONCEPT_NAMES",
    "DEFAULT_DRIVER_CHANNELS",
    "decode_utf8_rows",
    "export_segment_npz",
]
