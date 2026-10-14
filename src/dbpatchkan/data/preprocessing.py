"""Nissan Leaf telemetry ingestion, quality control, and resampling.

The functions in this module deliberately use only NumPy/Pandas (and PyArrow for
optional Parquet output).  They are deterministic and keep the source columns
alongside auditable, canonical derived columns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


NISSAN_SOURCE_COLUMNS: tuple[str, ...] = (
    "index", "Timestamp", "Speed", "Odometer", "RPM", "SOC", "SOH",
    "Torque", "Motor_Temp", "TP_FR", "TP_FL", "TP_RR", "TP_RL",
    "Aux_Pwr_100W", "AC_Pwr_250W", "EstPwr_AC_50W", "HV_Bat_Voltage",
    "Brake", "Acc_Pedal", "ECO", "ePedal", "HV_Bat_Current_2",
    "Ambient_C_Temp", "Pack_1_C_Temp", "Pack_2_C_Temp", "Pack_4_C_Temp",
    "Pack_Avg_C_Temp", "kWh_Remaining", "Horsepower", "Instant_kW",
    "Driver_ID", "Trip",
)

# A bitmask permits a source row to retain every applicable QC reason.
QC_FLAGS: Mapping[str, int] = {
    "invalid_timestamp": 1 << 0,
    "invalid_speed": 1 << 1,
    "invalid_soc": 1 << 2,
    "invalid_voltage": 1 << 3,
    "invalid_current": 1 << 4,
    "invalid_pack_temperature": 1 << 5,
    "rpm_sentinel": 1 << 6,
    "motor_temperature_sentinel": 1 << 7,
    "invalid_tire_pressure": 1 << 8,
    "invalid_pedal_or_brake": 1 << 9,
    "invalid_mode": 1 << 10,
    "invalid_context_sensor": 1 << 11,
    "invalid_odometer": 1 << 12,
}

QC_RULES: tuple[Mapping[str, Any], ...] = (
    {"Rule_ID": "Q01", "Name": "invalid_timestamp", "Field": "Timestamp", "Condition": "not a Unix-microsecond timestamp", "Severity": "error", "Action": "flag; start a new segment"},
    {"Rule_ID": "Q02", "Name": "invalid_speed", "Field": "Speed", "Condition": "outside 0..200 km/h", "Severity": "error", "Action": "replace with missing"},
    {"Rule_ID": "Q03", "Name": "invalid_soc", "Field": "SOC", "Condition": "outside 0..100%", "Severity": "error", "Action": "replace with missing"},
    {"Rule_ID": "Q04", "Name": "invalid_voltage", "Field": "HV_Bat_Voltage", "Condition": "0/655.35 sentinel or outside 240..420 V", "Severity": "error", "Action": "replace with missing"},
    {"Rule_ID": "Q05", "Name": "invalid_current", "Field": "HV_Bat_Current_2", "Condition": "absolute value above 1000 A", "Severity": "error", "Action": "replace with missing"},
    {"Rule_ID": "Q06", "Name": "invalid_pack_temperature", "Field": "Pack_*_C_Temp", "Condition": "outside -30..80 C", "Severity": "warning", "Action": "replace affected sensor with missing"},
    {"Rule_ID": "Q07", "Name": "rpm_sentinel", "Field": "RPM", "Condition": "absolute value 16383 or above 20000 rpm", "Severity": "warning", "Action": "replace with missing"},
    {"Rule_ID": "Q08", "Name": "motor_temperature_sentinel", "Field": "Motor_Temp", "Condition": "-40 sentinel or outside -30..200 C", "Severity": "warning", "Action": "replace with missing"},
    {"Rule_ID": "Q09", "Name": "invalid_tire_pressure", "Field": "TP_*", "Condition": "zero or outside 100..500 kPa", "Severity": "warning", "Action": "replace with missing"},
    {"Rule_ID": "Q10", "Name": "invalid_pedal_or_brake", "Field": "Acc_Pedal/Brake", "Condition": "pedal outside 0..200 or brake below zero", "Severity": "warning", "Action": "replace with missing"},
    {"Rule_ID": "Q11", "Name": "invalid_mode", "Field": "ECO/ePedal", "Condition": "not 0 or 1", "Severity": "warning", "Action": "replace with missing"},
    {"Rule_ID": "Q12", "Name": "invalid_context_sensor", "Field": "ambient/HVAC/auxiliary", "Condition": "outside documented engineering bounds", "Severity": "warning", "Action": "replace with missing"},
    {"Rule_ID": "Q13", "Name": "invalid_odometer", "Field": "Odometer", "Condition": "255-km sentinel, negative, or implausibly large", "Severity": "warning", "Action": "replace with missing; audit chronology before distance targets"},
)


@dataclass
class PreprocessState:
    """State carried between CSV chunks to preserve temporal segmentation."""

    last_trip: Any = None
    last_timestamp: pd.Timestamp | None = None
    last_segment_id: int = -1


@dataclass(frozen=True)
class PreprocessConfig:
    nominal_capacity_kwh: float = 40.0
    nominal_capacity_ah: float = 111.11
    max_gap_s: float = 2.0
    native_hz: float = 4.0
    segment_length_s: int = 256
    segment_stride_s: int = 128
    min_window_coverage: float = 0.80
    chunksize: int = 250_000
    dataset_id: str = "nissan_leaf_real_world"
    vehicle_id: str = "nissan_leaf_2018_40kwh_01"
    write_parquet: bool = True
    speed_bounds_kmh: tuple[float, float] = (0.0, 200.0)
    voltage_bounds_v: tuple[float, float] = (240.0, 420.0)
    soc_bounds_pct: tuple[float, float] = (0.0, 100.0)
    pack_temperature_bounds_c: tuple[float, float] = (-20.0, 70.0)
    tire_pressure_bounds_kpa: tuple[float, float] = (150.0, 400.0)
    dbsi_parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SequenceWindow:
    """One fixed-length, non-crossing temporal window."""

    window_id: str
    trip_key: str
    segment_id: int
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    coverage_pct: float
    data: pd.DataFrame


def _coerce_config(config: PreprocessConfig | Mapping[str, Any] | None) -> PreprocessConfig:
    if config is None:
        return PreprocessConfig()
    if isinstance(config, PreprocessConfig):
        return config
    config_dict = dict(config)
    # Accept the repository's complete YAML structure as well as a compact flat
    # preprocessing mapping.  Unrelated sections are intentionally ignored.
    if "data" in config_dict and isinstance(config_dict["data"], Mapping):
        full_config = config_dict
        data = dict(config_dict["data"])
        aliases = {
            "nominal_energy_kwh": "nominal_capacity_kwh",
            "nominal_capacity_ah": "nominal_capacity_ah",
            "segment_gap_seconds": "max_gap_s",
            "segment_length": "segment_length_s",
            "segment_stride": "segment_stride_s",
            "min_window_coverage": "min_window_coverage",
            "dataset_id": "dataset_id",
            "vehicle_id": "vehicle_id",
            "chunksize": "chunksize",
            "native_hz": "native_hz",
            "write_parquet": "write_parquet",
        }
        config_dict = {target: data[source] for source, target in aliases.items() if source in data}
        qc = dict(full_config.get("qc", {}))
        for source, target in {
            "speed_kmh": "speed_bounds_kmh",
            "voltage_v": "voltage_bounds_v",
            "soc_pct": "soc_bounds_pct",
            "pack_temperature_c": "pack_temperature_bounds_c",
            "tire_pressure_kpa": "tire_pressure_bounds_kpa",
        }.items():
            if source in qc:
                config_dict[target] = tuple(float(value) for value in qc[source])

        dbsi = dict(full_config.get("dbsi", {}))
        weights = dict(dbsi.get("weights", {}))
        dbsi_parameters: dict[str, Any] = {
            f"weight_{name}": float(value) for name, value in weights.items()
        }
        for source, targets in {
            "discharge_c_rate": ("discharge_c_lower", "discharge_c_upper"),
            "regen_c_rate": ("regen_c_lower", "regen_c_upper"),
            "high_temperature_c": ("hot_temp_lower_c", "hot_temp_upper_c"),
            "temperature_spread_c": ("temp_spread_lower_c", "temp_spread_upper_c"),
            "high_soc_pct": ("high_soc_lower_pct", "high_soc_upper_pct"),
            "low_soc_pct": ("low_soc_upper_pct", "low_soc_lower_pct"),
            "dod_pct": ("dod_lower_pct", "dod_upper_pct"),
            "c_rate_slew": ("slew_lower_c_per_s", "slew_upper_c_per_s"),
        }.items():
            if source in dbsi and len(dbsi[source]) == 2:
                dbsi_parameters[targets[0]] = float(dbsi[source][0])
                dbsi_parameters[targets[1]] = float(dbsi[source][1])
        if "cold_regen_below_c" in dbsi:
            dbsi_parameters["cold_regen_start_c"] = float(dbsi["cold_regen_below_c"])
        config_dict["dbsi_parameters"] = dbsi_parameters
    allowed = set(PreprocessConfig.__dataclass_fields__)
    unknown = set(config_dict) - allowed
    if unknown:
        raise ValueError(f"Unknown preprocessing configuration keys: {sorted(unknown)}")
    return PreprocessConfig(**config_dict)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _invalidate(
    frame: pd.DataFrame,
    column: str,
    invalid: pd.Series,
    flag_name: str,
    bitmask: np.ndarray,
) -> None:
    """Set invalid source values to NaN and update a NumPy QC bitmask."""

    if column not in frame:
        return
    mask = invalid.fillna(False).to_numpy(dtype=bool)
    if mask.any():
        frame.loc[mask, column] = np.nan
        bitmask[mask] |= np.uint32(QC_FLAGS[flag_name])


def preprocess_nissan_frame(
    frame: pd.DataFrame,
    *,
    state: PreprocessState | None = None,
    config: PreprocessConfig | Mapping[str, Any] | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Clean and enrich one Nissan telemetry frame.

    Raw negative ``HV_Bat_Current_2`` denotes propulsion/discharge in this
    dataset.  The canonical ``current_std_a`` and ``p_batt_kw_clean`` therefore
    use positive values for battery discharge and negative values for regen.
    Native-rate distance, energy, and Ah integrations are reset across trip
    boundaries, non-positive time deltas, and gaps greater than ``max_gap_s``.
    """

    cfg = _coerce_config(config)
    out = frame.copy(deep=True) if copy else frame
    if len(out) == 0:
        return out

    for col in NISSAN_SOURCE_COLUMNS:
        if col not in out:
            out[col] = np.nan

    # Coerce all source fields except no-op object identifiers to numeric.  The
    # official Nissan file is entirely numeric, and coercion makes adapters safe.
    for col in NISSAN_SOURCE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    bits = np.zeros(len(out), dtype=np.uint32)
    timestamp_num = _numeric(out, "Timestamp")
    out["datetime_utc"] = pd.to_datetime(timestamp_num, unit="us", utc=True, errors="coerce")
    bad_ts = out["datetime_utc"].isna()
    bits[bad_ts.to_numpy()] |= np.uint32(QC_FLAGS["invalid_timestamp"])

    _invalidate(
        out,
        "Speed",
        (_numeric(out, "Speed") < cfg.speed_bounds_kmh[0])
        | (_numeric(out, "Speed") > cfg.speed_bounds_kmh[1]),
        "invalid_speed",
        bits,
    )
    odometer = _numeric(out, "Odometer")
    _invalidate(
        out,
        "Odometer",
        odometer.eq(255) | odometer.lt(0) | odometer.gt(2_000_000),
        "invalid_odometer",
        bits,
    )
    _invalidate(
        out,
        "SOC",
        (_numeric(out, "SOC") < cfg.soc_bounds_pct[0])
        | (_numeric(out, "SOC") > cfg.soc_bounds_pct[1]),
        "invalid_soc",
        bits,
    )

    voltage = _numeric(out, "HV_Bat_Voltage")
    voltage_bad = (
        voltage.eq(0) | np.isclose(voltage, 655.35)
        | voltage.lt(cfg.voltage_bounds_v[0]) | voltage.gt(cfg.voltage_bounds_v[1])
    )
    _invalidate(out, "HV_Bat_Voltage", voltage_bad, "invalid_voltage", bits)
    current = _numeric(out, "HV_Bat_Current_2")
    _invalidate(out, "HV_Bat_Current_2", current.abs().gt(1000), "invalid_current", bits)

    rpm = _numeric(out, "RPM")
    _invalidate(out, "RPM", rpm.abs().eq(16383) | rpm.abs().gt(20_000), "rpm_sentinel", bits)
    motor_temp = _numeric(out, "Motor_Temp")
    _invalidate(out, "Motor_Temp", motor_temp.le(-40) | motor_temp.lt(-30) | motor_temp.gt(200), "motor_temperature_sentinel", bits)

    for col in ("Pack_1_C_Temp", "Pack_2_C_Temp", "Pack_4_C_Temp"):
        value = _numeric(out, col)
        _invalidate(
            out,
            col,
            value.lt(cfg.pack_temperature_bounds_c[0])
            | value.gt(cfg.pack_temperature_bounds_c[1]),
            "invalid_pack_temperature",
            bits,
        )
    for col in ("TP_FR", "TP_FL", "TP_RR", "TP_RL"):
        value = _numeric(out, col)
        _invalidate(
            out,
            col,
            value.le(0) | value.lt(cfg.tire_pressure_bounds_kpa[0])
            | value.gt(cfg.tire_pressure_bounds_kpa[1]),
            "invalid_tire_pressure",
            bits,
        )

    pedal = _numeric(out, "Acc_Pedal")
    _invalidate(out, "Acc_Pedal", pedal.lt(0) | pedal.gt(200), "invalid_pedal_or_brake", bits)
    brake = _numeric(out, "Brake")
    _invalidate(out, "Brake", brake.lt(0), "invalid_pedal_or_brake", bits)
    for col in ("ECO", "ePedal"):
        value = _numeric(out, col)
        _invalidate(out, col, value.notna() & ~value.isin([0, 1]), "invalid_mode", bits)

    ambient = _numeric(out, "Ambient_C_Temp")
    _invalidate(out, "Ambient_C_Temp", ambient.lt(-50) | ambient.gt(70), "invalid_context_sensor", bits)
    for col in ("Aux_Pwr_100W", "AC_Pwr_250W", "EstPwr_AC_50W"):
        value = _numeric(out, col)
        _invalidate(out, col, value.lt(0) | value.gt(20_000), "invalid_context_sensor", bits)

    # Segment assignment is vectorised but honours state from the preceding CSV
    # chunk.  A missing timestamp always breaks the segment.
    local_state = state if state is not None else PreprocessState()
    ts = out["datetime_utc"]
    trip = out["Trip"]
    dt = ts.diff().dt.total_seconds()
    trip_change = trip.ne(trip.shift())
    breaks = trip_change | dt.isna() | dt.le(0) | dt.gt(cfg.max_gap_s)
    if local_state.last_timestamp is not None and len(out):
        first_dt = (ts.iloc[0] - local_state.last_timestamp).total_seconds() if pd.notna(ts.iloc[0]) else np.nan
        same_trip = bool(pd.notna(trip.iloc[0]) and trip.iloc[0] == local_state.last_trip)
        first_break = (not same_trip) or (not np.isfinite(first_dt)) or first_dt <= 0 or first_dt > cfg.max_gap_s
        dt.iloc[0] = first_dt
        breaks.iloc[0] = first_break
    else:
        breaks.iloc[0] = True
        dt.iloc[0] = np.nan

    segment_ids = local_state.last_segment_id + breaks.astype("int64").cumsum()
    out["segment_id"] = segment_ids.astype("int64")
    valid_dt = dt.where(~breaks & dt.gt(0) & dt.le(cfg.max_gap_s), 0.0)
    out["dt_s"] = valid_dt.astype("float64")

    local_state.last_trip = trip.iloc[-1]
    local_state.last_timestamp = ts.iloc[-1] if pd.notna(ts.iloc[-1]) else None
    local_state.last_segment_id = int(segment_ids.iloc[-1])

    # Lineage and canonical engineering fields.
    out["dataset_id"] = cfg.dataset_id
    out["vehicle_id"] = cfg.vehicle_id
    out["source_row_id"] = out["index"]
    out["trip_key"] = (
        out["dataset_id"].astype(str) + ":" + out["vehicle_id"].astype(str)
        + ":" + out["Trip"].astype("Int64").astype(str)
    )
    out["qc_bitmask"] = bits
    out["qc_issue_count"] = pd.Series(bits, index=out.index).map(lambda x: int(x).bit_count()).astype("uint8")

    out["speed_mps"] = _numeric(out, "Speed") / 3.6
    out["speed_kmh"] = _numeric(out, "Speed")
    out["torque_nm"] = _numeric(out, "Torque")
    out["rpm_abs"] = _numeric(out, "RPM").abs()
    out["eco"] = _numeric(out, "ECO")
    out["epedal"] = _numeric(out, "ePedal")
    out["accel_pedal_norm"] = (_numeric(out, "Acc_Pedal") / 200.0).clip(0, 1)
    out["brake_active"] = _numeric(out, "Brake").gt(0).astype("uint8")
    out["stop_flag"] = _numeric(out, "Speed").lt(1).astype("uint8")
    out["high_speed_flag"] = _numeric(out, "Speed").ge(100).astype("uint8")

    out["current_std_a"] = -_numeric(out, "HV_Bat_Current_2")
    out["p_batt_kw_clean"] = _numeric(out, "HV_Bat_Voltage") * out["current_std_a"] / 1000.0
    out["p_dis_kw"] = out["p_batt_kw_clean"].clip(lower=0)
    out["p_regen_kw"] = (-out["p_batt_kw_clean"]).clip(lower=0)
    out["hv_battery_voltage_v"] = _numeric(out, "HV_Bat_Voltage")
    out["voltage_v"] = out["hv_battery_voltage_v"]
    out["current_discharge_a"] = out["current_std_a"]
    out["soc_pct"] = _numeric(out, "SOC")
    out["ambient_temp_c"] = _numeric(out, "Ambient_C_Temp")

    pack_cols = ["Pack_1_C_Temp", "Pack_2_C_Temp", "Pack_4_C_Temp"]
    pack = out[pack_cols].apply(pd.to_numeric, errors="coerce")
    valid_pack_count = pack.notna().sum(axis=1)
    out["pack_temp_avg_clean_c"] = pack.mean(axis=1).where(valid_pack_count >= 2)
    out["pack_temp_spread_c"] = (pack.max(axis=1) - pack.min(axis=1)).where(valid_pack_count >= 2)
    out["pack_temp_avg_c"] = out["pack_temp_avg_clean_c"]

    out["c_signed"] = out["current_std_a"] / cfg.nominal_capacity_ah
    out["c_dis"] = out["c_signed"].clip(lower=0)
    out["c_regen"] = (-out["c_signed"]).clip(lower=0)
    prev_c = out["c_signed"].shift()
    same_segment = out["segment_id"].eq(out["segment_id"].shift()) & out["dt_s"].gt(0)
    out["c_rate_slew"] = ((out["c_signed"] - prev_c).abs() / out["dt_s"].replace(0, np.nan)).where(same_segment)

    dt_hours = out["dt_s"] / 3600.0
    out["distance_step_km"] = _numeric(out, "Speed") * dt_hours
    out["energy_dis_step_kwh"] = out["p_dis_kw"] * dt_hours
    out["energy_regen_step_kwh"] = out["p_regen_kw"] * dt_hours
    out["net_energy_step_kwh"] = out["p_batt_kw_clean"] * dt_hours
    out["ah_throughput_step"] = out["current_std_a"].abs() * dt_hours
    out["efc_step"] = out["ah_throughput_step"] / (2.0 * cfg.nominal_capacity_ah)
    out["aux_power_kw"] = _numeric(out, "Aux_Pwr_100W") / 1000.0
    out["hvac_power_kw"] = _numeric(out, "AC_Pwr_250W") / 1000.0
    fallback_hvac = _numeric(out, "EstPwr_AC_50W") / 1000.0
    out["hvac_power_kw"] = out["hvac_power_kw"].fillna(fallback_hvac)
    out["hvac_power_w"] = out["hvac_power_kw"] * 1000.0
    out["hvac_aux_kw"] = out["hvac_power_kw"] + out["aux_power_kw"]
    out["aux_energy_step_kwh"] = out["aux_power_kw"] * dt_hours
    out["hvac_energy_step_kwh"] = out["hvac_power_kw"] * dt_hours

    out["soc_high_flag"] = _numeric(out, "SOC").gt(80).astype("uint8")
    out["soc_low_flag"] = _numeric(out, "SOC").lt(20).astype("uint8")
    out["hot_pack_flag"] = out["pack_temp_avg_clean_c"].gt(35).astype("uint8")
    out["cold_pack_flag"] = out["pack_temp_avg_clean_c"].lt(10).astype("uint8")
    out["key_sensor_valid"] = out[["Speed", "SOC", "HV_Bat_Voltage", "HV_Bat_Current_2", "pack_temp_avg_clean_c"]].notna().all(axis=1)
    out["valid_fraction"] = out["key_sensor_valid"].astype("float32")
    protected = out[["Speed", "HV_Bat_Current_2", "HV_Bat_Voltage", "SOC", "pack_temp_avg_clean_c"]]
    out["missing_fraction"] = protected.isna().mean(axis=1).astype("float32")

    elapsed = out.groupby("segment_id", sort=False)["dt_s"].cumsum()
    out["elapsed_segment_s"] = elapsed
    out["elapsed_trip_s"] = out.groupby("trip_key", sort=False)["dt_s"].cumsum()
    return out


def iter_nissan_csv(
    csv_path: str | Path,
    *,
    chunksize: int = 250_000,
    config: PreprocessConfig | Mapping[str, Any] | None = None,
    preprocess: bool = True,
    nrows: int | None = None,
    **read_csv_kwargs: Any,
) -> Iterator[pd.DataFrame]:
    """Yield source or cleaned Nissan CSV chunks with boundary-safe state."""

    cfg = _coerce_config(config)
    state = PreprocessState()
    remaining = nrows
    reader = pd.read_csv(csv_path, chunksize=chunksize, **read_csv_kwargs)
    for chunk in reader:
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
            remaining -= len(chunk)
        if len(chunk) == 0:
            break
        yield preprocess_nissan_frame(chunk, state=state, config=cfg) if preprocess else chunk
        if remaining == 0:
            break


def load_nissan_csv(
    csv_path: str | Path,
    *,
    chunksize: int | None = None,
    config: PreprocessConfig | Mapping[str, Any] | None = None,
    nrows: int | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load and preprocess a Nissan CSV, optionally through bounded chunks."""

    cfg = _coerce_config(config)
    if chunksize is None:
        raw = pd.read_csv(csv_path, nrows=nrows, **read_csv_kwargs)
        return preprocess_nissan_frame(raw, config=cfg)
    chunks = list(iter_nissan_csv(csv_path, chunksize=chunksize, config=cfg, nrows=nrows, **read_csv_kwargs))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def interpolate_short_gaps_1hz(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    max_gap_s: int = 2,
) -> pd.DataFrame:
    """Linearly fill complete, interior gaps of at most ``max_gap_s`` rows.

    The input must already be on a complete 1-Hz grid.  Long runs are left
    entirely missing (rather than partially filled), and interpolation is done
    independently inside each ``segment_id``.  Per-column ``*_imputed`` masks
    and an ``interpolated_1hz`` row flag retain full auditability.
    """

    if frame.empty:
        return frame.copy()
    if max_gap_s < 0:
        raise ValueError("max_gap_s must be non-negative")
    default_columns = (
        "Speed", "Odometer", "RPM", "SOC", "Torque", "Motor_Temp",
        "TP_FR", "TP_FL", "TP_RR", "TP_RL", "Aux_Pwr_100W",
        "AC_Pwr_250W", "EstPwr_AC_50W", "HV_Bat_Voltage", "Brake",
        "Acc_Pedal", "HV_Bat_Current_2", "Ambient_C_Temp",
        "Pack_1_C_Temp", "Pack_2_C_Temp", "Pack_4_C_Temp",
    )
    selected = [c for c in (columns or default_columns) if c in frame]
    out = frame.copy()
    out["interpolated_1hz"] = 0
    for col in selected:
        out[f"{col}_imputed"] = 0

    for _, index in out.groupby("segment_id", sort=False, dropna=False).groups.items():
        loc = pd.Index(index)
        for col in selected:
            values = pd.to_numeric(out.loc[loc, col], errors="coerce")
            missing = values.isna()
            if not missing.any() or missing.all() or max_gap_s == 0:
                continue
            # Each run is identified without leaking across a valid observation.
            run_id = (~missing).cumsum()
            run_length = missing.groupby(run_id).transform("sum")
            interior = values.ffill().notna() & values.bfill().notna()
            eligible = missing & interior & run_length.le(max_gap_s)
            if not eligible.any():
                continue
            interpolated = values.interpolate(method="linear", limit_area="inside")
            target_index = values.index[eligible]
            out.loc[target_index, col] = interpolated.loc[target_index]
            out.loc[target_index, f"{col}_imputed"] = 1
            out.loc[target_index, "interpolated_1hz"] = 1
    return out


def resample_telemetry_1hz(
    frame: pd.DataFrame,
    *,
    native_hz: float = 4.0,
    nominal_capacity_ah: float = 111.11,
    max_interpolation_gap_s: int = 2,
) -> pd.DataFrame:
    """Median-resample cleaned telemetry to 1 Hz without crossing segments."""

    if frame.empty:
        return frame.copy()
    required = {"segment_id", "datetime_utc"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Cannot resample without columns: {sorted(missing)}")

    data = frame.loc[frame["datetime_utc"].notna()].copy()
    if data.empty:
        return data
    data["datetime_utc_1s"] = data["datetime_utc"].dt.floor("s")
    keys = ["segment_id", "datetime_utc_1s"]
    id_cols = [c for c in ("dataset_id", "vehicle_id", "trip_key", "Trip", "Driver_ID") if c in data]
    state_cols = [c for c in ("ECO", "ePedal") if c in data]
    sum_cols = [c for c in data if c.endswith("_step_kwh") or c in {"ah_throughput_step", "efc_step", "distance_step_km"}]
    excluded = set(keys + id_cols + state_cols + sum_cols + ["datetime_utc", "source_row_id", "index", "Timestamp", "qc_bitmask"])
    numeric_cols = [c for c in data.select_dtypes(include=[np.number, "bool"]).columns if c not in excluded]
    agg: dict[str, Any] = {c: "median" for c in numeric_cols}
    agg.update({c: "first" for c in id_cols})
    agg.update({c: "last" for c in state_cols})
    agg.update({c: "sum" for c in sum_cols})
    if "source_row_id" in data:
        agg["source_row_id"] = "first"
    if "qc_bitmask" in data:
        agg["qc_bitmask"] = lambda x: np.bitwise_or.reduce(x.fillna(0).astype("uint32").to_numpy())
    agg["datetime_utc"] = "first"

    one = data.groupby(keys, sort=True, observed=True).agg(agg).reset_index()
    one["datetime_utc"] = one.pop("datetime_utc_1s")
    counts = data.groupby(keys, sort=True, observed=True).size().to_numpy()
    one["source_rows_1hz"] = counts
    one["coverage_1hz"] = np.minimum(counts / float(native_hz), 1.0)
    one["observed_1hz"] = 1
    one = one.sort_values(["segment_id", "datetime_utc"], kind="mergesort").reset_index(drop=True)

    # Materialize absent seconds first.  This makes short-gap interpolation exact
    # and gives the Transformer a stable time dimension without repeated values.
    expanded: list[pd.DataFrame] = []
    for segment_id, group in one.groupby("segment_id", sort=True):
        group = group.set_index("datetime_utc")
        full_index = pd.date_range(group.index.min(), group.index.max(), freq="1s")
        complete = group.reindex(full_index)
        complete.index.name = "datetime_utc"
        complete["segment_id"] = int(segment_id)
        for col in id_cols:
            complete[col] = complete[col].ffill().bfill()
        complete["observed_1hz"] = complete["observed_1hz"].fillna(0).astype("uint8")
        complete["source_rows_1hz"] = complete["source_rows_1hz"].fillna(0).astype("int32")
        complete["coverage_1hz"] = complete["coverage_1hz"].fillna(0.0)
        expanded.append(complete.reset_index())
    one = pd.concat(expanded, ignore_index=True)
    one = interpolate_short_gaps_1hz(one, max_gap_s=max_interpolation_gap_s)

    # Recompute canonical fields from interpolated source signals so voltage,
    # current, power, temperatures, and masks remain physically consistent.
    one["speed_mps"] = pd.to_numeric(one.get("Speed"), errors="coerce") / 3.6
    one["speed_kmh"] = pd.to_numeric(one.get("Speed"), errors="coerce")
    one["torque_nm"] = pd.to_numeric(one.get("Torque"), errors="coerce")
    one["rpm_abs"] = pd.to_numeric(one.get("RPM"), errors="coerce").abs()
    one["eco"] = pd.to_numeric(one.get("ECO"), errors="coerce")
    one["epedal"] = pd.to_numeric(one.get("ePedal"), errors="coerce")
    one["accel_pedal_norm"] = (pd.to_numeric(one.get("Acc_Pedal"), errors="coerce") / 200.0).clip(0, 1)
    one["brake_active"] = pd.to_numeric(one.get("Brake"), errors="coerce").gt(0).astype("uint8")
    one["stop_flag"] = pd.to_numeric(one.get("Speed"), errors="coerce").lt(1).astype("uint8")
    one["high_speed_flag"] = pd.to_numeric(one.get("Speed"), errors="coerce").ge(100).astype("uint8")
    one["current_std_a"] = -pd.to_numeric(one.get("HV_Bat_Current_2"), errors="coerce")
    one["hv_battery_voltage_v"] = pd.to_numeric(one.get("HV_Bat_Voltage"), errors="coerce")
    one["voltage_v"] = one["hv_battery_voltage_v"]
    one["current_discharge_a"] = one["current_std_a"]
    one["p_batt_kw_clean"] = one["voltage_v"] * one["current_std_a"] / 1000.0
    one["p_dis_kw"] = one["p_batt_kw_clean"].clip(lower=0)
    one["p_regen_kw"] = (-one["p_batt_kw_clean"]).clip(lower=0)
    pack_cols = ["Pack_1_C_Temp", "Pack_2_C_Temp", "Pack_4_C_Temp"]
    pack = one[pack_cols].apply(pd.to_numeric, errors="coerce")
    pack_count = pack.notna().sum(axis=1)
    one["pack_temp_avg_clean_c"] = pack.mean(axis=1).where(pack_count >= 2)
    one["pack_temp_avg_c"] = one["pack_temp_avg_clean_c"]
    one["pack_temp_spread_c"] = (pack.max(axis=1) - pack.min(axis=1)).where(pack_count >= 2)
    one["soc_pct"] = pd.to_numeric(one.get("SOC"), errors="coerce")
    one["ambient_temp_c"] = pd.to_numeric(one.get("Ambient_C_Temp"), errors="coerce")
    one["aux_power_kw"] = pd.to_numeric(one.get("Aux_Pwr_100W"), errors="coerce") / 1000.0
    one["hvac_power_kw"] = pd.to_numeric(one.get("AC_Pwr_250W"), errors="coerce") / 1000.0
    one["hvac_power_kw"] = one["hvac_power_kw"].fillna(
        pd.to_numeric(one.get("EstPwr_AC_50W"), errors="coerce") / 1000.0
    )
    one["hvac_power_w"] = one["hvac_power_kw"] * 1000.0
    one["hvac_aux_kw"] = one["hvac_power_kw"] + one["aux_power_kw"]
    one["key_sensor_valid"] = one[["Speed", "SOC", "HV_Bat_Voltage", "HV_Bat_Current_2", "pack_temp_avg_clean_c"]].notna().all(axis=1)
    one["valid_fraction"] = one["key_sensor_valid"].astype("float32")
    one["missing_fraction"] = one[["Speed", "SOC", "HV_Bat_Voltage", "HV_Bat_Current_2", "pack_temp_avg_clean_c"]].isna().mean(axis=1).astype("float32")

    dt = one.groupby("segment_id", sort=False)["datetime_utc"].diff().dt.total_seconds()
    one["dt_1hz_s"] = dt.where(dt.gt(0) & dt.le(2), 0.0).fillna(0.0)
    # The model-facing delta is the resampled cadence, not the median native dt.
    one["dt_s"] = one["dt_1hz_s"]
    speed = pd.to_numeric(one.get("speed_mps"), errors="coerce")
    # Three-point rolling median suppresses single-sample speed quantisation before
    # differentiating.  Rolling is strictly confined to a temporal segment.
    smoothed = speed.groupby(one["segment_id"], sort=False).transform(
        lambda x: x.rolling(3, center=True, min_periods=1).median()
    )
    ds = smoothed.groupby(one["segment_id"], sort=False).diff()
    one["speed_1hz_mps"] = smoothed
    one["accel_mps2"] = ds / one["dt_1hz_s"].replace(0, np.nan)
    da = one["accel_mps2"].groupby(one["segment_id"], sort=False).diff()
    one["jerk_mps3"] = da / one["dt_1hz_s"].replace(0, np.nan)
    one["harsh_accel_flag"] = one["accel_mps2"].ge(1.5).astype("uint8")
    one["harsh_brake_flag"] = one["accel_mps2"].le(-1.5).astype("uint8")
    # c-rate slew must follow interpolation and 1-Hz differentiation.
    one["c_signed"] = one["current_std_a"] / nominal_capacity_ah
    one["c_dis"] = one["c_signed"].clip(lower=0)
    one["c_regen"] = (-one["c_signed"]).clip(lower=0)
    one["c_rate_slew"] = one["c_signed"].groupby(one["segment_id"], sort=False).diff().abs() / one["dt_s"].replace(0, np.nan)
    return one


def iter_fixed_windows(
    one_hz: pd.DataFrame,
    *,
    length_s: int = 256,
    stride_s: int = 128,
    min_coverage: float = 0.80,
    include_partial: bool = True,
) -> Iterator[SequenceWindow]:
    """Yield exactly ``length_s`` rows per window, padded with missing seconds."""

    if length_s <= 0 or stride_s <= 0:
        raise ValueError("length_s and stride_s must be positive")
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must lie in [0, 1]")
    if one_hz.empty:
        return

    for segment_id, group in one_hz.groupby("segment_id", sort=True):
        group = group.sort_values("datetime_utc", kind="mergesort").drop_duplicates("datetime_utc", keep="last")
        first = group["datetime_utc"].min().floor("s")
        last = group["datetime_utc"].max().floor("s")
        available = int((last - first).total_seconds()) + 1
        if available < length_s and not include_partial:
            continue
        max_offset = max(available - length_s, 0)
        offsets = list(range(0, max_offset + 1, stride_s))
        if not offsets:
            offsets = [0]
        if available > length_s and offsets[-1] != max_offset:
            offsets.append(max_offset)

        indexed = group.set_index("datetime_utc")
        for offset in offsets:
            start = first + pd.Timedelta(seconds=offset)
            full_index = pd.date_range(start, periods=length_s, freq="1s", tz="UTC")
            window = indexed.reindex(full_index)
            observed = window["observed_1hz"].fillna(0).gt(0) if "observed_1hz" in window else window.notna().any(axis=1)
            coverage = float(observed.mean())
            if coverage < min_coverage:
                continue
            window.index.name = "datetime_utc"
            window = window.reset_index()
            window["observed_1hz"] = observed.to_numpy(dtype="uint8")
            window["validity_mask"] = (
                window.get("key_sensor_valid", pd.Series(False, index=window.index)).fillna(False).astype("uint8")
            )
            window["relative_position"] = np.arange(length_s, dtype="float32") / max(length_s - 1, 1)
            window["time_delta_s"] = np.where(np.arange(length_s) == 0, 0.0, 1.0)
            trip_key = str(group["trip_key"].iloc[0]) if "trip_key" in group else f"trip:{group.get('Trip', pd.Series(['unknown'])).iloc[0]}"
            window_id = f"{trip_key}:seg{int(segment_id)}:{start.value // 1000}"
            yield SequenceWindow(
                window_id=window_id,
                trip_key=trip_key,
                segment_id=int(segment_id),
                start_utc=start,
                end_utc=start + pd.Timedelta(seconds=length_s - 1),
                coverage_pct=coverage * 100.0,
                data=window,
            )


def window_to_arrays(
    window: SequenceWindow,
    driver_channels: Sequence[str],
    battery_channels: Sequence[str],
) -> dict[str, np.ndarray]:
    """Convert a fixed window into model-ready values and missingness masks."""

    def values_and_mask(columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        values = window.data.reindex(columns=list(columns)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        missing = ~np.isfinite(values)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), missing.astype(np.float32)

    driver, driver_missing = values_and_mask(driver_channels)
    battery, battery_missing = values_and_mask(battery_channels)
    return {
        "driver": driver,
        "battery": battery,
        "driver_missing": driver_missing,
        "battery_missing": battery_missing,
        "validity_mask": window.data["validity_mask"].to_numpy(dtype=np.float32),
        "time_delta_s": window.data["time_delta_s"].to_numpy(dtype=np.float32),
        "relative_position": window.data["relative_position"].to_numpy(dtype=np.float32),
    }


def deterministic_trip_sample(
    frame: pd.DataFrame,
    *,
    rows_per_trip: int = 200,
    anomaly_rows_per_trip: int = 25,
    max_rows: int = 30_000,
) -> pd.DataFrame:
    """Return an evenly spaced, anomaly-enriched, reproducible audit sample."""

    if frame.empty:
        return frame.copy()
    group_col = "trip_key" if "trip_key" in frame else "Trip"
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby(group_col, sort=True):
        n = min(rows_per_trip, len(group))
        positions = np.unique(np.linspace(0, len(group) - 1, n, dtype=int))
        regular = group.iloc[positions]
        if "qc_bitmask" in group:
            anomalous = group.loc[group["qc_bitmask"].fillna(0).ne(0)].head(anomaly_rows_per_trip)
            regular = pd.concat([regular, anomalous]).drop_duplicates()
        pieces.append(regular)
    sample = pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[:0].copy()
    if len(sample) > max_rows:
        keep = np.unique(np.linspace(0, len(sample) - 1, max_rows, dtype=int))
        sample = sample.iloc[keep]
    return sample.reset_index(drop=True)


def sha256_file(path: str | Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def run_nissan_pipeline(
    csv_path: str | Path,
    output_dir: str | Path,
    config: PreprocessConfig | Mapping[str, Any] | None = None,
    quick: bool = False,
) -> dict[str, Path]:
    """Run the complete deterministic CPU data pipeline and write audit tables.

    Full mode processes the CSV in chunks and finalises completed trips, avoiding
    a full-file in-memory load.  Quick mode reads at most 25,000 rows, disables
    Parquet output, and relaxes window coverage to make CI/smoke tests useful.
    """

    # Local imports avoid circular imports at module import time.
    from .dbsi import append_dbsi_by_group
    from .features import (
        aggregate_trip_features,
        aggregate_window_features,
        build_health_windows,
        build_source_profile,
        data_dictionary_frame,
    )

    cfg = _coerce_config(config)
    csv_path = Path(csv_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    nrows_limit = 25_000 if quick else None

    outputs: dict[str, Path] = {
        name: out_dir / name
        for name in (
            "profile.csv", "qc_summary.csv", "raw_sample.csv",
            "window_features.csv", "trip_features.csv", "health_windows.csv",
            "bsi_results.csv", "data_dictionary.csv", "manifest.json",
        )
    }

    # Preserve raw sentinel values in the profile.  Quantiles and unique counts
    # are explicitly sample-labelled; row/QC totals are accumulated by the main
    # streaming pass below.
    profile_nrows = 25_000 if quick else 100_000
    raw_profile_sample = pd.read_csv(csv_path, nrows=profile_nrows)
    raw_samples: list[pd.DataFrame] = []
    window_tables: list[pd.DataFrame] = []
    trip_tables: list[pd.DataFrame] = []
    qc_counts = {str(rule["Name"]): 0 for rule in QC_RULES}
    qc_trips: dict[str, set[str]] = {str(rule["Name"]): set() for rule in QC_RULES}
    carry = pd.DataFrame()
    row_count = 0
    clean_writer = None
    one_writer = None

    def write_parquet_piece(piece: pd.DataFrame, writer: Any, path: Path) -> Any:
        if quick or not cfg.write_parquet or piece.empty:
            return writer
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("PyArrow is required when write_parquet=True") from exc
        table = pa.Table.from_pandas(piece, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table.cast(writer.schema))
        return writer

    def finalise_trip(trip_frame: pd.DataFrame) -> None:
        nonlocal one_writer
        if trip_frame.empty:
            return
        one = resample_telemetry_1hz(
            trip_frame,
            native_hz=cfg.native_hz,
            nominal_capacity_ah=cfg.nominal_capacity_ah,
        )
        trip_tables.append(
            aggregate_trip_features(
                trip_frame,
                one_hz=one,
                nominal_capacity_ah=cfg.nominal_capacity_ah,
                dbsi_config=cfg.dbsi_parameters,
            )
        )
        window_tables.append(
            aggregate_window_features(
                one,
                native_frame=trip_frame,
                length_s=cfg.segment_length_s,
                stride_s=cfg.segment_stride_s,
                min_coverage=0.25 if quick else cfg.min_window_coverage,
                nominal_capacity_ah=cfg.nominal_capacity_ah,
                dbsi_config=cfg.dbsi_parameters,
            )
        )
        raw_samples.append(deterministic_trip_sample(trip_frame, max_rows=225))
        one_writer = write_parquet_piece(one, one_writer, out_dir / "telemetry_1hz.parquet")

    try:
        for clean in iter_nissan_csv(
            csv_path,
            chunksize=min(cfg.chunksize, 25_000) if quick else cfg.chunksize,
            config=cfg,
            nrows=nrows_limit,
        ):
            row_count += len(clean)
            chunk_bits = pd.to_numeric(clean["qc_bitmask"], errors="coerce").fillna(0).astype("uint32")
            for rule in QC_RULES:
                name = str(rule["Name"])
                affected = pd.Series(
                    (chunk_bits.to_numpy(dtype="uint32") & np.uint32(QC_FLAGS[name])) != 0,
                    index=clean.index,
                )
                qc_counts[name] += int(affected.sum())
                if affected.any():
                    qc_trips[name].update(clean.loc[affected, "trip_key"].dropna().astype(str).unique())
            clean_writer = write_parquet_piece(clean, clean_writer, out_dir / "clean_native.parquet")

            combined = pd.concat([carry, clean], ignore_index=True) if not carry.empty else clean
            trip_values = combined["Trip"].drop_duplicates().tolist()
            if len(trip_values) <= 1:
                carry = combined
                continue
            last_trip = combined["Trip"].iloc[-1]
            complete = combined.loc[combined["Trip"].ne(last_trip)]
            carry = combined.loc[combined["Trip"].eq(last_trip)].copy()
            for _, trip_frame in complete.groupby("Trip", sort=False):
                finalise_trip(trip_frame)
        finalise_trip(carry)
    finally:
        if clean_writer is not None:
            clean_writer.close()
        if one_writer is not None:
            one_writer.close()

    trips = pd.concat(trip_tables, ignore_index=True) if trip_tables else pd.DataFrame()
    if not trips.empty:
        trips = trips.sort_values("trip_start_utc", kind="mergesort").reset_index(drop=True)
        vehicle_cols = [c for c in ("dataset_id", "vehicle_id") if c in trips]
        if vehicle_cols:
            trips["cumulative_stress_dose"] = trips.groupby(
                vehicle_cols, sort=False, dropna=False
            )["stress_dose"].transform(lambda values: values.fillna(0).cumsum())
        else:
            trips["cumulative_stress_dose"] = trips["stress_dose"].fillna(0).cumsum()
    windows = pd.concat(window_tables, ignore_index=True) if window_tables else pd.DataFrame()
    bsi = append_dbsi_by_group(trips, group_col="trip_key", preaggregated=True) if not trips.empty else pd.DataFrame()
    # aggregate_trip_features already contains D-BSI; prefer its audit-complete
    # rows and use append_dbsi_by_group only as a compatibility fallback.
    if not trips.empty and "bsi_intensity" in trips:
        bsi_cols = [c for c in trips if c.startswith("bsi_") or c in {"trip_key", "Trip", "Driver_ID", "stress_dose", "cumulative_stress_dose", "duration_s"}]
        bsi = trips[bsi_cols].copy()
    health = build_health_windows(trips) if not trips.empty else pd.DataFrame()
    raw_sample = pd.concat(raw_samples, ignore_index=True) if raw_samples else pd.DataFrame()
    if len(raw_sample) > 30_000:
        indices = np.unique(np.linspace(0, len(raw_sample) - 1, 30_000, dtype=int))
        raw_sample = raw_sample.iloc[indices].reset_index(drop=True)

    build_source_profile(raw_profile_sample, total_rows=row_count).to_csv(outputs["profile.csv"], index=False)
    qc_summary = pd.DataFrame([
        {
            **rule,
            "Affected_Rows": qc_counts[str(rule["Name"])],
            "Affected_Trips": len(qc_trips[str(rule["Name"])]),
            "Rows_Remaining": row_count - qc_counts[str(rule["Name"])],
        }
        for rule in QC_RULES
    ])
    qc_summary.to_csv(outputs["qc_summary.csv"], index=False)
    raw_sample.to_csv(outputs["raw_sample.csv"], index=False)
    windows.to_csv(outputs["window_features.csv"], index=False)
    trips.to_csv(outputs["trip_features.csv"], index=False)
    health.to_csv(outputs["health_windows.csv"], index=False)
    bsi.to_csv(outputs["bsi_results.csv"], index=False)
    data_dictionary_frame().to_csv(outputs["data_dictionary.csv"], index=False)

    manifest = {
        "pipeline": "dbpatchkan-data-v1",
        "source_path": str(csv_path),
        "source_size_bytes": csv_path.stat().st_size,
        "source_sha256": sha256_file(csv_path),
        "rows_processed": row_count,
        "quick_mode": quick,
        "config": asdict(cfg),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "row_counts": {
            "raw_sample": len(raw_sample), "window_features": len(windows),
            "trip_features": len(trips), "health_windows": len(health),
            "bsi_results": len(bsi),
        },
    }
    outputs["manifest.json"].write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if not quick and cfg.write_parquet:
        outputs["clean_native.parquet"] = out_dir / "clean_native.parquet"
        outputs["telemetry_1hz.parquet"] = out_dir / "telemetry_1hz.parquet"
    return outputs


__all__ = [
    "NISSAN_SOURCE_COLUMNS", "QC_FLAGS", "QC_RULES", "PreprocessConfig",
    "PreprocessState", "SequenceWindow", "deterministic_trip_sample",
    "interpolate_short_gaps_1hz", "iter_fixed_windows", "iter_nissan_csv", "load_nissan_csv",
    "preprocess_nissan_frame", "resample_telemetry_1hz", "run_nissan_pipeline",
    "sha256_file", "window_to_arrays",
]
