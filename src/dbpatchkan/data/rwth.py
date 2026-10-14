"""RWTH fleet harmonisation and physics-linked driver--SOH features.

The RWTH field dataset contains measured driving telemetry and periodic 1 C
capacity tests.  This module preserves that distinction:

* capacity-based SOH is calculated only from the controlled discharge tests;
* exposure features use only telemetry occurring after one checkpoint and
  before the next checkpoint;
* pack temperature is not present in the field data, so thermal quantities are
  explicitly named proxies and never presented as measured temperature stress.

The physical pathway represented by the exported features is

``driver kinematics -> battery current/power -> C-rate/Joule-heating proxy ->
accumulated stress dose -> subsequent capacity loss``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .dbsi import high_severity, low_severity


@dataclass(frozen=True)
class RWTHVehicleSpec:
    """Battery constants required for capacity and C-rate calculations."""

    family: str
    nominal_capacity_ah: float
    cells_series: int
    window_cell_voltage_high_v: float
    window_cell_voltage_low_v: float
    initial_window_capacity_ah: float

    @property
    def window_pack_voltage_high_v(self) -> float:
        return self.cells_series * self.window_cell_voltage_high_v

    @property
    def window_pack_voltage_low_v(self) -> float:
        return self.cells_series * self.window_cell_voltage_low_v


SMART_SPEC = RWTHVehicleSpec(
    family="Smart",
    nominal_capacity_ah=52.0,
    cells_series=93,
    window_cell_voltage_high_v=3.914,
    window_cell_voltage_low_v=3.357,
    initial_window_capacity_ah=40.76,
)

IMIEV_SPEC = RWTHVehicleSpec(
    family="iMiEV_iOn",
    nominal_capacity_ah=50.0,
    cells_series=88,
    window_cell_voltage_high_v=3.933,
    window_cell_voltage_low_v=3.517,
    initial_window_capacity_ah=36.94,
)


# Tables D.13 and D.15 in Ruecker et al. (2024).  The paper values are rounded
# to 0.1 Ah and are the authoritative primary labels.  We also independently
# integrate the released time series and retain the difference as an audit.
RWTH_PUBLISHED_WINDOW_CAPACITY_AH: dict[tuple[str, int], float] = {
    ("Smart-1", 1): 40.8,
    ("Smart-1", 2): 39.3,
    ("Smart-1", 3): 38.6,
    ("Smart-1", 4): 38.0,
    ("Smart-2", 2): 40.5,
    ("Smart-2", 3): 39.8,
    ("Smart-2", 4): 39.8,
    ("Smart-3", 1): 39.9,
    ("Smart-3", 2): 37.2,
    ("Smart-3", 3): 38.8,
    ("Smart-4", 2): 39.2,
    ("Smart-5", 2): 40.8,
    ("Smart-6", 3): 39.5,
    ("iMiEV-1", 1): 36.9,
    ("iMiEV-1", 2): 37.6,
    ("iMiEV-1", 3): 38.2,
    ("iMiEV-1", 4): 37.8,
    ("iOn-1", 3): 34.6,
    ("iOn-2", 2): 34.4,
    ("iOn-2", 3): 32.2,
}


def rwth_vehicle_spec(vehicle_id: str) -> RWTHVehicleSpec:
    """Return the published pack constants for an RWTH vehicle identifier."""

    return SMART_SPEC if str(vehicle_id).startswith("Smart-") else IMIEV_SPEC


def matlab_datenum_to_datetime(values: Iterable[float]) -> pd.DatetimeIndex:
    """Convert MATLAB serial date numbers to naive UTC-like timestamps.

    RWTH timestamps are recorded in CET.  They are deliberately returned as
    timezone-naive wall-clock values because trip metadata uses the same CET
    convention and the dataset does not encode daylight-saving offsets.
    """

    numeric = np.asarray(values, dtype="float64")
    unix_days = numeric - 719529.0
    return pd.to_datetime(unix_days, unit="D", origin="unix", errors="coerce")


def _integral(
    time_days: np.ndarray,
    values: np.ndarray,
    *,
    max_gap_s: float,
    endpoint_mask: np.ndarray | None = None,
) -> float:
    """Trapezoidal integral over valid, contiguous sample pairs."""

    t = np.asarray(time_days, dtype="float64") * 86400.0
    y = np.asarray(values, dtype="float64")
    dt = np.diff(t)
    valid = (dt > 0.0) & (dt <= max_gap_s)
    valid &= np.isfinite(y[:-1]) & np.isfinite(y[1:])
    if endpoint_mask is not None:
        mask = np.asarray(endpoint_mask, dtype=bool)
        valid &= mask[:-1] & mask[1:]
    if not valid.any():
        return 0.0
    return float(np.sum(0.5 * (y[:-1][valid] + y[1:][valid]) * dt[valid]))


def capacity_checkpoints_from_frame(
    frame: pd.DataFrame,
    *,
    vehicle_id: str,
    max_gap_s: float = 1.0,
) -> pd.DataFrame:
    """Calculate auditable capacity checkpoints from one RWTH test table.

    Only controlled discharge tests (``test_direction == -1``) produce the
    primary SOH label.  Current is negative during discharge in the source.
    The fixed voltage windows and initial window capacities are those reported
    by Ruecker et al. (2024), allowing comparison across tests despite changes
    in the BMS usable voltage window.
    """

    required = {
        "time_num", "Odometer", "Current", "Voltage", "SoC_Real",
        "Temp_Ambient", "test_number", "test_direction",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"RWTH capacity frame is missing columns: {sorted(missing)}")
    spec = rwth_vehicle_spec(vehicle_id)
    records: list[dict[str, Any]] = []
    discharge = frame.loc[pd.to_numeric(frame["test_direction"], errors="coerce").eq(-1)].copy()
    for test_number, group in discharge.groupby("test_number", sort=True):
        group = group.sort_values("time_num", kind="mergesort")
        time_days = pd.to_numeric(group["time_num"], errors="coerce").to_numpy()
        current = pd.to_numeric(group["Current"], errors="coerce").to_numpy()
        voltage = pd.to_numeric(group["Voltage"], errors="coerce").to_numpy()
        pack_window = (
            (voltage >= spec.window_pack_voltage_low_v)
            & (voltage <= spec.window_pack_voltage_high_v)
        )
        capacity_usable_ah = -_integral(
            time_days, current, max_gap_s=max_gap_s
        ) / 3600.0
        capacity_window_ah_recalculated = -_integral(
            time_days, current, max_gap_s=max_gap_s, endpoint_mask=pack_window
        ) / 3600.0
        power = current * voltage
        energy_usable_kwh = -_integral(
            time_days, power, max_gap_s=max_gap_s
        ) / 3.6e6
        valid_time = time_days[np.isfinite(time_days)]
        start = matlab_datenum_to_datetime([valid_time.min()])[0]
        end = matlab_datenum_to_datetime([valid_time.max()])[0]
        published_capacity = RWTH_PUBLISHED_WINDOW_CAPACITY_AH.get(
            (vehicle_id, int(test_number))
        )
        capacity_window_ah = (
            float(published_capacity)
            if published_capacity is not None
            else capacity_window_ah_recalculated
        )
        records.append({
            "dataset_id": "rwth_commercial_ev_fleet",
            "vehicle_id": vehicle_id,
            "vehicle_family": spec.family,
            "test_number": int(test_number),
            "checkpoint_start": start,
            "checkpoint_end": end,
            "odometer_km": float(pd.to_numeric(group["Odometer"], errors="coerce").median()),
            "ambient_temp_c": float(
                pd.to_numeric(group["Temp_Ambient"], errors="coerce").median()
            ),
            "capacity_usable_ah": capacity_usable_ah,
            "capacity_window_ah": capacity_window_ah,
            "capacity_window_ah_recalculated": capacity_window_ah_recalculated,
            "capacity_recalculation_error_ah": capacity_window_ah_recalculated
            - capacity_window_ah,
            "energy_usable_kwh": energy_usable_kwh,
            "nominal_capacity_ah": spec.nominal_capacity_ah,
            "initial_window_capacity_ah": spec.initial_window_capacity_ah,
            "capacity_soh_pct": 100.0 * capacity_window_ah / spec.initial_window_capacity_ah,
            "window_pack_voltage_low_v": spec.window_pack_voltage_low_v,
            "window_pack_voltage_high_v": spec.window_pack_voltage_high_v,
            "sample_count": int(len(group)),
            "label_type": "1C_discharge_fixed_voltage_window",
            "label_source": (
                "published_tables_D13_D15" if published_capacity is not None
                else "recalculated_from_released_timeseries"
            ),
            "pack_temperature_observed": False,
        })
    return pd.DataFrame(records)


def build_capacity_checkpoints(capacity_dir: str | Path) -> pd.DataFrame:
    """Build primary capacity checkpoints for all RWTH vehicles."""

    capacity_dir = Path(capacity_dir)
    frames: list[pd.DataFrame] = []
    columns = [
        "time_num", "Odometer", "Current", "Voltage", "SoC_Real",
        "Temp_Ambient", "test_number", "test_direction",
    ]
    for path in sorted(capacity_dir.glob("*_capacity_tests.parquet")):
        vehicle_id = path.stem.replace("_capacity_tests", "")
        source = pq.read_table(path, columns=columns).to_pandas()
        frames.append(capacity_checkpoints_from_frame(source, vehicle_id=vehicle_id))
    if not frames:
        raise FileNotFoundError(f"No RWTH capacity parquet files found in {capacity_dir}")
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["vehicle_id", "checkpoint_start"], kind="mergesort").reset_index(
        drop=True
    )


def _arrhenius_factor(temperature_c: np.ndarray, *, activation_energy_j_mol: float) -> np.ndarray:
    """Relative Arrhenius rate versus 25 C using ambient temperature."""

    gas_constant = 8.314462618
    reference_k = 298.15
    temperature_k = np.clip(np.asarray(temperature_c, dtype="float64") + 273.15, 243.15, 333.15)
    return np.exp(
        activation_energy_j_mol / gas_constant * (1.0 / reference_k - 1.0 / temperature_k)
    )


def _safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _rms(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(np.sqrt(np.mean(np.square(numeric)))) if len(numeric) else float("nan")


def trip_physics_features(
    telemetry: pd.DataFrame,
    *,
    vehicle_id: str,
    thermal_time_constant_s: float = 1800.0,
    activation_energy_j_mol: float = 35_000.0,
) -> dict[str, Any]:
    """Aggregate one trip into driver and battery-physics exposure features."""

    if telemetry.empty:
        raise ValueError("telemetry must contain at least one row")
    spec = rwth_vehicle_spec(vehicle_id)
    group = telemetry.sort_values("datetime", kind="mergesort").reset_index(drop=True).copy()
    dt = group["datetime"].diff().dt.total_seconds().clip(lower=0.0, upper=5.0).fillna(0.0)
    speed = pd.to_numeric(group["Speed"], errors="coerce") / 3.6
    acceleration = speed.diff() / dt.replace(0.0, np.nan)
    acceleration = acceleration.where(acceleration.abs().le(8.0))
    jerk = acceleration.diff() / dt.replace(0.0, np.nan)
    jerk = jerk.where(jerk.abs().le(20.0))

    raw_current = pd.to_numeric(group["Current"], errors="coerce")
    current_discharge = -raw_current
    c_discharge = current_discharge.clip(lower=0.0) / spec.nominal_capacity_ah
    c_regen = (-current_discharge).clip(lower=0.0) / spec.nominal_capacity_ah
    c_abs = current_discharge.abs() / spec.nominal_capacity_ah
    c_slew = c_abs.diff().abs() / dt.replace(0.0, np.nan)
    voltage = pd.to_numeric(group["Voltage"], errors="coerce")
    power_discharge_kw = voltage * current_discharge / 1000.0
    soc = pd.to_numeric(group["SoC_Real"], errors="coerce")
    ambient = pd.to_numeric(group["Temp_Ambient"], errors="coerce")

    rate_severity = 0.75 * high_severity(c_discharge, 0.5, 3.0) + 0.25 * high_severity(
        c_regen, 0.3, 1.5
    )
    soc_severity = 0.7 * high_severity(soc, 80.0, 100.0) + 0.3 * low_severity(
        soc, 20.0, 0.0
    )
    transient_severity = high_severity(c_slew, 0.1, 1.0)
    arrhenius_ambient = pd.Series(
        _arrhenius_factor(ambient.to_numpy(), activation_energy_j_mol=activation_energy_j_mol),
        index=group.index,
    ).where(ambient.notna())

    # Dimensionless first-order thermal-memory proxy.  It is motivated by
    # C_th*dT/dt = I^2*R - hA*(T-T_amb), but is not a measured pack temperature.
    thermal_memory = np.zeros(len(group), dtype="float64")
    c_squared = np.nan_to_num(c_abs.to_numpy(dtype="float64"), nan=0.0) ** 2
    dt_values = dt.to_numpy(dtype="float64")
    for index in range(1, len(group)):
        decay = np.exp(-dt_values[index] / thermal_time_constant_s)
        thermal_memory[index] = thermal_memory[index - 1] * decay + c_squared[index] * (
            1.0 - decay
        )
    thermal_memory_series = pd.Series(thermal_memory, index=group.index)

    duration_h = float(dt.sum() / 3600.0)
    ah_throughput = float((raw_current.abs() * dt).sum() / 3600.0)
    efc = ah_throughput / (2.0 * spec.nominal_capacity_ah)
    dod = float(soc.max() - soc.min()) if soc.notna().any() else float("nan")
    cycle_severity = 0.5 * float(high_severity(dod, 20.0, 80.0))
    if duration_h > 0:
        cycle_severity += 0.5 * float(high_severity(efc / duration_h, 0.25, 1.0))

    observed_weight = 0.25 + 0.15 + 0.15 + 0.10
    partial_points = 100.0 * (
        0.25 * _safe_weighted_mean(pd.Series(rate_severity), dt.reset_index(drop=True))
        + 0.15 * _safe_weighted_mean(pd.Series(soc_severity), dt.reset_index(drop=True))
        + 0.15 * cycle_severity
        + 0.10 * _safe_weighted_mean(pd.Series(transient_severity), dt.reset_index(drop=True))
    )

    positive_accel = acceleration.clip(lower=0.0)
    negative_accel = (-acceleration.clip(upper=0.0))
    return {
        "duration_observed_h": duration_h,
        "speed_mean_kmh": float((speed * 3.6).mean()),
        "speed_max_kmh": float((speed * 3.6).max()),
        "speed_std_kmh": float((speed * 3.6).std()),
        "accel_positive_rms_mps2": _rms(positive_accel),
        "decel_rms_mps2": _rms(negative_accel),
        "jerk_rms_mps3": _rms(jerk),
        "harsh_accel_fraction": _safe_weighted_mean(acceleration.gt(1.5).astype(float), dt),
        "harsh_decel_fraction": _safe_weighted_mean(acceleration.lt(-1.5).astype(float), dt),
        "high_speed_fraction": _safe_weighted_mean(speed.gt(80.0 / 3.6).astype(float), dt),
        "stop_fraction": _safe_weighted_mean(speed.lt(0.5).astype(float), dt),
        "current_discharge_mean_a": _safe_weighted_mean(current_discharge.clip(lower=0.0), dt),
        "current_discharge_peak_a": float(current_discharge.max()),
        "c_rate_discharge_mean": _safe_weighted_mean(c_discharge, dt),
        "c_rate_discharge_peak": float(c_discharge.max()),
        "c_rate_regen_mean": _safe_weighted_mean(c_regen, dt),
        "c_rate_regen_peak": float(c_regen.max()),
        "high_c_rate_fraction": _safe_weighted_mean(c_discharge.gt(0.5).astype(float), dt),
        "ah_throughput": ah_throughput,
        "efc": efc,
        "energy_discharge_kwh": float((power_discharge_kw.clip(lower=0.0) * dt).sum() / 3600.0),
        "energy_regen_kwh": float((-power_discharge_kw.clip(upper=0.0) * dt).sum() / 3600.0),
        "soc_mean_pct": _safe_weighted_mean(soc, dt),
        "soc_high_fraction": _safe_weighted_mean(soc.gt(80.0).astype(float), dt),
        "soc_low_fraction": _safe_weighted_mean(soc.lt(20.0).astype(float), dt),
        "dod_pct": dod,
        "ambient_temp_mean_c": _safe_weighted_mean(ambient, dt),
        "arrhenius_ambient_factor_mean": _safe_weighted_mean(arrhenius_ambient, dt),
        "joule_c2_dose_h": float((c_squared * dt_values).sum() / 3600.0),
        "electrothermal_proxy_dose_h": float(
            (c_squared * np.nan_to_num(arrhenius_ambient, nan=1.0) * dt_values).sum() / 3600.0
        ),
        "thermal_memory_proxy_mean": _safe_weighted_mean(thermal_memory_series, dt),
        "thermal_memory_proxy_peak": float(np.nanmax(thermal_memory)),
        "bsi_rate": _safe_weighted_mean(pd.Series(rate_severity), dt.reset_index(drop=True)),
        "bsi_soc": _safe_weighted_mean(pd.Series(soc_severity), dt.reset_index(drop=True)),
        "bsi_cycle": float(np.clip(cycle_severity, 0.0, 1.0)),
        "bsi_transient": _safe_weighted_mean(
            pd.Series(transient_severity), dt.reset_index(drop=True)
        ),
        "dbsi_observed_lower_bound": partial_points,
        "dbsi_observed_weight_coverage": observed_weight,
        "dbsi_partial_renormalized": partial_points / observed_weight,
        "pack_temperature_observed": False,
        "thermal_feature_status": "latent_proxy_from_C_rate_squared_and_ambient",
    }


def _load_trip_metadata(trip_dir: Path) -> pd.DataFrame:
    frames = [
        pq.read_table(path).to_pandas()
        for path in sorted(trip_dir.glob("*_trip_data_datafile.parquet"))
    ]
    if not frames:
        raise FileNotFoundError(f"No datafile trip metadata found in {trip_dir}")
    trips = pd.concat(frames, ignore_index=True)
    trips["starttime"] = pd.to_datetime(trips["starttime"], errors="coerce")
    trips["endtime"] = trips["starttime"] + pd.to_timedelta(
        pd.to_numeric(trips["duration_seconds"], errors="coerce"), unit="s"
    )
    trips["trip_id"] = pd.to_numeric(trips["trip_id"], errors="coerce").astype("Int64")
    trips["trip_key"] = (
        "rwth_commercial_ev_fleet:" + trips["vehicle"].astype(str) + ":"
        + trips["trip_id"].astype(str)
    )
    return trips.sort_values(["vehicle", "starttime"], kind="mergesort").reset_index(drop=True)


def build_trip_features(
    field_dir: str | Path,
    trip_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map 1 Hz field telemetry to published trip intervals and aggregate it."""

    field_dir, trip_dir = Path(field_dir), Path(trip_dir)
    trips = _load_trip_metadata(trip_dir)
    output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    source_columns = [
        "time_num", "Speed", "Odometer", "Temp_Ambient", "SoC_Real",
        "Current", "Voltage", "Power",
    ]
    for path in sorted(field_dir.glob("*.parquet")):
        vehicle_id = path.stem
        telemetry = pq.read_table(path, columns=source_columns).to_pandas()
        telemetry["datetime"] = matlab_datenum_to_datetime(telemetry["time_num"])
        vehicle_trips = trips.loc[trips["vehicle"].eq(vehicle_id)].copy().reset_index(drop=True)
        # Pandas 3 may preserve source timestamps at microsecond resolution;
        # normalize all arrays to nanoseconds before integer interval matching.
        starts = vehicle_trips["starttime"].to_numpy(dtype="datetime64[ns]").astype("int64")
        ends = vehicle_trips["endtime"].to_numpy(dtype="datetime64[ns]").astype("int64")
        times = telemetry["datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")
        locations = np.searchsorted(starts, times, side="right") - 1
        valid = (locations >= 0) & (locations < len(vehicle_trips))
        safe_locations = np.clip(locations, 0, max(len(vehicle_trips) - 1, 0))
        valid &= times <= ends[safe_locations]
        telemetry = telemetry.loc[valid].copy()
        telemetry["trip_location"] = locations[valid]
        for location, group in telemetry.groupby("trip_location", sort=True):
            metadata = vehicle_trips.iloc[int(location)]
            features = trip_physics_features(group, vehicle_id=vehicle_id)
            output.append({
                "dataset_id": "rwth_commercial_ev_fleet",
                "vehicle_id": vehicle_id,
                "trip_id": int(metadata["trip_id"]),
                "trip_key": metadata["trip_key"],
                "trip_start": metadata["starttime"],
                "trip_end": metadata["endtime"],
                "fleet_type": metadata["fleet_type"],
                "distance_km": float(metadata["distance"]),
                "odometer_start_km": float(metadata["odometer_start"]),
                "soc_start_pct": float(metadata["soc_start"]),
                "soc_stop_pct": float(metadata["soc_stop"]),
                "source_sample_count": int(len(group)),
                **features,
            })
        audit.append({
            "vehicle_id": vehicle_id,
            "source_rows": int(len(valid)),
            "mapped_rows": int(valid.sum()),
            "mapped_fraction": float(valid.mean()),
            "published_trip_count": int(len(vehicle_trips)),
            "mapped_trip_count": int(telemetry["trip_location"].nunique()),
        })
    if not output:
        raise ValueError("No RWTH field rows mapped to published trip intervals")
    features = pd.DataFrame(output).sort_values(
        ["vehicle_id", "trip_start"], kind="mergesort"
    ).reset_index(drop=True)
    return features, pd.DataFrame(audit)


def build_driver_soh_intervals(
    checkpoints: pd.DataFrame,
    trip_features: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate strictly past-only driving exposure between SOH checkpoints."""

    records: list[dict[str, Any]] = []
    additive = [
        "duration_observed_h", "distance_km", "ah_throughput", "efc",
        "energy_discharge_kwh", "energy_regen_kwh", "joule_c2_dose_h",
        "electrothermal_proxy_dose_h",
    ]
    weighted = [
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
    ]
    for vehicle_id, vehicle_checkpoints in checkpoints.groupby("vehicle_id", sort=True):
        ordered = vehicle_checkpoints.sort_values("checkpoint_start", kind="mergesort").reset_index(
            drop=True
        )
        vehicle_trips = trip_features.loc[trip_features["vehicle_id"].eq(vehicle_id)]
        for index in range(1, len(ordered)):
            start, end = ordered.iloc[index - 1], ordered.iloc[index]
            exposure = vehicle_trips.loc[
                vehicle_trips["trip_start"].ge(start["checkpoint_end"])
                & vehicle_trips["trip_end"].le(end["checkpoint_start"])
            ].copy()
            interval_days = (end["checkpoint_start"] - start["checkpoint_end"]).total_seconds() / 86400
            odometer_delta = float(end["odometer_km"] - start["odometer_km"])
            soh_loss = float(start["capacity_soh_pct"] - end["capacity_soh_pct"])
            record: dict[str, Any] = {
                "dataset_id": "rwth_commercial_ev_fleet",
                "vehicle_id": vehicle_id,
                "vehicle_family": start["vehicle_family"],
                "interval_id": f"{vehicle_id}:{int(start['test_number'])}-{int(end['test_number'])}",
                "start_test_number": int(start["test_number"]),
                "end_test_number": int(end["test_number"]),
                "interval_start": start["checkpoint_end"],
                "interval_end": end["checkpoint_start"],
                "interval_days": interval_days,
                "odometer_delta_km": odometer_delta,
                "start_soh_pct": float(start["capacity_soh_pct"]),
                "end_soh_pct": float(end["capacity_soh_pct"]),
                "soh_loss_pp": soh_loss,
                "soh_loss_pp_per_1000km": (
                    soh_loss / odometer_delta * 1000.0 if odometer_delta > 0 else np.nan
                ),
                "soh_loss_pp_per_30d": (
                    soh_loss / interval_days * 30.0 if interval_days > 0 else np.nan
                ),
                "trip_count": int(len(exposure)),
                "pack_temperature_observed": False,
                "thermal_feature_status": "latent_proxy_from_C_rate_squared_and_ambient",
                "past_only_exposure": True,
            }
            for column in additive:
                record[f"sum_{column}"] = float(exposure[column].sum()) if len(exposure) else 0.0
            weights = exposure["duration_observed_h"] if len(exposure) else pd.Series(dtype=float)
            for column in weighted:
                record[f"mean_{column}"] = (
                    _safe_weighted_mean(exposure[column], weights) if len(exposure) else np.nan
                )
            trip_distance = record["sum_distance_km"]
            record["field_distance_coverage"] = (
                trip_distance / odometer_delta if odometer_delta > 0 else np.nan
            )
            records.append(record)
    return pd.DataFrame(records).sort_values(
        ["vehicle_id", "interval_start"], kind="mergesort"
    ).reset_index(drop=True)


def build_checkpoint_exposures(
    checkpoints: pd.DataFrame,
    trip_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build a secondary cumulative-exposure table for every health checkpoint.

    Unlike the primary interval table, this cross-sectional table can include
    vehicles with only one capacity test.  It is useful for current-SOH
    estimation but must not be interpreted as a within-vehicle degradation
    effect because initial battery condition is not observed for every vehicle.
    """

    additive = [
        "duration_observed_h", "distance_km", "ah_throughput", "efc",
        "energy_discharge_kwh", "energy_regen_kwh", "joule_c2_dose_h",
        "electrothermal_proxy_dose_h",
    ]
    weighted = [
        "speed_mean_kmh", "accel_positive_rms_mps2", "decel_rms_mps2",
        "jerk_rms_mps3", "harsh_accel_fraction", "harsh_decel_fraction",
        "high_speed_fraction", "stop_fraction", "c_rate_discharge_mean",
        "c_rate_discharge_peak", "c_rate_regen_mean", "c_rate_regen_peak",
        "high_c_rate_fraction", "soc_mean_pct", "soc_high_fraction",
        "soc_low_fraction", "dod_pct", "ambient_temp_mean_c",
        "arrhenius_ambient_factor_mean", "thermal_memory_proxy_mean",
        "thermal_memory_proxy_peak", "bsi_rate", "bsi_soc", "bsi_cycle",
        "bsi_transient", "dbsi_observed_lower_bound", "dbsi_partial_renormalized",
    ]
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints.itertuples(index=False):
        exposure = trip_features.loc[
            trip_features["vehicle_id"].eq(checkpoint.vehicle_id)
            & trip_features["trip_end"].le(checkpoint.checkpoint_start)
        ].copy()
        first_trip_start = exposure["trip_start"].min() if len(exposure) else pd.NaT
        observation_days = (
            (checkpoint.checkpoint_start - first_trip_start).total_seconds() / 86400.0
            if pd.notna(first_trip_start) else 0.0
        )
        record: dict[str, Any] = {
            "dataset_id": "rwth_commercial_ev_fleet",
            "vehicle_id": checkpoint.vehicle_id,
            "vehicle_family": checkpoint.vehicle_family,
            "checkpoint_id": f"{checkpoint.vehicle_id}:{int(checkpoint.test_number)}",
            "test_number": int(checkpoint.test_number),
            "checkpoint_start": checkpoint.checkpoint_start,
            "capacity_soh_pct": float(checkpoint.capacity_soh_pct),
            "capacity_window_ah": float(checkpoint.capacity_window_ah),
            "odometer_km": float(checkpoint.odometer_km),
            "ambient_temp_capacity_test_c": float(checkpoint.ambient_temp_c),
            "exposure_start": first_trip_start,
            "observation_days": observation_days,
            "trip_count": int(len(exposure)),
            "past_only_exposure": True,
            "analysis_role": "secondary_cross_sectional_current_soh",
            "pack_temperature_observed": False,
        }
        for column in additive:
            record[f"sum_{column}"] = float(exposure[column].sum()) if len(exposure) else 0.0
        weights = exposure["duration_observed_h"] if len(exposure) else pd.Series(dtype=float)
        for column in weighted:
            record[f"mean_{column}"] = (
                _safe_weighted_mean(exposure[column], weights) if len(exposure) else np.nan
            )
        start_odometer = (
            float(exposure["odometer_start_km"].min()) if len(exposure) else np.nan
        )
        observed_odometer_span = checkpoint.odometer_km - start_odometer
        record["field_distance_coverage"] = (
            record["sum_distance_km"] / observed_odometer_span
            if observed_odometer_span > 0 else np.nan
        )
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["vehicle_id", "checkpoint_start"], kind="mergesort"
    ).reset_index(drop=True)


def build_loov_folds(intervals: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic leave-one-vehicle-out lineage for interval rows."""

    vehicles = sorted(intervals.loc[intervals["trip_count"].gt(0), "vehicle_id"].unique())
    records: list[dict[str, Any]] = []
    for fold, held_out in enumerate(vehicles):
        for interval_id, vehicle_id in intervals[["interval_id", "vehicle_id"]].itertuples(
            index=False, name=None
        ):
            records.append({
                "outer_fold": fold,
                "held_out_vehicle": held_out,
                "interval_id": interval_id,
                "vehicle_id": vehicle_id,
                "split": "test" if vehicle_id == held_out else "train",
            })
    return pd.DataFrame(records)


def rwth_physics_parameters() -> dict[str, Any]:
    """Return constants and label semantics for manifests/workbooks."""

    return {
        "smart": asdict(SMART_SPEC),
        "imiev_ion": asdict(IMIEV_SPEC),
        "source_current_convention": "negative_discharge_positive_regen",
        "canonical_current_convention": "positive_discharge_negative_regen",
        "capacity_label": "1C discharge capacity in a fixed pack-voltage window",
        "thermal_time_constant_s": 1800.0,
        "ambient_arrhenius_activation_energy_j_mol": 35_000.0,
        "pack_temperature_observed": False,
        "thermal_proxy_equation": (
            "theta[k]=theta[k-1]*exp(-dt/tau)+C_rate[k]^2*(1-exp(-dt/tau))"
        ),
        "degradation_pathway": (
            "driver_kinematics->battery_current_power->electrical_and_latent_thermal_stress"
            "->cumulative_dose->future_capacity_loss"
        ),
    }


__all__ = [
    "IMIEV_SPEC", "RWTH_PUBLISHED_WINDOW_CAPACITY_AH", "SMART_SPEC",
    "RWTHVehicleSpec", "build_capacity_checkpoints", "build_checkpoint_exposures",
    "build_driver_soh_intervals", "build_loov_folds", "build_trip_features",
    "capacity_checkpoints_from_frame", "matlab_datenum_to_datetime",
    "rwth_physics_parameters", "rwth_vehicle_spec", "trip_physics_features",
]
