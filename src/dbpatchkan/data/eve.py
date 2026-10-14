"""Harmonise the public EVE Pilot Renault-Zoe telemetry for Stage 5B.

The source consists of one irregularly sampled file per CAN signal.  The
harmoniser uses speed as the observation backbone, resamples each signal before
joining, and exports trip-level physical concepts plus monthly BMS-SOH
checkpoints.  Raw BMS SOH is noisy and sometimes implausible; only values in
``[50, 105]`` are retained and a past-only three-month rolling median is stored
as a secondary trend target.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


EVE_DOI = "https://doi.org/10.57745/5O6QIH"
EVE_LICENSE = "Etalab Open License 2.0 (compatible with CC BY 2.0)"
EVE_DOWNLOAD_ROOT = "https://entrepot.recherche.data.gouv.fr/api/access/datafile"
EVE_NOMINAL_CAPACITY_AH = 65.0
EVE_FILES: dict[str, int] = {
    "speed": 627272,
    "odometer": 627237,
    "current": 627211,
    "voltage": 627172,
    "soc": 627209,
    "soh": 627254,
    "ambient": 627269,
    "module_temp_01": 627213,
    "module_temp_02": 627243,
    "module_temp_03": 627153,
    "module_temp_04": 627191,
    "module_temp_05": 627274,
    "module_temp_06": 627271,
    "module_temp_07": 627170,
    "module_temp_08": 627179,
    "module_temp_09": 627194,
    "module_temp_10": 627156,
    "module_temp_11": 627185,
    "module_temp_12": 627166,
}


def _update(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


def download_eve_files(
    destination: Path,
    *,
    reporter: Callable[[str], None] = _update,
    retries: int = 4,
) -> dict[str, Path]:
    """Download the selected public signals with resume and progress support."""

    destination.mkdir(parents=True, exist_ok=True)
    located: dict[str, Path] = {}
    for name, file_id in EVE_FILES.items():
        path = destination / f"{name}.tab"
        located[name] = path
        if path.exists() and path.stat().st_size > 100:
            reporter(f"EVE {name}: already present ({path.stat().st_size / 2**20:.1f} MB)")
            continue
        url = f"{EVE_DOWNLOAD_ROOT}/{file_id}"
        partial = path.with_suffix(".tab.partial")
        for attempt in range(1, retries + 1):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "DBPatchKAN/0.1"})
                started = time.time()
                with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                    total = int(response.headers.get("Content-Length", 0))
                    received = 0
                    last_report = 0.0
                    while True:
                        chunk = response.read(2**20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        received += len(chunk)
                        now = time.time()
                        if now - last_report >= 15:
                            percent = 100.0 * received / total if total else math.nan
                            reporter(
                                f"EVE {name}: {received / 2**20:.1f} MB "
                                f"({percent:.1f}% if size known)"
                            )
                            last_report = now
                partial.replace(path)
                reporter(
                    f"EVE {name}: downloaded {path.stat().st_size / 2**20:.1f} MB "
                    f"in {(time.time() - started) / 60:.1f} min"
                )
                break
            except Exception as error:
                if attempt == retries:
                    raise RuntimeError(f"Failed to download EVE {name} after {retries} attempts") from error
                reporter(f"EVE {name}: attempt {attempt} failed; retrying")
                time.sleep(2**attempt)
    return located


def _read_resampled(path: Path, name: str, frequency: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", usecols=["Datetime", "Value"], low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["Datetime"], errors="coerce")
    frame[name] = pd.to_numeric(frame["Value"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", name])[["timestamp", name]]
    frame["timestamp"] = frame.timestamp.dt.floor(frequency)
    return frame.groupby("timestamp", as_index=False, sort=True)[name].median()


def _asof(base: pd.DataFrame, right: pd.DataFrame, tolerance: str) -> pd.DataFrame:
    return pd.merge_asof(
        base.sort_values("timestamp"),
        right.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )


def load_eve_telemetry(paths: dict[str, Path], *, reporter: Callable[[str], None] = _update) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a five-second harmonised table without expanding parked gaps."""

    reporter("EVE harmonisation: resampling speed backbone")
    telemetry = _read_resampled(paths["speed"], "speed_kmh", "5s")
    specifications = (
        ("current", "current_source_a", "5s", "30s"),
        ("voltage", "voltage_v", "5s", "30s"),
        ("soc", "soc_pct", "30s", "5min"),
        ("odometer", "odometer_km", "30s", "5min"),
        ("ambient", "ambient_c", "5min", "90min"),
    )
    for key, column, frequency, tolerance in specifications:
        reporter(f"EVE harmonisation: resampling {key}")
        telemetry = _asof(telemetry, _read_resampled(paths[key], column, frequency), tolerance)

    reporter("EVE harmonisation: combining 12 measured module temperatures")
    temperatures = []
    for index in range(1, 13):
        key = f"module_temp_{index:02d}"
        temperatures.append(
            _read_resampled(paths[key], f"temp_{index:02d}_c", "60s").set_index("timestamp")
        )
    temperature = pd.concat(temperatures, axis=1).sort_index()
    temperature["pack_temp_mean_c"] = temperature.mean(axis=1, skipna=True)
    temperature["pack_temp_spread_c"] = temperature.max(axis=1, skipna=True) - temperature.min(axis=1, skipna=True)
    temperature = temperature[["pack_temp_mean_c", "pack_temp_spread_c"]].reset_index()
    telemetry = _asof(telemetry, temperature, "15min")

    soh = _read_resampled(paths["soh"], "bms_soh_pct_raw", "1D")
    soh.loc[~soh.bms_soh_pct_raw.between(50.0, 105.0), "bms_soh_pct_raw"] = np.nan
    soh = soh.dropna(subset=["bms_soh_pct_raw"])
    telemetry = pd.merge_asof(
        telemetry.sort_values("timestamp"),
        soh.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("45D"),
    )
    moving = telemetry.speed_kmh.gt(10) & telemetry.current_source_a.notna()
    moving_current_median = float(telemetry.loc[moving, "current_source_a"].median())
    multiplier = -1.0 if moving_current_median < 0 else 1.0
    telemetry["current_a"] = multiplier * telemetry.current_source_a
    telemetry["power_kw"] = telemetry.current_a * telemetry.voltage_v / 1000.0
    audit = {
        "dataset": "EVE Pilot Renault Zoe",
        "doi": EVE_DOI,
        "license": EVE_LICENSE,
        "rows_harmonised": int(len(telemetry)),
        "start": str(telemetry.timestamp.min()),
        "end": str(telemetry.timestamp.max()),
        "current_source_median_while_moving_a": moving_current_median,
        "canonical_current_multiplier": multiplier,
        "canonical_current_definition": "positive discharge, negative regeneration",
        "nominal_capacity_ah": EVE_NOMINAL_CAPACITY_AH,
        "temperature": "mean/spread of 12 measured battery-module temperatures",
    }
    return telemetry, audit


def _integral(values: np.ndarray, dt_s: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(dt_s) & (dt_s > 0) & (dt_s <= 30)
    return float(np.sum(values[valid] * dt_s[valid]))


def _trip_record(group: pd.DataFrame, trip_id: int) -> dict[str, object]:
    group = group.sort_values("timestamp")
    dt = group.timestamp.diff().dt.total_seconds().fillna(0).clip(0, 30).to_numpy(float)
    speed = group.speed_kmh.to_numpy(float)
    elapsed = (group.timestamp - group.timestamp.iloc[0]).dt.total_seconds().to_numpy(float)
    # Duplicate timestamps can remain after joins.  A tiny monotone offset keeps
    # numerical gradients well-defined without pretending to add observations.
    elapsed = np.maximum.accumulate(elapsed + np.arange(len(elapsed)) * 1e-6)
    acceleration = np.gradient(speed / 3.6, elapsed) if len(group) >= 3 else np.zeros(len(group))
    jerk = np.gradient(acceleration, elapsed) if len(group) >= 3 else np.zeros(len(group))
    current = group.current_a.to_numpy(float)
    voltage = group.voltage_v.to_numpy(float)
    c_rate = current / EVE_NOMINAL_CAPACITY_AH
    abs_c = np.abs(c_rate)
    power = current * voltage / 1000.0
    temp = group.pack_temp_mean_c.to_numpy(float)
    ambient = group.ambient_c.to_numpy(float)
    soc = group.soc_pct.to_numpy(float)
    arrhenius = np.exp(np.clip(35000.0 / 8.314 * (1 / 298.15 - 1 / (temp + 273.15)), -3, 3))
    distance_integrated = _integral(np.maximum(speed, 0) / 3600.0, dt)
    distance_odometer = float(group.odometer_km.max() - group.odometer_km.min())
    distance = distance_odometer if np.isfinite(distance_odometer) and distance_odometer > 0 else distance_integrated
    duration_h = _integral(np.ones(len(group)), dt) / 3600.0
    discharge_energy = _integral(np.maximum(power, 0), dt) / 3600.0
    regen_energy = _integral(np.maximum(-power, 0), dt) / 3600.0
    ah_throughput = _integral(np.abs(current), dt) / 3600.0
    efc = ah_throughput / (2.0 * EVE_NOMINAL_CAPACITY_AH)
    c2_dose = _integral(abs_c**2, dt) / 3600.0
    electrothermal = _integral(abs_c**2 * arrhenius, dt) / 3600.0
    transient = np.abs(np.diff(c_rate, prepend=c_rate[0])) / np.maximum(dt, 1.0)
    transient_dose = _integral(transient, dt)
    valid_temp = temp[np.isfinite(temp)]
    temp_rise = float(valid_temp[-1] - valid_temp[0]) if len(valid_temp) >= 2 else math.nan
    rate_component = float(np.nanmean(np.clip((abs_c - 0.5) / 1.5, 0, 1)))
    thermal_component = float(np.nanmean(np.clip((temp - 30.0) / 15.0, 0, 1)))
    soc_component = float(np.nanmean(np.clip((soc - 80.0) / 20.0, 0, 1)))
    cycle_component = float(np.clip(efc / 0.05, 0, 1))
    transient_component = float(np.nanmean(np.clip(transient / 0.05, 0, 1)))
    interaction_component = rate_component * thermal_component
    dbsi = 100.0 * (
        0.25 * rate_component
        + 0.25 * thermal_component
        + 0.15 * soc_component
        + 0.15 * cycle_component
        + 0.10 * transient_component
        + 0.10 * interaction_component
    )
    return {
        "dataset_id": "eve_pilot_renault_zoe",
        "vehicle_id": "Renault-Zoe-Q90-1",
        "vehicle_family": "Renault_Zoe_Q90",
        "trip_id": int(trip_id),
        "trip_key": f"eve_pilot_renault_zoe:Renault-Zoe-Q90-1:{trip_id}",
        "trip_start": group.timestamp.iloc[0],
        "trip_end": group.timestamp.iloc[-1],
        "sample_count": int(len(group)),
        "duration_h": duration_h,
        "distance_km": float(distance),
        "speed_mean_kmh": float(np.nanmean(speed)),
        "speed_p95_kmh": float(np.nanpercentile(speed, 95)),
        "speed_std_kmh": float(np.nanstd(speed)),
        "accel_positive_rms_mps2": float(np.sqrt(np.mean(np.maximum(acceleration, 0) ** 2))),
        "decel_rms_mps2": float(np.sqrt(np.mean(np.minimum(acceleration, 0) ** 2))),
        "jerk_rms_mps3": float(np.sqrt(np.mean(jerk**2))),
        "harsh_accel_fraction": float(np.mean(acceleration > 1.5)),
        "harsh_decel_fraction": float(np.mean(acceleration < -1.5)),
        "stop_fraction": float(np.mean(speed < 1.0)),
        "ambient_mean_c": float(np.nanmean(ambient)),
        "soc_mean_pct": float(np.nanmean(soc)),
        "soc_high_fraction": float(np.nanmean(soc >= 80.0)),
        "c_rate_abs_mean": float(np.nanmean(abs_c)),
        "c_rate_abs_p95": float(np.nanpercentile(abs_c, 95)),
        "c_rate_abs_peak": float(np.nanmax(abs_c)),
        "high_c_rate_fraction": float(np.nanmean(abs_c >= 1.0)),
        "ah_throughput": ah_throughput,
        "efc": efc,
        "energy_discharge_kwh": discharge_energy,
        "energy_regen_kwh": regen_energy,
        "energy_kwh_per_100km": 100.0 * discharge_energy / max(float(distance), 0.1),
        "pack_temp_start_c": float(valid_temp[0]) if len(valid_temp) else math.nan,
        "pack_temp_end_c": float(valid_temp[-1]) if len(valid_temp) else math.nan,
        "pack_temp_mean_c": float(np.nanmean(temp)),
        "pack_temp_max_c": float(np.nanmax(temp)),
        "pack_temp_spread_mean_c": float(np.nanmean(group.pack_temp_spread_c)),
        "pack_temp_rise_c": temp_rise,
        "c_rate_squared_dose_h": c2_dose,
        "electrothermal_dose_h": electrothermal,
        "transient_dose": transient_dose,
        "high_soc_hot_time_h": _integral(((soc >= 80) & (temp >= 30)).astype(float), dt) / 3600.0,
        "bsi_rate": 100.0 * rate_component,
        "bsi_thermal": 100.0 * thermal_component,
        "bsi_soc": 100.0 * soc_component,
        "bsi_cycle": 100.0 * cycle_component,
        "bsi_transient": 100.0 * transient_component,
        "bsi_interaction": 100.0 * interaction_component,
        "dbsi": dbsi,
        "stress_dose_h": dbsi / 100.0 * duration_h,
        "bms_soh_pct_raw": float(pd.to_numeric(group.bms_soh_pct_raw, errors="coerce").median()),
    }


def eve_trip_features(telemetry: pd.DataFrame, *, reporter: Callable[[str], None] = _update) -> pd.DataFrame:
    """Segment observed logging sessions into valid driving trips."""

    frame = telemetry.sort_values("timestamp").copy()
    frame["session"] = frame.timestamp.diff().dt.total_seconds().gt(300).cumsum()
    records = []
    candidate_count = frame.session.nunique()
    for _, group in frame.groupby("session", sort=True):
        duration = (group.timestamp.max() - group.timestamp.min()).total_seconds()
        if len(group) < 12 or duration < 60 or group.speed_kmh.max() < 5:
            continue
        record = _trip_record(group, len(records) + 1)
        if record["distance_km"] >= 0.2 and record["duration_h"] > 0:
            records.append(record)
        if len(records) and len(records) % 250 == 0:
            reporter(f"EVE trip extraction: {len(records)} valid trips")
    reporter(f"EVE trip extraction complete: {len(records)} valid of {candidate_count} sessions")
    return pd.DataFrame(records)


def eve_monthly_checkpoints(trips: pd.DataFrame, soh_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build past-only monthly BMS-SOH checkpoints and signed intervals."""

    health = pd.read_csv(soh_path, sep="\t", usecols=["Datetime", "Value"])
    health["timestamp"] = pd.to_datetime(health.Datetime, errors="coerce")
    health["bms_soh_pct"] = pd.to_numeric(health.Value, errors="coerce")
    health = health[health.bms_soh_pct.between(50, 105)].dropna(subset=["timestamp"])
    health["month"] = health.timestamp.dt.to_period("M")
    checkpoints = health.groupby("month").agg(
        checkpoint_time=("timestamp", "max"),
        bms_soh_pct=("bms_soh_pct", "median"),
        soh_observations=("bms_soh_pct", "size"),
    ).reset_index()
    checkpoints["bms_soh_trend_pct"] = checkpoints.bms_soh_pct.rolling(3, min_periods=1).median()
    checkpoints["checkpoint_id"] = "EVE:" + checkpoints.month.astype(str)
    first = checkpoints.checkpoint_time.min()
    checkpoints["calendar_days"] = (
        checkpoints.checkpoint_time - first
    ).dt.total_seconds() / 86400.0

    cumulative_sum = (
        "duration_h", "distance_km", "efc", "energy_discharge_kwh",
        "c_rate_squared_dose_h", "electrothermal_dose_h", "transient_dose",
        "high_soc_hot_time_h", "stress_dose_h",
    )
    mean_columns = (
        "speed_mean_kmh", "accel_positive_rms_mps2", "decel_rms_mps2",
        "jerk_rms_mps3", "harsh_accel_fraction", "harsh_decel_fraction",
        "c_rate_abs_p95", "high_c_rate_fraction", "pack_temp_mean_c",
        "pack_temp_max_c", "pack_temp_rise_c", "dbsi", "ambient_mean_c",
    )
    rows = []
    for checkpoint in checkpoints.itertuples(index=False):
        prior = trips[trips.trip_end.lt(checkpoint.checkpoint_time)]
        recent_start = checkpoint.checkpoint_time - pd.Timedelta(days=31)
        recent = prior[prior.trip_end.ge(recent_start)]
        row = checkpoint._asdict()
        row.update({
            "dataset_id": "eve_pilot_renault_zoe",
            "vehicle_id": "Renault-Zoe-Q90-1",
            "trip_count_cumulative": int(len(prior)),
            "trip_count_recent_31d": int(len(recent)),
        })
        for column in cumulative_sum:
            row[f"cum_{column}"] = float(pd.to_numeric(prior[column], errors="coerce").sum())
            row[f"recent_{column}"] = float(pd.to_numeric(recent[column], errors="coerce").sum())
        for column in mean_columns:
            row[f"cum_mean_{column}"] = float(pd.to_numeric(prior[column], errors="coerce").mean())
            row[f"recent_mean_{column}"] = float(pd.to_numeric(recent[column], errors="coerce").mean())
        row["odometer_proxy_km"] = row["cum_distance_km"]
        rows.append(row)
    checkpoint_table = pd.DataFrame(rows).sort_values("checkpoint_time").reset_index(drop=True)
    interval = checkpoint_table.copy()
    interval["previous_checkpoint_id"] = interval.checkpoint_id.shift()
    interval["previous_soh_pct"] = interval.bms_soh_pct.shift()
    interval["previous_soh_trend_pct"] = interval.bms_soh_trend_pct.shift()
    interval["signed_soh_loss_pp"] = interval.previous_soh_pct - interval.bms_soh_pct
    interval["signed_trend_loss_pp"] = interval.previous_soh_trend_pct - interval.bms_soh_trend_pct
    interval = interval.iloc[1:].reset_index(drop=True)
    return checkpoint_table, interval


def build_eve_stage(
    raw_dir: Path,
    output_dir: Path,
    *,
    download: bool = True,
    reporter: Callable[[str], None] = _update,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = download_eve_files(raw_dir, reporter=reporter) if download else {
        name: raw_dir / f"{name}.tab" for name in EVE_FILES
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing EVE signals: {missing}")
    telemetry, audit = load_eve_telemetry(paths, reporter=reporter)
    trips = eve_trip_features(telemetry, reporter=reporter)
    checkpoints, intervals = eve_monthly_checkpoints(trips, paths["soh"])
    trips.to_parquet(output_dir / "eve_trip_physics_features.parquet", index=False)
    trips.to_csv(output_dir / "eve_trip_physics_features.csv", index=False)
    checkpoints.to_csv(output_dir / "eve_monthly_soh_checkpoints.csv", index=False)
    intervals.to_csv(output_dir / "eve_monthly_soh_intervals.csv", index=False)
    file_manifest = []
    for name, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(2**20), b""):
                digest.update(chunk)
        file_manifest.append({
            "signal": name,
            "dataverse_file_id": EVE_FILES[name],
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    pd.DataFrame(file_manifest).to_csv(output_dir / "eve_source_manifest.csv", index=False)
    summary = {
        **audit,
        "trip_rows": int(len(trips)),
        "monthly_soh_checkpoints": int(len(checkpoints)),
        "monthly_soh_intervals": int(len(intervals)),
        "distinct_raw_bms_soh_values": int(checkpoints.bms_soh_pct.nunique()),
        "label_type": "Renault BMS-reported SOH; not pooled with capacity SOH",
        "causal_claim_authorized": False,
    }
    (output_dir / "EVE_HARMONISATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    reporter("EVE Stage 5B harmonisation complete")
    return summary


__all__ = [
    "EVE_DOI",
    "EVE_FILES",
    "EVE_LICENSE",
    "EVE_NOMINAL_CAPACITY_AH",
    "build_eve_stage",
    "download_eve_files",
    "eve_monthly_checkpoints",
    "eve_trip_features",
    "load_eve_telemetry",
]
