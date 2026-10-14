"""Transparent six-component Driving Battery Stress Index (D-BSI)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DBSIConfig:
    """Frozen v1 thresholds and weights for the explainable driving-stress index.

    Component scores are fractions in [0, 1].  ``bsi_intensity`` and individual
    contribution fields are points on a 0--100 scale.
    """

    nominal_capacity_ah: float = 111.11
    discharge_c_lower: float = 0.5
    discharge_c_upper: float = 3.0
    regen_c_lower: float = 0.3
    regen_c_upper: float = 1.5
    hot_temp_lower_c: float = 30.0
    hot_temp_upper_c: float = 45.0
    temp_spread_lower_c: float = 3.0
    temp_spread_upper_c: float = 10.0
    high_soc_lower_pct: float = 80.0
    high_soc_upper_pct: float = 100.0
    low_soc_upper_pct: float = 20.0
    low_soc_lower_pct: float = 0.0
    dod_lower_pct: float = 20.0
    dod_upper_pct: float = 80.0
    slew_lower_c_per_s: float = 0.1
    slew_upper_c_per_s: float = 1.0
    cold_regen_start_c: float = 10.0
    cold_regen_full_c: float = 0.0
    # Explicit editable v1 defaults.  Fleet-calibrated limits can replace these
    # in sensitivity analyses without changing the transparent primary index.
    efc_rate_lower_per_h: float = 0.25
    efc_rate_upper_per_h: float = 1.0
    weight_rate: float = 0.25
    weight_thermal: float = 0.25
    weight_soc: float = 0.15
    weight_cycle: float = 0.15
    weight_transient: float = 0.10
    weight_interaction: float = 0.10
    version: str = "D-BSI-v1.0"

    def __post_init__(self) -> None:
        total = (
            self.weight_rate + self.weight_thermal + self.weight_soc
            + self.weight_cycle + self.weight_transient + self.weight_interaction
        )
        if not np.isclose(total, 1.0):
            raise ValueError(f"D-BSI weights must sum to 1, got {total}")
        if self.nominal_capacity_ah <= 0:
            raise ValueError("nominal_capacity_ah must be positive")


def _coerce_config(config: DBSIConfig | Mapping[str, Any] | None) -> DBSIConfig:
    if config is None:
        return DBSIConfig()
    if isinstance(config, DBSIConfig):
        return config
    config_dict = dict(config)
    if "dbsi" in config_dict and isinstance(config_dict["dbsi"], Mapping):
        section = dict(config_dict["dbsi"])
        data_section = dict(config_dict.get("data", {})) if isinstance(config_dict.get("data"), Mapping) else {}
        flat: dict[str, Any] = {}
        pairs = {
            "discharge_c_rate": ("discharge_c_lower", "discharge_c_upper"),
            "regen_c_rate": ("regen_c_lower", "regen_c_upper"),
            "high_temperature_c": ("hot_temp_lower_c", "hot_temp_upper_c"),
            "temperature_spread_c": ("temp_spread_lower_c", "temp_spread_upper_c"),
            "high_soc_pct": ("high_soc_lower_pct", "high_soc_upper_pct"),
            # YAML expresses low-SOC limits in descending activation order.
            "low_soc_pct": ("low_soc_upper_pct", "low_soc_lower_pct"),
            "dod_pct": ("dod_lower_pct", "dod_upper_pct"),
            "c_rate_slew": ("slew_lower_c_per_s", "slew_upper_c_per_s"),
        }
        for source, targets in pairs.items():
            values = section.get(source)
            if isinstance(values, (list, tuple)) and len(values) == 2:
                flat[targets[0]], flat[targets[1]] = values
        if "cold_regen_below_c" in section:
            flat["cold_regen_start_c"] = section["cold_regen_below_c"]
        if "weights" in section and isinstance(section["weights"], Mapping):
            for name, value in section["weights"].items():
                flat[f"weight_{name}"] = value
        if "nominal_capacity_ah" in data_section:
            flat["nominal_capacity_ah"] = data_section["nominal_capacity_ah"]
        config_dict = flat
    allowed = set(DBSIConfig.__dataclass_fields__)
    unknown = set(config_dict) - allowed
    if unknown:
        raise ValueError(f"Unknown D-BSI configuration keys: {sorted(unknown)}")
    return DBSIConfig(**config_dict)


def high_severity(x: Any, lower: float, upper: float) -> Any:
    """Squared clipped ramp for stress that increases with a quantity."""

    if not upper > lower:
        raise ValueError("upper must exceed lower")
    result = np.square(np.clip((x - lower) / (upper - lower), 0.0, 1.0))
    return result


def low_severity(x: Any, upper: float, lower: float) -> Any:
    """Squared clipped ramp for stress that increases as a quantity falls."""

    if not upper > lower:
        raise ValueError("upper must exceed lower")
    return np.square(np.clip((upper - x) / (upper - lower), 0.0, 1.0))


def _series(frame: pd.DataFrame, name: str, fallback: Any = np.nan) -> pd.Series:
    if name in frame:
        return pd.to_numeric(frame[name], errors="coerce")
    if np.isscalar(fallback):
        return pd.Series(fallback, index=frame.index, dtype="float64")
    return pd.Series(fallback, index=frame.index, dtype="float64")


def _time_weighted_mean(values: Any, dt: pd.Series) -> float:
    values = pd.Series(values, index=dt.index, dtype="float64")
    valid = values.notna() & dt.notna() & dt.gt(0)
    denom = float(dt.loc[valid].sum())
    if denom <= 0:
        # A one-row aggregate or synthetic example can still have a meaningful
        # severity.  Use an unweighted mean rather than silently returning zero.
        return float(values.mean()) if values.notna().any() else 0.0
    return float(np.average(values.loc[valid], weights=dt.loc[valid]))


def dbsi_row_severities(
    telemetry: pd.DataFrame,
    config: DBSIConfig | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Calculate all instantaneous severity terms used by D-BSI."""

    cfg = _coerce_config(config)
    current = _series(telemetry, "current_std_a", -_series(telemetry, "HV_Bat_Current_2"))
    c_dis = _series(telemetry, "c_dis", current.clip(lower=0) / cfg.nominal_capacity_ah)
    c_regen = _series(telemetry, "c_regen", (-current).clip(lower=0) / cfg.nominal_capacity_ah)
    temp = _series(telemetry, "pack_temp_avg_clean_c", _series(telemetry, "Pack_Avg_C_Temp"))
    spread = _series(telemetry, "pack_temp_spread_c")
    soc = _series(telemetry, "SOC", _series(telemetry, "soc_pct"))
    slew = _series(telemetry, "c_rate_slew")

    out = pd.DataFrame(index=telemetry.index)
    out["severity_discharge_rate"] = high_severity(c_dis, cfg.discharge_c_lower, cfg.discharge_c_upper)
    out["severity_regen_rate"] = high_severity(c_regen, cfg.regen_c_lower, cfg.regen_c_upper)
    out["severity_hot_temperature"] = high_severity(temp, cfg.hot_temp_lower_c, cfg.hot_temp_upper_c)
    out["severity_cold_temperature"] = low_severity(temp, cfg.cold_regen_start_c, cfg.cold_regen_full_c)
    # Cold temperature contributes only while meaningful regenerative current is
    # present; at rest or discharge it is not a cold-regen stress term.
    out["severity_cold_regen"] = out["severity_cold_temperature"] * np.clip(
        c_regen / cfg.regen_c_lower, 0.0, 1.0
    )
    out["severity_temperature_spread"] = high_severity(
        spread, cfg.temp_spread_lower_c, cfg.temp_spread_upper_c
    )
    out["severity_high_soc"] = high_severity(soc, cfg.high_soc_lower_pct, cfg.high_soc_upper_pct)
    out["severity_low_soc"] = low_severity(soc, cfg.low_soc_upper_pct, cfg.low_soc_lower_pct)
    out["severity_transient"] = high_severity(slew, cfg.slew_lower_c_per_s, cfg.slew_upper_c_per_s)
    out["severity_hot_discharge_interaction"] = (
        out["severity_discharge_rate"] * out["severity_hot_temperature"]
    )
    out["severity_regen_boundary_interaction"] = out["severity_regen_rate"] * np.maximum(
        out["severity_cold_temperature"], out["severity_high_soc"]
    )
    return out.clip(lower=0.0, upper=1.0)


def compute_dbsi(
    telemetry: pd.DataFrame,
    config: DBSIConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute six auditable D-BSI components and the accumulated stress dose."""

    cfg = _coerce_config(config)
    if telemetry.empty:
        return _empty_dbsi_result(cfg)
    dt = _series(telemetry, "dt_s", _series(telemetry, "dt_1hz_s", 1.0)).clip(lower=0)
    severity = dbsi_row_severities(telemetry, cfg)

    discharge = _time_weighted_mean(severity["severity_discharge_rate"], dt)
    regen = _time_weighted_mean(severity["severity_regen_rate"], dt)
    hot = _time_weighted_mean(severity["severity_hot_temperature"], dt)
    cold_regen = _time_weighted_mean(severity["severity_cold_regen"], dt)
    spread = _time_weighted_mean(severity["severity_temperature_spread"], dt)
    high_soc = _time_weighted_mean(severity["severity_high_soc"], dt)
    low_soc = _time_weighted_mean(severity["severity_low_soc"], dt)
    transient = _time_weighted_mean(severity["severity_transient"], dt)
    hot_discharge = _time_weighted_mean(severity["severity_hot_discharge_interaction"], dt)
    boundary_regen = _time_weighted_mean(severity["severity_regen_boundary_interaction"], dt)

    soc = _series(telemetry, "SOC", _series(telemetry, "soc_pct"))
    if "dod_pct" in telemetry and telemetry["dod_pct"].notna().any():
        dod_pct = float(pd.to_numeric(telemetry["dod_pct"], errors="coerce").median())
    else:
        valid_soc = soc.dropna()
        dod_pct = float(valid_soc.max() - valid_soc.min()) if len(valid_soc) else 0.0

    duration_s = float(dt.sum())
    if duration_s <= 0 and "duration_s" in telemetry:
        duration_s = float(pd.to_numeric(telemetry["duration_s"], errors="coerce").median())
    ah = float(_series(telemetry, "ah_throughput_step", 0.0).sum(min_count=1))
    if (not np.isfinite(ah) or ah == 0.0) and "ah_throughput" in telemetry:
        ah = float(pd.to_numeric(telemetry["ah_throughput"], errors="coerce").median())
    ah = 0.0 if not np.isfinite(ah) else ah
    efc = ah / (2.0 * cfg.nominal_capacity_ah)
    if "efc" in telemetry and telemetry["efc"].notna().any():
        efc = float(pd.to_numeric(telemetry["efc"], errors="coerce").median())
    efc_rate = efc / (duration_s / 3600.0) if duration_s > 0 else 0.0

    s_rate = 0.75 * discharge + 0.25 * regen
    s_thermal = 0.70 * hot + 0.20 * cold_regen + 0.10 * spread
    s_soc = 0.70 * high_soc + 0.30 * low_soc
    s_cycle = 0.50 * high_severity(dod_pct, cfg.dod_lower_pct, cfg.dod_upper_pct) + 0.50 * high_severity(
        efc_rate, cfg.efc_rate_lower_per_h, cfg.efc_rate_upper_per_h
    )
    s_transient = transient
    s_interaction = 0.50 * hot_discharge + 0.50 * boundary_regen

    components = {
        "bsi_rate": float(np.clip(s_rate, 0, 1)),
        "bsi_thermal": float(np.clip(s_thermal, 0, 1)),
        "bsi_soc": float(np.clip(s_soc, 0, 1)),
        "bsi_cycle": float(np.clip(s_cycle, 0, 1)),
        "bsi_transient": float(np.clip(s_transient, 0, 1)),
        "bsi_interaction": float(np.clip(s_interaction, 0, 1)),
    }
    weights = {
        "rate": cfg.weight_rate, "thermal": cfg.weight_thermal,
        "soc": cfg.weight_soc, "cycle": cfg.weight_cycle,
        "transient": cfg.weight_transient, "interaction": cfg.weight_interaction,
    }
    contributions = {
        f"bsi_contribution_{name}": 100.0 * weights[name] * components[f"bsi_{name}"]
        for name in weights
    }
    intensity = float(sum(contributions.values()))
    if intensity < 25:
        band = "low"
    elif intensity < 50:
        band = "moderate"
    elif intensity < 75:
        band = "high"
    else:
        band = "severe"
    return {
        **components,
        **contributions,
        "bsi_intensity": intensity,
        "stress_dose": intensity / 100.0 * duration_s / 3600.0,
        "bsi_band": band,
        "bsi_version": cfg.version,
        "duration_s": duration_s,
        "dod_pct_bsi": dod_pct,
        "efc_bsi": efc,
        "efc_rate_per_h_bsi": efc_rate,
    }


def _empty_dbsi_result(cfg: DBSIConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("rate", "thermal", "soc", "cycle", "transient", "interaction"):
        result[f"bsi_{name}"] = 0.0
        result[f"bsi_contribution_{name}"] = 0.0
    result.update({
        "bsi_intensity": 0.0, "stress_dose": 0.0, "bsi_band": "low",
        "bsi_version": cfg.version, "duration_s": 0.0, "dod_pct_bsi": 0.0,
        "efc_bsi": 0.0, "efc_rate_per_h_bsi": 0.0,
    })
    return result


def append_dbsi_by_group(
    frame: pd.DataFrame,
    *,
    group_col: str = "trip_key",
    config: DBSIConfig | Mapping[str, Any] | None = None,
    preaggregated: bool = False,
) -> pd.DataFrame:
    """Return one D-BSI result per group, or enrich a preaggregated table.

    ``preaggregated=True`` is intended for already summarized trip tables.  If
    D-BSI fields are already present they are preserved verbatim.
    """

    cfg = _coerce_config(config)
    if frame.empty:
        return pd.DataFrame()
    if group_col not in frame:
        raise ValueError(f"Missing group column {group_col!r}")
    if preaggregated and "bsi_intensity" in frame:
        out = frame.copy()
    else:
        records: list[dict[str, Any]] = []
        for key, group in frame.groupby(group_col, sort=False, dropna=False):
            metadata = {group_col: key}
            for col in ("Trip", "Driver_ID", "vehicle_id", "dataset_id"):
                if col in group:
                    metadata[col] = group[col].iloc[0]
            records.append({**metadata, **compute_dbsi(group, cfg)})
        out = pd.DataFrame(records)
    if "stress_dose" in out:
        sort_cols = [c for c in ("trip_start_utc", "window_start_utc", group_col) if c in out]
        ordered = out.sort_values(sort_cols, kind="mergesort") if sort_cols else out
        vehicle_cols = [c for c in ("dataset_id", "vehicle_id") if c in ordered]
        if vehicle_cols:
            ordered["cumulative_stress_dose"] = ordered.groupby(
                vehicle_cols, sort=False, dropna=False
            )["stress_dose"].transform(lambda values: values.fillna(0).cumsum())
        else:
            ordered["cumulative_stress_dose"] = ordered["stress_dose"].fillna(0).cumsum()
        out = ordered.sort_index() if sort_cols else ordered
    return out


def dbsi_parameter_frame(config: DBSIConfig | Mapping[str, Any] | None = None) -> pd.DataFrame:
    cfg = _coerce_config(config)
    return pd.DataFrame(
        [{"Parameter": key, "Value": value, "Version": cfg.version} for key, value in asdict(cfg).items()]
    )


__all__ = [
    "DBSIConfig", "append_dbsi_by_group", "compute_dbsi",
    "dbsi_parameter_frame", "dbsi_row_severities", "high_severity", "low_severity",
]
