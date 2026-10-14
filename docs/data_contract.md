# Data contract

## Required Nissan source columns

`Timestamp`, `Speed`, `Odometer`, `RPM`, `SOC`, `SOH`, `Torque`,
`Motor_Temp`, `Aux_Pwr_100W`, `AC_Pwr_250W`, `EstPwr_AC_50W`,
`HV_Bat_Voltage`, `Brake`, `Acc_Pedal`, `ECO`, `ePedal`,
`HV_Bat_Current_2`, `Ambient_C_Temp`, `Pack_1_C_Temp`,
`Pack_2_C_Temp`, `Pack_4_C_Temp`, `Driver_ID`, and `Trip`.

The four tire-pressure fields and source-derived audit fields are accepted but are
not required model inputs.

## Canonical sequence schema

Every dataset adapter should emit, when available:

- identifiers: `dataset_id`, `vehicle_id`, `trip_id`, `timestamp_utc`;
- common core: `speed_mps`, `accel_mps2`, `jerk_mps3`, `voltage_v`,
  `current_discharge_a`, `soc_pct`, `pack_temp_avg_c`;
- optional driver fields: `torque_nm`, `accel_pedal_norm`, `brake_active`,
  `rpm_abs`, `eco`, `epedal`;
- optional battery/context fields: `pack_temp_spread_c`, `ambient_temp_c`,
  `hvac_power_w`;
- model controls: `delta_t_s`, `validity_mask`, and per-channel missing masks;
- targets/controls: `soh_pct`, `odometer_km`, `vehicle_age_days` when known.

Missing optional fields are represented by zeros after fold-fitted scaling and an
explicit missing-channel mask. Low-cadence data retain their native cadence and
use a separate patch adapter.

## Output tables

- `window_features`: one row per valid 256-second segment.
- `trip_features`: one row per source trip.
- `health_windows`: one row per current/future health pair.
- `bsi_results`: D-BSI components, weighted contributions, dose, and cumulative
  dose.
- `feature_selection`: fold-safe selection evidence and rejection reason.
- `soh_predictions`: target, prediction, interval, residual, fold, model, and
  split lineage.

