# DB-PatchKAN research protocol

## Scientific objectives

1. Quantify associations between driving behaviour and immediate electrical,
   thermal, cycling, SOC, and transient battery stress.
2. Discover stable driving archetypes without using driver identity, then train
   classifiers for Eco/Balanced/Aggressive behaviour and a secondary identity
   diagnostic.
3. Calculate a transparent six-component Driving Battery Stress Index (D-BSI)
   and cumulative stress dose.
4. Estimate current and future SOH using leakage-safe operational data, named
   concepts, and a compact spline KAN.
5. Validate chronologically within Nissan and, when provided, by held-out vehicle
   on external capacity-labelled fleets.

## Unit of analysis

The raw telemetry row is not an independent SOH observation. The project uses:

- native-rate rows for physically correct time integration;
- 1 Hz segments for deep representation learning;
- one record per trip for behaviour and current-SOH analyses;
- one record per health interval/checkpoint for degradation forecasting.

Every overlapping segment from a trip stays in the same split. Feature selection,
scaling, imputation, PCA, clustering, and calibration are fitted on training data
only.

## Canonical sign and audit policy

The Nissan source uses negative high-voltage current during traction. The cleaned
field `current_discharge_a` reverses that sign, so positive means discharge and
negative means regenerative charging. Recalculated battery power is:

`battery_power_kw = voltage_v * current_discharge_a / 1000`

Positive recalculated power is traction/discharge. Source `Instant_kW`,
`Horsepower`, and `Pack_Avg_C_Temp` are retained only for audit comparisons.

## Leakage boundary

The following never enter core SOH predictors: row index, raw timestamp, Trip,
Driver_ID, current SOH, `kWh_Remaining`, source horsepower, source instantaneous
power, and source pack-average temperature. Age and odometer are allowed only as
explicit aging controls and standalone baselines. The last known SOH is allowed
only for future-SOH forecasting.

## D-BSI v1

Each component is bounded to [0, 1] and reported separately:

- electrical rate (25%);
- thermal (25%);
- SOC exposure (15%);
- throughput/cycling (15%);
- transients (10%);
- interaction (10%).

`D-BSI = 100 * weighted component sum`

`StressDose = (D-BSI / 100) * duration_hours`

D-BSI is a driving-stress index. It is not a complete calendar-aging or charging
stress model, and causal language is not justified by this observational dataset.

## Model hierarchy

The recommended-deployment model is selected empirically:

1. age/mileage, Elastic Net, random forest, XGBoost, and EBM-style baselines;
2. patch Transformer with an MLP concept head;
3. DB-PatchKAN with driver/battery streams, bidirectional cross-attention, named
   concept supervision, and a compact float32 spline KAN.

The deep model succeeds only if paired held-out results improve over both the best
operational tabular model and the parameter-matched concept MLP. Otherwise, the
simpler model is retained.

## Validation gates

- Nissan: four expanding chronological folds on the first 80% of trips, followed
  by an untouched final 20% test set.
- Forecasting: purge 30 days or 500 km around split boundaries as applicable.
- External fleets: leave-one-vehicle-out outer validation and chronological inner
  tuning.
- Reporting: fold/vehicle/seed means with paired bootstrap 95% confidence
  intervals, bias, tail error, interval coverage, efficiency, and class-balanced
  metrics.

## Current evidence boundary

The included Nissan file represents one vehicle. It supports within-vehicle
behaviour/stress analysis and proof-of-concept SOH modelling. Cross-vehicle
architecture claims remain pending until external capacity-labelled data are
placed behind the implemented adapter interface.

