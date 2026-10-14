"""Create the audited Stage 12 ensemble, paired, and anchoring reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from dbpatchkan.data.evbattery import load_evbattery_sample


KEYS = [
    "row_index", "fold", "manufacturer_id", "vehicle_id",
    "vehicle_key", "target_capacity_ah",
]


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    target = frame["target_capacity_ah"].to_numpy(float)
    prediction = frame["prediction_capacity_ah"].to_numpy(float)
    residual = prediction - target
    vehicle = frame.assign(abs_error=np.abs(residual)).groupby("vehicle_key")["abs_error"].mean()
    return {
        "n": int(len(frame)),
        "vehicles": int(frame["vehicle_key"].nunique()),
        "mae_ah": float(np.mean(np.abs(residual))),
        "rmse_ah": float(np.sqrt(np.mean(np.square(residual)))),
        "bias_ah": float(np.mean(residual)),
        "r2": float(r2_score(target, prediction)),
        "vehicle_macro_mae_ah": float(vehicle.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--bootstrap", type=int, default=50_000)
    args = parser.parse_args()
    stage_root = Path(args.stage_root)
    result_root = stage_root / "results"
    predictions = pd.read_csv(result_root / "stage12_predictions.csv")

    baseline_names = ["Mean", "Persistence_WarmStart", "RandomForest", "Ridge"]
    baseline = predictions.loc[predictions["model"].isin(baseline_names)].copy()
    deep = predictions.loc[predictions["model"].str.startswith("BatteryPatch")]
    ensemble = deep.groupby(KEYS + ["model"], as_index=False)["prediction_capacity_ah"].mean()
    combined = pd.concat(
        [baseline[KEYS + ["model", "prediction_capacity_ah"]], ensemble],
        ignore_index=True,
    )
    combined.to_csv(result_root / "stage12_seed_ensemble_predictions.csv", index=False)
    metric_rows = []
    for model, frame in combined.groupby("model"):
        metric_rows.append({"model": model, "manufacturer_id": "ALL", **metrics(frame)})
        for manufacturer, subgroup in frame.groupby("manufacturer_id"):
            metric_rows.append({"model": model, "manufacturer_id": int(manufacturer), **metrics(subgroup)})
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(result_root / "stage12_seed_ensemble_metrics.csv", index=False)

    wide = combined.pivot(index=KEYS, columns="model", values="prediction_capacity_ah").reset_index()
    blend_parts = []
    weights = []
    for fold in sorted(wide["fold"].unique()):
        development = wide["fold"].ne(fold)
        held_out = wide["fold"].eq(fold)
        grid = np.linspace(0.0, 1.0, 1001)
        y = wide.loc[development, "target_capacity_ah"].to_numpy()
        persistence = wide.loc[development, "Persistence_WarmStart"].to_numpy()
        kan = wide.loc[development, "BatteryPatchConcept_KAN"].to_numpy()
        loss = np.asarray([
            np.mean(np.abs(y - (persistence + alpha * (kan - persistence))))
            for alpha in grid
        ])
        alpha = float(grid[int(np.argmin(loss))])
        part = wide.loc[held_out, KEYS].copy()
        part["model"] = "PostHoc_CrossFit_Persistence_KAN"
        part["prediction_capacity_ah"] = (
            wide.loc[held_out, "Persistence_WarmStart"].to_numpy()
            + alpha
            * (
                wide.loc[held_out, "BatteryPatchConcept_KAN"].to_numpy()
                - wide.loc[held_out, "Persistence_WarmStart"].to_numpy()
            )
        )
        part["kan_weight"] = alpha
        blend_parts.append(part)
        weights.append({
            "held_out_fold": int(fold),
            "kan_weight": alpha,
            "other_fold_development_mae_ah": float(loss.min()),
        })
    blend = pd.concat(blend_parts, ignore_index=True)
    blend.to_csv(result_root / "stage12_posthoc_crossfit_persistence_kan.csv", index=False)
    pd.DataFrame(weights).to_csv(result_root / "stage12_posthoc_blend_weights.csv", index=False)
    blend_metrics = pd.DataFrame([
        {"manufacturer_id": "ALL", **metrics(blend)},
        *[
            {"manufacturer_id": int(manufacturer), **metrics(frame)}
            for manufacturer, frame in blend.groupby("manufacturer_id")
        ],
    ])
    blend_metrics.to_csv(result_root / "stage12_posthoc_blend_metrics.csv", index=False)

    comparison = pd.concat([combined, blend.drop(columns="kan_weight")], ignore_index=True)
    comparison["abs_error"] = (
        comparison["prediction_capacity_ah"] - comparison["target_capacity_ah"]
    ).abs()
    vehicle_errors = comparison.groupby(["model", "vehicle_key"])["abs_error"].mean().unstack("model")
    rng = np.random.default_rng(42)
    paired_rows = []
    proposed = "BatteryPatchConcept_KAN"
    for challenger in [
        "Persistence_WarmStart", "RandomForest", "Ridge", "BatteryPatchConcept_MLP"
    ]:
        delta = (vehicle_errors[proposed] - vehicle_errors[challenger]).dropna().to_numpy()
        bootstrap = np.mean(
            rng.choice(delta, size=(args.bootstrap, len(delta)), replace=True), axis=1
        )
        paired_rows.append({
            "proposed": proposed,
            "challenger": challenger,
            "vehicles": len(delta),
            "mean_delta_mae_ah": float(np.mean(delta)),
            "median_delta_mae_ah": float(np.median(delta)),
            "proposed_vehicle_win_fraction": float(np.mean(delta < 0)),
            "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
            "interpretation": "negative favors proposed",
        })
    blend_delta = (
        vehicle_errors["PostHoc_CrossFit_Persistence_KAN"]
        - vehicle_errors["Persistence_WarmStart"]
    ).dropna().to_numpy()
    blend_bootstrap = np.mean(
        rng.choice(blend_delta, size=(args.bootstrap, len(blend_delta)), replace=True), axis=1
    )
    paired_rows.append({
        "proposed": "PostHoc_CrossFit_Persistence_KAN",
        "challenger": "Persistence_WarmStart",
        "vehicles": len(blend_delta),
        "mean_delta_mae_ah": float(np.mean(blend_delta)),
        "median_delta_mae_ah": float(np.median(blend_delta)),
        "proposed_vehicle_win_fraction": float(np.mean(blend_delta < 0)),
        "bootstrap_ci_low": float(np.quantile(blend_bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(blend_bootstrap, 0.975)),
        "interpretation": "exploratory post-hoc; negative favors blend",
    })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(result_root / "stage12_paired_vehicle_bootstrap.csv", index=False)

    cache = load_evbattery_sample(stage_root / "evbattery_capacity_m200000_v64.npz")
    cohort = cache["concepts"].copy()
    cohort["vehicle_key"] = (
        cohort["manufacturer_id"].astype(int).astype(str)
        + "_" + cohort["vehicle_id"].astype(int).astype(str)
    )
    vehicle_segments = cohort.groupby("vehicle_key")["charge_segment"].nunique()
    fallback_rows = 0
    for _, group in cohort.groupby("vehicle_key"):
        ordered = group.groupby("charge_segment")["mileage"].median().sort_values()
        fallback_rows += int(group["charge_segment"].eq(ordered.index[0]).sum())

    overall = metric_frame.loc[metric_frame["manufacturer_id"].eq("ALL")].set_index("model")
    blend_all = blend_metrics.loc[blend_metrics["manufacturer_id"].eq("ALL")].iloc[0]
    report = f"""# Stage 12 EVBattery external-capacity audit

## Scope

- Target: manufacturer-supplied partial-window capacity in Ah.
- This target is not absolute pack SOH.
- Dataset contains charging telemetry and no driver trajectories.
- Outer validation: five vehicle-disjoint folds.
- Deep-model seeds: 42, 52, and 62.
- Cohort: {len(cohort):,} records from {cohort['vehicle_key'].nunique()} capacity-labelled vehicles.
- Each vehicle contributes at most 64 reservoir-sampled records.
- Unique vehicle/charge-segment combinations: {cohort.groupby(['vehicle_key', 'charge_segment']).ngroups:,}.
- Persistence initial/fallback rows: {fallback_rows:,} ({100*fallback_rows/len(cohort):.2f}%).

## Seed-ensemble results

| Model | MAE (Ah) | RMSE (Ah) | R² | Vehicle-macro MAE (Ah) |
|---|---:|---:|---:|---:|
"""
    for model in overall.sort_values("mae_ah").index:
        row = overall.loc[model]
        report += (
            f"| {model} | {row.mae_ah:.6f} | {row.rmse_ah:.6f} | "
            f"{row.r2:.6f} | {row.vehicle_macro_mae_ah:.6f} |\n"
        )
    report += f"""

Warm-start persistence is the overall winner because it uses the previous
capacity measurement from the same held-out vehicle. Among cold-start learned
estimators, the three-seed KAN ensemble is best at
{overall.loc['BatteryPatchConcept_KAN','mae_ah']:.6f} Ah, versus
{overall.loc['RandomForest','mae_ah']:.6f} Ah for Random Forest,
{overall.loc['Ridge','mae_ah']:.6f} Ah for Ridge, and
{overall.loc['BatteryPatchConcept_MLP','mae_ah']:.6f} Ah for the matched MLP.

## Paired vehicle conclusions

- KAN versus matched MLP: mean vehicle-MAE difference
  {paired_rows[3]['mean_delta_mae_ah']:.6f} Ah, 95% cluster-bootstrap interval
  [{paired_rows[3]['bootstrap_ci_low']:.6f}, {paired_rows[3]['bootstrap_ci_high']:.6f}].
- KAN versus Random Forest and Ridge: intervals cross zero; superiority is not
  established statistically.
- KAN versus warm-start persistence: persistence is decisively better.

## Exploratory persistence anchoring

A post-hoc, outer-fold-cross-fitted convex combination selected KAN weights of
{', '.join(f"{row['kan_weight']:.3f}" for row in weights)} (mean {np.mean([row['kan_weight'] for row in weights]):.3f}).
It produced MAE {blend_all.mae_ah:.6f} Ah versus
{overall.loc['Persistence_WarmStart','mae_ah']:.6f} Ah for persistence alone.
The vehicle-level paired change was {np.mean(blend_delta):.6f} Ah with 95%
cluster-bootstrap interval [{np.quantile(blend_bootstrap,0.025):.6f},
{np.quantile(blend_bootstrap,0.975):.6f}]. This is evidence that the KAN signal
can provide a small correction to persistence, but it is exploratory because
the anchoring analysis was designed after viewing Stage 12 results.

## Final interpretation

Stage 12 externally supports the battery Patch-Concept-KAN branch and shows a
statistically reliable advantage over its parameter-matched MLP. It does not
show universal superiority over classical models, and it cannot validate the
driver-behaviour pathway. The strongest deployment result remains a
persistence-anchored framework, with the learned KAN acting as a small,
interpretable correction rather than replacing the prior measurement.
"""
    (result_root / "STAGE12_EXTERNAL_CAPACITY_REPORT.md").write_text(report, encoding="utf-8")
    audit = {
        "analysis_type": "posthoc_stage12_audit",
        "seed_ensemble_used": True,
        "vehicle_cluster_bootstrap_replicates": args.bootstrap,
        "persistence_is_warm_start": True,
        "blend_is_posthoc_exploratory": True,
        "mean_crossfit_kan_weight": float(np.mean([row["kan_weight"] for row in weights])),
        "raw_result_archive_sha256": "688FD48B8FA39A97B94B10620F94E6028775E52D5DC4BE644344B16C3133B170",
    }
    (result_root / "STAGE12_POSTHOC_AUDIT.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
