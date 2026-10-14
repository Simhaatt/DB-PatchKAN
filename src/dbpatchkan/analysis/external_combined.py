"""Combine RWTH Stage 5A and EVE Stage 5B without pooling label definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _metric_table(path: Path, dataset: str, task_family: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.insert(0, "dataset", dataset)
    frame.insert(1, "task_family", task_family)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rwth-dir", type=Path, required=True)
    parser.add_argument("--eve-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rwth = args.rwth_dir.resolve()
    eve = args.eve_dir.resolve()
    rwth_decision = json.loads((rwth / "RWTH_EXTERNAL_ACCEPTANCE.json").read_text())
    eve_decision = json.loads((eve / "EVE_EXTERNAL_ACCEPTANCE.json").read_text())

    tables = [
        _metric_table(rwth / "rwth_trip_response_loov_metrics.csv", "RWTH", "immediate_response"),
        _metric_table(rwth / "rwth_capacity_soh_loov_metrics.csv", "RWTH", "current_capacity_soh"),
        _metric_table(rwth / "rwth_signed_degradation_loov_metrics.csv", "RWTH", "signed_degradation"),
        _metric_table(eve / "eve_trip_response_temporal_metrics.csv", "EVE", "immediate_response"),
        _metric_table(eve / "eve_bms_soh_trend_temporal_metrics.csv", "EVE", "current_bms_soh_trend"),
        _metric_table(eve / "eve_signed_monthly_trend_loss_temporal_metrics.csv", "EVE", "signed_degradation"),
    ]
    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined.to_csv(output / "external_combined_metrics_LONG.csv", index=False)

    evidence = pd.DataFrame([
        {
            "claim": "Driver behaviour predicts immediate electrical stress on unseen vehicles",
            "dataset": "RWTH",
            "evidence": "leave-one-vehicle-out driver increment with vehicle-bootstrap CI",
            "supported": bool(rwth_decision["driver_increment_significant_tasks"]),
            "scope": "association/prediction; not causal",
        },
        {
            "claim": "Driver behaviour predicts measured battery thermal response",
            "dataset": "EVE",
            "evidence": "expanding temporal folds; measured 12-module temperature",
            "supported": eve_decision["driver_improves_measured_temp_rise_over_context_ridge"],
            "scope": "one Renault Zoe; cross-manufacturer replication",
        },
        {
            "claim": "Physics KAN improves external current capacity SOH beyond aging",
            "dataset": "RWTH",
            "evidence": "16 checkpoints, nine-vehicle LOOV",
            "supported": rwth_decision["capacity_physics_kan_beats_aging_ridge"],
            "scope": "small-sample proof-of-concept",
        },
        {
            "claim": "Physics KAN improves signed degradation forecasting",
            "dataset": "RWTH",
            "evidence": "eight exposure intervals, four-vehicle LOOV",
            "supported": rwth_decision["interval_physics_kan_beats_context_ridge"],
            "scope": "very small sample; capacity recalibration retained",
        },
        {
            "claim": "Physics KAN improves BMS-SOH trend across time",
            "dataset": "EVE",
            "evidence": "expanding temporal folds on monthly trend checkpoints",
            "supported": eve_decision["physics_kan_beats_aging_current_soh"],
            "scope": "one vehicle; BMS label kept separate",
        },
    ])
    evidence.to_csv(output / "external_evidence_matrix.csv", index=False)

    increment = pd.read_csv(rwth / "rwth_incremental_driver_value.csv")
    challenger = increment[increment.challenger.eq("DriverContext-XGBoost")].copy()
    challenger = challenger.sort_values("delta_mae_b_minus_a")
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(challenger))
    ax.barh(y, challenger.delta_mae_b_minus_a, color=np.where(challenger.driver_adds_value, "#167D8D", "#A0A0A0"))
    ax.errorbar(
        challenger.delta_mae_b_minus_a,
        y,
        xerr=np.vstack([
            challenger.delta_mae_b_minus_a - challenger.ci_low,
            challenger.ci_high - challenger.delta_mae_b_minus_a,
        ]),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, challenger.task)
    ax.set_xlabel("Vehicle-macro MAE change: driver+context minus context (lower is better)")
    ax.set_title("RWTH external driver contribution to battery-stress prediction")
    fig.tight_layout()
    fig.savefig(output / "rwth_driver_increment_external.png", dpi=180)
    plt.close(fig)

    health = combined[combined.task_family.isin([
        "current_capacity_soh", "current_bms_soh_trend", "signed_degradation"
    ])].dropna(subset=["mae"]).copy()
    health["label"] = health.dataset + " | " + health.task_family + " | " + health.model
    health = health.sort_values(["dataset", "task_family", "mae"])
    fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(health))))
    ax.barh(np.arange(len(health)), health.mae, color=np.where(health.model.str.contains("KAN"), "#D47A2C", "#4E79A7"))
    ax.set_yticks(np.arange(len(health)), health.label, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("MAE in percentage points (labels are not pooled)")
    ax.set_title("External health-model comparisons by dataset and label definition")
    fig.tight_layout()
    fig.savefig(output / "external_health_model_mae.png", dpi=180)
    plt.close(fig)

    final = {
        "driver_to_immediate_stress_pathway_supported_rwth": bool(
            rwth_decision["driver_increment_significant_tasks"]
        ),
        "measured_thermal_pathway_replicated_eve": bool(
            eve_decision["driver_improves_measured_temp_rise_over_context_ridge"]
        ),
        "external_health_kan_supported_rwth_current": bool(
            rwth_decision["capacity_physics_kan_beats_aging_ridge"]
        ),
        "external_health_kan_supported_rwth_forecast": bool(
            rwth_decision["interval_physics_kan_beats_context_ridge"]
        ),
        "labels_pooled": False,
        "causal_claim_authorized": False,
        "nissan_locked_test_used": False,
        "deployment_rule": (
            "Select the simplest model that wins within each dataset/task. KAN remains the "
            "explainable proposed model only where external comparison supports it."
        ),
    }
    (output / "EXTERNAL_COMBINED_DECISION.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    report = f"""# Stage 5 combined external-validation report

RWTH and EVE are analysed jointly at the evidence level, never by pooling their
health labels. RWTH uses controlled fixed-voltage-window capacity SOH; EVE uses
cleaned Renault BMS-reported SOH summarized monthly.

## Driver-to-battery physics pathway

RWTH unseen-vehicle support: **{final['driver_to_immediate_stress_pathway_supported_rwth']}**.
EVE measured-temperature replication: **{final['measured_thermal_pathway_replicated_eve']}**.

These results support the proposed mediation chain
`driver action -> current/C-rate -> electrical and thermal stress -> accumulated dose`.
They do not by themselves establish a causal effect on long-term degradation.

## Health layer

RWTH current capacity-SOH physics-KAN support: **{final['external_health_kan_supported_rwth_current']}**.
RWTH signed degradation physics-KAN support: **{final['external_health_kan_supported_rwth_forecast']}**.

Any negative result is retained.  With only 16 capacity checkpoints and eight
exposure-bearing intervals, a simpler aging/context model remains the external
recommendation whenever it has lower vehicle-macro MAE.

The Nissan locked test was not accessed in Stage 5.
"""
    (output / "STAGE5_EXTERNAL_COMBINED_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
