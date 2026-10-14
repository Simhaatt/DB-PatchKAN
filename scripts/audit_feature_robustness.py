"""Audit consensus features with fold-wise removal and EBM response direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

from dbpatchkan.analysis.baselines import build_regression_baselines
from dbpatchkan.analysis.explainability import ebm_response_functions
from dbpatchkan.analysis.feature_selection import apply_semantic_exclusions, default_permutation_estimator
from dbpatchkan.analysis.protocol import _expanding_splits, _structured_frame, _target_specs
from dbpatchkan.analysis.validation import temporal_group_holdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dbpatchkan"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    health_raw = pd.read_csv(output / "health_windows.csv")
    report = pd.read_csv(output / "feature_selection.csv")
    health, time_col, group_col, _ = _structured_frame(
        health_raw,
        time_candidates=("anchor_time_utc", "trip_start_utc", "datetime_utc", "Timestamp"),
        group_candidates=("health_window_id", "trip_key", "Trip"),
        prefix="health",
    )
    development, _ = temporal_group_holdout(
        health, time_col=time_col, group_col=group_col, test_fraction=0.20
    )
    distance_col = next(
        (name for name in ("odometer_clean_km", "odometer_km", "Odometer", "cumulative_distance_km") if name in health),
        None,
    )
    excluded, _ = apply_semantic_exclusions(
        health,
        extra_exclusions=(
            time_col, group_col, "dataset_id", "vehicle_id", "qc_grade",
            "target_quality_flag", "soh_change_pp", "loss_rate_pp_per_1000km",
            "soh_variation_flag",
        ),
    )
    numeric = excluded.apply(pd.to_numeric, errors="coerce")
    specs = {spec.task: spec for spec in _target_specs(health)[0]}
    candidates = report.loc[
        (report["Stage"] == "development_consensus")
        & (report["Final_Status"] == "selected"),
        ["Task", "Feature_Name", "Fold_Selection_Frequency", "Positive_Permutation_Frequency"],
    ].drop_duplicates()
    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for _, candidate in candidates.iterrows():
        task = str(candidate["Task"])
        feature = str(candidate["Feature_Name"])
        spec = specs.get(task)
        if spec is None or feature not in numeric:
            continue
        target = pd.to_numeric(health[spec.column], errors="coerce").to_numpy(dtype=float)
        valid_development = development[np.isfinite(target[development])]
        folds = _expanding_splits(
            health,
            valid_development,
            time_col=time_col,
            group_col=group_col,
            purge_days=spec.purge_days,
            distance_col=distance_col,
            purge_distance_km=spec.purge_distance_km,
            n_splits=4,
        )
        directions: list[int] = []
        ablation_positive: list[bool] = []
        for fold, (train_index, validation_index) in enumerate(folds):
            selected = report.loc[
                (report["Task"] == task)
                & (report["Stage"] == "development_fold")
                & (pd.to_numeric(report["Fold"], errors="coerce") == fold)
                & (report["Fold_Model_Status"] == "selected"),
                "Feature_Name",
            ].dropna().astype(str).tolist()
            selected = [name for name in selected if name in numeric]
            if feature not in selected:
                delta = 0.0
                direction = 0
                status = "not_selected_in_fold"
            else:
                estimator = default_permutation_estimator(args.seed + fold)
                estimator.fit(numeric.iloc[train_index][selected], target[train_index])
                baseline = mean_absolute_error(
                    target[validation_index], estimator.predict(numeric.iloc[validation_index][selected])
                )
                reduced = [name for name in selected if name != feature]
                if reduced:
                    ablated = clone(estimator)
                    ablated.fit(numeric.iloc[train_index][reduced], target[train_index])
                    ablated_prediction = ablated.predict(numeric.iloc[validation_index][reduced])
                else:
                    ablated = DummyRegressor(strategy="mean")
                    ablated.fit(np.zeros((len(train_index), 1)), target[train_index])
                    ablated_prediction = ablated.predict(np.zeros((len(validation_index), 1)))
                delta = float(mean_absolute_error(target[validation_index], ablated_prediction) - baseline)

                ebm = build_regression_baselines(
                    random_state=args.seed + fold, include_xgboost=False, include_ebm=True
                ).get("EBM")
                direction = 0
                if ebm is not None:
                    ebm.fit(numeric.iloc[train_index][[feature]], target[train_index])
                    curve = ebm_response_functions(ebm)
                    x = pd.to_numeric(curve.get("Feature_Value"), errors="coerce")
                    y = pd.to_numeric(curve.get("Effect"), errors="coerce")
                    finite = x.notna() & y.notna()
                    if finite.sum() >= 3:
                        correlation = float(spearmanr(x[finite], y[finite]).statistic)
                        direction = int(np.sign(correlation)) if abs(correlation) >= 0.10 else 0
                status = "audited"
            directions.append(direction)
            ablation_positive.append(delta > 0)
            fold_rows.append(
                {
                    "Task": task,
                    "Feature_Name": feature,
                    "Fold": fold,
                    "Selected_In_Fold": feature in selected,
                    "Ablation_Delta_MAE_pp": delta,
                    "Ablation_Degraded": delta > 0,
                    "EBM_Response_Direction": {1: "increasing", -1: "decreasing", 0: "unstable_or_flat"}[direction],
                    "Status": status,
                }
            )
        positive_fraction = float(np.mean(ablation_positive)) if ablation_positive else 0.0
        nonzero = [value for value in directions if value]
        direction_fraction = (
            max(nonzero.count(1), nonzero.count(-1)) / len(directions) if directions else 0.0
        )
        stable_direction = direction_fraction >= 0.70
        robust = bool(
            float(candidate["Fold_Selection_Frequency"]) >= 0.70
            and float(candidate["Positive_Permutation_Frequency"]) >= 0.70
            and positive_fraction >= 0.70
            and stable_direction
        )
        summary_rows.append(
            {
                "Task": task,
                "Feature_Name": feature,
                "Fold_Selection_Frequency": float(candidate["Fold_Selection_Frequency"]),
                "Positive_Permutation_Frequency": float(candidate["Positive_Permutation_Frequency"]),
                "Positive_Ablation_Frequency": positive_fraction,
                "Stable_Response_Frequency": direction_fraction,
                "Stable_Response": stable_direction,
                "Robustly_Important": robust,
            }
        )
    fold_table = pd.DataFrame(fold_rows)
    summary_table = pd.DataFrame(summary_rows)
    audit = pd.concat(
        [summary_table.assign(Record_Type="Summary"), fold_table.assign(Record_Type="Fold")],
        ignore_index=True,
        sort=False,
    )
    audit.to_csv(output / "feature_robustness_audit.csv", index=False)
    for column, default in (
        ("Positive_Ablation_Frequency", np.nan),
        ("Stable_Response_Frequency", np.nan),
        ("Stable_Response", False),
        ("Robustly_Important", False),
    ):
        report[column] = default
    for _, row in summary_table.iterrows():
        selected = (
            (report["Task"] == row["Task"])
            & (report["Feature_Name"] == row["Feature_Name"])
            & (report["Stage"] == "development_consensus")
        )
        for column in (
            "Positive_Ablation_Frequency", "Stable_Response_Frequency",
            "Stable_Response", "Robustly_Important",
        ):
            report.loc[selected, column] = row[column]
    report.to_csv(output / "feature_selection.csv", index=False)
    print(json.dumps({"candidates": len(summary_table), "robust": int(summary_table.get("Robustly_Important", pd.Series(dtype=bool)).sum())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
