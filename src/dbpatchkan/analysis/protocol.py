"""End-to-end leakage-safe classical analysis protocol.

This module owns orchestration only.  All learned preprocessing and selection is
re-fitted inside the relevant training boundary; the chronological final 20%
holdout is touched once, after development choices have been frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .baselines import (
    build_classification_baselines,
    build_regression_baselines,
    fit_evaluate_classifiers,
    fit_evaluate_regressors,
    soh_ablation_feature_sets,
)
from .clustering import (
    BehaviorLevelMapper,
    aggression_score_from_clusters,
    discover_behavior_archetypes,
    infer_behavior_features,
)
from .explainability import ebm_global_terms, ebm_response_functions, tree_shap_summary
from .feature_selection import (
    apply_semantic_exclusions,
    consensus_feature_selection,
    default_permutation_estimator,
    elastic_net_stability_selection,
    heldout_permutation_importance,
    mrmr_ranking,
    redundancy_filter,
    tune_feature_budget,
)
from .validation import (
    PurgedExpandingWindowSplit,
    block_bootstrap_confidence_interval,
    purge_training_groups,
    temporal_group_holdout,
)


@dataclass(frozen=True)
class TargetSpec:
    task: str
    column: str
    unit: str
    purge_days: float = 0.0
    purge_distance_km: float = 0.0


def _read_table(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if path.suffix.casefold() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.casefold() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def _first_existing(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame), None)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _structured_frame(
    frame: pd.DataFrame,
    *,
    time_candidates: Sequence[str],
    group_candidates: Sequence[str],
    prefix: str,
) -> tuple[pd.DataFrame, str, str, list[str]]:
    result = frame.reset_index(drop=True).copy()
    assumptions: list[str] = []
    group_col = _first_existing(result, group_candidates)
    if group_col is None:
        group_col = f"__{prefix}_group__"
        result[group_col] = np.arange(len(result))
        assumptions.append("synthetic_unique_groups")
    time_col = _first_existing(result, time_candidates)
    if time_col is None:
        time_col = f"__{prefix}_time__"
        result[time_col] = pd.Timestamp("2000-01-01", tz="UTC") + pd.to_timedelta(
            np.arange(len(result)), unit="D"
        )
        assumptions.append("synthetic_chronology_from_row_order")
    return result, time_col, group_col, assumptions


def _expanding_splits(
    frame: pd.DataFrame,
    positions: np.ndarray,
    *,
    time_col: str,
    group_col: str,
    purge_days: float = 0.0,
    distance_col: str | None = None,
    purge_distance_km: float = 0.0,
    n_splits: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    subset = frame.iloc[positions]
    splitter = PurgedExpandingWindowSplit(
        n_splits=n_splits,
        time_col=time_col,
        group_col=group_col,
        purge_days=purge_days,
        distance_col=distance_col,
        purge_distance_km=purge_distance_km,
    )
    return [(positions[train], positions[validation]) for train, validation in splitter.split(subset)]


def _compact_regression_models(random_state: int) -> dict[str, BaseEstimator]:
    suite = build_regression_baselines(
        random_state=random_state, include_xgboost=False, include_ebm=False
    )
    models = {name: suite[name] for name in ("Persistence", "ElasticNet", "RandomForest")}
    forest = models["RandomForest"]
    try:
        forest.set_params(randomforestregressor__n_estimators=160)
    except ValueError:
        pass
    return models


def _compact_classification_models(random_state: int) -> dict[str, BaseEstimator]:
    suite = build_classification_baselines(
        random_state=random_state, include_xgboost=False, include_ebm=False
    )
    models = {
        name: suite[name] for name in ("ElasticNetLogistic", "RandomForest") if name in suite
    }
    if "RandomForest" in models:
        try:
            models["RandomForest"].set_params(randomforestclassifier__n_estimators=200)
        except ValueError:
            pass
    return models


def _metric_confidence_intervals(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    group_col: str,
    classification: bool = False,
    random_state: int,
) -> pd.DataFrame:
    output = metrics.copy()
    if output.empty or predictions.empty:
        return output
    if classification:
        output["Macro_F1_CI_Lower"] = np.nan
        output["Macro_F1_CI_Upper"] = np.nan
    else:
        output["MAE_CI_Lower"] = np.nan
        output["MAE_CI_Upper"] = np.nan
        output["RMSE_CI_Lower"] = np.nan
        output["RMSE_CI_Upper"] = np.nan
    for row_index, metric_row in output.iterrows():
        if metric_row.get("Status") != "ok":
            continue
        model_predictions = predictions.loc[predictions["Model"] == metric_row["Model"]].copy()
        if model_predictions.empty:
            continue
        source_positions = model_predictions["Row_Index"].to_numpy(dtype=int)
        model_predictions["__group__"] = frame.iloc[source_positions][group_col].to_numpy()
        if classification:
            def macro_f1(sample: pd.DataFrame) -> float:
                from sklearn.metrics import f1_score

                return float(
                    f1_score(
                        sample["Observed"], sample["Predicted"], average="macro", zero_division=0
                    )
                )

            _, lower, upper = block_bootstrap_confidence_interval(
                model_predictions,
                group_col="__group__",
                statistic=macro_f1,
                n_bootstraps=200,
                random_state=random_state,
            )
            output.loc[row_index, ["Macro_F1_CI_Lower", "Macro_F1_CI_Upper"]] = [lower, upper]
        else:
            def mae(sample: pd.DataFrame) -> float:
                return float(
                    np.mean(np.abs(sample["soh_observed_pct"] - sample["soh_pred_pct"]))
                )

            def rmse(sample: pd.DataFrame) -> float:
                residual = sample["soh_observed_pct"] - sample["soh_pred_pct"]
                return float(np.sqrt(np.mean(np.square(residual))))

            _, mae_lower, mae_upper = block_bootstrap_confidence_interval(
                model_predictions,
                group_col="__group__",
                statistic=mae,
                n_bootstraps=200,
                random_state=random_state,
            )
            _, rmse_lower, rmse_upper = block_bootstrap_confidence_interval(
                model_predictions,
                group_col="__group__",
                statistic=rmse,
                n_bootstraps=200,
                random_state=random_state + 1,
            )
            output.loc[
                row_index,
                ["MAE_CI_Lower", "MAE_CI_Upper", "RMSE_CI_Lower", "RMSE_CI_Upper"],
            ] = [mae_lower, mae_upper, rmse_lower, rmse_upper]
    return output


def _behavior_scores(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    *,
    features: Sequence[str],
    random_state: int,
    bootstrap_iterations: int,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, BehaviorLevelMapper]:
    discovery = discover_behavior_archetypes(
        train,
        feature_names=features,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )
    train_scores, contributions = aggression_score_from_clusters(
        train,
        discovery.labels,
        discovery.probabilities,
        feature_names=features,
    )
    mapping = contributions.attrs["cluster_mapping"].sort_values("natural_cluster_id")
    intensities = mapping["cluster_aggression_intensity"].to_numpy(dtype=float)
    evaluate_labels, evaluate_probabilities = discovery.predict(evaluate)
    evaluate_scores = evaluate_probabilities @ intensities
    mapper = BehaviorLevelMapper().fit(train_scores)
    return (
        discovery,
        train_scores,
        mapper.transform(train_scores),
        evaluate_labels,
        evaluate_scores,
        mapper,
    )


def _select_classifier(benchmarks: pd.DataFrame, task: str, default: str) -> str:
    required = {"Task", "split_role", "Status", "Model", "Macro_F1"}
    if benchmarks.empty or not required.issubset(benchmarks.columns):
        return default
    rows = benchmarks.loc[
        (benchmarks["Task"] == task)
        & (benchmarks["split_role"] == "development_validation")
        & (benchmarks["Status"] == "ok")
    ]
    if rows.empty or "Macro_F1" not in rows:
        return default
    scores = rows.groupby("Model")["Macro_F1"].mean().sort_values(ascending=False)
    return str(scores.index[0]) if len(scores) else default


def _behavior_protocol(
    trips_input: pd.DataFrame,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trips, time_col, group_col, assumptions = _structured_frame(
        trips_input,
        time_candidates=("trip_start_utc", "window_start_utc", "datetime_utc", "Timestamp"),
        group_candidates=("trip_key", "Trip", "window_id"),
        prefix="behavior",
    )
    features = infer_behavior_features(trips)
    if len(trips) < 5 or not features or trips[group_col].nunique(dropna=False) < 5:
        return (
            pd.DataFrame(
                columns=[
                    "natural_cluster_id",
                    "natural_cluster_probability",
                    "aggression_score",
                    "behavior_level",
                ]
            ),
            pd.DataFrame(),
            {
                "status": "insufficient_data",
                "behavior_features": features,
                "assumptions": assumptions,
            },
        )
    development, final_holdout = temporal_group_holdout(
        trips, time_col=time_col, group_col=group_col, test_fraction=0.20
    )
    folds = _expanding_splits(
        trips,
        development,
        time_col=time_col,
        group_col=group_col,
        n_splits=4,
    )
    benchmark_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    clustering_folds: list[dict[str, Any]] = []
    classifiers = _compact_classification_models(random_state)
    fixed_driver_ids = {0, 1, 2, 4, "0", "1", "2", "4"}
    eligible_ids: list[Any] = []
    driver_col = "Driver_ID" if "Driver_ID" in trips else None
    if driver_col:
        development_trips = trips.iloc[development]
        counts = development_trips.groupby(driver_col)[group_col].nunique()
        eligible_ids = [driver for driver, count in counts.items() if driver in fixed_driver_ids and count >= 5]

    for fold_id, (train_index, validation_index) in enumerate(folds):
        try:
            discovery, _, train_levels, _, validation_scores, mapper = _behavior_scores(
                trips.iloc[train_index],
                trips.iloc[validation_index],
                features=features,
                random_state=random_state + fold_id,
                bootstrap_iterations=8,
            )
        except ValueError:
            continue
        validation_levels = mapper.transform(validation_scores)
        if np.unique(train_levels).size >= 2:
            evaluation = fit_evaluate_classifiers(
                trips.iloc[train_index][features],
                train_levels,
                trips.iloc[validation_index][features],
                validation_levels,
                models=classifiers,
                random_state=random_state + fold_id,
            )
            metrics = _metric_confidence_intervals(
                evaluation.metrics,
                evaluation.predictions,
                trips,
                group_col=group_col,
                classification=True,
                random_state=random_state + fold_id,
            )
            metrics["Task"] = "behavior_level"
            metrics["Fold"] = fold_id
            metrics["split_role"] = "development_validation"
            benchmark_tables.append(metrics)
            predictions = evaluation.predictions.copy()
            predictions["Task"] = "behavior_level"
            predictions["Fold"] = fold_id
            prediction_tables.append(predictions)
        if driver_col and len(eligible_ids) >= 2:
            train_mask = trips.iloc[train_index][driver_col].isin(eligible_ids).to_numpy()
            validation_mask = trips.iloc[validation_index][driver_col].isin(eligible_ids).to_numpy()
            driver_train = train_index[train_mask]
            driver_validation = validation_index[validation_mask]
            if (
                len(driver_train) >= 4
                and len(driver_validation) >= 1
                and trips.iloc[driver_train][driver_col].nunique() >= 2
            ):
                evaluation = fit_evaluate_classifiers(
                    trips.iloc[driver_train][features],
                    trips.iloc[driver_train][driver_col].astype(str),
                    trips.iloc[driver_validation][features],
                    trips.iloc[driver_validation][driver_col].astype(str),
                    models=classifiers,
                    random_state=random_state + fold_id,
                )
                metrics = _metric_confidence_intervals(
                    evaluation.metrics,
                    evaluation.predictions,
                    trips,
                    group_col=group_col,
                    classification=True,
                    random_state=random_state + 100 + fold_id,
                )
                metrics["Task"] = "driver_id"
                metrics["Fold"] = fold_id
                metrics["split_role"] = "development_validation"
                benchmark_tables.append(metrics)
                predictions = evaluation.predictions.copy()
                predictions["Task"] = "driver_id"
                predictions["Fold"] = fold_id
                prediction_tables.append(predictions)
        clustering_folds.append(
            {
                "fold": fold_id,
                "train_groups": int(trips.iloc[train_index][group_col].nunique()),
                "validation_groups": int(trips.iloc[validation_index][group_col].nunique()),
                "selected_k": discovery.selected_k,
                "bootstrap_ari": discovery.bootstrap_ari,
                "exploratory": discovery.exploratory,
            }
        )

    benchmarks = pd.concat(benchmark_tables, ignore_index=True, sort=False) if benchmark_tables else pd.DataFrame()
    fold_predictions = (
        pd.concat(prediction_tables, ignore_index=True, sort=False) if prediction_tables else pd.DataFrame()
    )
    behavior_model = _select_classifier(benchmarks, "behavior_level", "ElasticNetLogistic")
    driver_model = _select_classifier(benchmarks, "driver_id", "ElasticNetLogistic")

    try:
        discovery, development_scores, development_levels, final_labels, final_scores, mapper = _behavior_scores(
            trips.iloc[development],
            trips.iloc[final_holdout],
            features=features,
            random_state=random_state,
            bootstrap_iterations=20,
        )
    except ValueError as error:
        return pd.DataFrame(), benchmarks, {
            "status": "final_development_clustering_failed",
            "message": str(error),
            "folds": clustering_folds,
        }
    final_levels = mapper.transform(final_scores)
    all_labels = np.full(len(trips), -1, dtype=int)
    all_scores = np.full(len(trips), np.nan)
    all_levels = np.full(len(trips), "", dtype=object)
    all_probabilities = np.full(len(trips), np.nan)
    all_labels[development] = discovery.labels
    all_scores[development] = development_scores
    all_levels[development] = development_levels
    all_probabilities[development] = discovery.probabilities.max(axis=1)
    final_cluster_labels, final_cluster_probabilities = discovery.predict(trips.iloc[final_holdout])
    all_labels[final_holdout] = final_cluster_labels
    all_scores[final_holdout] = final_scores
    all_levels[final_holdout] = final_levels
    all_probabilities[final_holdout] = final_cluster_probabilities.max(axis=1)

    identifiers = [
        column
        for column in ("dataset_id", "vehicle_id", "trip_key", "Trip", "Driver_ID", time_col)
        if column in trips
    ]
    output = trips[identifiers].copy()
    output["natural_cluster_id"] = all_labels
    output["natural_cluster_probability"] = all_probabilities
    output["aggression_score"] = all_scores
    output["behavior_level"] = all_levels
    output["behavior_level_pred"] = pd.Series(pd.NA, index=output.index, dtype="object")
    output["driver_id_pred"] = pd.Series(pd.NA, index=output.index, dtype="object")
    output["split_role"] = "development"
    output.loc[final_holdout, "split_role"] = "final_holdout"

    if not fold_predictions.empty:
        behavior_oof = fold_predictions.loc[
            (fold_predictions["Task"] == "behavior_level")
            & (fold_predictions["Model"] == behavior_model)
        ]
        for _, row in behavior_oof.iterrows():
            output.loc[int(row["Row_Index"]), "behavior_level_pred"] = row["Predicted"]
        driver_oof = fold_predictions.loc[
            (fold_predictions["Task"] == "driver_id")
            & (fold_predictions["Model"] == driver_model)
        ]
        for _, row in driver_oof.iterrows():
            output.loc[int(row["Row_Index"]), "driver_id_pred"] = row["Predicted"]

    if len(final_holdout) and np.unique(development_levels).size >= 2:
        final_evaluation = fit_evaluate_classifiers(
            trips.iloc[development][features],
            development_levels,
            trips.iloc[final_holdout][features],
            final_levels,
            models=classifiers,
            random_state=random_state,
        )
        metrics = _metric_confidence_intervals(
            final_evaluation.metrics,
            final_evaluation.predictions,
            trips,
            group_col=group_col,
            classification=True,
            random_state=random_state,
        )
        metrics["Task"] = "behavior_level"
        metrics["Fold"] = "final"
        metrics["split_role"] = "final_holdout"
        metrics["Development_Selected_Model"] = metrics["Model"] == behavior_model
        benchmarks = pd.concat([benchmarks, metrics], ignore_index=True, sort=False)
        selected_predictions = final_evaluation.predictions.loc[
            final_evaluation.predictions["Model"] == behavior_model
        ]
        for _, row in selected_predictions.iterrows():
            output.loc[int(row["Row_Index"]), "behavior_level_pred"] = row["Predicted"]

    if driver_col and len(eligible_ids) >= 2:
        development_driver = development[trips.iloc[development][driver_col].isin(eligible_ids).to_numpy()]
        final_driver = final_holdout[trips.iloc[final_holdout][driver_col].isin(eligible_ids).to_numpy()]
        if len(final_driver) and trips.iloc[development_driver][driver_col].nunique() >= 2:
            evaluation = fit_evaluate_classifiers(
                trips.iloc[development_driver][features],
                trips.iloc[development_driver][driver_col].astype(str),
                trips.iloc[final_driver][features],
                trips.iloc[final_driver][driver_col].astype(str),
                models=classifiers,
                random_state=random_state,
            )
            metrics = _metric_confidence_intervals(
                evaluation.metrics,
                evaluation.predictions,
                trips,
                group_col=group_col,
                classification=True,
                random_state=random_state + 100,
            )
            metrics["Task"] = "driver_id"
            metrics["Fold"] = "final"
            metrics["split_role"] = "final_holdout"
            metrics["Development_Selected_Model"] = metrics["Model"] == driver_model
            benchmarks = pd.concat([benchmarks, metrics], ignore_index=True, sort=False)
            selected_predictions = evaluation.predictions.loc[evaluation.predictions["Model"] == driver_model]
            for _, row in selected_predictions.iterrows():
                output.loc[int(row["Row_Index"]), "driver_id_pred"] = row["Predicted"]

    output["identity_eligible"] = (
        output[driver_col].isin(eligible_ids) if driver_col else False
    )
    summary = {
        "status": "ok",
        "behavior_features": features,
        "development_groups": int(trips.iloc[development][group_col].nunique()),
        "final_holdout_groups": int(trips.iloc[final_holdout][group_col].nunique()),
        "development_folds_completed": len(clustering_folds),
        "folds": clustering_folds,
        "selected_behavior_classifier": behavior_model,
        "selected_driver_classifier": driver_model if len(eligible_ids) >= 2 else None,
        "eligible_driver_ids": list(map(str, eligible_ids)),
        "final_cluster_k": discovery.selected_k,
        "final_cluster_bootstrap_ari": discovery.bootstrap_ari,
        "final_cluster_exploratory": discovery.exploratory,
        "final_gmm_candidates": discovery.candidate_metrics.to_dict(orient="records"),
        "final_clusterer_comparison": discovery.comparison_metrics.to_dict(orient="records"),
        "assumptions": assumptions,
        "behavior_to_health_fold_integration": "pending_no_safe_cross_table_key",
    }
    return output, benchmarks, summary


def _target_specs(frame: pd.DataFrame) -> tuple[list[TargetSpec], list[dict[str, str]]]:
    specs: list[TargetSpec] = []
    unavailable: list[dict[str, str]] = []
    current = _first_existing(
        frame, ("soh_target_trend_pct", "soh_target_raw_pct", "soh_observed_pct", "SOH")
    )
    if current:
        specs.append(TargetSpec("current_soh", current, "SOH percentage points"))
    else:
        unavailable.append({"task": "current_soh", "status": "target_unavailable"})
    candidates = (
        TargetSpec("forecast_30d_soh", "target_soh_30d_pct", "SOH percentage points", purge_days=30),
        TargetSpec("forecast_30d_loss", "target_loss_30d_pp", "loss percentage points", purge_days=30),
        TargetSpec("forecast_500km_soh", "target_soh_500km_pct", "SOH percentage points", purge_distance_km=500),
        TargetSpec(
            "forecast_500km_loss_rate",
            "target_loss_rate_pp_per_1000km",
            "SOH percentage points per 1000 km",
            purge_distance_km=500,
        ),
    )
    for spec in candidates:
        if spec.column in frame and pd.to_numeric(frame[spec.column], errors="coerce").notna().any():
            specs.append(spec)
        else:
            unavailable.append({"task": spec.task, "status": "target_unavailable"})
    return specs, unavailable


def _inner_budget(
    numeric: pd.DataFrame,
    target: np.ndarray,
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    *,
    time_col: str,
    group_col: str,
    distance_col: str | None,
    spec: TargetSpec,
    random_state: int,
) -> tuple[int, pd.DataFrame]:
    local_metadata = metadata.iloc[train_index]
    if local_metadata[group_col].nunique(dropna=False) < 4:
        return 8, pd.DataFrame()
    splitter = PurgedExpandingWindowSplit(
        n_splits=min(3, local_metadata[group_col].nunique() - 1),
        time_col=time_col,
        group_col=group_col,
        purge_days=spec.purge_days,
        distance_col=distance_col,
        purge_distance_km=spec.purge_distance_km,
    )
    splits = list(splitter.split(local_metadata))
    if not splits:
        return 8, pd.DataFrame()
    estimator = _compact_regression_models(random_state)["ElasticNet"]
    tuned = tune_feature_budget(
        numeric.iloc[train_index],
        target[train_index],
        splits,
        budgets=(8, 12, 16, 20, 24, 32),
        estimator=estimator,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
    )
    return tuned.selected_budget, tuned.summary


def _fold_feature_selection(
    numeric: pd.DataFrame,
    target: np.ndarray,
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    *,
    task: str,
    fold_id: int,
    time_col: str,
    group_col: str,
    distance_col: str | None,
    spec: TargetSpec,
    random_state: int,
) -> tuple[list[str], pd.DataFrame, int, pd.DataFrame]:
    screen = redundancy_filter(numeric.iloc[train_index])
    if not screen.selected_features:
        report = screen.report.copy()
        report["Task"] = task
        report["Fold"] = fold_id
        report["Stage"] = "development_fold"
        return [], report, 8, pd.DataFrame()
    ranking = mrmr_ranking(
        numeric.iloc[train_index][screen.selected_features],
        target[train_index],
        max_features=min(40, len(screen.selected_features)),
        random_state=random_state + fold_id,
    )
    stability = elastic_net_stability_selection(
        numeric.iloc[train_index][screen.selected_features],
        target[train_index],
        groups=metadata.iloc[train_index][group_col].to_numpy(),
        n_bootstraps=100,
        random_state=random_state + fold_id,
    )
    budget, budget_summary = _inner_budget(
        numeric,
        target,
        metadata,
        train_index,
        time_col=time_col,
        group_col=group_col,
        distance_col=distance_col,
        spec=spec,
        random_state=random_state + fold_id,
    )
    ranked = ranking["Feature_Name"].tolist()
    stable = set(
        stability.loc[
            stability["ElasticNet_Selection_Frequency"] >= 0.70, "Feature_Name"
        ].tolist()
    )
    model_features = [feature for feature in ranked if feature in stable][:budget]
    if not model_features:
        model_features = ranked[:budget]
    permutation = pd.DataFrame(
        columns=["Feature_Name", "Permutation_Importance", "Permutation_Importance_Std"]
    )
    if model_features and len(validation_index):
        estimator = default_permutation_estimator(random_state + fold_id)
        try:
            estimator.fit(numeric.iloc[train_index][model_features], target[train_index])
            permutation = heldout_permutation_importance(
                estimator,
                numeric.iloc[validation_index][model_features],
                target[validation_index],
                scoring="neg_mean_absolute_error",
                n_repeats=8,
                random_state=random_state + fold_id,
            )
        except (ValueError, FloatingPointError):
            pass
    report = (
        screen.report.merge(ranking, on="Feature_Name", how="left")
        .merge(stability, on="Feature_Name", how="left")
        .merge(permutation, on="Feature_Name", how="left")
    )
    report["mRMR_Selected"] = report["mRMR_Rank"].le(40).fillna(False)
    report["ElasticNet_Selected"] = (
        report["ElasticNet_Selection_Frequency"].ge(0.70).fillna(False)
    )
    report["Permutation_Selected"] = report["Permutation_Importance"].gt(0).fillna(False)
    report["Selector_Support"] = report[
        ["mRMR_Selected", "ElasticNet_Selected", "Permutation_Selected"]
    ].sum(axis=1)
    report["Fold_Selected"] = report["Selector_Support"] >= 2
    report["Fold_Model_Status"] = np.where(
        report["Feature_Name"].isin(model_features), "selected", "not_selected"
    )
    report["Chosen_Budget"] = budget
    report["Task"] = task
    report["Fold"] = fold_id
    report["Stage"] = "development_fold"
    if not budget_summary.empty:
        budget_summary = budget_summary.copy()
        budget_summary["Task"] = task
        budget_summary["Outer_Fold"] = fold_id
    return model_features, report, budget, budget_summary


def _regression_evaluation(
    numeric: pd.DataFrame,
    target: np.ndarray,
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    features: Sequence[str],
    models: Mapping[str, BaseEstimator],
    *,
    task: str,
    target_column: str,
    target_unit: str,
    feature_set: str,
    fold: int | str,
    split_role: str,
    group_col: str,
    random_state: int,
    purge_days: float = 0.0,
    purge_distance_km: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, BaseEstimator]]:
    if not features or len(train_index) < 2 or len(validation_index) < 1:
        return pd.DataFrame(), pd.DataFrame(), {}
    evaluation = fit_evaluate_regressors(
        numeric.iloc[train_index][list(features)],
        target[train_index],
        numeric.iloc[validation_index][list(features)],
        target[validation_index],
        models=models,
        random_state=random_state,
    )
    metrics = _metric_confidence_intervals(
        evaluation.metrics,
        evaluation.predictions,
        metadata,
        group_col=group_col,
        random_state=random_state,
    )
    for table in (metrics, evaluation.predictions):
        table["Task"] = task
        table["Target"] = target_column
        table["Target_Unit"] = target_unit
        table["Feature_Set"] = feature_set
        table["Fold"] = fold
        table["split_role"] = split_role
        table["Feature_Count"] = len(features)
        table["Train_Group_Count"] = metadata.iloc[train_index][group_col].nunique(dropna=False)
        table["Validation_Group_Count"] = metadata.iloc[validation_index][group_col].nunique(
            dropna=False
        )
        table["Purge_Days"] = purge_days
        table["Purge_Distance_km"] = purge_distance_km
    if not evaluation.predictions.empty:
        evaluation.predictions["target_observed"] = evaluation.predictions["soh_observed_pct"]
        evaluation.predictions["target_predicted"] = evaluation.predictions["soh_pred_pct"]
    return metrics, evaluation.predictions, evaluation.fitted_models


def _ablation_evaluations(
    numeric: pd.DataFrame,
    target: np.ndarray,
    metadata: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    available_features: Sequence[str],
    *,
    spec: TargetSpec,
    fold: int | str,
    split_role: str,
    group_col: str,
    random_state: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    metrics_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    sets = soh_ablation_feature_sets(available_features)
    model = {"ElasticNet": _compact_regression_models(random_state)["ElasticNet"]}
    for set_name, features in sets.items():
        if not features:
            metrics_tables.append(
                pd.DataFrame(
                    [
                        {
                            "Task": spec.task,
                            "Target": spec.column,
                            "Target_Unit": spec.unit,
                            "Feature_Set": set_name,
                            "Fold": fold,
                            "split_role": split_role,
                            "Model": "ElasticNet",
                            "Status": "unavailable_no_matching_features",
                            "Feature_Count": 0,
                        }
                    ]
                )
            )
            continue
        metrics, predictions, _ = _regression_evaluation(
            numeric,
            target,
            metadata,
            train_index,
            validation_index,
            features,
            model,
            task=spec.task,
            target_column=spec.column,
            target_unit=spec.unit,
            feature_set=set_name,
            fold=fold,
            split_role=split_role,
            group_col=group_col,
            random_state=random_state,
            purge_days=spec.purge_days,
            purge_distance_km=spec.purge_distance_km,
        )
        metrics_tables.append(metrics)
        prediction_tables.append(predictions)
    return metrics_tables, prediction_tables


def _development_selected_model(benchmarks: pd.DataFrame, task: str) -> str:
    required = {"Task", "Feature_Set", "split_role", "Status", "Model", "MAE_pp"}
    if benchmarks.empty or not required.issubset(benchmarks.columns):
        return "ElasticNet"
    rows = benchmarks.loc[
        (benchmarks["Task"] == task)
        & (benchmarks["Feature_Set"] == "Consensus")
        & (benchmarks["split_role"] == "development_validation")
        & (benchmarks["Status"] == "ok")
    ]
    if rows.empty:
        return "ElasticNet"
    mean_error = rows.groupby("Model")["MAE_pp"].mean().sort_values()
    return str(mean_error.index[0]) if len(mean_error) else "ElasticNet"


def _soh_protocol(
    health_input: pd.DataFrame,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    health, time_col, group_col, assumptions = _structured_frame(
        health_input,
        time_candidates=("anchor_time_utc", "trip_start_utc", "datetime_utc", "Timestamp"),
        group_candidates=("health_window_id", "trip_key", "Trip"),
        prefix="health",
    )
    if len(health) < 5 or health[group_col].nunique(dropna=False) < 5:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_data",
            "assumptions": assumptions,
        }
    development, final_holdout = temporal_group_holdout(
        health, time_col=time_col, group_col=group_col, test_fraction=0.20
    )
    distance_col = _first_existing(
        health, ("odometer_clean_km", "odometer_km", "Odometer", "cumulative_distance_km")
    )
    excluded, semantic_audit = apply_semantic_exclusions(
        health,
        extra_exclusions=(
            time_col,
            group_col,
            "dataset_id",
            "vehicle_id",
            "qc_grade",
            "target_quality_flag",
            "soh_change_pp",
            "loss_rate_pp_per_1000km",
            "soh_variation_flag",
        ),
    )
    numeric = excluded.apply(pd.to_numeric, errors="coerce")
    specs, unavailable = _target_specs(health)
    all_feature_reports: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_benchmarks: list[pd.DataFrame] = []
    all_explanations: list[pd.DataFrame] = []
    task_summaries: list[dict[str, Any]] = list(unavailable)
    budget_tables: list[pd.DataFrame] = []

    for spec_index, spec in enumerate(specs):
        target_series = pd.to_numeric(health[spec.column], errors="coerce")
        target = target_series.to_numpy(dtype=float)
        valid = np.isfinite(target)
        development_valid = development[valid[development]]
        holdout_valid = final_holdout[valid[final_holdout]]
        if len(development_valid) < 4 or np.unique(target[development_valid]).size < 2:
            task_summaries.append(
                {"task": spec.task, "status": "insufficient_development_target_variation"}
            )
            continue
        folds = _expanding_splits(
            health,
            development_valid,
            time_col=time_col,
            group_col=group_col,
            purge_days=spec.purge_days,
            distance_col=distance_col,
            purge_distance_km=spec.purge_distance_km,
            n_splits=4,
        )
        fold_reports: list[pd.DataFrame] = []
        fold_benchmarks: list[pd.DataFrame] = []
        chosen_budgets: list[int] = []
        completed_folds = 0
        for fold_id, (train_index, validation_index) in enumerate(folds):
            if len(train_index) < 3 or np.unique(target[train_index]).size < 2:
                continue
            features, report, budget, budget_summary = _fold_feature_selection(
                numeric,
                target,
                health,
                train_index,
                validation_index,
                task=spec.task,
                fold_id=fold_id,
                time_col=time_col,
                group_col=group_col,
                distance_col=distance_col,
                spec=spec,
                random_state=random_state + spec_index * 100,
            )
            fold_reports.append(report)
            chosen_budgets.append(budget)
            if not budget_summary.empty:
                budget_tables.append(budget_summary)
            if not features:
                continue
            metrics, predictions, _ = _regression_evaluation(
                numeric,
                target,
                health,
                train_index,
                validation_index,
                features,
                _compact_regression_models(random_state + fold_id),
                task=spec.task,
                target_column=spec.column,
                target_unit=spec.unit,
                feature_set="Consensus",
                fold=fold_id,
                split_role="development_validation",
                group_col=group_col,
                random_state=random_state + fold_id,
                purge_days=spec.purge_days,
                purge_distance_km=spec.purge_distance_km,
            )
            fold_benchmarks.append(metrics)
            all_predictions.append(predictions)
            ablation_metrics, ablation_predictions = _ablation_evaluations(
                numeric,
                target,
                health,
                train_index,
                validation_index,
                [feature for feature in report.loc[report["Final_Status"] == "selected", "Feature_Name"]],
                spec=spec,
                fold=fold_id,
                split_role="development_validation",
                group_col=group_col,
                random_state=random_state + fold_id,
            )
            fold_benchmarks.extend(ablation_metrics)
            all_predictions.extend(ablation_predictions)
            completed_folds += 1
        if not fold_reports:
            task_summaries.append({"task": spec.task, "status": "no_valid_development_folds"})
            continue
        reports = pd.concat(fold_reports, ignore_index=True, sort=False)
        all_feature_reports.append(reports)
        consensus_input = reports[
            [
                "Feature_Name",
                "Fold",
                "mRMR_Selected",
                "ElasticNet_Selected",
                "Permutation_Importance",
            ]
        ].copy()
        consensus = consensus_feature_selection(consensus_input)
        consensus["Task"] = spec.task
        consensus["Fold"] = "development_consensus"
        consensus["Stage"] = "development_consensus"
        all_feature_reports.append(consensus)
        if fold_benchmarks:
            task_benchmarks = pd.concat(fold_benchmarks, ignore_index=True, sort=False)
            all_benchmarks.append(task_benchmarks)
        else:
            task_benchmarks = pd.DataFrame()
        chosen_budget = int(pd.Series(chosen_budgets).mode().min()) if chosen_budgets else 8
        selected_model = _development_selected_model(task_benchmarks, spec.task)

        if len(holdout_valid) == 0:
            task_summaries.append(
                {
                    "task": spec.task,
                    "status": "development_only_no_final_labels",
                    "development_folds_completed": completed_folds,
                    "chosen_budget": chosen_budget,
                    "selected_model": selected_model,
                }
            )
            continue
        final_train = purge_training_groups(
            health,
            development_valid,
            holdout_valid,
            time_col=time_col,
            group_col=group_col,
            purge_days=spec.purge_days,
            distance_col=distance_col,
            purge_distance_km=spec.purge_distance_km,
        )
        if len(final_train) < 3 or np.unique(target[final_train]).size < 2:
            task_summaries.append(
                {
                    "task": spec.task,
                    "status": "final_training_empty_after_purge",
                    "development_folds_completed": completed_folds,
                }
            )
            continue
        final_screen = redundancy_filter(numeric.iloc[final_train])
        final_ranking = mrmr_ranking(
            numeric.iloc[final_train][final_screen.selected_features],
            target[final_train],
            max_features=min(40, len(final_screen.selected_features)),
            random_state=random_state,
        )
        final_stability = elastic_net_stability_selection(
            numeric.iloc[final_train][final_screen.selected_features],
            target[final_train],
            groups=health.iloc[final_train][group_col].to_numpy(),
            n_bootstraps=100,
            random_state=random_state,
        )
        ordered_final = final_ranking["Feature_Name"].tolist()
        consensus_selected = consensus.loc[
            consensus["Final_Status"] == "selected", "Feature_Name"
        ].tolist()
        final_features = [feature for feature in consensus_selected if feature in ordered_final]
        if not final_features:
            consensus_order = consensus.sort_values(
                ["Fold_Selection_Frequency", "Permutation_Importance"], ascending=False
            )["Feature_Name"].tolist()
            final_features = [feature for feature in consensus_order if feature in ordered_final]
        if not final_features:
            final_features = ordered_final
        final_features = final_features[:chosen_budget]
        final_report = (
            final_screen.report.merge(final_ranking, on="Feature_Name", how="left")
            .merge(final_stability, on="Feature_Name", how="left")
        )
        final_report = final_report.merge(
            consensus[
                [
                    "Feature_Name",
                    "Fold_Selection_Frequency",
                    "Positive_Permutation_Frequency",
                    "Permutation_Importance",
                    "Consensus_Rank",
                ]
            ],
            on="Feature_Name",
            how="left",
        )
        final_report["Final_Status"] = np.where(
            final_report["Feature_Name"].isin(final_features), "selected", "rejected"
        )
        final_report["Rejection_Reason"] = np.where(
            final_report["Feature_Name"].isin(final_features),
            "",
            final_report["Rejection_Reason"].replace("", "outside_frozen_consensus_budget"),
        )
        final_report["Task"] = spec.task
        final_report["Fold"] = "final_fit"
        final_report["Stage"] = "final_fit_development_only"
        final_report["Chosen_Budget"] = chosen_budget
        all_feature_reports.append(final_report)

        full_suite = build_regression_baselines(random_state=random_state)
        metrics, predictions, fitted = _regression_evaluation(
            numeric,
            target,
            health,
            final_train,
            holdout_valid,
            final_features,
            full_suite,
            task=spec.task,
            target_column=spec.column,
            target_unit=spec.unit,
            feature_set="Consensus",
            fold="final",
            split_role="final_holdout",
            group_col=group_col,
            random_state=random_state,
            purge_days=spec.purge_days,
            purge_distance_km=spec.purge_distance_km,
        )
        metrics["Development_Selected_Model"] = metrics["Model"] == selected_model
        all_benchmarks.append(metrics)
        all_predictions.append(predictions)
        ablation_metrics, ablation_predictions = _ablation_evaluations(
            numeric,
            target,
            health,
            final_train,
            holdout_valid,
            final_screen.selected_features,
            spec=spec,
            fold="final",
            split_role="final_holdout",
            group_col=group_col,
            random_state=random_state,
        )
        for table in ablation_metrics:
            table["Development_Selected_Model"] = False
        all_benchmarks.extend(ablation_metrics)
        all_predictions.extend(ablation_predictions)

        explanation_model = fitted.get(selected_model)
        if explanation_model is None:
            explanation_model = fitted.get("RandomForest") or next(iter(fitted.values()), None)
        if explanation_model is not None and final_features:
            try:
                permutation = heldout_permutation_importance(
                    explanation_model,
                    numeric.iloc[holdout_valid][final_features],
                    target[holdout_valid],
                    scoring="neg_mean_absolute_error",
                    n_repeats=10,
                    random_state=random_state,
                ).rename(columns={"Permutation_Importance": "Importance"})
                permutation["Explanation_Type"] = "final_holdout_permutation_diagnostic"
                permutation["Model"] = selected_model
                permutation["Task"] = spec.task
                permutation["Status"] = "diagnostic_not_used_for_selection"
                all_explanations.append(permutation)
            except ValueError:
                pass
            shap = tree_shap_summary(
                explanation_model,
                numeric.iloc[holdout_valid][final_features],
                random_state=random_state,
            )
            shap_table = shap.importance.rename(columns={"SHAP_Importance": "Importance"})
            shap_table["Explanation_Type"] = "TreeSHAP"
            shap_table["Model"] = selected_model
            shap_table["Task"] = spec.task
            shap_table["Status"] = shap.status
            all_explanations.append(shap_table)
        ebm_model = fitted.get("EBM")
        if ebm_model is not None:
            ebm_terms = ebm_global_terms(ebm_model).rename(
                columns={"EBM_Importance": "Importance"}
            )
            ebm_terms["Explanation_Type"] = "EBM_global_term"
            ebm_terms["Model"] = "EBM"
            ebm_terms["Task"] = spec.task
            all_explanations.append(ebm_terms)
            ebm_curves = ebm_response_functions(ebm_model)
            if not ebm_curves.empty:
                ebm_curves["Importance"] = ebm_curves["Effect"].abs()
                ebm_curves["Explanation_Type"] = "EBM_response_curve"
                ebm_curves["Model"] = "EBM"
                ebm_curves["Task"] = spec.task
                all_explanations.append(ebm_curves)
        task_summaries.append(
            {
                "task": spec.task,
                "status": "ok",
                "target": spec.column,
                "development_folds_completed": completed_folds,
                "chosen_budget": chosen_budget,
                "selected_model": selected_model,
                "final_features": final_features,
                "final_train_groups_after_purge": int(health.iloc[final_train][group_col].nunique()),
                "final_holdout_groups": int(health.iloc[holdout_valid][group_col].nunique()),
                "purge_days": spec.purge_days,
                "purge_distance_km": spec.purge_distance_km,
            }
        )

    semantic_audit["Task"] = "all"
    semantic_audit["Fold"] = "semantic"
    semantic_audit["Stage"] = "semantic_exclusion"
    all_feature_reports.insert(0, semantic_audit)
    if budget_tables:
        budget_audit = pd.concat(budget_tables, ignore_index=True, sort=False)
        budget_audit = budget_audit.rename(columns={"Requested_Budget": "Chosen_Budget"})
        budget_audit["Feature_Name"] = "__feature_budget__"
        budget_audit["Fold"] = budget_audit["Outer_Fold"]
        budget_audit["Stage"] = "inner_budget_tuning"
        all_feature_reports.append(budget_audit)
    summary = {
        "status": "ok" if specs else "no_targets",
        "development_groups": int(health.iloc[development][group_col].nunique()),
        "final_holdout_groups": int(health.iloc[final_holdout][group_col].nunique()),
        "final_holdout_policy": "untouched_chronological_last_20_percent_one_frozen_batch",
        "tasks": task_summaries,
        "assumptions": assumptions,
        "behavior_to_health_fold_integration": "pending_unless_past_only_behavior_features_preexist",
    }
    return (
        pd.concat(all_feature_reports, ignore_index=True, sort=False) if all_feature_reports else pd.DataFrame(),
        pd.concat(all_predictions, ignore_index=True, sort=False) if all_predictions else pd.DataFrame(),
        pd.concat(all_benchmarks, ignore_index=True, sort=False) if all_benchmarks else pd.DataFrame(),
        pd.concat(all_explanations, ignore_index=True, sort=False) if all_explanations else pd.DataFrame(),
        summary,
    )


def run_classical_analysis_protocol(
    trip_features: pd.DataFrame | str | Path,
    health_windows: pd.DataFrame | str | Path,
    output_dir: str | Path,
    random_state: int = 42,
) -> dict[str, Path]:
    """Execute the full development/final-holdout classical protocol."""

    trips = _read_table(trip_features)
    health = _read_table(health_windows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    behavior, behavior_benchmarks, behavior_summary = _behavior_protocol(
        trips, random_state=random_state
    )
    features, predictions, soh_benchmarks, explanations, soh_summary = _soh_protocol(
        health, random_state=random_state
    )
    benchmarks = pd.concat(
        [behavior_benchmarks, soh_benchmarks], ignore_index=True, sort=False
    )
    artifacts = {
        "feature_selection": features,
        "behavior_clusters": behavior,
        "soh_predictions": predictions,
        "benchmarks": benchmarks,
        "explainability": explanations,
    }
    required_empty_columns = {
        "feature_selection": ["Feature_Name", "Final_Status", "Rejection_Reason", "Task", "Fold", "Stage"],
        "behavior_clusters": ["natural_cluster_id", "aggression_score", "behavior_level", "split_role"],
        "soh_predictions": ["Row_Index", "Task", "Model", "Fold", "split_role", "target_observed", "target_predicted"],
        "benchmarks": ["Task", "Model", "Feature_Set", "Fold", "split_role", "Status"],
        "explainability": ["Task", "Model", "Feature_Name", "Explanation_Type", "Importance", "Status"],
    }
    paths: dict[str, Path] = {}
    for name, table in artifacts.items():
        if table.empty:
            table = pd.DataFrame(columns=required_empty_columns[name])
        path = destination / f"{name}.csv"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            table.to_csv(path, index=False)
        paths[name] = path
    summary = {
        "framework": "DB-PatchKAN",
        "protocol_version": "classical-v2-leakage-safe",
        "random_state": int(random_state),
        "trip_rows": int(len(trips)),
        "health_rows": int(len(health)),
        "behavior": behavior_summary,
        "soh": soh_summary,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    summary_path = destination / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    paths["analysis_summary"] = summary_path
    return paths
