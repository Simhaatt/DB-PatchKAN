"""Classical analysis, validation, and explainability public API."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .baselines import (
    BaselineEvaluation,
    PersistenceRegressor,
    build_classification_baselines,
    build_regression_baselines,
    classification_metrics,
    fit_evaluate_classifiers,
    fit_evaluate_regressors,
    regression_metrics,
    soh_ablation_feature_sets,
)
from .clustering import (
    BehaviorClusteringResult,
    BehaviorLevelMapper,
    BehaviorLevelThresholds,
    aggression_score_from_clusters,
    compare_clusterers,
    discover_behavior_archetypes,
    eligible_driver_ids,
    infer_behavior_features,
    label_behavior_levels,
)
from .explainability import (
    ShapResult,
    ebm_global_terms,
    ebm_response_functions,
    grouped_permutation_importance,
    occlusion_importance,
    robust_importance_summary,
    tree_shap_summary,
)
from .deep_explainability import (
    OUTPUT_NAMES,
    aggregate_response_curves,
    channel_gate_table,
    concept_integrated_gradients,
    concept_response_curves,
    kan_edge_strength_table,
    kan_path_importance_table,
    trip_predictions,
    validation_occlusion_tables,
)
from .feature_selection import (
    DEFAULT_SEMANTIC_EXCLUSIONS,
    FoldSafeFeatureSelector,
    RedundancyResult,
    FeatureBudgetResult,
    apply_semantic_exclusions,
    consensus_feature_selection,
    default_permutation_estimator,
    elastic_net_stability_selection,
    heldout_permutation_importance,
    identify_semantic_exclusions,
    mrmr_ranking,
    redundancy_filter,
    tune_feature_budget,
    variance_inflation_factors,
)
from .validation import (
    FoldIndices,
    PurgedExpandingWindowSplit,
    TemporalValidationPlan,
    assert_no_group_leakage,
    block_bootstrap_confidence_interval,
    build_temporal_validation_plan,
    chronological_group_order,
    paired_block_bootstrap_difference,
    purge_training_groups,
    split_conformal_interval,
    temporal_group_holdout,
)


def _read_table(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    suffix = path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def _first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in frame), None)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _empty_artifacts() -> dict[str, pd.DataFrame]:
    return {
        "feature_selection": pd.DataFrame(
            columns=["Feature_Name", "Final_Status", "Rejection_Reason"]
        ),
        "behavior_clusters": pd.DataFrame(
            columns=["natural_cluster_id", "natural_cluster_probability", "aggression_score", "behavior_level"]
        ),
        "soh_predictions": pd.DataFrame(
            columns=["Row_Index", "Model", "soh_observed_pct", "soh_pred_pct", "residual_pp", "absolute_error_pp"]
        ),
        "benchmarks": pd.DataFrame(columns=["Task", "Model", "Status"]),
        "explainability": pd.DataFrame(
            columns=["Explanation_Type", "Feature_Name", "Importance", "Status"]
        ),
    }


def _behavior_analysis(
    trips: pd.DataFrame,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features = infer_behavior_features(trips)
    empty = pd.DataFrame(
        columns=["natural_cluster_id", "natural_cluster_probability", "aggression_score", "behavior_level"]
    )
    if len(trips) < 4 or len(features) < 1:
        return empty, pd.DataFrame(), {"status": "insufficient_data", "behavior_features": features}

    time_col = _first_existing(trips, ["trip_start_utc", "window_start_utc", "datetime_utc", "Timestamp"])
    group_col = _first_existing(trips, ["trip_key", "Trip", "window_id"])
    if time_col and group_col and trips[group_col].nunique(dropna=False) >= 4:
        try:
            train_index, test_index = temporal_group_holdout(
                trips, time_col=time_col, group_col=group_col, test_fraction=0.20
            )
        except (ValueError, KeyError):
            train_index, test_index = np.arange(len(trips)), np.array([], dtype=int)
    else:
        train_index, test_index = np.arange(len(trips)), np.array([], dtype=int)

    try:
        discovery = discover_behavior_archetypes(
            trips.iloc[train_index],
            feature_names=features,
            bootstrap_iterations=20,
            random_state=random_state,
        )
    except ValueError as error:
        return empty, pd.DataFrame(), {
            "status": "clustering_failed",
            "message": str(error),
            "behavior_features": features,
        }
    all_labels, all_probabilities = discovery.predict(trips)
    _, train_contributions = aggression_score_from_clusters(
        trips.iloc[train_index],
        discovery.labels,
        discovery.probabilities,
        feature_names=features,
    )
    mapping = train_contributions.attrs["cluster_mapping"].sort_values("natural_cluster_id")
    intensities = mapping["cluster_aggression_intensity"].to_numpy(dtype=float)
    all_scores = all_probabilities @ intensities
    mapper = BehaviorLevelMapper().fit(all_scores[train_index])
    levels = mapper.transform(all_scores)

    identifiers = [
        column
        for column in ["dataset_id", "vehicle_id", "trip_key", "Trip", "Driver_ID", "trip_start_utc"]
        if column in trips
    ]
    output = trips[identifiers].copy()
    output["natural_cluster_id"] = all_labels
    output["natural_cluster_probability"] = all_probabilities.max(axis=1)
    for cluster in range(all_probabilities.shape[1]):
        output[f"cluster_probability_C{cluster:02d}"] = all_probabilities[:, cluster]
    output["aggression_score"] = all_scores
    output["behavior_level"] = levels
    output["split_role"] = "development"
    if test_index.size:
        output.iloc[test_index, output.columns.get_loc("split_role")] = "holdout"

    benchmarks = pd.DataFrame()
    if test_index.size and np.unique(levels[train_index]).size >= 2:
        evaluation = fit_evaluate_classifiers(
            trips.iloc[train_index][features],
            levels[train_index],
            trips.iloc[test_index][features],
            levels[test_index],
            random_state=random_state,
        )
        benchmarks = evaluation.metrics.copy()
        if not benchmarks.empty:
            benchmarks.insert(0, "Task", "behavior_level")
    summary = {
        "status": "ok",
        "behavior_features": features,
        "selected_k": discovery.selected_k,
        "covariance_type": discovery.covariance_type,
        "bootstrap_ari": discovery.bootstrap_ari,
        "exploratory": discovery.exploratory,
        "thresholds": {
            "eco_upper": mapper.thresholds_.lower,
            "aggressive_lower": mapper.thresholds_.upper,
        },
        "candidate_metrics": discovery.candidate_metrics.to_dict(orient="records"),
        "comparison_metrics": discovery.comparison_metrics.to_dict(orient="records"),
    }
    return output, benchmarks, summary


def _soh_analysis(
    health: pd.DataFrame,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target = _first_existing(
        health,
        ["soh_target_trend_pct", "soh_target_raw_pct", "soh_observed_pct", "SOH"],
    )
    if target is None or len(health) < 4:
        return (
            pd.DataFrame(columns=["Feature_Name", "Final_Status", "Rejection_Reason"]),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"status": "insufficient_data", "target": target},
        )
    y = pd.to_numeric(health[target], errors="coerce")
    valid_positions = np.flatnonzero(y.notna().to_numpy())
    health = health.iloc[valid_positions].copy()
    y = y.iloc[valid_positions].to_numpy(dtype=float)
    if len(health) < 4 or np.unique(y).size < 2:
        return (
            pd.DataFrame(columns=["Feature_Name", "Final_Status", "Rejection_Reason"]),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"status": "insufficient_target_variation", "target": target},
        )

    time_col = _first_existing(health, ["anchor_time_utc", "trip_start_utc", "datetime_utc", "Timestamp"])
    group_col = _first_existing(health, ["health_window_id", "trip_key", "Trip"])
    if group_col is None:
        group_col = "__analysis_group__"
        health[group_col] = np.arange(len(health))
    if time_col is None:
        time_col = "__analysis_time__"
        health[time_col] = pd.Timestamp("2000-01-01", tz="UTC") + pd.to_timedelta(
            np.arange(len(health)), unit="D"
        )
    try:
        train_index, test_index = temporal_group_holdout(
            health, time_col=time_col, group_col=group_col, test_fraction=0.20
        )
    except ValueError:
        split = max(2, len(health) - max(1, int(np.ceil(0.2 * len(health)))))
        train_index, test_index = np.arange(split), np.arange(split, len(health))

    excluded, semantic_audit = apply_semantic_exclusions(
        health,
        target_columns=[target],
        extra_exclusions=[time_col, group_col, "dataset_id", "vehicle_id", "qc_grade", "target_quality_flag"],
    )
    numeric = excluded.apply(pd.to_numeric, errors="coerce")
    train_frame = numeric.iloc[train_index]
    screen = redundancy_filter(train_frame)
    selected = screen.selected_features
    if not selected:
        feature_report = pd.concat([semantic_audit, screen.report], ignore_index=True)
        return feature_report, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "no_features_after_screening",
            "target": target,
        }

    ranking = mrmr_ranking(
        train_frame[selected], y[train_index], max_features=min(40, len(selected)), random_state=random_state
    )
    stability = elastic_net_stability_selection(
        train_frame[selected],
        y[train_index],
        groups=health.iloc[train_index][group_col].to_numpy(),
        n_bootstraps=100,
        random_state=random_state,
    )
    ranked = ranking["Feature_Name"].tolist()
    stable = stability.loc[
        stability["ElasticNet_Selection_Frequency"] >= 0.70, "Feature_Name"
    ].tolist()
    model_features = [feature for feature in ranked if feature in set(stable)][:20]
    if not model_features:
        model_features = ranked[: min(20, len(ranked))] or selected[: min(20, len(selected))]

    evaluation = fit_evaluate_regressors(
        numeric.iloc[train_index][model_features],
        y[train_index],
        numeric.iloc[test_index][model_features],
        y[test_index],
        random_state=random_state,
    )
    benchmarks = evaluation.metrics.copy()
    if not benchmarks.empty:
        benchmarks.insert(0, "Task", target)
        benchmarks["Feature_Set"] = "fold_safe_consensus"
    predictions = evaluation.predictions.copy()
    if not predictions.empty:
        predictions["Target"] = target
        predictions["split_role"] = "holdout"

    permutation = pd.DataFrame(
        columns=["Feature_Name", "Permutation_Importance", "Permutation_Importance_Std"]
    )
    explanation_rows: list[pd.DataFrame] = []
    preferred = next(
        (name for name in ["XGBoost", "EBM", "RandomForest"] if name in evaluation.fitted_models),
        None,
    )
    if preferred is not None and len(test_index):
        estimator = evaluation.fitted_models[preferred]
        try:
            permutation = heldout_permutation_importance(
                estimator,
                numeric.iloc[test_index][model_features],
                y[test_index],
                scoring="neg_mean_absolute_error",
                random_state=random_state,
            )
            item = permutation.rename(columns={"Permutation_Importance": "Importance"})
            item["Explanation_Type"] = "heldout_permutation"
            item["Model"] = preferred
            item["Status"] = "ok"
            explanation_rows.append(item)
        except ValueError:
            pass
        shap_result = tree_shap_summary(
            estimator, numeric.iloc[test_index][model_features], random_state=random_state
        )
        shap_table = shap_result.importance.rename(columns={"SHAP_Importance": "Importance"})
        shap_table["Explanation_Type"] = "TreeSHAP"
        shap_table["Model"] = preferred
        shap_table["Status"] = shap_result.status
        explanation_rows.append(shap_table)
    explanations = pd.concat(explanation_rows, ignore_index=True) if explanation_rows else pd.DataFrame(
        columns=["Explanation_Type", "Feature_Name", "Importance", "Model", "Status"]
    )

    report = screen.report.merge(ranking, on="Feature_Name", how="left").merge(
        stability, on="Feature_Name", how="left"
    ).merge(permutation, on="Feature_Name", how="left")
    report["Final_Status"] = np.where(report["Feature_Name"].isin(model_features), "selected", "rejected")
    report["Rejection_Reason"] = np.where(
        report["Feature_Name"].isin(model_features),
        "",
        report["Rejection_Reason"].replace("", "not_in_consensus_budget"),
    )
    report["Consensus_Rank"] = report["Feature_Name"].map(
        {feature: rank for rank, feature in enumerate(model_features, start=1)}
    )
    feature_report = pd.concat([semantic_audit, report], ignore_index=True, sort=False)
    summary = {
        "status": "ok",
        "target": target,
        "train_rows": len(train_index),
        "holdout_rows": len(test_index),
        "model_features": model_features,
        "successful_models": list(evaluation.fitted_models),
    }
    return feature_report, predictions, benchmarks, explanations, summary


def run_classical_analysis(
    trip_features: pd.DataFrame | str | Path,
    health_windows: pd.DataFrame | str | Path,
    output_dir: str | Path,
    random_state: int = 42,
) -> dict[str, Path]:
    """Execute the leakage-safe expanding-fold/final-holdout protocol."""

    from .protocol import run_classical_analysis_protocol

    return run_classical_analysis_protocol(
        trip_features, health_windows, output_dir, random_state=random_state
    )


__all__ = [
    "DEFAULT_SEMANTIC_EXCLUSIONS",
    "BaselineEvaluation",
    "BehaviorClusteringResult",
    "BehaviorLevelMapper",
    "BehaviorLevelThresholds",
    "OUTPUT_NAMES",
    "FoldIndices",
    "FeatureBudgetResult",
    "FoldSafeFeatureSelector",
    "PersistenceRegressor",
    "PurgedExpandingWindowSplit",
    "RedundancyResult",
    "ShapResult",
    "TemporalValidationPlan",
    "aggression_score_from_clusters",
    "aggregate_response_curves",
    "apply_semantic_exclusions",
    "assert_no_group_leakage",
    "block_bootstrap_confidence_interval",
    "build_classification_baselines",
    "build_regression_baselines",
    "build_temporal_validation_plan",
    "chronological_group_order",
    "classification_metrics",
    "channel_gate_table",
    "compare_clusterers",
    "consensus_feature_selection",
    "concept_integrated_gradients",
    "concept_response_curves",
    "discover_behavior_archetypes",
    "ebm_global_terms",
    "ebm_response_functions",
    "elastic_net_stability_selection",
    "eligible_driver_ids",
    "fit_evaluate_classifiers",
    "fit_evaluate_regressors",
    "grouped_permutation_importance",
    "heldout_permutation_importance",
    "identify_semantic_exclusions",
    "infer_behavior_features",
    "label_behavior_levels",
    "kan_edge_strength_table",
    "kan_path_importance_table",
    "mrmr_ranking",
    "occlusion_importance",
    "paired_block_bootstrap_difference",
    "purge_training_groups",
    "redundancy_filter",
    "regression_metrics",
    "robust_importance_summary",
    "run_classical_analysis",
    "soh_ablation_feature_sets",
    "split_conformal_interval",
    "temporal_group_holdout",
    "tree_shap_summary",
    "trip_predictions",
    "tune_feature_budget",
    "variance_inflation_factors",
    "validation_occlusion_tables",
]
