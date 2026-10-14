from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dbpatchkan.analysis import (
    BehaviorLevelMapper,
    PurgedExpandingWindowSplit,
    aggression_score_from_clusters,
    apply_semantic_exclusions,
    assert_no_group_leakage,
    build_classification_baselines,
    build_regression_baselines,
    classification_metrics,
    consensus_feature_selection,
    discover_behavior_archetypes,
    elastic_net_stability_selection,
    fit_evaluate_classifiers,
    fit_evaluate_regressors,
    grouped_permutation_importance,
    mrmr_ranking,
    redundancy_filter,
    regression_metrics,
    run_classical_analysis,
    purge_training_groups,
    temporal_group_holdout,
)


def test_semantic_redundancy_and_supervised_selection_are_auditable() -> None:
    rng = np.random.default_rng(7)
    n = 60
    signal = rng.normal(size=n)
    target = 3 * signal + rng.normal(scale=0.1, size=n)
    frame = pd.DataFrame(
        {
            "index": np.arange(n),
            "Timestamp": np.arange(n) * 1_000_000,
            "SOH": target,
            "kWh_Remaining": target / 10,
            "soh_change_pp": target / 100,
            "loss_rate_pp_per_1000km": target / 200,
            "soh_variation_flag": False,
            "signal": signal,
            "signal_copy": signal + rng.normal(scale=1e-5, size=n),
            "noise": rng.normal(size=n),
            "mostly_missing": [np.nan] * 40 + list(rng.normal(size=20)),
            "constant": 1.0,
        }
    )
    clean, semantic = apply_semantic_exclusions(frame)
    assert {"index", "Timestamp", "SOH", "kWh_Remaining"}.isdisjoint(clean.columns)
    assert set(semantic["Feature_Name"]) >= {
        "SOH",
        "kWh_Remaining",
        "soh_change_pp",
        "loss_rate_pp_per_1000km",
        "soh_variation_flag",
    }

    screened = redundancy_filter(clean, feature_priority=["signal"])
    assert "signal" in screened.selected_features
    assert "signal_copy" not in screened.selected_features
    assert "mostly_missing" not in screened.selected_features
    assert "constant" not in screened.selected_features
    assert screened.report.loc[
        screened.report.Feature_Name == "signal_copy", "Rejection_Reason"
    ].iloc[0] == "correlated_redundancy"

    ranking = mrmr_ranking(clean[screened.selected_features], target, max_features=2)
    assert ranking.iloc[0]["Feature_Name"] == "signal"
    stability = elastic_net_stability_selection(
        clean[screened.selected_features],
        target,
        groups=np.repeat(np.arange(12), 5),
        n_bootstraps=12,
        alpha=0.05,
    )
    signal_frequency = stability.loc[
        stability.Feature_Name == "signal", "ElasticNet_Selection_Frequency"
    ].iloc[0]
    assert signal_frequency >= 0.8


def test_consensus_requires_stability_positive_permutation_and_two_selectors() -> None:
    records = pd.DataFrame(
        {
            "Fold": np.tile(np.arange(4), 2),
            "Feature_Name": np.repeat(["good", "unstable"], 4),
            "mRMR_Selected": [True] * 4 + [True, False, False, False],
            "ElasticNet_Selected": [True] * 4 + [False] * 4,
            "Permutation_Importance": [0.2] * 4 + [-0.1] * 4,
        }
    )
    result = consensus_feature_selection(records)
    status = result.set_index("Feature_Name")["Final_Status"]
    assert status["good"] == "selected"
    assert status["unstable"] == "rejected"


def test_behavior_discovery_and_training_only_level_thresholds() -> None:
    rng = np.random.default_rng(9)
    centers = np.repeat([0.2, 1.0, 2.0], 25)
    trips = pd.DataFrame(
        {
            "accel_pos_rms": centers + rng.normal(0, 0.04, len(centers)),
            "jerk_rms": centers * 0.8 + rng.normal(0, 0.04, len(centers)),
            "pedal_p95": centers * 0.4 + rng.normal(0, 0.02, len(centers)),
            "power_dis_p95_kw": centers * 10 + rng.normal(0, 0.3, len(centers)),
        }
    )
    result = discover_behavior_archetypes(
        trips,
        cluster_range=range(2, 5),
        n_init=3,
        bootstrap_iterations=5,
        random_state=9,
    )
    assert result.labels.shape == (75,)
    assert result.probabilities.shape == (75, result.selected_k)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    scores, contributions = aggression_score_from_clusters(
        trips, result.labels, result.probabilities
    )
    mapper = BehaviorLevelMapper().fit(scores[:60])
    levels = mapper.transform(scores)
    assert set(levels) == {"Eco", "Balanced", "Aggressive"}
    assert "cluster_mapping" in contributions.attrs


def test_temporal_splits_keep_complete_groups_and_holdout_is_last() -> None:
    base = pd.Timestamp("2024-01-01", tz="UTC")
    frame = pd.DataFrame(
        {
            "trip_key": np.repeat([f"T{i:02d}" for i in range(12)], 2),
            "datetime_utc": [base + pd.Timedelta(days=i) for i in range(12) for _ in range(2)],
            "odometer_clean_km": np.repeat(np.arange(12) * 100.0, 2),
        }
    )
    development, holdout = temporal_group_holdout(
        frame, time_col="datetime_utc", group_col="trip_key", test_fraction=0.2
    )
    assert set(frame.iloc[holdout].trip_key) == {"T09", "T10", "T11"}
    assert set(frame.iloc[development].trip_key).isdisjoint(frame.iloc[holdout].trip_key)

    splitter = PurgedExpandingWindowSplit(
        3, time_col="datetime_utc", group_col="trip_key"
    )
    for train, validation in splitter.split(frame.iloc[development]):
        local = frame.iloc[development].reset_index(drop=True)
        assert_no_group_leakage(
            local, train, validation, group_col="trip_key", time_col="datetime_utc"
        )

    purged = purge_training_groups(
        frame,
        np.flatnonzero(frame.trip_key.isin(["T00", "T01", "T02", "T03"])),
        np.flatnonzero(frame.trip_key == "T05"),
        time_col="datetime_utc",
        group_col="trip_key",
        purge_days=2,
    )
    assert set(frame.iloc[purged].trip_key) == {"T00", "T01", "T02"}


def test_baselines_metrics_and_grouped_permutation() -> None:
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"signal": np.arange(40, dtype=float), "noise": rng.normal(size=40)})
    y = 95 - 0.1 * X["signal"].to_numpy() + rng.normal(scale=0.02, size=40)
    models = build_regression_baselines(include_xgboost=False, include_ebm=False)
    models = {key: value for key, value in models.items() if key in {"Linear", "RandomForest"}}
    evaluation = fit_evaluate_regressors(X.iloc[:30], y[:30], X.iloc[30:], y[30:], models=models)
    assert set(evaluation.metrics["Status"]) == {"ok"}
    assert regression_metrics(y[30:], evaluation.predictions.query("Model == 'Linear'").soh_pred_pct)[
        "MAE_pp"
    ] < 0.1

    fitted = evaluation.fitted_models["RandomForest"]
    importance = grouped_permutation_importance(
        fitted,
        X.iloc[30:],
        y[30:],
        feature_groups={"driver_signal": ["signal"], "noise": ["noise"]},
        scoring="neg_mean_absolute_error",
        n_repeats=5,
    )
    assert set(importance.Feature_Group) == {"driver_signal", "noise"}

    labels = np.array(["Eco"] * 15 + ["Aggressive"] * 15)
    classifiers = build_classification_baselines(include_xgboost=False, include_ebm=False)
    classifiers = {"ElasticNetLogistic": classifiers["ElasticNetLogistic"]}
    classified = fit_evaluate_classifiers(
        X.iloc[:24], labels[:24], X.iloc[24:30], labels[24:], models=classifiers
    )
    assert not classified.metrics.empty
    direct = classification_metrics([0, 1, 1], [0, 1, 0], probabilities=np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]))
    assert 0 <= direct["Macro_F1"] <= 1
    assert "Brier_Score" in direct


def test_runner_always_writes_contract_artifacts_for_sparse_data(tmp_path) -> None:
    trips = pd.DataFrame(
        {
            "Trip": [1, 2, 3],
            "trip_start_utc": pd.date_range("2025-01-01", periods=3, tz="UTC"),
            "accel_pos_rms": [0.1, 0.2, 0.3],
        }
    )
    health = pd.DataFrame(
        {
            "health_window_id": [1, 2, 3],
            "anchor_time_utc": pd.date_range("2025-01-01", periods=3, tz="UTC"),
            "soh_target_trend_pct": [95.0, 94.9, 94.8],
        }
    )
    paths = run_classical_analysis(trips, health, tmp_path)
    expected = {
        "feature_selection",
        "behavior_clusters",
        "soh_predictions",
        "benchmarks",
        "explainability",
        "analysis_summary",
    }
    assert set(paths) == expected
    assert all(path.exists() for path in paths.values())
    summary = json.loads(paths["analysis_summary"].read_text(encoding="utf-8"))
    assert summary["trip_rows"] == 3
