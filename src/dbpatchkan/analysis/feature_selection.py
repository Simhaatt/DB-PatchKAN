"""Leakage-aware, fold-local feature screening and selection utilities.

The functions in this module deliberately operate on *training data only*.  They
do not know about a test fold and therefore cannot accidentally fit an imputer,
correlation matrix, or selector on held-out observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import check_scoring
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_SEMANTIC_EXCLUSIONS = frozenset(
    {
        "index",
        "timestamp",
        "trip",
        "trip_key",
        "driver_id",
        "soh",
        "soh_target_raw_pct",
        "soh_target_trend_pct",
        "soh_observed_pct",
        "soh_pred_pct",
        "soh_pred_lower90",
        "soh_pred_upper90",
        "last_known_soh",
        "last_known_soh_pct",
        "lagged_soh",
        "soh_change_pp",
        "loss_rate_pp_per_1000km",
        "soh_variation_flag",
        "target_soh_30d_pct",
        "target_loss_30d_pp",
        "target_soh_500km_pct",
        "target_loss_500km_pp",
        "target_loss_rate_pp_per_1000km",
        "kwh_remaining",
        "horsepower",
        "instant_kw",
        "pack_avg_c_temp",
    }
)


def _as_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric, finite copy while retaining the original columns/order."""

    converted = {
        column: pd.to_numeric(frame[column], errors="coerce") for column in frame.columns
    }
    return pd.DataFrame(converted, index=frame.index).replace([np.inf, -np.inf], np.nan)


def identify_semantic_exclusions(
    columns: Iterable[str],
    *,
    extra_exclusions: Iterable[str] = (),
    target_columns: Iterable[str] = (),
) -> dict[str, str]:
    """Identify identifier, target, and deterministic-leakage columns.

    Matching is case-insensitive.  ``last_known_soh`` is intentionally not in the
    default set because it can be valid for a forecasting task when calculated
    strictly from the past.
    """

    base = set(DEFAULT_SEMANTIC_EXCLUSIONS)
    base.update(str(value).casefold() for value in extra_exclusions)
    targets = {str(value).casefold() for value in target_columns}
    result: dict[str, str] = {}
    for column in columns:
        folded = str(column).casefold()
        if folded in targets:
            result[str(column)] = "target"
        elif folded in base or folded.startswith("target_") or folded.startswith("soh_target"):
            if folded in {"index", "timestamp", "trip", "trip_key", "driver_id"}:
                result[str(column)] = "identifier_or_group"
            elif folded in {"kwh_remaining", "horsepower", "instant_kw", "pack_avg_c_temp"}:
                result[str(column)] = "leakage_or_redundant_source"
            else:
                result[str(column)] = "current_or_future_target"
    return result


def apply_semantic_exclusions(
    frame: pd.DataFrame,
    *,
    extra_exclusions: Iterable[str] = (),
    target_columns: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove semantic leakage fields and return a transparent audit table."""

    excluded = identify_semantic_exclusions(
        frame.columns,
        extra_exclusions=extra_exclusions,
        target_columns=target_columns,
    )
    kept = [column for column in frame.columns if str(column) not in excluded]
    audit = pd.DataFrame(
        [
            {"Feature_Name": column, "Final_Status": "rejected", "Rejection_Reason": reason}
            for column, reason in excluded.items()
        ],
        columns=["Feature_Name", "Final_Status", "Rejection_Reason"],
    )
    return frame.loc[:, kept].copy(), audit


@dataclass
class RedundancyResult:
    """Output of missingness, constancy, and correlation screening."""

    selected_features: list[str]
    rejected_features: list[str]
    report: pd.DataFrame
    correlation_matrix: pd.DataFrame

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.reindex(columns=self.selected_features)


@dataclass
class FeatureBudgetResult:
    selected_budget: int
    summary: pd.DataFrame
    fold_results: pd.DataFrame


def variance_inflation_factors(frame: pd.DataFrame) -> pd.Series:
    """Return VIF diagnostics using a stable pseudo-inverse correlation matrix."""

    numeric = _as_numeric_frame(frame)
    if numeric.shape[1] == 0:
        return pd.Series(dtype=float, name="VIF")
    if numeric.shape[1] == 1:
        return pd.Series([1.0], index=numeric.columns, name="VIF")
    values = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(numeric)
    scale = np.nanstd(values, axis=0)
    valid = scale > 1e-12
    result = np.full(numeric.shape[1], np.inf, dtype=float)
    if valid.sum() == 1:
        result[valid] = 1.0
    elif valid.sum() > 1:
        standardized = (values[:, valid] - np.mean(values[:, valid], axis=0)) / scale[valid]
        correlation = np.corrcoef(standardized, rowvar=False)
        try:
            inverse = np.linalg.inv(correlation)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(correlation)
        result[valid] = np.maximum(1.0, np.diag(inverse))
    return pd.Series(result, index=numeric.columns, name="VIF")


def _connected_components(adjacency: Mapping[str, set[str]], order: Sequence[str]) -> list[list[str]]:
    seen: set[str] = set()
    components: list[list[str]] = []
    for root in order:
        if root in seen:
            continue
        stack = [root]
        component: list[str] = []
        seen.add(root)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adjacency.get(node, set()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component, key=order.index))
    return components


def redundancy_filter(
    frame: pd.DataFrame,
    *,
    protected_features: Iterable[str] = (),
    missing_threshold: float = 0.40,
    dominant_threshold: float = 0.995,
    correlation_threshold: float = 0.90,
    feature_priority: Sequence[str] = (),
) -> RedundancyResult:
    """Screen features using only the supplied (normally training-fold) rows.

    A correlation group is a connected component, not merely a pair.  This makes
    the outcome deterministic in cases such as A~B and B~C.  The representative
    is chosen by explicit priority, then protected status, missingness, and source
    column order.
    """

    if not 0 <= missing_threshold <= 1:
        raise ValueError("missing_threshold must be in [0, 1]")
    if not 0 <= dominant_threshold <= 1:
        raise ValueError("dominant_threshold must be in [0, 1]")
    if not 0 <= correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in [0, 1]")

    numeric = _as_numeric_frame(frame)
    protected = {str(value) for value in protected_features}
    priority = {str(feature): rank for rank, feature in enumerate(feature_priority)}
    original_order = {str(feature): rank for rank, feature in enumerate(numeric.columns)}
    missing = numeric.isna().mean() if len(numeric) else pd.Series(1.0, index=numeric.columns)
    dominant: dict[str, float] = {}
    for feature in numeric.columns:
        counts = numeric[feature].dropna().value_counts(normalize=True)
        dominant[str(feature)] = float(counts.iloc[0]) if len(counts) else 1.0

    rejected_reason: dict[str, str] = {}
    candidates: list[str] = []
    for feature in map(str, numeric.columns):
        if feature not in protected and float(missing[feature]) > missing_threshold:
            rejected_reason[feature] = "missingness"
        elif feature not in protected and dominant[feature] > dominant_threshold:
            rejected_reason[feature] = "near_constant"
        else:
            candidates.append(feature)

    if candidates:
        correlation = numeric[candidates].corr(method="spearman", min_periods=2).abs()
    else:
        correlation = pd.DataFrame(index=[], columns=[], dtype=float)
    adjacency = {feature: set() for feature in candidates}
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            value = correlation.at[left, right]
            if pd.notna(value) and float(value) >= correlation_threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)

    components = _connected_components(adjacency, candidates)
    correlation_groups: dict[str, str] = {}
    representatives: dict[str, str] = {}
    selected: list[str] = []
    for group_number, component in enumerate(components, start=1):
        group_id = f"CG{group_number:03d}" if len(component) > 1 else ""
        representative = min(
            component,
            key=lambda feature: (
                priority.get(feature, len(priority) + 1),
                0 if feature in protected else 1,
                float(missing[feature]),
                -dominant[feature],
                original_order[feature],
            ),
        )
        selected.append(representative)
        for feature in component:
            correlation_groups[feature] = group_id
            representatives[feature] = representative
            if feature != representative:
                rejected_reason[feature] = "correlated_redundancy"

    selected = [feature for feature in candidates if feature in set(selected)]
    vif = variance_inflation_factors(numeric[selected]) if selected else pd.Series(dtype=float)
    rows: list[dict[str, object]] = []
    for feature in map(str, numeric.columns):
        is_selected = feature in selected
        rows.append(
            {
                "Feature_Name": feature,
                "Protected_Feature": feature in protected,
                "Missing_Pct": 100.0 * float(missing[feature]),
                "Dominant_Value_Pct": 100.0 * dominant[feature],
                "Correlation_Group": correlation_groups.get(feature, ""),
                "Representative_Feature": representatives.get(feature, ""),
                "VIF": float(vif.get(feature, np.nan)),
                "VIF_Warning": bool(vif.get(feature, 0.0) > 10),
                "Final_Status": "selected" if is_selected else "rejected",
                "Rejection_Reason": "" if is_selected else rejected_reason.get(feature, "screened"),
            }
        )
    report = pd.DataFrame(rows)
    return RedundancyResult(
        selected_features=selected,
        rejected_features=[feature for feature in numeric.columns if feature not in selected],
        report=report,
        correlation_matrix=correlation,
    )


def _prepare_supervised(
    frame: pd.DataFrame, target: Sequence[object] | pd.Series | np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    numeric = _as_numeric_frame(frame)
    y = np.asarray(target)
    if y.ndim != 1 or len(y) != len(numeric):
        raise ValueError("target must be one-dimensional and aligned with frame")
    target_valid = ~pd.isna(y)
    numeric = numeric.loc[target_valid]
    y = y[target_valid]
    if numeric.empty:
        return numeric, y
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    values = imputer.fit_transform(numeric)
    return pd.DataFrame(values, columns=numeric.columns, index=numeric.index), y


def mrmr_ranking(
    frame: pd.DataFrame,
    target: Sequence[object] | pd.Series | np.ndarray,
    *,
    max_features: int = 40,
    task: str = "regression",
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a deterministic mutual-information/minimum-redundancy ranking.

    This is a compact mRMR-style implementation: relevance is mutual information
    and redundancy is mean absolute Spearman correlation with already selected
    variables.  Both terms are scaled to [0, 1] before subtraction.
    """

    X, y = _prepare_supervised(frame, target)
    columns = list(map(str, X.columns))
    output_columns = ["Feature_Name", "mRMR_Rank", "MI_Relevance", "Mean_Redundancy", "mRMR_Score"]
    if not columns or len(X) < 3 or np.unique(y).size < 2:
        return pd.DataFrame(columns=output_columns)
    max_features = max(0, min(int(max_features), len(columns)))
    if max_features == 0:
        return pd.DataFrame(columns=output_columns)
    neighbours = max(1, min(3, len(X) - 1))
    if task == "regression":
        relevance = mutual_info_regression(
            X, y.astype(float), n_neighbors=neighbours, random_state=random_state
        )
    elif task == "classification":
        relevance = mutual_info_classif(
            X, y, n_neighbors=neighbours, random_state=random_state
        )
    else:
        raise ValueError("task must be 'regression' or 'classification'")
    relevance = np.nan_to_num(np.asarray(relevance, dtype=float), nan=0.0)
    maximum = float(relevance.max(initial=0.0))
    relevance_scaled = relevance / maximum if maximum > 0 else relevance
    correlations = X.corr(method="spearman").abs().fillna(0.0)

    remaining = list(range(len(columns)))
    selected: list[int] = []
    rows: list[dict[str, object]] = []
    for rank in range(1, max_features + 1):
        best_index = -1
        best_tuple = (-np.inf, -np.inf, 0)
        best_redundancy = 0.0
        for index in remaining:
            redundancy = (
                float(correlations.iloc[index, selected].mean()) if selected else 0.0
            )
            score = float(relevance_scaled[index]) - redundancy
            # The negative index ensures original column order breaks exact ties.
            candidate = (score, float(relevance_scaled[index]), -index)
            if candidate > best_tuple:
                best_tuple = candidate
                best_index = index
                best_redundancy = redundancy
        selected.append(best_index)
        remaining.remove(best_index)
        rows.append(
            {
                "Feature_Name": columns[best_index],
                "mRMR_Rank": rank,
                "MI_Relevance": float(relevance[best_index]),
                "Mean_Redundancy": best_redundancy,
                "mRMR_Score": best_tuple[0],
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def _bootstrap_positions(
    n_rows: int, groups: Sequence[object] | np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    if groups is None:
        return rng.integers(0, n_rows, size=n_rows)
    group_values = np.asarray(groups)
    if len(group_values) != n_rows:
        raise ValueError("groups must be aligned with frame")
    unique_groups = pd.unique(group_values)
    sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    blocks = [np.flatnonzero(group_values == group) for group in sampled]
    return np.concatenate(blocks) if blocks else np.array([], dtype=int)


def elastic_net_stability_selection(
    frame: pd.DataFrame,
    target: Sequence[object] | pd.Series | np.ndarray,
    *,
    groups: Sequence[object] | np.ndarray | None = None,
    n_bootstraps: int = 100,
    task: str = "regression",
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    coefficient_tolerance: float = 1e-8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate elastic-net selection frequency using row or group bootstraps."""

    X, y = _prepare_supervised(frame, target)
    columns = list(map(str, X.columns))
    result_columns = [
        "Feature_Name",
        "ElasticNet_Selection_Frequency",
        "Median_Abs_Coefficient",
        "Successful_Bootstraps",
    ]
    if not columns or len(X) < 3 or np.unique(y).size < 2 or n_bootstraps <= 0:
        return pd.DataFrame(
            {
                "Feature_Name": columns,
                "ElasticNet_Selection_Frequency": 0.0,
                "Median_Abs_Coefficient": 0.0,
                "Successful_Bootstraps": 0,
            },
            columns=result_columns,
        )
    # _prepare_supervised can drop invalid targets; align groups through its retained index.
    aligned_groups = None
    if groups is not None:
        original_groups = np.asarray(groups)
        if len(original_groups) != len(frame):
            raise ValueError("groups must be aligned with frame")
        target_valid = ~pd.isna(np.asarray(target))
        aligned_groups = original_groups[target_valid]

    rng = np.random.default_rng(random_state)
    coefficients: list[np.ndarray] = []
    for _ in range(int(n_bootstraps)):
        positions = _bootstrap_positions(len(X), aligned_groups, rng)
        if positions.size < 3 or np.unique(y[positions]).size < 2:
            continue
        if task == "regression":
            estimator: BaseEstimator = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=10_000,
                random_state=random_state,
            )
        elif task == "classification":
            estimator = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=l1_ratio,
                C=max(1e-6, 1.0 / max(alpha, 1e-6)),
                max_iter=5_000,
                random_state=random_state,
            )
        else:
            raise ValueError("task must be 'regression' or 'classification'")
        pipeline = make_pipeline(StandardScaler(), estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pipeline.fit(X.iloc[positions], y[positions])
            except (ValueError, FloatingPointError):
                continue
        coefficient = np.asarray(pipeline[-1].coef_, dtype=float)
        if coefficient.ndim == 2:
            coefficient = np.max(np.abs(coefficient), axis=0)
        else:
            coefficient = np.abs(coefficient)
        coefficients.append(coefficient)
    if coefficients:
        matrix = np.vstack(coefficients)
        frequency = np.mean(matrix > coefficient_tolerance, axis=0)
        median = np.median(matrix, axis=0)
    else:
        frequency = np.zeros(len(columns))
        median = np.zeros(len(columns))
    return pd.DataFrame(
        {
            "Feature_Name": columns,
            "ElasticNet_Selection_Frequency": frequency,
            "Median_Abs_Coefficient": median,
            "Successful_Bootstraps": len(coefficients),
        },
        columns=result_columns,
    ).sort_values(
        ["ElasticNet_Selection_Frequency", "Median_Abs_Coefficient", "Feature_Name"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def heldout_permutation_importance(
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    target: Sequence[object] | pd.Series | np.ndarray,
    *,
    scoring: str | None = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate permutation importance on a held-out fold."""

    X = _as_numeric_frame(frame)
    y = np.asarray(target)
    if X.empty or len(X) != len(y):
        return pd.DataFrame(
            columns=["Feature_Name", "Permutation_Importance", "Permutation_Importance_Std"]
        )
    result = permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return pd.DataFrame(
        {
            "Feature_Name": X.columns,
            "Permutation_Importance": result.importances_mean,
            "Permutation_Importance_Std": result.importances_std,
        }
    ).sort_values("Permutation_Importance", ascending=False, ignore_index=True)


def consensus_feature_selection(
    records: pd.DataFrame,
    *,
    min_fold_frequency: float = 0.70,
    min_selector_support: int = 2,
    max_features: int | None = None,
) -> pd.DataFrame:
    """Aggregate fold-wise mRMR, elastic-net, and permutation evidence.

    Expected columns are ``Feature_Name`` and ``Fold`` plus any of
    ``mRMR_Selected``, ``ElasticNet_Selected``, and ``Permutation_Importance``.
    Missing selector columns are treated as unsupported rather than as errors.
    """

    if "Feature_Name" not in records:
        raise ValueError("records must contain Feature_Name")
    working = records.copy()
    if "Fold" not in working:
        working["Fold"] = 0
    for name in ("mRMR_Selected", "ElasticNet_Selected"):
        if name not in working:
            working[name] = False
        working[name] = working[name].fillna(False).astype(bool)
    if "Permutation_Importance" not in working:
        working["Permutation_Importance"] = np.nan
    working["Permutation_Selected"] = working["Permutation_Importance"].fillna(0.0) > 0
    selector_columns = ["mRMR_Selected", "ElasticNet_Selected", "Permutation_Selected"]
    working["Selector_Support"] = working[selector_columns].sum(axis=1)
    working["Fold_Selected"] = working["Selector_Support"] >= min_selector_support

    rows: list[dict[str, object]] = []
    for feature, subset in working.groupby("Feature_Name", sort=False):
        selector_frequency = subset[selector_columns].mean()
        fold_frequency = float(subset["Fold_Selected"].mean())
        median_importance = float(subset["Permutation_Importance"].median(skipna=True))
        if not np.isfinite(median_importance):
            median_importance = 0.0
        supported_selectors = int((selector_frequency >= min_fold_frequency).sum())
        selected = (
            fold_frequency >= min_fold_frequency
            and median_importance > 0
            and supported_selectors >= min_selector_support
        )
        rows.append(
            {
                "Feature_Name": feature,
                "mRMR_Selection_Frequency": float(selector_frequency["mRMR_Selected"]),
                "ElasticNet_Selection_Frequency": float(
                    selector_frequency["ElasticNet_Selected"]
                ),
                "Positive_Permutation_Frequency": float(
                    selector_frequency["Permutation_Selected"]
                ),
                "Permutation_Importance": median_importance,
                "Fold_Selection_Frequency": fold_frequency,
                "Selector_Support": supported_selectors,
                "Final_Status": "selected" if selected else "rejected",
                "Rejection_Reason": "" if selected else "insufficient_stability_or_importance",
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output = output.sort_values(
        ["Final_Status", "Fold_Selection_Frequency", "Permutation_Importance", "Feature_Name"],
        ascending=[False, False, False, True],
        ignore_index=True,
    )
    output["Consensus_Rank"] = np.arange(1, len(output) + 1)
    if max_features is not None:
        selected_indices = output.index[output["Final_Status"] == "selected"]
        if len(selected_indices) > max_features:
            output.loc[selected_indices[max_features:], "Final_Status"] = "rejected"
            output.loc[selected_indices[max_features:], "Rejection_Reason"] = "feature_budget"
    return output


def tune_feature_budget(
    frame: pd.DataFrame,
    target: Sequence[object] | pd.Series | np.ndarray,
    splits: Iterable[tuple[Sequence[int], Sequence[int]]],
    *,
    budgets: Sequence[int] = (8, 12, 16, 20, 24, 32),
    estimator: BaseEstimator | None = None,
    task: str = "regression",
    scoring: str | None = None,
    random_state: int = 42,
) -> FeatureBudgetResult:
    """Tune a feature count using fold-local screening and mRMR ranking.

    Each fold recomputes missingness, correlations, imputation, and supervised
    ranking exclusively on its training rows.  The smallest budget wins exact
    score ties.
    """

    X = pd.DataFrame(frame).copy()
    y = np.asarray(target)
    if len(X) != len(y):
        raise ValueError("target must be aligned with frame")
    candidate_budgets = sorted({int(value) for value in budgets if int(value) > 0})
    if not candidate_budgets:
        raise ValueError("at least one positive feature budget is required")
    if estimator is None:
        if task == "regression":
            estimator = default_permutation_estimator(random_state)
            scoring = scoring or "neg_mean_absolute_error"
        elif task == "classification":
            estimator = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                LogisticRegression(max_iter=5_000, class_weight="balanced", random_state=random_state),
            )
            scoring = scoring or "balanced_accuracy"
        else:
            raise ValueError("task must be 'regression' or 'classification'")
    rows: list[dict[str, object]] = []
    for fold_id, (train_indices, validation_indices) in enumerate(splits):
        train_positions = np.asarray(train_indices, dtype=int)
        validation_positions = np.asarray(validation_indices, dtype=int)
        if train_positions.size < 3 or validation_positions.size < 1:
            continue
        screened = redundancy_filter(X.iloc[train_positions])
        ranking = mrmr_ranking(
            X.iloc[train_positions][screened.selected_features],
            y[train_positions],
            max_features=max(candidate_budgets),
            task=task,
            random_state=random_state + fold_id,
        )
        ordered = ranking["Feature_Name"].tolist()
        for budget in candidate_budgets:
            features = ordered[: min(budget, len(ordered))]
            if not features:
                continue
            model = clone(estimator)
            try:
                model.fit(X.iloc[train_positions][features], y[train_positions])
                scorer = check_scoring(model, scoring=scoring)
                score = float(
                    scorer(model, X.iloc[validation_positions][features], y[validation_positions])
                )
                status = "ok"
            except (ValueError, FloatingPointError) as error:
                score = np.nan
                status = f"failed: {type(error).__name__}"
            rows.append(
                {
                    "Fold": fold_id,
                    "Requested_Budget": budget,
                    "Actual_Features": len(features),
                    "Score": score,
                    "Status": status,
                    "Features": "|".join(features),
                }
            )
    fold_results = pd.DataFrame(rows)
    valid = fold_results.loc[fold_results.get("Status", pd.Series(dtype=object)) == "ok"]
    if valid.empty:
        return FeatureBudgetResult(candidate_budgets[0], pd.DataFrame(), fold_results)
    summary = (
        valid.groupby("Requested_Budget", as_index=False)
        .agg(Mean_Score=("Score", "mean"), Score_Std=("Score", "std"), Folds=("Fold", "nunique"))
        .sort_values(["Mean_Score", "Requested_Budget"], ascending=[False, True], ignore_index=True)
    )
    summary["Selected"] = False
    summary.loc[0, "Selected"] = True
    return FeatureBudgetResult(int(summary.loc[0, "Requested_Budget"]), summary, fold_results)


class FoldSafeFeatureSelector(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible semantic and redundancy selector.

    Supervised ranking is intentionally separate so model selection can combine
    multiple evidence sources without hiding fold leakage in a transformer.
    """

    def __init__(
        self,
        *,
        protected_features: Sequence[str] = (),
        extra_exclusions: Sequence[str] = (),
        target_columns: Sequence[str] = (),
        missing_threshold: float = 0.40,
        dominant_threshold: float = 0.995,
        correlation_threshold: float = 0.90,
        feature_priority: Sequence[str] = (),
    ) -> None:
        self.protected_features = protected_features
        self.extra_exclusions = extra_exclusions
        self.target_columns = target_columns
        self.missing_threshold = missing_threshold
        self.dominant_threshold = dominant_threshold
        self.correlation_threshold = correlation_threshold
        self.feature_priority = feature_priority

    def fit(self, X: pd.DataFrame, y: object = None) -> "FoldSafeFeatureSelector":
        frame = pd.DataFrame(X).copy()
        semantic, self.semantic_audit_ = apply_semantic_exclusions(
            frame,
            extra_exclusions=self.extra_exclusions,
            target_columns=self.target_columns,
        )
        self.screening_result_ = redundancy_filter(
            semantic,
            protected_features=self.protected_features,
            missing_threshold=self.missing_threshold,
            dominant_threshold=self.dominant_threshold,
            correlation_threshold=self.correlation_threshold,
            feature_priority=self.feature_priority,
        )
        self.selected_features_ = self.screening_result_.selected_features
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("FoldSafeFeatureSelector must be fitted before transform")
        return _as_numeric_frame(pd.DataFrame(X).reindex(columns=self.selected_features_))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("FoldSafeFeatureSelector must be fitted first")
        return np.asarray(self.selected_features_, dtype=object)


def default_permutation_estimator(random_state: int = 42) -> BaseEstimator:
    """Small-data-safe nonlinear estimator used by the analysis runner."""

    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    )
