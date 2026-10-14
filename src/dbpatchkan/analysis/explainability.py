"""Held-out explanation helpers that do not treat attention as importance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import check_scoring


@dataclass
class ShapResult:
    values: np.ndarray | None
    base_values: np.ndarray | None
    importance: pd.DataFrame
    status: str
    message: str = ""


def grouped_permutation_importance(
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    target: Sequence[object],
    *,
    feature_groups: Mapping[str, Sequence[str]] | None = None,
    scoring: str | Callable[..., float] | None = None,
    n_repeats: int = 10,
    block_labels: Sequence[object] | None = None,
    permute_within_blocks: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Jointly permute related columns and report held-out score reduction.

    ``feature_groups`` addresses correlated engineered variables (for example all
    thermal features).  ``permute_within_blocks`` can additionally restrict row
    permutations to trips/vehicles.  It is disabled by default because a feature
    constant within a trip would otherwise have zero measurable importance.
    """

    X = pd.DataFrame(frame).copy()
    y = np.asarray(target)
    if len(X) != len(y):
        raise ValueError("target must be aligned with frame")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    groups = (
        {str(column): [str(column)] for column in X.columns}
        if feature_groups is None
        else {str(name): list(columns) for name, columns in feature_groups.items()}
    )
    unknown = sorted({column for columns in groups.values() for column in columns if column not in X})
    if unknown:
        raise KeyError(f"unknown feature columns: {unknown}")
    blocks = None if block_labels is None else np.asarray(block_labels)
    if blocks is not None and len(blocks) != len(X):
        raise ValueError("block_labels must be aligned with frame")

    scorer = check_scoring(estimator, scoring=scoring)
    baseline = float(scorer(estimator, X, y))
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for group_name, columns in groups.items():
        reductions: list[float] = []
        for _ in range(n_repeats):
            permuted = X.copy()
            if permute_within_blocks and blocks is not None:
                source = np.arange(len(X))
                for block in pd.unique(blocks):
                    positions = np.flatnonzero(blocks == block)
                    source[positions] = rng.permutation(positions)
            else:
                source = rng.permutation(len(X))
            permuted.loc[:, columns] = X.iloc[source][columns].to_numpy()
            reductions.append(baseline - float(scorer(estimator, permuted, y)))
        rows.append(
            {
                "Feature_Group": group_name,
                "Features": "|".join(columns),
                "Permutation_Importance": float(np.mean(reductions)),
                "Permutation_Importance_Std": float(np.std(reductions, ddof=1)) if len(reductions) > 1 else 0.0,
                "Baseline_Score": baseline,
                "Repeats": n_repeats,
            }
        )
    return pd.DataFrame(rows).sort_values("Permutation_Importance", ascending=False, ignore_index=True)


def _unwrap_pipeline(estimator: BaseEstimator, frame: pd.DataFrame) -> tuple[BaseEstimator, np.ndarray, list[str]]:
    if hasattr(estimator, "steps") and len(estimator.steps) > 1:  # sklearn Pipeline
        transformer = estimator[:-1]
        model = estimator[-1]
        transformed = np.asarray(transformer.transform(frame), dtype=float)
        try:
            names = list(map(str, transformer.get_feature_names_out(frame.columns)))
        except (AttributeError, ValueError):
            names = list(map(str, frame.columns))
    else:
        model = estimator
        transformed = np.asarray(frame, dtype=float)
        names = list(map(str, frame.columns))
    if transformed.ndim != 2:
        raise ValueError("SHAP input must be two-dimensional")
    if len(names) != transformed.shape[1]:
        names = [f"feature_{index}" for index in range(transformed.shape[1])]
    return model, transformed, names


def tree_shap_summary(
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    *,
    max_samples: int = 1_000,
    random_state: int = 42,
) -> ShapResult:
    """Compute TreeSHAP, with a transparent feature-importance fallback."""

    X = pd.DataFrame(frame)
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state)
    try:
        model, transformed, names = _unwrap_pipeline(estimator, X)
    except (TypeError, ValueError) as error:
        return ShapResult(
            None,
            None,
            pd.DataFrame(columns=["Feature_Name", "SHAP_Importance"]),
            "failed",
            str(error),
        )
    try:
        import shap

        explanation = shap.TreeExplainer(model)(transformed)
        values = np.asarray(explanation.values)
        # Multiclass releases have used both (n, p, k) and list[k](n, p).
        if values.ndim == 3:
            feature_axis = 1 if values.shape[1] == len(names) else 2
            reduce_axes = tuple(axis for axis in range(values.ndim) if axis != feature_axis)
            importance_values = np.mean(np.abs(values), axis=reduce_axes)
        elif values.ndim == 2:
            importance_values = np.mean(np.abs(values), axis=0)
        else:
            raise ValueError(f"unsupported SHAP value shape {values.shape}")
        importance = pd.DataFrame(
            {"Feature_Name": names, "SHAP_Importance": importance_values}
        ).sort_values("SHAP_Importance", ascending=False, ignore_index=True)
        return ShapResult(
            values,
            np.asarray(explanation.base_values),
            importance,
            "ok",
        )
    except Exception as error:
        feature_importance = getattr(model, "feature_importances_", None)
        if feature_importance is not None and len(feature_importance) == len(names):
            importance = pd.DataFrame(
                {"Feature_Name": names, "SHAP_Importance": np.asarray(feature_importance, dtype=float)}
            ).sort_values("SHAP_Importance", ascending=False, ignore_index=True)
            return ShapResult(
                None,
                None,
                importance,
                "fallback_model_importance",
                f"SHAP unavailable: {type(error).__name__}: {error}",
            )
        return ShapResult(
            None,
            None,
            pd.DataFrame(columns=["Feature_Name", "SHAP_Importance"]),
            "unavailable",
            f"{type(error).__name__}: {error}",
        )


def ebm_global_terms(estimator: BaseEstimator) -> pd.DataFrame:
    """Extract EBM global term importances without requiring interpret at import."""

    model = estimator[-1] if hasattr(estimator, "steps") else estimator
    if not hasattr(model, "explain_global"):
        return pd.DataFrame(columns=["Feature_Name", "EBM_Importance", "Status"])
    try:
        data = model.explain_global().data()
        names = data.get("names", getattr(model, "term_names_", []))
        scores = data.get("scores", [])
        return pd.DataFrame(
            {
                "Feature_Name": list(map(str, names)),
                "EBM_Importance": np.asarray(scores, dtype=float),
                "Status": "ok",
            }
        ).sort_values("EBM_Importance", ascending=False, ignore_index=True)
    except Exception as error:
        return pd.DataFrame(
            [{"Feature_Name": "", "EBM_Importance": np.nan, "Status": f"failed: {error}"}]
        )


def ebm_response_functions(estimator: BaseEstimator) -> pd.DataFrame:
    """Extract auditable one-dimensional EBM response-curve points."""

    model = estimator[-1] if hasattr(estimator, "steps") else estimator
    columns = [
        "Feature_Name", "Feature_Value", "Effect", "Lower_Bound", "Upper_Bound", "Status"
    ]
    if not hasattr(model, "explain_global"):
        return pd.DataFrame(columns=columns)
    try:
        explanation = model.explain_global()
        rows: list[dict[str, object]] = []
        for index, term_name in enumerate(map(str, getattr(model, "term_names_", []))):
            data = explanation.data(index)
            values = data.get("names", []) if isinstance(data, dict) else []
            scores = data.get("scores", []) if isinstance(data, dict) else []
            score_array = np.asarray(scores, dtype=object)
            # Two-dimensional interaction surfaces are represented by their
            # global importance; only named one-dimensional terms form curves.
            if score_array.ndim != 1:
                continue
            lower = data.get("lower_bounds", []) if isinstance(data, dict) else []
            upper = data.get("upper_bounds", []) if isinstance(data, dict) else []
            for point in range(min(len(values), len(scores))):
                rows.append(
                    {
                        "Feature_Name": term_name,
                        "Feature_Value": values[point],
                        "Effect": float(scores[point]),
                        "Lower_Bound": float(lower[point]) if point < len(lower) else np.nan,
                        "Upper_Bound": float(upper[point]) if point < len(upper) else np.nan,
                        "Status": "ok",
                    }
                )
        return pd.DataFrame(rows, columns=columns)
    except Exception as error:
        return pd.DataFrame(
            [{
                "Feature_Name": "",
                "Feature_Value": np.nan,
                "Effect": np.nan,
                "Lower_Bound": np.nan,
                "Upper_Bound": np.nan,
                "Status": f"failed: {error}",
            }],
            columns=columns,
        )


def occlusion_importance(
    predict: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    target: Sequence[float],
    *,
    channel_names: Sequence[str] | None = None,
    baseline: np.ndarray | float | None = None,
    loss: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> pd.DataFrame:
    """Leave-one-channel-out importance for sequence models.

    Inputs may be ``(batch, time, channels)`` or ``(batch, channels)``.  This
    helper uses prediction degradation, not attention weights.
    """

    array = np.asarray(inputs, dtype=float)
    if array.ndim not in (2, 3):
        raise ValueError("inputs must have shape (batch, channels) or (batch, time, channels)")
    y = np.asarray(target, dtype=float)
    if len(array) != len(y):
        raise ValueError("target must align with inputs")
    names = list(channel_names) if channel_names is not None else [
        f"channel_{index}" for index in range(array.shape[-1])
    ]
    if len(names) != array.shape[-1]:
        raise ValueError("channel_names must match the final input dimension")
    metric = loss or (lambda actual, estimate: float(np.mean(np.abs(actual - estimate))))
    base_prediction = np.asarray(predict(array), dtype=float).reshape(-1)
    base_loss = float(metric(y, base_prediction))
    if baseline is None:
        axes = tuple(range(array.ndim - 1))
        fill = np.nanmedian(array, axis=axes)
    else:
        fill = np.asarray(baseline, dtype=float)
    rows: list[dict[str, object]] = []
    for channel, name in enumerate(names):
        occluded = array.copy()
        occluded[..., channel] = fill if np.ndim(fill) == 0 else fill[channel]
        prediction = np.asarray(predict(occluded), dtype=float).reshape(-1)
        occluded_loss = float(metric(y, prediction))
        rows.append(
            {
                "Feature_Name": name,
                "Baseline_Loss": base_loss,
                "Occluded_Loss": occluded_loss,
                "Channel_Occlusion_Delta": occluded_loss - base_loss,
            }
        )
    return pd.DataFrame(rows).sort_values("Channel_Occlusion_Delta", ascending=False, ignore_index=True)


def robust_importance_summary(
    records: pd.DataFrame,
    *,
    minimum_fold_frequency: float = 0.70,
) -> pd.DataFrame:
    """Apply the study's four-part definition of robust feature importance."""

    required = {"Feature_Name", "Fold"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"records are missing {sorted(missing)}")
    work = records.copy()
    defaults = {
        "Selected": False,
        "Permutation_Importance": 0.0,
        "Ablation_Delta": 0.0,
        "Stable_Response": False,
    }
    for column, default in defaults.items():
        if column not in work:
            work[column] = default
    rows: list[dict[str, object]] = []
    for feature, subset in work.groupby("Feature_Name", sort=False):
        selection_frequency = float(subset["Selected"].fillna(False).astype(bool).mean())
        permutation = float(subset["Permutation_Importance"].fillna(0.0).median())
        ablation_frequency = float((subset["Ablation_Delta"].fillna(0.0) > 0).mean())
        response_frequency = float(subset["Stable_Response"].fillna(False).astype(bool).mean())
        robust = (
            selection_frequency >= minimum_fold_frequency
            and permutation > 0
            and ablation_frequency >= minimum_fold_frequency
            and response_frequency >= minimum_fold_frequency
        )
        rows.append(
            {
                "Feature_Name": feature,
                "Fold_Selection_Frequency": selection_frequency,
                "Permutation_Importance": permutation,
                "Positive_Ablation_Frequency": ablation_frequency,
                "Stable_Response_Frequency": response_frequency,
                "Robustly_Important": robust,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Robustly_Important", "Permutation_Importance"], ascending=[False, False], ignore_index=True
    )
