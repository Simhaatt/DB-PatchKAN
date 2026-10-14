"""Classical regression/classification baselines and consistent metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR


_AGING_TOKENS = ("age", "days_since", "odometer", "mileage")
_DRIVER_TOKENS = (
    "speed",
    "accel",
    "decel",
    "jerk",
    "pedal",
    "brake",
    "torque",
    "rpm",
    "eco",
    "epedal",
    "behavior",
)
_CLUSTER_TOKENS = ("cluster",)
_DBSI_TOKENS = ("bsi_", "stress_dose")
_BATTERY_TOKENS = (
    "soc",
    "current",
    "power",
    "voltage",
    "temp",
    "thermal",
    "energy",
    "regen",
    "ah_",
    "efc",
    "dod",
    "hvac",
)


class PersistenceRegressor(RegressorMixin, BaseEstimator):
    """Forecast the last finite training target for every future observation."""

    def fit(self, X: object, y: Sequence[float]) -> "PersistenceRegressor":
        values = np.asarray(y, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("PersistenceRegressor requires at least one finite target")
        self.last_value_ = float(finite[-1])
        self.n_features_in_ = np.shape(X)[1] if np.ndim(X) == 2 else 0
        return self

    def predict(self, X: object) -> np.ndarray:
        if not hasattr(self, "last_value_"):
            raise RuntimeError("PersistenceRegressor must be fitted first")
        return np.full(len(X), self.last_value_, dtype=float)


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.to_numeric(frame[column], errors="coerce") for column in frame.columns},
        index=frame.index,
    ).replace([np.inf, -np.inf], np.nan)


def _scaled(estimator: BaseEstimator) -> BaseEstimator:
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        estimator,
    )


def _tree_pipeline(estimator: BaseEstimator) -> BaseEstimator:
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), estimator)


def build_regression_baselines(
    *,
    random_state: int = 42,
    include_xgboost: bool = True,
    include_ebm: bool = True,
) -> dict[str, BaseEstimator]:
    """Build the required small-data classical SOH baselines."""

    models: dict[str, BaseEstimator] = {
        "Persistence": PersistenceRegressor(),
        "Linear": _scaled(LinearRegression()),
        "ElasticNet": _scaled(
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20_000, random_state=random_state)
        ),
        "RBF-SVR": _scaled(SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale")),
        "GaussianProcess": _scaled(
            GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
                normalize_y=True,
                random_state=random_state,
            )
        ),
        "RandomForest": _tree_pipeline(
            RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                max_features=0.8,
                n_jobs=-1,
                random_state=random_state,
            )
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = _tree_pipeline(
                XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.03,
                    max_depth=3,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.05,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=random_state,
                )
            )
        except ImportError:
            pass
    if include_ebm:
        try:
            from interpret.glassbox import ExplainableBoostingRegressor

            models["EBM"] = _tree_pipeline(
                ExplainableBoostingRegressor(
                    interactions=10,
                    outer_bags=8,
                    learning_rate=0.03,
                    random_state=random_state,
                )
            )
        except ImportError:
            pass
    return models


def build_classification_baselines(
    *,
    random_state: int = 42,
    include_xgboost: bool = True,
    include_ebm: bool = True,
) -> dict[str, BaseEstimator]:
    """Build behaviour/driver classifiers with probabilities for calibration metrics."""

    models: dict[str, BaseEstimator] = {
        "ElasticNetLogistic": _scaled(
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=1.0,
                class_weight="balanced",
                max_iter=5_000,
                random_state=random_state,
            )
        ),
        "RBF-SVC": _scaled(
            SVC(C=10.0, kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)
        ),
        "RandomForest": _tree_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            )
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = _tree_pipeline(
                XGBClassifier(
                    n_estimators=350,
                    learning_rate=0.03,
                    max_depth=3,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.05,
                    reg_lambda=1.0,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    n_jobs=-1,
                    random_state=random_state,
                )
            )
        except ImportError:
            pass
    if include_ebm:
        try:
            from interpret.glassbox import ExplainableBoostingClassifier

            models["EBM"] = _tree_pipeline(
                ExplainableBoostingClassifier(
                    interactions=10,
                    outer_bags=8,
                    learning_rate=0.03,
                    random_state=random_state,
                )
            )
        except ImportError:
            pass
    return models


def soh_ablation_feature_sets(columns: Sequence[str]) -> dict[str, list[str]]:
    """Construct the predefined M0--M5 feature families from available columns."""

    available = list(map(str, columns))

    def matching(tokens: Sequence[str]) -> list[str]:
        return [column for column in available if any(token in column.casefold() for token in tokens)]

    aging = matching(_AGING_TOKENS)
    clusters = matching(_CLUSTER_TOKENS)
    dbsi = matching(_DBSI_TOKENS)
    driver = [column for column in matching(_DRIVER_TOKENS) if column not in clusters]
    battery = [
        column
        for column in matching(_BATTERY_TOKENS)
        if column not in dbsi and column not in driver
    ]

    def union(*groups: Sequence[str]) -> list[str]:
        included = {feature for group in groups for feature in group}
        return [feature for feature in available if feature in included]

    return {
        "M0_age_mileage": union(aging),
        "M1_operational": union(aging, battery),
        "M2_driver": union(aging, battery, driver),
        # M3 is specifically the incremental cluster representation.  When no
        # leakage-safe cluster feature is available, returning M1's fields
        # would create a duplicate result mislabeled as a cluster ablation.
        "M3_clusters": union(aging, battery, clusters) if clusters else [],
        "M4_dbsi": union(aging, battery, dbsi),
        "M5_full": union(aging, battery, driver, clusters, dbsi),
    }


def regression_metrics(
    observed: Sequence[float],
    predicted: Sequence[float],
    *,
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
) -> dict[str, float]:
    """SOH metrics in percentage points with optional interval diagnostics."""

    y = np.asarray(observed, dtype=float)
    prediction = np.asarray(predicted, dtype=float)
    valid = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[valid], prediction[valid]
    if y.size == 0:
        return {
            "N": 0,
            "MAE_pp": np.nan,
            "RMSE_pp": np.nan,
            "R2": np.nan,
            "Bias_pp": np.nan,
            "P95_Absolute_Error_pp": np.nan,
            "MAPE_pct": np.nan,
        }
    residual = prediction - y
    result = {
        "N": int(y.size),
        "MAE_pp": float(mean_absolute_error(y, prediction)),
        "RMSE_pp": float(np.sqrt(mean_squared_error(y, prediction))),
        "R2": float(r2_score(y, prediction)) if y.size >= 2 and np.ptp(y) > 0 else np.nan,
        "Bias_pp": float(np.mean(residual)),
        "P95_Absolute_Error_pp": float(np.quantile(np.abs(residual), 0.95)),
        "MAPE_pct": float(np.mean(np.abs(residual[y != 0] / y[y != 0])) * 100) if np.any(y != 0) else np.nan,
    }
    if lower is not None and upper is not None:
        lower_array = np.asarray(lower, dtype=float)[valid]
        upper_array = np.asarray(upper, dtype=float)[valid]
        interval_valid = np.isfinite(lower_array) & np.isfinite(upper_array)
        result["Interval_Coverage"] = (
            float(np.mean((y[interval_valid] >= lower_array[interval_valid]) & (y[interval_valid] <= upper_array[interval_valid])))
            if interval_valid.any()
            else np.nan
        )
        result["Interval_Mean_Width_pp"] = (
            float(np.mean(upper_array[interval_valid] - lower_array[interval_valid]))
            if interval_valid.any()
            else np.nan
        )
    return result


def classification_metrics(
    observed: Sequence[object],
    predicted: Sequence[object],
    *,
    probabilities: np.ndarray | None = None,
    classes: Sequence[object] | None = None,
) -> dict[str, object]:
    """Balanced multiclass metrics, per-class recall, and probability scores."""

    y = np.asarray(observed)
    prediction = np.asarray(predicted)
    labels = np.asarray(list(classes) if classes is not None else sorted(pd.unique(y), key=str))
    result: dict[str, object] = {
        "N": int(len(y)),
        "Accuracy": float(accuracy_score(y, prediction)),
        "Macro_F1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, prediction)),
    }
    _, recall, _, _ = precision_recall_fscore_support(
        y, prediction, labels=labels, zero_division=0
    )
    for label, value in zip(labels, recall, strict=True):
        result[f"Recall_{label}"] = float(value)
    result["Confusion_Matrix"] = confusion_matrix(y, prediction, labels=labels).tolist()
    if probabilities is not None:
        probability = np.asarray(probabilities, dtype=float)
        if probability.shape != (len(y), len(labels)):
            raise ValueError("probabilities must align with observations and classes")
        probability = np.clip(probability, 1e-12, 1.0)
        probability /= probability.sum(axis=1, keepdims=True)
        result["Log_Loss"] = float(log_loss(y, probability, labels=labels))
        class_to_index = {label: index for index, label in enumerate(labels)}
        one_hot = np.zeros_like(probability)
        for row, label in enumerate(y):
            one_hot[row, class_to_index[label]] = 1.0
        result["Brier_Score"] = float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))
    return result


@dataclass
class BaselineEvaluation:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    fitted_models: dict[str, BaseEstimator]


def fit_evaluate_regressors(
    X_train: pd.DataFrame,
    y_train: Sequence[float],
    X_test: pd.DataFrame,
    y_test: Sequence[float],
    *,
    models: Mapping[str, BaseEstimator] | None = None,
    random_state: int = 42,
) -> BaselineEvaluation:
    """Fit models on one fold, continuing cleanly when an optional model fails."""

    train = _numeric(pd.DataFrame(X_train))
    test = _numeric(pd.DataFrame(X_test).reindex(columns=train.columns))
    y_train_array = np.asarray(y_train, dtype=float)
    y_test_array = np.asarray(y_test, dtype=float)
    train_valid = np.isfinite(y_train_array)
    test_valid = np.isfinite(y_test_array)
    train, y_train_array = train.iloc[train_valid], y_train_array[train_valid]
    test_eval, y_test_eval = test.iloc[test_valid], y_test_array[test_valid]
    suite = dict(models) if models is not None else build_regression_baselines(random_state=random_state)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    fitted: dict[str, BaseEstimator] = {}
    for name, template in suite.items():
        if name == "GaussianProcess" and len(train) > 2_000:
            metric_rows.append({"Model": name, "Status": "skipped_sample_limit"})
            continue
        estimator = clone(template)
        started = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                estimator.fit(train, y_train_array)
                prediction = np.asarray(estimator.predict(test_eval), dtype=float)
            except Exception as error:  # Optional model/library failures are reportable data.
                metric_rows.append(
                    {"Model": name, "Status": "failed", "Error": f"{type(error).__name__}: {error}"}
                )
                continue
        elapsed = perf_counter() - started
        metric_rows.append(
            {"Model": name, "Status": "ok", "Fit_Predict_Time_s": elapsed, **regression_metrics(y_test_eval, prediction)}
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "Row_Index": test_eval.index,
                    "Model": name,
                    "soh_observed_pct": y_test_eval,
                    "soh_pred_pct": prediction,
                    "residual_pp": prediction - y_test_eval,
                    "absolute_error_pp": np.abs(prediction - y_test_eval),
                }
            )
        )
        fitted[name] = estimator
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame(
        columns=["Row_Index", "Model", "soh_observed_pct", "soh_pred_pct", "residual_pp", "absolute_error_pp"]
    )
    return BaselineEvaluation(pd.DataFrame(metric_rows), predictions, fitted)


def fit_evaluate_classifiers(
    X_train: pd.DataFrame,
    y_train: Sequence[object],
    X_test: pd.DataFrame,
    y_test: Sequence[object],
    *,
    models: Mapping[str, BaseEstimator] | None = None,
    random_state: int = 42,
) -> BaselineEvaluation:
    """Fit probabilistic behaviour classifiers on one grouped/temporal fold."""

    train = _numeric(pd.DataFrame(X_train))
    test = _numeric(pd.DataFrame(X_test).reindex(columns=train.columns))
    if len(train) != len(y_train) or len(test) != len(y_test):
        raise ValueError("feature frames and classification targets must be aligned")
    # Fit the label mapping on training labels only.  Unseen held-out classes are
    # retained as explicit evaluation-only labels with zero predicted probability.
    train_labels = np.asarray(y_train).astype(str)
    test_labels = np.asarray(y_test).astype(str)
    encoder = LabelEncoder().fit(train_labels)
    train_target = encoder.transform(train_labels)
    unseen = sorted(set(test_labels) - set(encoder.classes_))
    evaluation_classes = list(map(str, encoder.classes_)) + unseen
    class_to_index = {label: index for index, label in enumerate(evaluation_classes)}
    test_target = np.asarray([class_to_index[label] for label in test_labels], dtype=int)
    suite = dict(models) if models is not None else build_classification_baselines(random_state=random_state)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    fitted: dict[str, BaseEstimator] = {}
    for name, template in suite.items():
        estimator = clone(template)
        started = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                estimator.fit(train, train_target)
                encoded_prediction = np.asarray(estimator.predict(test), dtype=int)
                probability_raw = estimator.predict_proba(test) if hasattr(estimator, "predict_proba") else None
            except Exception as error:
                metric_rows.append(
                    {"Model": name, "Status": "failed", "Error": f"{type(error).__name__}: {error}"}
                )
                continue
        probability = None
        if probability_raw is not None:
            model_classes = np.asarray(getattr(estimator, "classes_", np.arange(probability_raw.shape[1])), dtype=int)
            probability = np.full((len(test), len(evaluation_classes)), 1e-12, dtype=float)
            probability[:, model_classes] = probability_raw
            probability /= probability.sum(axis=1, keepdims=True)
        elapsed = perf_counter() - started
        metrics = classification_metrics(
            test_target,
            encoded_prediction,
            probabilities=probability,
            classes=np.arange(len(evaluation_classes)),
        )
        metric_rows.append({"Model": name, "Status": "ok", "Fit_Predict_Time_s": elapsed, **metrics})
        prediction_rows.append(
            pd.DataFrame(
                {
                    "Row_Index": test.index,
                    "Model": name,
                    "Observed": test_labels,
                    "Predicted": np.asarray(evaluation_classes, dtype=object)[encoded_prediction],
                }
            )
        )
        fitted[name] = estimator
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame(
        columns=["Row_Index", "Model", "Observed", "Predicted"]
    )
    return BaselineEvaluation(pd.DataFrame(metric_rows), predictions, fitted)
