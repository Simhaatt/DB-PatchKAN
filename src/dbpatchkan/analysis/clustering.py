"""Driver-behaviour discovery and explainable aggression-level mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


DEFAULT_BEHAVIOR_PATTERNS = (
    "speed_std",
    "speed_p95",
    "accel_pos",
    "decel_abs",
    "jerk",
    "pedal",
    "brake_active",
    "harsh_accel",
    "harsh_brake",
    "torque_pos",
    "power_dis",
    "current_dis",
)


@dataclass
class BehaviorClusteringResult:
    """Fitted preprocessing/model objects and auditable discovery outputs."""

    labels: np.ndarray
    probabilities: np.ndarray
    transformed: np.ndarray
    feature_names: list[str]
    candidate_metrics: pd.DataFrame
    comparison_metrics: pd.DataFrame
    scaler: RobustScaler
    imputer: SimpleImputer
    pca: PCA
    model: GaussianMixture
    selected_k: int
    covariance_type: str
    bootstrap_ari: float
    exploratory: bool

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = _numeric_behavior_frame(frame, self.feature_names)
        values = self.imputer.transform(numeric)
        return self.pca.transform(self.scaler.transform(values))

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        transformed = self.transform(frame)
        return self.model.predict(transformed), self.model.predict_proba(transformed)


@dataclass(frozen=True)
class BehaviorLevelThresholds:
    lower: float
    upper: float


class BehaviorLevelMapper:
    """Training-fold tertile mapper for Eco/Balanced/Aggressive labels."""

    def __init__(self, lower_quantile: float = 1 / 3, upper_quantile: float = 2 / 3) -> None:
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, scores: Sequence[float]) -> "BehaviorLevelMapper":
        values = np.asarray(scores, dtype=float)
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            self.thresholds_ = BehaviorLevelThresholds(1 / 3, 2 / 3)
        else:
            lower, upper = np.quantile(valid, [self.lower_quantile, self.upper_quantile])
            if np.isclose(lower, upper):
                spread = max(float(np.std(valid)), 1e-9)
                lower, upper = float(lower - spread / 2), float(upper + spread / 2)
            self.thresholds_ = BehaviorLevelThresholds(float(lower), float(upper))
        return self

    def transform(self, scores: Sequence[float]) -> np.ndarray:
        if not hasattr(self, "thresholds_"):
            raise RuntimeError("BehaviorLevelMapper must be fitted before transform")
        values = np.asarray(scores, dtype=float)
        labels = np.full(values.shape, "Balanced", dtype=object)
        labels[values <= self.thresholds_.lower] = "Eco"
        labels[values > self.thresholds_.upper] = "Aggressive"
        labels[~np.isfinite(values)] = "Balanced"
        return labels

    def fit_transform(self, scores: Sequence[float]) -> np.ndarray:
        return self.fit(scores).transform(scores)


def _numeric_behavior_frame(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    missing = [feature for feature in features if feature not in frame]
    if missing:
        raise KeyError(f"missing behavior features: {missing}")
    return pd.DataFrame(
        {feature: pd.to_numeric(frame[feature], errors="coerce") for feature in features},
        index=frame.index,
    ).replace([np.inf, -np.inf], np.nan)


def infer_behavior_features(frame: pd.DataFrame) -> list[str]:
    """Find numeric driver-style features without using context/identity fields."""

    forbidden = ("driver_id", "trip", "timestamp", "datetime", "soh", "odometer", "age")
    matches: list[str] = []
    for column in frame.columns:
        name = str(column).casefold()
        if any(token in name for token in forbidden):
            continue
        if any(pattern in name for pattern in DEFAULT_BEHAVIOR_PATTERNS):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().any():
                matches.append(str(column))
    return matches


def _cluster_metrics(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    valid = labels >= 0
    unique = np.unique(labels[valid])
    if valid.sum() < 3 or unique.size < 2 or unique.size >= valid.sum():
        return {"Silhouette": np.nan, "Davies_Bouldin": np.nan, "Calinski_Harabasz": np.nan}
    return {
        "Silhouette": float(silhouette_score(values[valid], labels[valid])),
        "Davies_Bouldin": float(davies_bouldin_score(values[valid], labels[valid])),
        "Calinski_Harabasz": float(calinski_harabasz_score(values[valid], labels[valid])),
    }


def _bootstrap_gmm_stability(
    values: np.ndarray,
    reference_labels: np.ndarray,
    *,
    n_components: int,
    covariance_type: str,
    n_init: int,
    iterations: int,
    random_state: int,
) -> float:
    if iterations <= 0:
        return np.nan
    rng = np.random.default_rng(random_state)
    scores: list[float] = []
    for iteration in range(iterations):
        sample = rng.integers(0, len(values), size=len(values))
        if np.unique(sample).size < n_components:
            continue
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=max(1, min(n_init, 5)),
            reg_covar=1e-5,
            random_state=random_state + iteration + 1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(values[sample])
                labels = model.predict(values)
            except (ValueError, np.linalg.LinAlgError):
                continue
        scores.append(float(adjusted_rand_score(reference_labels, labels)))
    return float(np.mean(scores)) if scores else np.nan


def compare_clusterers(
    values: np.ndarray,
    *,
    n_clusters: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare K-means, DBSCAN, and optional HDBSCAN in a shared PCA space."""

    rows: list[dict[str, object]] = []
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(values)
    rows.append(
        {
            "Algorithm": "KMeans",
            "Clusters": int(np.unique(kmeans_labels).size),
            "Noise_Fraction": 0.0,
            **_cluster_metrics(values, kmeans_labels),
        }
    )

    neighbours = max(2, min(5, len(values) - 1))
    if len(values) > neighbours:
        distances = NearestNeighbors(n_neighbors=neighbours).fit(values).kneighbors()[0][:, -1]
        eps = float(np.quantile(distances, 0.80))
        eps = max(eps, np.finfo(float).eps)
        dbscan_labels = DBSCAN(eps=eps, min_samples=neighbours).fit_predict(values)
        rows.append(
            {
                "Algorithm": "DBSCAN",
                "Clusters": int(np.unique(dbscan_labels[dbscan_labels >= 0]).size),
                "Noise_Fraction": float(np.mean(dbscan_labels < 0)),
                **_cluster_metrics(values, dbscan_labels),
            }
        )

    try:
        from hdbscan import HDBSCAN  # type: ignore[import-not-found]

        hdbscan_labels = HDBSCAN(min_cluster_size=max(3, int(np.ceil(0.05 * len(values))))).fit_predict(
            values
        )
        rows.append(
            {
                "Algorithm": "HDBSCAN",
                "Clusters": int(np.unique(hdbscan_labels[hdbscan_labels >= 0]).size),
                "Noise_Fraction": float(np.mean(hdbscan_labels < 0)),
                **_cluster_metrics(values, hdbscan_labels),
            }
        )
    except (ImportError, ValueError):
        rows.append(
            {
                "Algorithm": "HDBSCAN",
                "Clusters": np.nan,
                "Noise_Fraction": np.nan,
                "Silhouette": np.nan,
                "Davies_Bouldin": np.nan,
                "Calinski_Harabasz": np.nan,
                "Status": "optional_dependency_unavailable",
            }
        )
    return pd.DataFrame(rows)


def discover_behavior_archetypes(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str] | None = None,
    cluster_range: Iterable[int] = range(2, 7),
    covariance_types: Sequence[str] = ("diag", "full"),
    n_init: int = 20,
    bootstrap_iterations: int = 30,
    stability_threshold: float = 0.75,
    min_cluster_share: float = 0.05,
    random_state: int = 42,
) -> BehaviorClusteringResult:
    """Fit RobustScaler -> PCA(95%) -> selected Gaussian mixture.

    If no candidate satisfies stability and balance constraints, the lowest-BIC
    valid model is returned with ``exploratory=True``.  This explicit fallback is
    important for sparse temporal folds.
    """

    features = list(feature_names) if feature_names is not None else infer_behavior_features(frame)
    if not features:
        raise ValueError("no usable behavior features were found")
    numeric = _numeric_behavior_frame(frame, features)
    if len(numeric) < 3:
        raise ValueError("at least three observations are required for behavior discovery")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = imputer.fit_transform(numeric)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(imputed)
    max_components = max(1, min(scaled.shape[0] - 1, scaled.shape[1]))
    if max_components == 1:
        pca = PCA(n_components=1, random_state=random_state)
    else:
        pca = PCA(n_components=0.95, svd_solver="full", random_state=random_state)
    transformed = pca.fit_transform(scaled)

    candidates: list[tuple[GaussianMixture, np.ndarray, dict[str, object]]] = []
    for components in sorted(set(int(value) for value in cluster_range)):
        if components < 2 or components >= len(transformed):
            continue
        for covariance_type in covariance_types:
            model = GaussianMixture(
                n_components=components,
                covariance_type=covariance_type,
                n_init=n_init,
                reg_covar=1e-5,
                random_state=random_state,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    labels = model.fit_predict(transformed)
                except (ValueError, np.linalg.LinAlgError):
                    continue
            counts = np.bincount(labels, minlength=components)
            share = float(counts.min() / len(labels))
            stability = _bootstrap_gmm_stability(
                transformed,
                labels,
                n_components=components,
                covariance_type=covariance_type,
                n_init=n_init,
                iterations=bootstrap_iterations,
                random_state=random_state + 100 * components,
            )
            metrics: dict[str, object] = {
                "Algorithm": "GMM",
                "Clusters": components,
                "Covariance_Type": covariance_type,
                "BIC": float(model.bic(transformed)),
                "AIC": float(model.aic(transformed)),
                "Minimum_Cluster_Share": share,
                "Bootstrap_ARI": stability,
                "Eligible": bool(
                    share >= min_cluster_share
                    and np.isfinite(stability)
                    and stability >= stability_threshold
                ),
                **_cluster_metrics(transformed, labels),
            }
            candidates.append((model, labels, metrics))
    if not candidates:
        raise ValueError("no valid GMM candidate could be fitted")
    eligible = [candidate for candidate in candidates if bool(candidate[2]["Eligible"])]
    pool = eligible or candidates
    selected_model, labels, selected_metrics = min(pool, key=lambda item: float(item[2]["BIC"]))
    candidate_metrics = pd.DataFrame([item[2] for item in candidates]).sort_values(
        ["Eligible", "BIC"], ascending=[False, True], ignore_index=True
    )
    comparison = compare_clusterers(
        transformed,
        n_clusters=int(selected_metrics["Clusters"]),
        random_state=random_state,
    )
    comparison = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "Algorithm": "GMM-selected",
                        "Clusters": selected_metrics["Clusters"],
                        "Noise_Fraction": 0.0,
                        "Silhouette": selected_metrics["Silhouette"],
                        "Davies_Bouldin": selected_metrics["Davies_Bouldin"],
                        "Calinski_Harabasz": selected_metrics["Calinski_Harabasz"],
                    }
                ]
            ),
            comparison,
        ],
        ignore_index=True,
    )
    return BehaviorClusteringResult(
        labels=np.asarray(labels, dtype=int),
        probabilities=selected_model.predict_proba(transformed),
        transformed=transformed,
        feature_names=features,
        candidate_metrics=candidate_metrics,
        comparison_metrics=comparison,
        scaler=scaler,
        imputer=imputer,
        pca=pca,
        model=selected_model,
        selected_k=int(selected_metrics["Clusters"]),
        covariance_type=str(selected_metrics["Covariance_Type"]),
        bootstrap_ari=float(selected_metrics["Bootstrap_ARI"]),
        exploratory=not bool(selected_metrics["Eligible"]),
    )


def aggression_score_from_clusters(
    frame: pd.DataFrame,
    labels: Sequence[int],
    probabilities: np.ndarray | None = None,
    *,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute posterior-weighted cluster aggression and feature contributions."""

    features = list(feature_names) if feature_names is not None else infer_behavior_features(frame)
    if not features:
        raise ValueError("no aggression features were found")
    numeric = _numeric_behavior_frame(frame, features)
    medians = numeric.median()
    filled = numeric.fillna(medians)
    center = filled.median()
    scale = filled.quantile(0.75) - filled.quantile(0.25)
    scale = scale.mask(scale.abs() < 1e-12, 1.0)
    robust_z = ((filled - center) / scale).clip(-5, 5)
    direct_score = robust_z.mean(axis=1).to_numpy(dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    if len(labels_array) != len(frame):
        raise ValueError("labels must be aligned with frame")
    clusters = np.arange(labels_array.max(initial=-1) + 1)
    cluster_raw = np.array(
        [np.nanmean(direct_score[labels_array == cluster]) for cluster in clusters], dtype=float
    )
    finite = cluster_raw[np.isfinite(cluster_raw)]
    if finite.size and float(finite.max() - finite.min()) > 1e-12:
        intensity = (cluster_raw - finite.min()) / (finite.max() - finite.min())
    else:
        intensity = np.full(cluster_raw.shape, 0.5)
    intensity = np.nan_to_num(intensity, nan=0.5)
    if probabilities is not None:
        probability_array = np.asarray(probabilities, dtype=float)
        if probability_array.shape != (len(frame), len(intensity)):
            raise ValueError("probabilities must have shape (rows, clusters)")
        score = probability_array @ intensity
    else:
        score = intensity[labels_array]

    contribution = robust_z.copy()
    contribution.columns = [f"aggression_contribution__{column}" for column in features]
    contribution.insert(0, "aggression_score", score)
    contribution.insert(0, "natural_cluster_id", labels_array)
    mapping = pd.DataFrame(
        {
            "natural_cluster_id": clusters,
            "cluster_aggression_raw": cluster_raw,
            "cluster_aggression_intensity": intensity,
        }
    )
    contribution.attrs["cluster_mapping"] = mapping
    return score, contribution


def label_behavior_levels(
    scores: Sequence[float],
    *,
    training_scores: Sequence[float] | None = None,
) -> tuple[np.ndarray, BehaviorLevelThresholds]:
    """Convenience function that freezes thresholds from training scores."""

    mapper = BehaviorLevelMapper().fit(training_scores if training_scores is not None else scores)
    return mapper.transform(scores), mapper.thresholds_


def eligible_driver_ids(
    frame: pd.DataFrame,
    *,
    driver_col: str = "Driver_ID",
    trip_col: str = "Trip",
    minimum_trips: int = 5,
) -> list[object]:
    """Return driver IDs with enough independent trips for scored identification."""

    if driver_col not in frame or trip_col not in frame:
        return []
    counts = frame.groupby(driver_col, dropna=False)[trip_col].nunique()
    return counts[counts >= minimum_trips].index.tolist()
