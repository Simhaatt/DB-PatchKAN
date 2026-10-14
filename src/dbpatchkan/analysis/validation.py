"""Leakage-safe temporal/group validation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


@dataclass(frozen=True)
class FoldIndices:
    """One validation fold expressed as positional row indices."""

    fold_id: int
    train_index: np.ndarray
    validation_index: np.ndarray


@dataclass(frozen=True)
class TemporalValidationPlan:
    """Development folds plus one untouched chronological holdout."""

    folds: tuple[FoldIndices, ...]
    development_index: np.ndarray
    holdout_index: np.ndarray


def _coerce_time(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, utc=True, errors="coerce")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.95:
        finite = numeric.dropna().abs()
        median = float(finite.median()) if len(finite) else 0.0
        unit = "us" if median > 1e14 else "ms" if median > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def chronological_group_order(
    frame: pd.DataFrame,
    *,
    time_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Return one row per group, sorted by its first observation time."""

    if time_col not in frame or group_col not in frame:
        raise KeyError(f"frame must contain {time_col!r} and {group_col!r}")
    work = pd.DataFrame(
        {
            "group": frame[group_col].astype(object),
            "time": _coerce_time(frame[time_col]),
            "position": np.arange(len(frame)),
        }
    )
    summary = (
        work.groupby("group", dropna=False, sort=False)
        .agg(start_time=("time", "min"), end_time=("time", "max"), first_position=("position", "min"))
        .reset_index()
    )
    # Invalid times remain deterministic and sort after valid chronological data.
    return summary.sort_values(
        ["start_time", "first_position"], na_position="last", kind="stable", ignore_index=True
    )


def temporal_group_holdout(
    frame: pd.DataFrame,
    *,
    time_col: str,
    group_col: str,
    test_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Split complete groups into chronological development and final test sets."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    order = chronological_group_order(frame, time_col=time_col, group_col=group_col)
    n_groups = len(order)
    if n_groups < 2:
        raise ValueError("at least two groups are required for a holdout")
    n_test = min(n_groups - 1, max(1, int(np.ceil(n_groups * test_fraction))))
    test_groups = set(order.iloc[-n_test:]["group"].tolist())
    mask = frame[group_col].isin(test_groups).to_numpy()
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def purge_training_groups(
    frame: pd.DataFrame,
    train_index: Sequence[int],
    validation_index: Sequence[int],
    *,
    time_col: str,
    group_col: str,
    purge_days: float = 0.0,
    distance_col: str | None = None,
    purge_distance_km: float = 0.0,
) -> np.ndarray:
    """Remove complete training groups whose exposure can overlap a forecast target."""

    train_positions = np.asarray(train_index, dtype=int)
    validation_positions = np.asarray(validation_index, dtype=int)
    if train_positions.size == 0 or validation_positions.size == 0:
        return train_positions
    train_groups = set(frame.iloc[train_positions][group_col].tolist())
    times = _coerce_time(frame[time_col])
    if purge_days > 0:
        validation_start = times.iloc[validation_positions].min()
        if pd.notna(validation_start):
            cutoff = validation_start - pd.Timedelta(days=float(purge_days))
            group_end = (
                pd.DataFrame({"group": frame[group_col], "value": times})
                .groupby("group", dropna=False)["value"]
                .max()
            )
            train_groups = {
                group
                for group in train_groups
                if group in group_end.index
                and pd.notna(group_end.loc[group])
                and group_end.loc[group] < cutoff
            }
    if purge_distance_km > 0 and distance_col and distance_col in frame:
        distance = pd.to_numeric(frame[distance_col], errors="coerce")
        validation_distance = distance.iloc[validation_positions]
        if validation_distance.notna().any():
            cutoff = float(validation_distance.min()) - float(purge_distance_km)
            group_end = (
                pd.DataFrame({"group": frame[group_col], "value": distance})
                .groupby("group", dropna=False)["value"]
                .max()
            )
            train_groups = {
                group
                for group in train_groups
                if group in group_end.index
                and pd.notna(group_end.loc[group])
                and float(group_end.loc[group]) < cutoff
            }
    keep = frame.iloc[train_positions][group_col].isin(train_groups).to_numpy()
    return train_positions[keep]


class PurgedExpandingWindowSplit(BaseCrossValidator):
    """Expanding temporal folds that keep groups intact and purge forecast overlap."""

    def __init__(
        self,
        n_splits: int = 4,
        *,
        time_col: str = "datetime_utc",
        group_col: str = "trip_key",
        purge_days: float = 0.0,
        distance_col: str | None = None,
        purge_distance_km: float = 0.0,
        test_size_groups: int | None = None,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be positive")
        self.n_splits = n_splits
        self.time_col = time_col
        self.group_col = group_col
        self.purge_days = purge_days
        self.distance_col = distance_col
        self.purge_distance_km = purge_distance_km
        self.test_size_groups = test_size_groups

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y: object = None,
        groups: Sequence[object] | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        frame = pd.DataFrame(X).copy()
        if groups is not None:
            if len(groups) != len(frame):
                raise ValueError("groups must be aligned with X")
            internal_group = "__validation_group__"
            while internal_group in frame:
                internal_group += "_"
            frame[internal_group] = np.asarray(groups)
            group_col = internal_group
        else:
            group_col = self.group_col
        order = chronological_group_order(frame, time_col=self.time_col, group_col=group_col)
        n_groups = len(order)
        if n_groups < 2:
            return
        effective_splits = min(self.n_splits, n_groups - 1)
        test_size = self.test_size_groups or max(1, n_groups // (effective_splits + 1))
        test_size = min(test_size, max(1, (n_groups - 1) // effective_splits))
        first_validation = n_groups - effective_splits * test_size
        group_values = frame[group_col].astype(object)
        times = _coerce_time(frame[self.time_col])
        distances = (
            pd.to_numeric(frame[self.distance_col], errors="coerce")
            if self.distance_col and self.distance_col in frame
            else None
        )

        for fold in range(effective_splits):
            validation_start = first_validation + fold * test_size
            validation_stop = min(n_groups, validation_start + test_size)
            train_groups = set(order.iloc[:validation_start]["group"].tolist())
            validation_groups = set(order.iloc[validation_start:validation_stop]["group"].tolist())
            validation_mask = group_values.isin(validation_groups).to_numpy()
            if self.purge_days > 0 and validation_mask.any():
                validation_min_time = times[validation_mask].min()
                if pd.notna(validation_min_time):
                    cutoff = validation_min_time - pd.Timedelta(days=float(self.purge_days))
                    group_end_times = (
                        pd.DataFrame({"group": group_values, "time": times})
                        .groupby("group", dropna=False)["time"]
                        .max()
                    )
                    train_groups = {
                        group
                        for group in train_groups
                        if group in group_end_times.index
                        and pd.notna(group_end_times.loc[group])
                        and group_end_times.loc[group] < cutoff
                    }
            if (
                self.purge_distance_km > 0
                and distances is not None
                and validation_mask.any()
                and distances[validation_mask].notna().any()
            ):
                cutoff_distance = (
                    float(distances[validation_mask].min()) - float(self.purge_distance_km)
                )
                group_end_distance = (
                    pd.DataFrame({"group": group_values, "distance": distances})
                    .groupby("group", dropna=False)["distance"]
                    .max()
                )
                train_groups = {
                    group
                    for group in train_groups
                    if group in group_end_distance.index
                    and pd.notna(group_end_distance.loc[group])
                    and float(group_end_distance.loc[group]) < cutoff_distance
                }

            train_mask = group_values.isin(train_groups).to_numpy()
            train_index = np.flatnonzero(train_mask)
            validation_index = np.flatnonzero(validation_mask)
            if train_index.size and validation_index.size:
                yield train_index, validation_index


def build_temporal_validation_plan(
    frame: pd.DataFrame,
    *,
    time_col: str,
    group_col: str,
    n_splits: int = 4,
    final_test_fraction: float = 0.20,
    purge_days: float = 0.0,
    distance_col: str | None = None,
    purge_distance_km: float = 0.0,
) -> TemporalValidationPlan:
    """Create the Nissan-style 80/20 holdout and expanding development folds."""

    development, holdout = temporal_group_holdout(
        frame,
        time_col=time_col,
        group_col=group_col,
        test_fraction=final_test_fraction,
    )
    development_frame = frame.iloc[development].copy()
    splitter = PurgedExpandingWindowSplit(
        n_splits=n_splits,
        time_col=time_col,
        group_col=group_col,
        purge_days=purge_days,
        distance_col=distance_col,
        purge_distance_km=purge_distance_km,
    )
    folds: list[FoldIndices] = []
    for fold_id, (local_train, local_validation) in enumerate(splitter.split(development_frame)):
        folds.append(
            FoldIndices(
                fold_id=fold_id,
                train_index=development[local_train],
                validation_index=development[local_validation],
            )
        )
    return TemporalValidationPlan(tuple(folds), development, holdout)


def assert_no_group_leakage(
    frame: pd.DataFrame,
    train_index: Sequence[int],
    validation_index: Sequence[int],
    *,
    group_col: str,
    time_col: str | None = None,
) -> None:
    """Raise an informative assertion when a split leaks groups or chronology."""

    train_groups = set(frame.iloc[np.asarray(train_index)][group_col].tolist())
    validation_groups = set(frame.iloc[np.asarray(validation_index)][group_col].tolist())
    overlap = train_groups & validation_groups
    if overlap:
        raise AssertionError(f"group leakage detected: {sorted(map(str, overlap))[:5]}")
    if time_col and len(train_index) and len(validation_index):
        train_time = _coerce_time(frame.iloc[np.asarray(train_index)][time_col]).max()
        validation_time = _coerce_time(frame.iloc[np.asarray(validation_index)][time_col]).min()
        if pd.notna(train_time) and pd.notna(validation_time) and train_time >= validation_time:
            raise AssertionError("training observations are not strictly earlier than validation")


def block_bootstrap_confidence_interval(
    values: pd.DataFrame,
    *,
    group_col: str,
    statistic: Callable[[pd.DataFrame], float],
    n_bootstraps: int = 1_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Point estimate and percentile interval from complete-group bootstraps."""

    if group_col not in values:
        raise KeyError(group_col)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    point = float(statistic(values))
    groups = list(pd.unique(values[group_col]))
    if not groups or n_bootstraps <= 0:
        return point, np.nan, np.nan
    positions = {group: np.flatnonzero(values[group_col].to_numpy() == group) for group in groups}
    rng = np.random.default_rng(random_state)
    estimates: list[float] = []
    for _ in range(n_bootstraps):
        sample = rng.choice(groups, size=len(groups), replace=True)
        index = np.concatenate([positions[group] for group in sample])
        try:
            estimate = float(statistic(values.iloc[index]))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return point, np.nan, np.nan
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return point, float(lower), float(upper)


def paired_block_bootstrap_difference(
    values: pd.DataFrame,
    *,
    group_col: str,
    observed_col: str,
    prediction_a_col: str,
    prediction_b_col: str,
    loss: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 1_000,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Return loss(A)-loss(B) with a paired block-bootstrap interval."""

    def difference(sample: pd.DataFrame) -> float:
        observed = sample[observed_col].to_numpy(dtype=float)
        return float(
            loss(observed, sample[prediction_a_col].to_numpy(dtype=float))
            - loss(observed, sample[prediction_b_col].to_numpy(dtype=float))
        )

    return block_bootstrap_confidence_interval(
        values,
        group_col=group_col,
        statistic=difference,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )


def split_conformal_interval(
    calibration_observed: Sequence[float],
    calibration_predicted: Sequence[float],
    test_predicted: Sequence[float],
    *,
    coverage: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Finite-sample split-conformal symmetric prediction interval."""

    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    observed = np.asarray(calibration_observed, dtype=float)
    predicted = np.asarray(calibration_predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    residuals = np.abs(observed[valid] - predicted[valid])
    if residuals.size == 0:
        test = np.asarray(test_predicted, dtype=float)
        return np.full_like(test, np.nan), np.full_like(test, np.nan), np.nan
    # ceil((n+1)*coverage)/n implements the standard finite-sample correction.
    quantile_level = min(1.0, np.ceil((residuals.size + 1) * coverage) / residuals.size)
    radius = float(np.quantile(residuals, quantile_level, method="higher"))
    test = np.asarray(test_predicted, dtype=float)
    return test - radius, test + radius, radius
