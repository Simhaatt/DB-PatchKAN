"""Create leakage-safe train/validation/test NPZ files for GPU notebooks.

The split is chronological by whole trip: the final 20% is untouched test
data, and the last expanding development fold is validation.  Behaviour
archetypes and tertile thresholds are fitted on training trips only, then
applied to validation and test trips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from dbpatchkan.analysis.clustering import (
    BehaviorLevelMapper,
    aggression_score_from_clusters,
    discover_behavior_archetypes,
    infer_behavior_features,
)
from dbpatchkan.data.tensors import decode_utf8_rows, export_segment_npz


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.", suffix=".tmp.npz", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _split_npz(source: Path, destination: Path, selected: np.ndarray) -> None:
    with np.load(source, allow_pickle=False) as archive:
        n_samples = int(archive["driver"].shape[0])
        arrays: dict[str, np.ndarray] = {}
        for name in archive.files:
            value = archive[name]
            arrays[name] = value[selected] if value.ndim and value.shape[0] == n_samples else value
    _atomic_npz(destination, arrays)


def _behaviour_table(
    trips: pd.DataFrame,
    train_keys: set[str],
    validation_keys: set[str],
    test_keys: set[str],
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train = trips.loc[trips["trip_key"].isin(train_keys)].copy()
    features = infer_behavior_features(train)
    discovery = discover_behavior_archetypes(
        train,
        feature_names=features,
        bootstrap_iterations=30,
        random_state=seed,
    )
    train_scores, contributions = aggression_score_from_clusters(
        train,
        discovery.labels,
        discovery.probabilities,
        feature_names=features,
    )
    cluster_mapping = contributions.attrs["cluster_mapping"].sort_values(
        "natural_cluster_id"
    )
    intensities = cluster_mapping["cluster_aggression_intensity"].to_numpy(dtype=float)
    mapper = BehaviorLevelMapper().fit(train_scores)

    rows: list[pd.DataFrame] = []
    split_sets = (
        ("train", train_keys),
        ("validation", validation_keys),
        ("untouched_test", test_keys),
    )
    for role, keys in split_sets:
        part = trips.loc[trips["trip_key"].isin(keys)].copy()
        if role == "train":
            labels = discovery.labels
            probabilities = discovery.probabilities
            scores = train_scores
        else:
            labels, probabilities = discovery.predict(part)
            scores = probabilities @ intensities
        part = part[["dataset_id", "vehicle_id", "trip_key", "Trip", "Driver_ID", "trip_start_utc"]]
        part["natural_cluster_id"] = labels
        part["natural_cluster_probability"] = probabilities.max(axis=1)
        part["aggression_score"] = scores
        part["behavior_level"] = mapper.transform(scores)
        part["behavior_target"] = part["behavior_level"].map(
            {"Eco": 0, "Balanced": 1, "Aggressive": 2}
        )
        part["split_role"] = role
        part["clustering_fit_scope"] = "train_only"
        rows.append(part)
    behavior = pd.concat(rows, ignore_index=True).sort_values("trip_start_utc")
    metadata = {
        "fit_trip_count": int(len(train)),
        "behavior_features": features,
        "selected_k": int(discovery.selected_k),
        "bootstrap_ari": float(discovery.bootstrap_ari),
        "exploratory": bool(discovery.exploratory),
        "tertile_lower": float(mapper.thresholds_.lower),
        "tertile_upper": float(mapper.thresholds_.upper),
    }
    candidates = discovery.candidate_metrics.copy()
    candidates["Fit_Scope"] = "gpu_training_trips_only"
    return behavior, candidates, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dbpatchkan"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir.resolve()

    trips = pd.read_csv(output / "trip_features.csv")
    health = pd.read_csv(output / "health_windows.csv")
    trips["trip_start_utc"] = pd.to_datetime(trips["trip_start_utc"], utc=True, errors="coerce")
    trips = trips.sort_values(["trip_start_utc", "trip_key"], kind="mergesort").reset_index(drop=True)
    if trips["trip_key"].duplicated().any():
        raise ValueError("trip_features.csv must contain one row per trip_key")

    n_test = max(1, int(math.ceil(0.20 * len(trips))))
    development = trips.iloc[:-n_test]
    test = trips.iloc[-n_test:]
    n_validation = max(1, len(development) // 5)
    train = development.iloc[:-n_validation]
    validation = development.iloc[-n_validation:]
    split_keys = {
        "train": set(train["trip_key"].astype(str)),
        "validation": set(validation["trip_key"].astype(str)),
        "test": set(test["trip_key"].astype(str)),
    }
    if any(split_keys[left] & split_keys[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("trip split overlap detected")

    behavior, candidates, behavior_meta = _behaviour_table(
        trips,
        split_keys["train"],
        split_keys["validation"],
        split_keys["test"],
        seed=args.seed,
    )
    behavior.to_csv(output / "gpu_behavior_labels.csv", index=False)
    candidates.to_csv(output / "gpu_behavior_cluster_selection.csv", index=False)

    target_columns = [
        "trip_key",
        "target_loss_30d_pp",
        "target_loss_500km_pp",
        "target_soh_30d_pct",
        "target_soh_500km_pct",
        "target_quality_flag",
    ]
    available = [column for column in target_columns if column in health]
    health_targets = health[available].drop_duplicates("trip_key")
    labels = trips.merge(health_targets, on="trip_key", how="left", validate="one_to_one")
    first_time = labels.groupby(["dataset_id", "vehicle_id"])["trip_start_utc"].transform("min")
    labels["calendar_age_days"] = (labels["trip_start_utc"] - first_time).dt.total_seconds() / 86400.0
    labels["last_known_soh"] = labels.groupby(["dataset_id", "vehicle_id"])[
        "soh_target_raw_pct"
    ].shift(1)
    labels = labels.merge(
        behavior[["trip_key", "behavior_level", "behavior_target", "split_role"]],
        on="trip_key",
        how="left",
        validate="one_to_one",
    )
    labels.to_csv(output / "gpu_trip_labels.csv", index=False)

    temporary_all = output / ".segments_all_split_source.npz"
    export_segment_npz(
        output / "telemetry_1hz.parquet",
        temporary_all,
        trip_features=labels,
        behavior_clusters=behavior,
        segment_length=256,
        stride=64,
        min_coverage=0.80,
    )
    with np.load(temporary_all, allow_pickle=False) as archive:
        trip_keys = np.asarray(decode_utf8_rows(archive["trip_key_utf8"]), dtype=object)
        n_windows = int(len(trip_keys))
    files: dict[str, dict[str, object]] = {}
    for role, filename in (
        ("train", "segments_train.npz"),
        ("validation", "segments_validation.npz"),
        ("test", "segments_test_UNTOUCHED.npz"),
    ):
        selected = np.flatnonzero(np.isin(trip_keys, list(split_keys[role])))
        destination = output / filename
        _split_npz(temporary_all, destination, selected)
        with np.load(destination, allow_pickle=False) as archive:
            target_counts = {
                name: int(np.asarray(archive[name], dtype=bool).sum())
                for name in ("current_soh_mask", "loss_30d_mask", "loss_500km_mask")
                if name in archive.files
            }
        files[role] = {
            "path": str(destination),
            "trips": len(split_keys[role]),
            "windows": int(len(selected)),
            "target_windows": target_counts,
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }
    temporary_all.unlink(missing_ok=True)

    manifest = {
        "protocol": "gpu-tensors-v1-whole-trip-chronological",
        "seed": args.seed,
        "total_trips": int(len(trips)),
        "total_windows": n_windows,
        "split_rule": "first 81 train / next 20 validation / final 26 untouched test",
        "test_policy": "excluded from SSL, tuning, normalization, feature fitting, and early stopping",
        "behavior": behavior_meta,
        "files": files,
    }
    (output / "gpu_tensor_split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
