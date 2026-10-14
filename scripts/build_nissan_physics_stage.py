"""Build the leakage-safe Nissan physics and hybrid-model development tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dbpatchkan.data.nissan_physics import (
    append_past_only_exposures,
    nissan_physics_parameters,
    nissan_trip_physics_features,
)


PHYSICS_COLUMNS = [
    "dataset_id", "vehicle_id", "trip_key", "Trip", "datetime_utc", "segment_id",
    "dt_s", "current_std_a", "voltage_v", "soc_pct",
    "pack_temp_avg_clean_c", "ambient_temp_c",
]


def update(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_utf8_matrix(values: np.ndarray) -> list[str]:
    array = np.asarray(values, dtype=np.uint8)
    return [bytes(row[row != 0]).decode("utf-8") for row in array]


def trip_keys_from_npz(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as arrays:
        return set(decode_utf8_matrix(arrays["trip_key_utf8"]))


def build_physics_table(clean_native: Path, cache: Path) -> pd.DataFrame:
    if cache.exists():
        cached = pd.read_parquet(cache)
        if "soc_high_hot_time_h" in cached:
            update(f"Loading cached trip-local physics: {cache.name}")
            return cached
        update("Cached physics predates high-SOC/high-temperature exposure; rebuilding")
    available = set(pq.ParquetFile(clean_native).schema.names)
    missing = sorted(set(PHYSICS_COLUMNS) - available)
    if missing:
        raise ValueError(f"Clean native parquet is missing physics columns: {missing}")
    update("Reading required clean-native physics columns")
    telemetry = pd.read_parquet(clean_native, columns=PHYSICS_COLUMNS)
    update(f"Loaded {len(telemetry):,} native rows; aggregating 127 trips")
    rows: list[dict[str, object]] = []
    groups = telemetry.groupby("trip_key", sort=False, dropna=False)
    for index, (_, group) in enumerate(groups, start=1):
        rows.append(nissan_trip_physics_features(group))
        if index == 1 or index % 10 == 0 or index == groups.ngroups:
            update(f"Physics aggregation {index}/{groups.ngroups} trips")
    result = pd.DataFrame(rows)
    result.to_parquet(cache, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/dbpatchkan"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/nissan_physics_hybrid")
    )
    args = parser.parse_args()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    required = {
        "clean_native": source / "clean_native.parquet",
        "trips": source / "trip_features.csv",
        "health": source / "health_windows.csv",
        "train": source / "segments_train.npz",
        "validation": source / "segments_validation.npz",
        "test": source / "segments_test_UNTOUCHED.npz",
        "split_manifest": source / "gpu_tensor_split_manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Nissan pipeline artifacts: {missing}")

    physics = build_physics_table(
        required["clean_native"], output / "nissan_trip_local_physics.parquet"
    )
    trips = pd.read_csv(required["trips"])
    health = pd.read_csv(required["health"])
    health_targets = health[[
        "trip_key", "health_window_id", "soh_target_trend_pct",
        "target_soh_30d_pct", "target_loss_30d_pp",
        "target_soh_500km_pct", "target_loss_500km_pp",
        "target_loss_rate_pp_per_1000km", "target_quality_flag",
    ]].copy()
    combined = trips.merge(physics, on=["dataset_id", "vehicle_id", "trip_key", "Trip"],
                           how="left", validate="one_to_one", suffixes=("", "_physics"))
    combined = combined.merge(health_targets, on="trip_key", how="left", validate="one_to_one")
    combined["current_soh_target"] = pd.to_numeric(
        combined["soh_target_raw_pct"], errors="coerce"
    )
    combined["current_degradation_from_100_pp"] = (
        100.0 - combined["current_soh_target"]
    ).clip(lower=0.0)
    combined = append_past_only_exposures(combined)
    combined = combined.sort_values("trip_end_utc", kind="mergesort").reset_index(drop=True)
    combined["last_known_soh"] = combined.groupby(
        ["dataset_id", "vehicle_id"], sort=False, dropna=False
    )["current_soh_target"].shift(1)

    train_keys = trip_keys_from_npz(required["train"])
    validation_keys = trip_keys_from_npz(required["validation"])
    test_keys = trip_keys_from_npz(required["test"])
    if train_keys & validation_keys or train_keys & test_keys or validation_keys & test_keys:
        raise AssertionError("Tensor split trip keys overlap")
    combined["split"] = np.select(
        [combined["trip_key"].isin(train_keys), combined["trip_key"].isin(validation_keys),
         combined["trip_key"].isin(test_keys)],
        ["train", "validation", "test_UNTOUCHED"], default="unmapped",
    )
    if set(combined["split"]) != {"train", "validation", "test_UNTOUCHED"}:
        raise AssertionError(f"Unexpected split mapping: {combined['split'].value_counts().to_dict()}")

    # Full table remains local for the eventual one-time locked evaluation.
    full_path = output / "nissan_physics_model_table_FULL_LOCAL.parquet"
    combined.to_parquet(full_path, index=False)
    development = combined.loc[combined["split"].isin(["train", "validation"])].copy()
    if development["split"].eq("test_UNTOUCHED").any():
        raise AssertionError("Locked test rows entered the development table")
    development.to_parquet(output / "nissan_physics_model_table_DEV.parquet", index=False)
    development.to_csv(output / "nissan_physics_model_table_DEV.csv", index=False)
    physics.to_csv(output / "nissan_trip_local_physics.csv", index=False)

    parameters = nissan_physics_parameters()
    (output / "nissan_physics_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )
    split_audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_trips": len(train_keys),
        "validation_trips": len(validation_keys),
        "test_trips_locked_local_only": len(test_keys),
        "development_rows": len(development),
        "development_current_soh_labels": int(development["current_soh_target"].notna().sum()),
        "development_30d_labels": int(development["target_loss_30d_pp"].notna().sum()),
        "development_500km_labels": int(development["target_loss_500km_pp"].notna().sum()),
        "resistance_available_trips": int(development["resistance_estimate_ohm"].notna().sum()),
        "pack_temperature_observed": True,
        "test_rows_exported_to_development_package": 0,
        "source_hashes": {
            name: sha256(path) for name, path in required.items() if path.is_file()
        },
    }
    (output / "NISSAN_PHYSICS_SPLIT_AUDIT.json").write_text(
        json.dumps(split_audit, indent=2), encoding="utf-8"
    )
    update(f"Complete: {json.dumps(split_audit, indent=2)}")


if __name__ == "__main__":
    main()
