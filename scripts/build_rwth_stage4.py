"""Build Stage 4 RWTH harmonisation and driver--SOH interval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from dbpatchkan.data.rwth import (
    build_capacity_checkpoints,
    build_checkpoint_exposures,
    build_driver_soh_intervals,
    build_loov_folds,
    build_trip_features,
    rwth_physics_parameters,
)


def update(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_profile(raw_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in sorted(raw_root.rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        for field in schema:
            records.append({
                "relative_path": path.relative_to(raw_root).as_posix(),
                "database": path.parent.name,
                "file_name": path.name,
                "row_count": parquet.metadata.num_rows,
                "column_name": field.name,
                "data_type": str(field.type),
                "nullable": field.nullable,
            })
    return pd.DataFrame(records)


def source_manifest(source_zip: Path, raw_root: Path) -> dict[str, object]:
    files = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_file():
            files.append({
                "relative_path": path.relative_to(raw_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive": str(source_zip.resolve()),
        "source_archive_bytes": source_zip.stat().st_size,
        "source_archive_sha256": sha256(source_zip),
        "extracted_files": files,
    }


def write_physics_spec(path: Path) -> None:
    path.write_text(
        r"""# Physics-linked driver behaviour to battery-health model

## Structural pathway

Driver behaviour is represented by speed, positive acceleration, deceleration,
jerk, stopping and high-speed exposure. These variables are not connected to
SOH through an unrestricted shortcut in the primary scientific model. They
first produce measured electrical demand and named stress concepts:

\[
P_{batt}(t)=V(t)I(t), \qquad C(t)=\frac{|I(t)|}{Q_{nom}}
\]

The current creates an irreversible throughput dose and a Joule-heating proxy:

\[
D_{Ah}=\int |I(t)|dt, \qquad D_{J}=\int C(t)^2dt.
\]

Because RWTH does not report pack temperature during field operation, the
first-order thermal state is explicitly a latent proxy:

\[
\Theta_k=\Theta_{k-1}e^{-\Delta t/\tau_{th}}+C_k^2
\left(1-e^{-\Delta t/\tau_{th}}\right).
\]

Ambient temperature modulates the proxy with a relative Arrhenius factor:

\[
A(T)=\exp\left[\frac{E_a}{R}\left(\frac{1}{T_{ref}}-\frac{1}{T}\right)\right].
\]

For each pair of controlled capacity tests, only trips after the first test and
before the second test are aggregated. The observed label is:

\[
\Delta SOH_j=SOH_{j,start}-SOH_{j,end}, \qquad
SOH=100\frac{Q_{window}}{Q_{initial,window}}.
\]

The Stage 5 constrained predictor will decompose non-negative future loss into
calendar, cycling, transient and interaction terms:

\[
\widehat{\Delta SOH}=\operatorname{softplus}
\left(g_{cal}+g_{cycle}+g_{transient}+g_{interaction}+r_{KAN}\right).
\]

KAN splines operate on named concepts. Monotonicity is imposed on cumulative
throughput and positive stress dose in the future-loss branch, while a small
regularised residual permits measurement noise and unobserved factors.

## Claim boundary

This design estimates a temporally ordered, physically mediated association.
It does not establish causal driver effects without randomized or sufficiently
rich quasi-experimental data. Pack-temperature effects remain proxies for RWTH
and must be validated on a dataset containing measured battery temperature.
""",
        encoding="utf-8",
    )


def build(raw_root: Path, source_zip: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    update("Profiling parquet schemas and creating immutable manifest")
    schema = schema_profile(raw_root)
    schema.to_csv(output_dir / "rwth_schema_profile.csv", index=False)
    manifest = source_manifest(source_zip, raw_root)
    (output_dir / "rwth_source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    update("Reconstructing fixed-voltage 1C capacity checkpoints")
    checkpoints = build_capacity_checkpoints(raw_root / "capacity_test")
    checkpoints.to_csv(output_dir / "rwth_capacity_checkpoints.csv", index=False)
    checkpoint_audit = checkpoints[[
        "vehicle_id", "test_number", "capacity_window_ah",
        "capacity_window_ah_recalculated", "capacity_recalculation_error_ah",
        "capacity_soh_pct", "label_source",
    ]].copy()
    checkpoint_audit["absolute_recalculation_error_ah"] = checkpoint_audit[
        "capacity_recalculation_error_ah"
    ].abs()
    checkpoint_audit.to_csv(output_dir / "rwth_capacity_recalculation_audit.csv", index=False)

    cached_trips = output_dir / "rwth_trip_physics_features.parquet"
    cached_mapping = output_dir / "rwth_trip_mapping_audit.csv"
    if cached_trips.exists() and cached_mapping.exists():
        update("Loading cached 1 Hz-to-trip physics features")
        trips = pd.read_parquet(cached_trips)
        mapping_audit = pd.read_csv(cached_mapping)
    else:
        update("Mapping 1 Hz field telemetry to published trip intervals")
        trips, mapping_audit = build_trip_features(
            raw_root / "field_test", raw_root / "trip_data"
        )
        trips.to_parquet(cached_trips, index=False)
        trips.to_csv(output_dir / "rwth_trip_physics_features.csv", index=False)
        mapping_audit.to_csv(cached_mapping, index=False)

    update("Building past-only driver exposure between capacity tests")
    intervals = build_driver_soh_intervals(checkpoints, trips)
    intervals.to_csv(output_dir / "rwth_driver_soh_intervals.csv", index=False)
    folds = build_loov_folds(intervals)
    folds.to_csv(output_dir / "rwth_loov_folds.csv", index=False)

    update("Building secondary cumulative exposure at every capacity checkpoint")
    checkpoint_exposures = build_checkpoint_exposures(checkpoints, trips)
    checkpoint_exposures.to_csv(output_dir / "rwth_checkpoint_exposures.csv", index=False)
    checkpoint_fold_records = []
    eligible_checkpoint_vehicles = sorted(
        checkpoint_exposures.loc[
            checkpoint_exposures["trip_count"].gt(0), "vehicle_id"
        ].unique()
    )
    for fold, held_out in enumerate(eligible_checkpoint_vehicles):
        for row in checkpoint_exposures.itertuples(index=False):
            checkpoint_fold_records.append({
                "outer_fold": fold,
                "held_out_vehicle": held_out,
                "checkpoint_id": row.checkpoint_id,
                "vehicle_id": row.vehicle_id,
                "split": "test" if row.vehicle_id == held_out else "train",
            })
    pd.DataFrame(checkpoint_fold_records).to_csv(
        output_dir / "rwth_checkpoint_loov_folds.csv", index=False
    )

    parameters = rwth_physics_parameters()
    (output_dir / "rwth_physics_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )
    write_physics_spec(output_dir / "PHYSICS_LINKED_DRIVER_SOH_MODEL.md")

    repeated = checkpoints.groupby("vehicle_id").size()
    usable_intervals = intervals.loc[intervals["trip_count"].gt(0)]
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "RWTH commercial EV fleet",
        "vehicles": int(checkpoints["vehicle_id"].nunique()),
        "field_rows": int(mapping_audit["source_rows"].sum()),
        "mapped_field_rows": int(mapping_audit["mapped_rows"].sum()),
        "trip_feature_rows": int(len(trips)),
        "primary_capacity_checkpoints": int(len(checkpoints)),
        "vehicles_with_repeated_capacity_tests": int(repeated.gt(1).sum()),
        "degradation_intervals": int(len(intervals)),
        "intervals_with_field_exposure": int(len(usable_intervals)),
        "loov_outer_folds": int(folds["held_out_vehicle"].nunique()),
        "checkpoint_exposure_rows": int(len(checkpoint_exposures)),
        "checkpoints_with_prior_field_exposure": int(
            checkpoint_exposures["trip_count"].gt(0).sum()
        ),
        "checkpoint_loov_vehicles": int(len(eligible_checkpoint_vehicles)),
        "capacity_label": "published fixed-voltage-window 1C discharge capacity",
        "capacity_recalculation_median_abs_error_ah": float(
            checkpoint_audit["absolute_recalculation_error_ah"].median()
        ),
        "capacity_recalculation_max_abs_error_ah": float(
            checkpoint_audit["absolute_recalculation_error_ah"].max()
        ),
        "pack_temperature_observed": False,
        "thermal_model_status": "latent proxy; not measured pack temperature",
        "causal_claim_authorized": False,
        "next_stage": "Stage 5 fold-safe driver-stress-SOH modelling and external validation",
    }
    (output_dir / "STAGE4_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    update(json.dumps(summary, indent=2))

    archive_base = output_dir.parent / "DBPatchKAN_stage4_rwth_harmonisation"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", output_dir.parent, output_dir.name))
    update(f"Stage 4 archive created: {archive_path} ({archive_path.stat().st_size / 2**20:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path,
        default=Path("data/external/rwth/raw"),
    )
    parser.add_argument(
        "--source-zip", type=Path,
        default=Path("data/external/rwth/Electric_Vehicle_and_Battery_Data.zip"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/stage4_rwth_harmonisation"),
    )
    args = parser.parse_args()
    build(args.raw_root.resolve(), args.source_zip.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
