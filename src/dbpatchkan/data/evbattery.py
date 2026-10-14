"""Streaming adapter for the public Tsinghua EVBattery field dataset.

The release stores every 128-step charging snippet as a separate legacy
``torch.save`` pickle inside one of three compressed tar archives.  Extracting
the release creates roughly 1.28 million tiny files, which is slow and wasteful
on both Windows and Colab.  This module therefore reads snippets directly from
the archives and converts only selected records to contiguous arrays.

Scientific scope
----------------
EVBattery contains charging telemetry and manufacturer-supplied partial-window
capacity labels.  It does *not* contain driving trajectories.  It can validate
the battery-stream/capacity branch of DB-PatchKAN, but not driver discovery or
the complete driver-to-degradation pathway.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import codecs
import hashlib
import io
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
import pandas as pd
import torch


EVBATTERY_CHANNELS: tuple[str, ...] = (
    "average_cell_voltage_v",
    "charging_current_a",
    "maximum_cell_voltage_v",
    "minimum_cell_voltage_v",
    "maximum_cell_temperature_c",
    "minimum_cell_temperature_c",
    "soc_pct",
    "timestamp_s",
)


@dataclass(frozen=True)
class EVBatteryRecord:
    """One audited charging snippet and its released metadata."""

    manufacturer_id: int
    member_name: str
    snippet_id: str
    sequence: np.ndarray
    vehicle_id: int
    charge_segment: str
    mileage: float
    capacity_ah: float
    anomaly_label: int
    source_label: str


def sha256_file(path: str | Path, *, chunk_bytes: int = 2**20) -> str:
    """Return a reproducible SHA-256 digest without loading a file into RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy_safe_globals() -> list[Any]:
    """Allow only the NumPy objects required by the legacy tensor format.

    ``weights_only=True`` is intentionally retained.  Using a regular pickle
    loader, or ``torch.load(..., weights_only=False)``, would permit arbitrary
    code execution from a malicious dataset file.
    """

    globals_: list[Any] = [
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        (np._core.multiarray._reconstruct, "numpy._core.multiarray._reconstruct"),
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np._core.multiarray.scalar, "numpy._core.multiarray.scalar"),
        np.ndarray,
        np.dtype,
        codecs.encode,
    ]
    globals_.extend(
        value
        for name in dir(np.dtypes)
        if name.endswith("DType")
        and isinstance((value := getattr(np.dtypes, name)), type)
    )
    return globals_


def safe_load_evbattery_payload(payload: bytes) -> tuple[np.ndarray, OrderedDict[str, Any]]:
    """Decode an official EVBattery snippet using PyTorch's restricted loader."""

    with torch.serialization.safe_globals(_numpy_safe_globals()):
        value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"Unexpected EVBattery payload type: {type(value)!r}")
    sequence, metadata = value
    if not isinstance(sequence, np.ndarray) or sequence.shape != (128, 8):
        raise ValueError(f"Expected a (128, 8) sequence, received {getattr(sequence, 'shape', None)}")
    if not isinstance(metadata, (dict, OrderedDict)):
        raise ValueError(f"Unexpected EVBattery metadata type: {type(metadata)!r}")
    array = np.asarray(sequence, dtype="float32")
    if not np.isfinite(array).all():
        array = np.where(np.isfinite(array), array, np.nan).astype("float32")
    return array, OrderedDict(metadata)


def _manufacturer_id(path: str | Path) -> int:
    stem = Path(path).name.casefold()
    for value in (1, 2, 3):
        if f"dataset{value}" in stem:
            return value
    raise ValueError(f"Cannot infer EVBattery manufacturer from {path}")


def read_vehicle_labels(archive_path: str | Path) -> pd.DataFrame:
    """Read the vehicle-level anomaly labels without extracting the archive."""

    archive_path = Path(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        candidates = [
            member for member in archive.getmembers()
            if member.isfile() and member.name.endswith("/label/label.csv")
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one label.csv in {archive_path}, found {len(candidates)}")
        handle = archive.extractfile(candidates[0])
        if handle is None:
            raise OSError(f"Unable to read {candidates[0].name}")
        labels = pd.read_csv(handle)
    labels = labels.loc[:, ~labels.columns.str.startswith("Unnamed:")].copy()
    required = {"car", "label"}
    if required - set(labels):
        raise ValueError(f"Vehicle label table is missing {sorted(required - set(labels))}")
    labels["car"] = pd.to_numeric(labels["car"], errors="raise").astype("int64")
    labels["label"] = pd.to_numeric(labels["label"], errors="raise").astype("int8")
    labels["manufacturer_id"] = _manufacturer_id(archive_path)
    if labels["car"].duplicated().any():
        raise ValueError(f"Duplicate vehicle IDs in {archive_path}")
    return labels[["manufacturer_id", "car", "label"]]


def iter_evbattery_records(
    archive_path: str | Path,
    *,
    capacity_only: bool = False,
    max_members: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    progress_every: int = 5_000,
) -> Iterator[EVBatteryRecord]:
    """Yield snippets sequentially from a compressed archive.

    Parameters
    ----------
    capacity_only:
        Skip records whose released capacity equals zero, the dataset's marker
        for an unavailable capacity label.
    max_members:
        Optional smoke-test bound on the number of pickle members decoded.
    """

    archive_path = Path(archive_path)
    manufacturer = _manufacturer_id(archive_path)
    vehicle_labels = read_vehicle_labels(archive_path).set_index("car")["label"].to_dict()
    decoded = 0
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".pkl"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            sequence, metadata = safe_load_evbattery_payload(handle.read())
            decoded += 1
            if progress is not None and (decoded == 1 or decoded % progress_every == 0):
                progress({
                    "archive": archive_path.name,
                    "decoded": decoded,
                    "member": member.name,
                })
            vehicle = int(metadata["car"])
            metadata_label = str(metadata.get("label", ""))
            anomaly = int(vehicle_labels.get(vehicle, int(metadata_label != "00")))
            capacity = float(metadata.get("capacity", 0.0) or 0.0)
            if not capacity_only or capacity > 0:
                yield EVBatteryRecord(
                    manufacturer_id=manufacturer,
                    member_name=member.name,
                    snippet_id=Path(member.name).stem,
                    sequence=sequence,
                    vehicle_id=vehicle,
                    charge_segment=str(metadata.get("charge_segment", "")),
                    mileage=float(metadata.get("mileage", np.nan)),
                    capacity_ah=capacity,
                    anomaly_label=anomaly,
                    source_label=metadata_label,
                )
            if max_members is not None and decoded >= max_members:
                break


def snippet_concepts(sequence: np.ndarray, *, mileage: float) -> dict[str, float]:
    """Calculate named, interpretable charging concepts from one snippet."""

    values = np.asarray(sequence, dtype="float64")
    if values.shape != (128, 8):
        raise ValueError(f"Expected (128, 8), received {values.shape}")
    avg_v, current, max_v, min_v, max_t, min_t, soc, time_s = values.T
    cell_spread = max_v - min_v
    mean_temp = 0.5 * (max_t + min_t)
    temp_spread = max_t - min_t

    def first_last_delta(x: np.ndarray) -> float:
        finite = x[np.isfinite(x)]
        return float(finite[-1] - finite[0]) if len(finite) >= 2 else np.nan

    dt = np.diff(time_s)
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    return {
        "average_cell_voltage_mean_v": float(np.nanmean(avg_v)),
        "average_cell_voltage_std_v": float(np.nanstd(avg_v)),
        "average_cell_voltage_rise_v": first_last_delta(avg_v),
        "charging_current_mean_a": float(np.nanmean(current)),
        "charging_current_std_a": float(np.nanstd(current)),
        "charging_current_peak_abs_a": float(np.nanmax(np.abs(current))),
        "cell_voltage_spread_mean_v": float(np.nanmean(cell_spread)),
        "cell_voltage_spread_max_v": float(np.nanmax(cell_spread)),
        "cell_temperature_mean_c": float(np.nanmean(mean_temp)),
        "cell_temperature_rise_c": first_last_delta(mean_temp),
        "cell_temperature_spread_mean_c": float(np.nanmean(temp_spread)),
        "cell_temperature_spread_max_c": float(np.nanmax(temp_spread)),
        "soc_start_pct": float(soc[np.flatnonzero(np.isfinite(soc))[0]]) if np.isfinite(soc).any() else np.nan,
        "soc_end_pct": float(soc[np.flatnonzero(np.isfinite(soc))[-1]]) if np.isfinite(soc).any() else np.nan,
        "soc_gain_pct": first_last_delta(soc),
        "duration_s": float(np.nansum(valid_dt)) if len(valid_dt) else np.nan,
        "median_sample_interval_s": float(np.nanmedian(valid_dt)) if len(valid_dt) else np.nan,
        "mileage": float(mileage),
    }


def build_evbattery_sample(
    archive_paths: Sequence[str | Path],
    *,
    capacity_only: bool = True,
    max_members_per_archive: int | None = None,
    max_records_per_vehicle: int = 256,
    seed: int = 42,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build a deterministic vehicle-stratified reservoir sample.

    Limiting records per vehicle prevents vehicles with many overlapping
    sliding windows from dominating training while preserving every vehicle as
    an independent validation unit.
    """

    if max_records_per_vehicle <= 0:
        raise ValueError("max_records_per_vehicle must be positive")
    rng = np.random.default_rng(seed)
    reservoirs: dict[tuple[int, int], list[EVBatteryRecord]] = defaultdict(list)
    seen: dict[tuple[int, int], int] = defaultdict(int)
    for archive_path in archive_paths:
        for record in iter_evbattery_records(
            archive_path,
            capacity_only=capacity_only,
            max_members=max_members_per_archive,
            progress=progress,
        ):
            key = (record.manufacturer_id, record.vehicle_id)
            seen[key] += 1
            bucket = reservoirs[key]
            if len(bucket) < max_records_per_vehicle:
                bucket.append(record)
            else:
                replacement = int(rng.integers(0, seen[key]))
                if replacement < max_records_per_vehicle:
                    bucket[replacement] = record

    records = [record for key in sorted(reservoirs) for record in reservoirs[key]]
    if not records:
        raise ValueError("No EVBattery records matched the requested filters")
    sequences = np.stack([record.sequence for record in records]).astype("float32")
    concepts = pd.DataFrame([
        {
            "manufacturer_id": record.manufacturer_id,
            "vehicle_id": record.vehicle_id,
            "snippet_id": record.snippet_id,
            "charge_segment": record.charge_segment,
            "capacity_ah": record.capacity_ah,
            "anomaly_label": record.anomaly_label,
            "source_label": record.source_label,
            **snippet_concepts(record.sequence, mileage=record.mileage),
        }
        for record in records
    ])
    return {
        "sequence": sequences,
        "concepts": concepts,
        "channels": np.asarray(EVBATTERY_CHANNELS, dtype="U40"),
        "seen_per_vehicle": {f"M{m}_V{v}": int(count) for (m, v), count in seen.items()},
    }


def save_evbattery_sample(path: str | Path, sample: dict[str, Any]) -> Path:
    """Save contiguous tensors and metadata without recreating tiny files."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    concepts = sample["concepts"]
    arrays: dict[str, np.ndarray] = {
        "sequence": np.asarray(sample["sequence"], dtype="float32"),
        "channels": np.asarray(sample["channels"]),
        "concept_columns": np.asarray(concepts.columns, dtype="U80"),
    }
    for column in concepts:
        series = concepts[column]
        if pd.api.types.is_numeric_dtype(series):
            arrays[f"meta__{column}"] = pd.to_numeric(series, errors="coerce").to_numpy()
        else:
            arrays[f"meta__{column}"] = series.fillna("").astype(str).to_numpy(dtype="U80")
    np.savez_compressed(path, **arrays)
    return path


def load_evbattery_sample(path: str | Path) -> dict[str, Any]:
    """Load a contiguous EVBattery cache without enabling pickle support."""

    with np.load(Path(path), allow_pickle=False) as archive:
        sequence = np.asarray(archive["sequence"], dtype="float32")
        channels = np.asarray(archive["channels"])
        columns = [str(value) for value in archive["concept_columns"]]
        concepts = pd.DataFrame({column: archive[f"meta__{column}"] for column in columns})
    if len(sequence) != len(concepts):
        raise ValueError("EVBattery cache sequence/metadata row mismatch")
    return {"sequence": sequence, "channels": channels, "concepts": concepts}
