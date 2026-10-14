"""Tensor datasets and deterministic synthetic data for DB-PatchKAN."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .patchkan import DBPatchKANConfig


DRIVER_CHANNELS = (
    "speed_mps",
    "accel_mps2",
    "jerk_mps3",
    "torque_nm",
    "accel_pedal_norm",
    "brake_active",
    "rpm_abs",
    "eco",
    "epedal",
    "valid_fraction",
    "dt_s",
    "relative_position",
    "missing_fraction",
)

BATTERY_CHANNELS = (
    "voltage_v",
    "current_discharge_a",
    "soc_pct",
    "pack_temp_avg_c",
    "pack_temp_spread_c",
    "ambient_temp_c",
    "hvac_power_w",
    "valid_fraction",
    "dt_s",
    "relative_position",
    "missing_fraction",
)


def _as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


class EVSegmentDataset(Dataset[dict[str, Tensor]]):
    """In-memory fixed-length EV segment dataset.

    The production preprocessing stage can persist these arrays to compressed
    NPZ.  Labels are optional, allowing the same dataset to support masked
    self-supervised pretraining.
    """

    def __init__(
        self,
        driver: Tensor | np.ndarray,
        battery: Tensor | np.ndarray,
        *,
        driver_mask: Tensor | np.ndarray | None = None,
        battery_mask: Tensor | np.ndarray | None = None,
        concept_targets: Tensor | np.ndarray | None = None,
        concept_mask: Tensor | np.ndarray | None = None,
        observed_concepts: Tensor | np.ndarray | None = None,
        observed_concept_mask: Tensor | np.ndarray | None = None,
        targets: Mapping[str, Tensor | np.ndarray] | None = None,
    ) -> None:
        self.driver = _as_tensor(driver, dtype=torch.float32)
        self.battery = _as_tensor(battery, dtype=torch.float32)
        if self.driver.ndim != 3 or self.battery.ndim != 3:
            raise ValueError("Driver and battery arrays must be [N, T, C]")
        if self.driver.shape[:2] != self.battery.shape[:2]:
            raise ValueError("Driver and battery arrays must share N and T")
        self.n_samples = int(self.driver.shape[0])
        self.driver_mask = self._optional_array(driver_mask, torch.bool, self.driver.shape)
        self.battery_mask = self._optional_array(battery_mask, torch.bool, self.battery.shape)

        self.concept_targets = self._optional_array(
            concept_targets, torch.float32, None, leading=self.n_samples
        )
        self.concept_mask = self._optional_array(
            concept_mask,
            torch.bool,
            None if self.concept_targets is None else self.concept_targets.shape,
            leading=self.n_samples,
        )
        self.observed_concepts = self._optional_array(
            observed_concepts, torch.float32, None, leading=self.n_samples
        )
        self.observed_concept_mask = self._optional_array(
            observed_concept_mask,
            torch.bool,
            None if self.observed_concepts is None else self.observed_concepts.shape,
            leading=self.n_samples,
        )
        if (
            self.concept_targets is not None
            and self.observed_concepts is not None
            and self.concept_targets.shape != self.observed_concepts.shape
        ):
            raise ValueError("Target and observed concept arrays must have equal shape")

        self.targets: dict[str, Tensor] = {}
        for name, value in (targets or {}).items():
            is_class = name in {
                "behavior_target",
                "behavior_level",
                "driver_target",
                "driver_id_target",
            }
            tensor = _as_tensor(value, dtype=torch.long if is_class else torch.float32)
            if tensor.ndim == 0 or tensor.shape[0] != self.n_samples:
                raise ValueError(f"Target {name!r} must begin with N={self.n_samples}")
            self.targets[name] = tensor

    def _optional_array(
        self,
        value: Tensor | np.ndarray | None,
        dtype: torch.dtype,
        shape: torch.Size | tuple[int, ...] | None,
        *,
        leading: int | None = None,
    ) -> Tensor | None:
        if value is None:
            return None
        tensor = _as_tensor(value, dtype=dtype)
        if shape is not None and tensor.shape != shape:
            # Time-only masks are supported and expanded by the model.
            time_only = len(shape) == 3 and tensor.shape == shape[:2]
            if not time_only:
                raise ValueError(f"Array shape {tuple(tensor.shape)} != expected {tuple(shape)}")
        if leading is not None and (tensor.ndim == 0 or tensor.shape[0] != leading):
            raise ValueError(f"Array must begin with N={leading}")
        return tensor

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item: dict[str, Tensor] = {
            "driver": self.driver[index],
            "battery": self.battery[index],
        }
        optional = {
            "driver_mask": self.driver_mask,
            "battery_mask": self.battery_mask,
            "concept_targets": self.concept_targets,
            "concept_mask": self.concept_mask,
            "observed_concepts": self.observed_concepts,
            "observed_concept_mask": self.observed_concept_mask,
        }
        item.update({name: value[index] for name, value in optional.items() if value is not None})
        item.update({name: value[index] for name, value in self.targets.items()})
        return item

    @classmethod
    def from_npz(cls, path: str | Path) -> "EVSegmentDataset":
        """Load the documented tensor exchange format without pickle objects."""

        with np.load(Path(path), allow_pickle=False) as arrays:
            names = set(arrays.files)

            def choose(*candidates: str) -> np.ndarray | None:
                key = next((candidate for candidate in candidates if candidate in names), None)
                return None if key is None else arrays[key]

            driver = choose("driver", "driver_values")
            battery = choose("battery", "battery_values")
            if driver is None or battery is None:
                raise ValueError("NPZ must contain driver and battery arrays")
            driver_missing = choose("driver_missing")
            battery_missing = choose("battery_missing")
            source_driver_mask = choose("driver_mask")
            source_battery_mask = choose("battery_mask")
            if driver_missing is None and source_driver_mask is not None:
                driver_missing = ~np.asarray(source_driver_mask, dtype=bool)
            if battery_missing is None and source_battery_mask is not None:
                battery_missing = ~np.asarray(source_battery_mask, dtype=bool)
            validity = choose("validity_mask")
            time_delta = choose("time_delta_s", "dt_s")
            relative_position = choose("relative_position")

            # The CPU window exporter keeps the four shared channels separate
            # from the 9/7 physical streams.  Assemble the canonical 13/11
            # tensors here so the exchange format is directly GPU-ready.
            if driver.shape[-1] == 9 and battery.shape[-1] == 7:
                n, time, _ = driver.shape

                def shared(value: np.ndarray | None, default: float) -> np.ndarray:
                    if value is None:
                        return np.full((n, time, 1), default, dtype=np.float32)
                    array = np.asarray(value, dtype=np.float32)
                    if array.ndim == 2:
                        array = array[..., None]
                    if array.shape != (n, time, 1):
                        raise ValueError("Shared sequence metadata must have shape [N,T]")
                    return array

                driver_missing_fraction = (
                    np.asarray(driver_missing, dtype=np.float32).mean(axis=-1, keepdims=True)
                    if driver_missing is not None
                    else np.zeros((n, time, 1), dtype=np.float32)
                )
                battery_missing_fraction = (
                    np.asarray(battery_missing, dtype=np.float32).mean(axis=-1, keepdims=True)
                    if battery_missing is not None
                    else np.zeros((n, time, 1), dtype=np.float32)
                )
                common = (
                    shared(validity, 1.0),
                    shared(time_delta, 1.0),
                    shared(relative_position, 0.0),
                )
                driver = np.concatenate((driver, *common, driver_missing_fraction), axis=-1)
                battery = np.concatenate((battery, *common, battery_missing_fraction), axis=-1)

                if driver_missing is not None:
                    driver_mask = np.concatenate(
                        (
                            ~np.asarray(driver_missing, dtype=bool),
                            *[np.isfinite(part) for part in common],
                            np.isfinite(driver_missing_fraction),
                        ),
                        axis=-1,
                    )
                else:
                    driver_mask = np.isfinite(driver)
                if battery_missing is not None:
                    battery_mask = np.concatenate(
                        (
                            ~np.asarray(battery_missing, dtype=bool),
                            *[np.isfinite(part) for part in common],
                            np.isfinite(battery_missing_fraction),
                        ),
                        axis=-1,
                    )
                else:
                    battery_mask = np.isfinite(battery)
            else:
                driver_mask = source_driver_mask
                battery_mask = source_battery_mask

            reserved = {
                "driver",
                "driver_values",
                "battery",
                "battery_values",
                "driver_mask",
                "battery_mask",
                "driver_missing",
                "battery_missing",
                "validity_mask",
                "time_delta_s",
                "dt_s",
                "relative_position",
                "concept_targets",
                "concept_mask",
                "observed_concepts",
                "observed_concept_mask",
                # Export-only lineage and audit metadata.  These arrays share
                # the sample dimension but are not learning targets; loading
                # them into every GPU batch wastes memory and can silently
                # turn UTF-8 byte matrices into floating-point tensors.
                "window_index",
                "window_start_unix_ms",
                "window_end_unix_ms",
                "segment_id",
                "coverage_pct",
                "window_id_utf8",
                "trip_key_utf8",
                "dataset_id_utf8",
                "vehicle_id_utf8",
            }
            # The tensor exporter writes one scalar audit column per named
            # concept in addition to the canonical concept matrix.  Keep only
            # the matrix in the training dataset.
            reserved.update(
                name
                for name in names
                if name.startswith("concept_") or name.startswith("concept_mask_")
            )
            targets = {name: arrays[name] for name in names - reserved}
            return cls(
                driver,
                battery,
                driver_mask=driver_mask,
                battery_mask=battery_mask,
                concept_targets=choose("concept_targets"),
                concept_mask=choose("concept_mask"),
                observed_concepts=choose("observed_concepts"),
                observed_concept_mask=choose("observed_concept_mask"),
                targets=targets,
            )

    def to_npz(self, path: str | Path, *, compressed: bool = True) -> Path:
        """Persist the tensor exchange format used by Colab and Kaggle."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "driver": self.driver.cpu().numpy(),
            "battery": self.battery.cpu().numpy(),
        }
        optional = {
            "driver_mask": self.driver_mask,
            "battery_mask": self.battery_mask,
            "concept_targets": self.concept_targets,
            "concept_mask": self.concept_mask,
            "observed_concepts": self.observed_concepts,
            "observed_concept_mask": self.observed_concept_mask,
        }
        arrays.update(
            {name: value.cpu().numpy() for name, value in optional.items() if value is not None}
        )
        arrays.update({name: value.cpu().numpy() for name, value in self.targets.items()})
        writer = np.savez_compressed if compressed else np.savez
        writer(destination, **arrays)
        return destination

    def concept_statistics(self) -> tuple[Tensor, Tensor]:
        """Finite mean/std for training-fold concept normalisation."""

        values = self.concept_targets
        if values is None:
            values = self.observed_concepts
        if values is None:
            raise ValueError("No concept arrays are available")
        center = torch.zeros(values.shape[1], dtype=torch.float32)
        scale = torch.ones(values.shape[1], dtype=torch.float32)
        for column in range(values.shape[1]):
            current = values[:, column]
            finite = torch.isfinite(current)
            if torch.any(finite):
                center[column] = current[finite].mean()
                std = current[finite].std(unbiased=False)
                scale[column] = std if std > 1.0e-6 else 1.0
        return center, scale


def _decode_utf8_matrix(values: np.ndarray) -> list[str]:
    """Decode the numeric, pickle-free text representation used by tensor exports."""

    array = np.asarray(values, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("Encoded trip metadata must have shape [N, bytes]")
    return [bytes(row[row != 0]).decode("utf-8") for row in array]


def _collapse_consistent(
    values: Tensor,
    *,
    valid: Tensor,
    missing_value: float | int,
    label: str,
    group_id: tuple[str, str, str],
) -> Tensor:
    """Collapse repeated window labels without averaging inconsistent values."""

    if values.ndim == 1:
        values = values.unsqueeze(-1)
        valid = valid.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    output = torch.full(
        values.shape[1:],
        missing_value,
        dtype=values.dtype,
        device=values.device,
    )
    flat_values = values.reshape(values.shape[0], -1)
    flat_valid = valid.reshape(valid.shape[0], -1)
    flat_output = output.reshape(-1)
    for column in range(flat_values.shape[1]):
        current = flat_values[flat_valid[:, column], column]
        if current.numel() == 0:
            continue
        first = current[0]
        if current.is_floating_point():
            consistent = torch.isclose(current, first, rtol=1.0e-5, atol=1.0e-5).all()
        else:
            consistent = (current == first).all()
        if not bool(consistent):
            raise ValueError(
                f"Inconsistent repeated {label!r} values within trip group {group_id!r}"
            )
        flat_output[column] = first
    return output.squeeze(0) if squeeze else output


class EVTripBagDataset(Dataset[dict[str, Tensor]]):
    """Variable-size bags of overlapping segments, with one label row per trip.

    Group identity is the composite ``(dataset_id, vehicle_id, trip_key)`` so
    common trip numbers from different vehicles can never be mixed.  Repeated
    window labels are collapsed only when they agree; conflicting labels fail
    loudly instead of being averaged.  Train and validation files should be
    loaded independently and checked with :func:`assert_disjoint_trip_bags`.
    """

    def __init__(
        self,
        segments: EVSegmentDataset,
        trip_keys: Sequence[str],
        *,
        dataset_ids: Sequence[str] | None = None,
        vehicle_ids: Sequence[str] | None = None,
        window_starts: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        n_segments = len(segments)
        if len(trip_keys) != n_segments:
            raise ValueError("trip_keys must contain one value per segment")
        dataset_ids = [""] * n_segments if dataset_ids is None else list(dataset_ids)
        vehicle_ids = [""] * n_segments if vehicle_ids is None else list(vehicle_ids)
        if len(dataset_ids) != n_segments or len(vehicle_ids) != n_segments:
            raise ValueError("Dataset and vehicle metadata must match the segment count")
        if n_segments == 0:
            raise ValueError("A trip-bag dataset cannot be empty")

        self.segments = segments
        starts = (
            np.arange(n_segments, dtype=np.int64)
            if window_starts is None
            else np.asarray(window_starts, dtype=np.int64)
        )
        if starts.shape != (n_segments,):
            raise ValueError("window_starts must contain one scalar per segment")

        grouped: dict[tuple[str, str, str], list[int]] = {}
        for index, parts in enumerate(zip(dataset_ids, vehicle_ids, trip_keys, strict=True)):
            group_id = tuple(str(part) for part in parts)
            if not group_id[2]:
                raise ValueError("trip_key cannot be empty")
            grouped.setdefault(group_id, []).append(index)
        ordered = sorted(
            grouped.items(),
            key=lambda pair: (min(starts[pair[1]]), pair[0]),
        )
        self.group_ids = tuple(group_id for group_id, _ in ordered)
        self._bag_indices = tuple(
            torch.as_tensor(
                sorted(indices, key=lambda index: (int(starts[index]), index)),
                dtype=torch.long,
            )
            for _, indices in ordered
        )
        self._trip_level = tuple(
            self._collapse_trip_values(group_id, indices)
            for group_id, indices in zip(self.group_ids, self._bag_indices, strict=True)
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "EVTripBagDataset":
        """Load a segment NPZ and group it by its exported composite trip key."""

        source = Path(path)
        segments = EVSegmentDataset.from_npz(source)
        with np.load(source, allow_pickle=False) as arrays:
            if "trip_key_utf8" not in arrays:
                raise ValueError("Trip-bag loading requires trip_key_utf8 metadata")
            trip_keys = _decode_utf8_matrix(arrays["trip_key_utf8"])
            dataset_ids = (
                _decode_utf8_matrix(arrays["dataset_id_utf8"])
                if "dataset_id_utf8" in arrays
                else None
            )
            vehicle_ids = (
                _decode_utf8_matrix(arrays["vehicle_id_utf8"])
                if "vehicle_id_utf8" in arrays
                else None
            )
            starts = (
                np.asarray(arrays["window_start_unix_ms"], dtype=np.int64).copy()
                if "window_start_unix_ms" in arrays
                else None
            )
        return cls(
            segments,
            trip_keys,
            dataset_ids=dataset_ids,
            vehicle_ids=vehicle_ids,
            window_starts=starts,
        )

    def _collapse_trip_values(
        self, group_id: tuple[str, str, str], indices: Tensor
    ) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        for value_name, mask_name in (
            ("concept_targets", "concept_mask"),
            ("observed_concepts", "observed_concept_mask"),
        ):
            values = getattr(self.segments, value_name)
            if values is None:
                continue
            selected = values[indices]
            source_mask = getattr(self.segments, mask_name)
            valid = torch.isfinite(selected)
            if source_mask is not None:
                valid &= source_mask[indices].bool()
            collapsed = _collapse_consistent(
                selected,
                valid=valid,
                missing_value=float("nan"),
                label=value_name,
                group_id=group_id,
            )
            result[value_name] = collapsed
            result[mask_name] = torch.isfinite(collapsed)

        for name, values in self.segments.targets.items():
            selected = values[indices]
            if name.endswith("_mask"):
                result[name] = selected.bool().any(dim=0)
                continue
            is_class = name in {
                "behavior_target",
                "behavior_level",
                "driver_target",
                "driver_id_target",
            }
            valid = selected.ne(-100) if is_class else torch.isfinite(selected)
            result[name] = _collapse_consistent(
                selected,
                valid=valid,
                missing_value=-100 if is_class else float("nan"),
                label=name,
                group_id=group_id,
            )
        return result

    def __len__(self) -> int:
        return len(self._bag_indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        indices = self._bag_indices[index]
        driver_mask = self.segments.driver_mask
        battery_mask = self.segments.battery_mask
        item: dict[str, Tensor] = {
            "driver": self.segments.driver[indices],
            "battery": self.segments.battery[indices],
            "driver_mask": (
                torch.isfinite(self.segments.driver[indices])
                if driver_mask is None
                else driver_mask[indices]
            ),
            "battery_mask": (
                torch.isfinite(self.segments.battery[indices])
                if battery_mask is None
                else battery_mask[indices]
            ),
            "segment_indices": indices,
            "trip_index": torch.tensor(index, dtype=torch.long),
        }
        item.update(self._trip_level[index])
        return item

    @staticmethod
    def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        return collate_trip_bags(batch)


def collate_trip_bags(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad only the segment dimension; retain exactly one target row per trip."""

    if not batch:
        raise ValueError("Cannot collate an empty trip batch")
    lengths = torch.tensor([item["driver"].shape[0] for item in batch], dtype=torch.long)
    max_segments = int(lengths.max())
    segment_mask = torch.arange(max_segments).unsqueeze(0) < lengths.unsqueeze(1)
    output: dict[str, Tensor] = {
        "segment_mask": segment_mask,
        "segment_count": lengths,
    }
    for name in ("driver", "battery", "driver_mask", "battery_mask"):
        padding = False if name.endswith("_mask") else 0.0
        output[name] = pad_sequence(
            [item[name] for item in batch], batch_first=True, padding_value=padding
        )
    output["segment_indices"] = pad_sequence(
        [item["segment_indices"] for item in batch],
        batch_first=True,
        padding_value=-1,
    )
    sequence_keys = {
        "driver",
        "battery",
        "driver_mask",
        "battery_mask",
        "segment_indices",
    }
    for name in sorted(set.intersection(*(set(item) for item in batch)) - sequence_keys):
        output[name] = torch.stack([item[name] for item in batch])
    return output


def assert_disjoint_trip_bags(
    training: EVTripBagDataset, validation: EVTripBagDataset
) -> None:
    """Reject cross-file trip overlap before a hierarchical training run."""

    overlap = set(training.group_ids) & set(validation.group_ids)
    if overlap:
        examples = sorted(overlap)[:3]
        raise ValueError(
            "Training and validation NPZ files share trip groups; overlapping "
            f"windows would leak across folds. Examples: {examples!r}"
        )


def make_synthetic_dataset(
    config: DBPatchKANConfig | None = None,
    *,
    n_samples: int = 32,
    seed: int = 42,
    missing_probability: float = 0.02,
) -> EVSegmentDataset:
    """Generate deterministic, learnable data for CI and GPU smoke runs."""

    cfg = config or DBPatchKANConfig()
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    generator = torch.Generator().manual_seed(seed)
    n, time = n_samples, cfg.segment_length
    position = torch.linspace(0.0, 1.0, time).view(1, time)
    aggression = torch.rand(n, 1, generator=generator)
    phase = 2 * torch.pi * torch.rand(n, 1, generator=generator)

    speed_kmh = (
        35.0
        + 25.0 * aggression
        + 18.0 * torch.sin(2 * torch.pi * position * 3 + phase)
        + 3.0 * torch.randn(n, time, generator=generator)
    ).clamp(0.0, 130.0)
    speed = speed_kmh / 3.6
    accel = torch.diff(speed, dim=1, prepend=speed[:, :1])
    jerk = torch.diff(accel, dim=1, prepend=accel[:, :1])

    driver = 0.1 * torch.randn(
        n, time, cfg.driver_channels, generator=generator
    )
    if cfg.driver_channels > 0:
        driver[:, :, 0] = speed
    if cfg.driver_channels > 1:
        driver[:, :, 1] = accel
    if cfg.driver_channels > 2:
        driver[:, :, 2] = jerk
    if cfg.driver_channels > 3:
        driver[:, :, 3] = accel * 70.0
    if cfg.driver_channels > 4:
        driver[:, :, 4] = (speed_kmh / 130.0 + accel.clamp_min(0) / 3.0).clamp(0, 1)
    if cfg.driver_channels > 5:
        driver[:, :, 5] = (accel < -0.8).float()
    if cfg.driver_channels > 6:
        driver[:, :, 6] = speed_kmh * 90.0
    if cfg.driver_channels > 7:
        driver[:, :, 7] = (aggression < 0.45).float()
    if cfg.driver_channels > 8:
        driver[:, :, 8] = (aggression < 0.6).float()

    soc_start = 85.0 - 10.0 * torch.rand(n, 1, generator=generator)
    soc = soc_start - position * (4.0 + 7.0 * aggression)
    current = 8.0 + speed_kmh * (0.25 + 0.2 * aggression) + accel.clamp_min(0) * 20.0
    voltage = 380.0 - (85.0 - soc) * 0.45 - current * 0.025
    ambient = 16.0 + 14.0 * torch.rand(n, 1, generator=generator)
    pack_temp = ambient + 5.0 + position * current.mean(dim=1, keepdim=True) * 0.04
    battery = 0.1 * torch.randn(
        n, time, cfg.battery_channels, generator=generator
    )
    structured_battery = (voltage, current, soc, pack_temp)
    for index, value in enumerate(structured_battery):
        if cfg.battery_channels > index:
            battery[:, :, index] = value
    if cfg.battery_channels > 4:
        battery[:, :, 4] = 1.0 + 2.0 * aggression
    if cfg.battery_channels > 5:
        battery[:, :, 5] = ambient.expand(-1, time)
    if cfg.battery_channels > 6:
        battery[:, :, 6] = 0.4 + 1.5 * (ambient > 25.0).float()

    # Populate the four shared metadata channels when the canonical layout is used.
    if cfg.driver_channels >= 13:
        driver[:, :, 9] = 1.0
        driver[:, :, 10] = 1.0
        driver[:, :, 11] = position
        driver[:, :, 12] = 0.0
    if cfg.battery_channels >= 11:
        battery[:, :, 7] = 1.0
        battery[:, :, 8] = 1.0
        battery[:, :, 9] = position
        battery[:, :, 10] = 0.0

    driver_mask = torch.rand(driver.shape, generator=generator) > missing_probability
    battery_mask = torch.rand(battery.shape, generator=generator) > missing_probability
    driver = driver.masked_fill(~driver_mask, float("nan"))
    battery = battery.masked_fill(~battery_mask, float("nan"))

    concepts = torch.zeros(n, cfg.n_concepts)
    bsi = (100.0 * aggression.squeeze(1)).clamp(0, 100)
    for column in range(min(6, cfg.n_concepts)):
        concepts[:, column] = (bsi * (0.7 + 0.05 * column)).clamp(0, 100)
    if cfg.n_concepts >= 9:
        behavior_logits = torch.cat(
            (1.0 - aggression, 1.0 - (aggression - 0.5).abs(), aggression), dim=1
        ) * 6.0
        concepts[:, 6:9] = torch.softmax(behavior_logits, dim=1)
    age_days = torch.linspace(0.0, 600.0, n)
    if cfg.n_concepts > 9:
        concepts[:, 9] = age_days
    if cfg.n_concepts > 10:
        concepts[:, 10] = 1000.0 + age_days * 18.0
    if cfg.n_concepts > 11:
        concepts[:, 11] = age_days * (0.01 + 0.01 * aggression.squeeze(1))
    current_soh = (
        98.0
        - 0.006 * age_days
        - 0.012 * bsi
        + 0.05 * torch.randn(n, generator=generator)
    )
    if cfg.n_concepts > 12:
        concepts[:, 12] = current_soh + 0.1
    for column in range(13, cfg.n_concepts):
        concepts[:, column] = (
            (column - 12) * aggression.squeeze(1)
            + 0.05 * torch.randn(n, generator=generator)
        )

    loss_30d = (0.03 + 0.0015 * bsi).clamp_min(0)
    loss_500km = (0.02 + 0.0010 * bsi).clamp_min(0)
    behavior_target = torch.bucketize(
        aggression.squeeze(1), torch.tensor([1.0 / 3.0, 2.0 / 3.0])
    )
    driver_target = torch.arange(n) % cfg.num_drivers
    observed_mask = torch.zeros_like(concepts, dtype=torch.bool)
    if cfg.n_concepts > 9:
        observed_mask[:, 9 : min(13, cfg.n_concepts)] = True

    return EVSegmentDataset(
        driver,
        battery,
        driver_mask=driver_mask,
        battery_mask=battery_mask,
        concept_targets=concepts,
        concept_mask=torch.ones_like(concepts, dtype=torch.bool),
        observed_concepts=concepts,
        observed_concept_mask=observed_mask,
        targets={
            "current_soh_target": current_soh,
            "loss_30d_target": loss_30d,
            "loss_500km_target": loss_500km,
            "behavior_target": behavior_target,
            "driver_target": driver_target,
        },
    )


__all__ = [
    "DRIVER_CHANNELS",
    "BATTERY_CHANNELS",
    "EVSegmentDataset",
    "EVTripBagDataset",
    "collate_trip_bags",
    "assert_disjoint_trip_bags",
    "make_synthetic_dataset",
]
