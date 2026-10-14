from __future__ import annotations

from collections import OrderedDict
import io
import tarfile

import numpy as np
import pandas as pd
import torch

from dbpatchkan.data.evbattery import (
    EVBATTERY_CHANNELS,
    build_evbattery_sample,
    iter_evbattery_records,
    load_evbattery_sample,
    read_vehicle_labels,
    safe_load_evbattery_payload,
    save_evbattery_sample,
    snippet_concepts,
)


def _payload(*, car: int, capacity: float) -> bytes:
    sequence = np.zeros((128, 8), dtype="float64")
    sequence[:, 0] = np.linspace(3.8, 3.9, 128)
    sequence[:, 1] = -20.0
    sequence[:, 2] = sequence[:, 0] + 0.01
    sequence[:, 3] = sequence[:, 0] - 0.01
    sequence[:, 4] = np.linspace(25, 27, 128)
    sequence[:, 5] = np.linspace(24, 26, 128)
    sequence[:, 6] = np.linspace(60, 70, 128)
    sequence[:, 7] = np.arange(128) * 10
    buffer = io.BytesIO()
    torch.save(
        (sequence, OrderedDict(label="00", car=car, charge_segment="2", mileage=1234.5, capacity=capacity)),
        buffer,
        _use_new_zipfile_serialization=False,
    )
    return buffer.getvalue()


def _archive(path, manufacturer: int = 1):
    root = f"battery_dataset{manufacturer}"
    with tarfile.open(path, "w:gz") as archive:
        labels = pd.DataFrame({"Unnamed: 0": [0], "car": [7], "label": [0]}).to_csv(index=False).encode()
        info = tarfile.TarInfo(f"{root}/label/label.csv")
        info.size = len(labels)
        archive.addfile(info, io.BytesIO(labels))
        for index, capacity in enumerate((0.0, 38.5, 38.0)):
            payload = _payload(car=7, capacity=capacity)
            info = tarfile.TarInfo(f"{root}/data/{index}.pkl")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_safe_loader_and_concepts():
    sequence, metadata = safe_load_evbattery_payload(_payload(car=7, capacity=38.5))
    assert sequence.shape == (128, len(EVBATTERY_CHANNELS))
    assert sequence.dtype == np.float32
    assert metadata["car"] == 7
    concepts = snippet_concepts(sequence, mileage=1234.5)
    assert np.isclose(concepts["soc_gain_pct"], 10.0)
    assert np.isclose(concepts["cell_voltage_spread_mean_v"], 0.02)
    assert np.isclose(concepts["cell_temperature_rise_c"], 2.0)


def test_stream_and_reservoir(tmp_path):
    archive = tmp_path / "battery_dataset1.tar.gz"
    _archive(archive)
    labels = read_vehicle_labels(archive)
    assert labels.to_dict("records") == [{"manufacturer_id": 1, "car": 7, "label": 0}]
    records = list(iter_evbattery_records(archive, capacity_only=True))
    assert [record.capacity_ah for record in records] == [38.5, 38.0]
    sample = build_evbattery_sample([archive], max_records_per_vehicle=1, seed=42)
    assert sample["sequence"].shape == (1, 128, 8)
    assert sample["concepts"]["vehicle_id"].tolist() == [7]
    cache = save_evbattery_sample(tmp_path / "sample.npz", sample)
    restored = load_evbattery_sample(cache)
    assert restored["sequence"].shape == (1, 128, 8)
    assert restored["concepts"]["vehicle_id"].tolist() == [7]
