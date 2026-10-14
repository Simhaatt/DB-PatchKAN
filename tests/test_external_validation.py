from __future__ import annotations

import numpy as np
import pandas as pd

from dbpatchkan.analysis.eve_external import expanding_splits
from dbpatchkan.analysis.rwth_external import _trip_table


def test_expanding_splits_never_use_future_rows() -> None:
    folds = expanding_splits(30, min_train=15, folds=4)
    assert folds
    for train, test in folds:
        assert train.max() < test.min()
        assert set(train).isdisjoint(set(test))


def test_rwth_trip_targets_use_only_physical_outputs() -> None:
    frame = pd.DataFrame({
        "vehicle_id": ["Smart-1"],
        "duration_observed_h": [2.0],
        "distance_km": [100.0],
        "c_rate_discharge_peak": [1.2],
        "high_c_rate_fraction": [0.1],
        "electrothermal_proxy_dose_h": [0.6],
        "thermal_memory_proxy_peak": [0.3],
        "energy_discharge_kwh": [18.0],
        "dbsi_partial_renormalized": [42.0],
    })
    table = _trip_table(frame)
    assert table.loc[0, "vehicle_family"] == "Smart"
    assert np.isclose(table.loc[0, "electrothermal_intensity"], 0.3)
    assert np.isclose(table.loc[0, "energy_kwh_per_100km"], 18.0)
    assert table.loc[0, "dbsi_partial"] == 42.0

