"""Add physical lineage and available explanation evidence to selection rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def feature_group(name: str) -> str:
    text = name.casefold()
    rules = (
        ("D-BSI", ("bsi", "stress_dose")),
        ("Thermal", ("temp", "ambient", "hot_", "cold_")),
        ("SOC", ("soc", "dod")),
        ("Cycling", ("ah_", "efc")),
        ("Driver", ("speed", "accel", "decel", "jerk", "pedal", "brake", "torque", "rpm", "eco", "stop", "harsh")),
        ("Electrical", ("current", "power", "energy", "regen", "voltage", "aux_", "hvac")),
        ("Aging", ("odometer", "days_since", "calendar_age", "cumulative_distance")),
        ("Quality", ("coverage", "source_rows", "qc_")),
        ("Context", ("distance", "duration", "moving_time")),
    )
    return next((group for group, tokens in rules if any(token in text for token in tokens)), "Context")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dbpatchkan"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    selection = pd.read_csv(output / "feature_selection.csv")
    dictionary = pd.read_csv(output / "data_dictionary.csv")
    explanation = pd.read_csv(output / "explainability.csv")
    dictionary = dictionary.drop_duplicates("Column_Name").set_index("Column_Name")
    definition = dictionary.get("Definition", pd.Series(dtype=object)).to_dict()
    algorithm = dictionary.get("Formula_or_Algorithm", pd.Series(dtype=object)).to_dict()
    selection["Feature_Group"] = selection["Feature_Name"].astype(str).map(feature_group)
    selection["Source_Columns"] = selection["Feature_Name"].map(algorithm)
    selection["Interpretation"] = selection["Feature_Name"].map(definition)
    fallback_definition = selection["Feature_Name"].astype(str).str.replace("_", " ").str.strip().str.capitalize()
    selection["Interpretation"] = selection["Interpretation"].fillna(
        fallback_definition + "; engineered/audit field documented in the feature pipeline"
    )
    selection["Source_Columns"] = selection["Source_Columns"].fillna(
        "Cleaned telemetry and/or past-only trip history; see features.py"
    )
    selection["Protected_Feature"] = selection["Feature_Name"].astype(str).str.contains(
        r"^(?:odometer_clean_km|bsi_(?:rate|thermal|soc|cycle|transient|interaction)|cumulative_stress_dose)$",
        regex=True,
    )
    for output_column, explanation_type in (
        ("SHAP_Importance", "TreeSHAP"),
        ("EBM_Importance", "EBM_global_term"),
    ):
        values = explanation.loc[
            explanation["Explanation_Type"].eq(explanation_type)
            & explanation["Feature_Name"].notna(),
            ["Task", "Feature_Name", "Importance"],
        ].copy()
        values["Importance"] = pd.to_numeric(values["Importance"], errors="coerce")
        values = values.groupby(["Task", "Feature_Name"], as_index=False)["Importance"].median()
        mapping = values.set_index(["Task", "Feature_Name"])["Importance"].to_dict()
        selection[output_column] = [
            mapping.get((task, feature), np.nan)
            for task, feature in zip(selection["Task"], selection["Feature_Name"], strict=True)
        ]
    for column in ("STG_Gate_Probability", "Channel_Occlusion_Delta"):
        if column not in selection:
            selection[column] = np.nan
    selection.to_csv(output / "feature_selection.csv", index=False)
    print({"rows": len(selection), "with_interpretation": int(selection["Interpretation"].notna().sum())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
