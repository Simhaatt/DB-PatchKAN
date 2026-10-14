"""Build the direct-upload package for the combined Stage 5 Colab notebook."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
HARMONISED = OUTPUT / "stage4_rwth_harmonisation"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    code_zip = OUTPUT / "dbpatchkan_stage5_colab_code.zip"
    with zipfile.ZipFile(code_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in ["pyproject.toml", "README.md"]:
            archive.write(ROOT / relative, relative)
        for base in [ROOT / "src", ROOT / "scripts"]:
            for path in sorted(base.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    archive.write(path, path.relative_to(ROOT).as_posix())

    files = {
        "dbpatchkan_stage5_colab_code.zip": code_zip,
        "rwth_trip_physics_features.parquet": HARMONISED / "rwth_trip_physics_features.parquet",
        "rwth_checkpoint_exposures.csv": HARMONISED / "rwth_checkpoint_exposures.csv",
        "rwth_driver_soh_intervals.csv": HARMONISED / "rwth_driver_soh_intervals.csv",
        "rwth_capacity_checkpoints.csv": HARMONISED / "rwth_capacity_checkpoints.csv",
        "rwth_physics_parameters.json": HARMONISED / "rwth_physics_parameters.json",
        "RWTH_STAGE4_SUMMARY.json": HARMONISED / "STAGE4_SUMMARY.json",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Stage 5 package inputs are missing: {missing}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "5A_RWTH_LOOV_plus_5B_EVE_temporal_replication",
        "nissan_locked_test_included": False,
        "nissan_locked_test_used": False,
        "eve_raw_included": False,
        "eve_download": {
            "doi": "https://doi.org/10.57745/5O6QIH",
            "method": "public Dataverse API inside Colab",
            "selected_raw_download_approx_mb": 270,
        },
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for name, path in files.items()
        },
    }
    manifest_path = OUTPUT / "STAGE5_UPLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    package = OUTPUT / "DBPatchKAN_Stage5_RWTH_EVE_External_Validation_Upload.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.write(manifest_path, manifest_path.name)
    print(json.dumps({
        "package": str(package),
        "package_mb": package.stat().st_size / 2**20,
        "files": list(files),
        "nissan_locked_test_included": False,
    }, indent=2))


if __name__ == "__main__":
    main()
