"""Create the one-file Stage 4N-B corrected Nissan upload package."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from build_colab_code_bundle import build_bundle

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "outputs" / "nissan_corrected_upload"
OUTPUT = ROOT / "outputs" / "DBPatchKAN_Nissan_Physics_Corrected_Upload.zip"
V1 = ROOT / "outputs" / "colab_completed" / "stage4n_nissan_physics_hybrid" / "results" / "stage4n_nissan_physics_hybrid"
SEEDS = (42, 52, 62, 72, 82)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    build_bundle(ROOT, BUILD / "dbpatchkan_colab_code.zip")
    sources = {
        ROOT / "outputs" / "dbpatchkan" / "segments_train.npz": BUILD / "segments_train.npz",
        ROOT / "outputs" / "dbpatchkan" / "segments_validation.npz": BUILD / "segments_validation.npz",
        ROOT / "outputs" / "nissan_physics_hybrid" / "nissan_physics_model_table_DEV.parquet": BUILD / "nissan_physics_model_table_DEV.parquet",
    }
    for seed in SEEDS:
        sources[V1 / f"seed_{seed}" / "best.pt"] = BUILD / f"stage4n_v1_seed_{seed}.pt"
    for source, destination in sources.items():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
    files = sorted(
        path for path in BUILD.iterdir()
        if path.is_file() and path.name != "UPLOAD_MANIFEST.json"
    )
    prohibited = [
        path.name for path in files
        if "untouched" in path.name.lower() or "segments_test" in path.name.lower()
    ]
    if prohibited:
        raise AssertionError(prohibited)
    manifest = {
        "stage": "4N-B",
        "architecture": "PI-DB-PatchKAN-v2-corrected",
        "seeds": list(SEEDS),
        "locked_test_included": False,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }
    manifest_path = BUILD / "UPLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted([*files, manifest_path]):
            archive.write(path, arcname=path.name)
    print(OUTPUT)
    print(f"bytes={OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
