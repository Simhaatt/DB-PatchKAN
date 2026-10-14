"""Create the single-upload Nissan PI-DB-PatchKAN Colab package."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from build_colab_code_bundle import build_bundle


ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "outputs" / "colab_completed" / "stage1b2_full_selection" / "DBPatchKAN_stage1b2_full_selection_ALL_MODELS_RESULTS.zip"
BUILD = ROOT / "outputs" / "nissan_physics_hybrid_upload"
OUTPUT = ROOT / "outputs" / "DBPatchKAN_Nissan_Physics_Hybrid_Upload.zip"
SEEDS = (42, 52, 62, 72, 82)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    code_bundle = BUILD / "dbpatchkan_colab_code.zip"
    build_bundle(ROOT, code_bundle)
    sources = {
        ROOT / "outputs" / "dbpatchkan" / "segments_train.npz": BUILD / "segments_train.npz",
        ROOT / "outputs" / "dbpatchkan" / "segments_validation.npz": BUILD / "segments_validation.npz",
        ROOT / "outputs" / "nissan_physics_hybrid" / "nissan_physics_model_table_DEV.parquet": BUILD / "nissan_physics_model_table_DEV.parquet",
        ROOT / "outputs" / "nissan_physics_hybrid" / "nissan_physics_parameters.json": BUILD / "nissan_physics_parameters.json",
        ROOT / "outputs" / "nissan_physics_hybrid" / "NISSAN_PHYSICS_SPLIT_AUDIT.json": BUILD / "NISSAN_PHYSICS_SPLIT_AUDIT.json",
    }
    for source, destination in sources.items():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)

    with ZipFile(STAGE2) as archive:
        for seed in SEEDS:
            member = (
                "stage1b2_full_selection/stage2/aging_residual/"
                f"seed_{seed}/shared_pretraining/pretrain_latest.pt"
            )
            extracted = BUILD / f"ssl_pretrained_seed_{seed}.pt"
            temporary = BUILD / f"full_pretrain_seed_{seed}.pt"
            temporary.write_bytes(archive.read(member))
            checkpoint = torch.load(temporary, map_location="cpu", weights_only=False)
            torch.save({
                "model": checkpoint["model"],
                "model_config": checkpoint["model_config"],
                "stage": "self_supervised_pretraining_only",
                "seed": seed,
                "soh_labels_used": False,
                "source_member": member,
            }, extracted)
            temporary.unlink()

    required = sorted(
        path for path in BUILD.iterdir()
        if path.is_file() and path.name != "UPLOAD_MANIFEST.json"
    )
    prohibited = [path.name for path in required if "untouched" in path.name.lower() or "segments_test" in path.name.lower()]
    if prohibited:
        raise AssertionError(f"Locked-test files entered upload build: {prohibited}")
    manifest = {
        "package_role": "Nissan PI-DB-PatchKAN development only",
        "seeds": list(SEEDS),
        "pretraining": "self-supervised checkpoints; no SOH-finetuned checkpoint",
        "locked_test_included": False,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in required
        },
    }
    manifest_path = BUILD / "UPLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive_files = [*required, manifest_path]
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(archive_files):
            archive.write(path, arcname=path.name)
    print(OUTPUT)
    print(f"bytes={OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
