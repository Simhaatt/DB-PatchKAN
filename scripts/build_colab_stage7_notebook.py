"""Build the Stage 7 external sequence-transfer notebook and upload package."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
UPLOAD = OUTPUT / "stage7_external_sequence_upload"
NOTEBOOK = ROOT / "notebooks" / "08_colab_a100_stage7_external_full_sequence.ipynb"
PACKAGE = OUTPUT / "DBPatchKAN_Stage7_External_Full_Sequence_Upload.zip"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def build_code_zip() -> Path:
    path = UPLOAD / "dbpatchkan_stage7_colab_code.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in ("pyproject.toml", "README.md"):
            archive.write(ROOT / relative, relative)
        for base in (ROOT / "src", ROOT / "scripts"):
            for file in sorted(base.rglob("*")):
                if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                    archive.write(file, file.relative_to(ROOT).as_posix())
    return path


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {"name": NOTEBOOK.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Stage 7 — External full-sequence DB-PatchKAN validation

This notebook performs the missing external experiment:

- RWTH 128-step time-normalized trip sequences derived from 1-Hz telemetry,
  with leave-one-vehicle-out health validation;
- EVE 128-step time-normalized sequences from harmonised telemetry, with
  chronological validation;
- dual-stream Patch Transformer concept pretraining;
- named-concept KAN and parameter-matched MLP health heads;
- inner-fold persistence gating;
- paired cluster bootstrap;
- no access to the locked Nissan test.

Expected A100 runtime is approximately **60–110 minutes**, including the public
EVE download. A learned gate of zero is a valid result: it means the external
training data selected persistence."""
        ),
        code_cell(
            """
# BLOCK 1 — Configuration and runtime verification
from pathlib import Path
from datetime import datetime, timezone
import os, json, shutil, subprocess, sys, time, zipfile

RUN_ID = "stage7_external_full_sequence_seed42"
SEED = 42
CONCEPT_EPOCHS = 18
HEALTH_EPOCHS = 300
BATCH_SIZE = 128
INPUT_ROOT = Path("/content/dbpatchkan_stage7_input")
PROJECT_EXTRACT = Path("/content/dbpatchkan_stage7_project")
RUN_ROOT = Path("/content/dbpatchkan_stage7_runs") / RUN_ID
EVE_RAW = Path("/content/dbpatchkan_eve_raw")
EVE_PREPARED = Path("/content/dbpatchkan_eve_prepared")

def update(message):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)

update("BLOCK 1 starting")
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
EVE_RAW.mkdir(parents=True, exist_ok=True)
EVE_PREPARED.mkdir(parents=True, exist_ok=True)
update(f"Configuration ready: run={RUN_ID}, concept_epochs={CONCEPT_EPOCHS}")
"""
        ),
        code_cell(
            """
# BLOCK 2 — Upload and extract the single Stage 7 package
from google.colab import files

update("BLOCK 2 starting: select DBPatchKAN_Stage7_External_Full_Sequence_Upload.zip")
uploaded = files.upload()
zip_items = [
    (name, payload) for name, payload in uploaded.items()
    if name.lower().endswith(".zip")
]
assert len(zip_items) == 1, f"Upload exactly one ZIP; received {[name for name, _ in zip_items]}"
name, payload = zip_items[0]
package_path = INPUT_ROOT / "stage7_upload.zip"
package_path.write_bytes(payload)
update(f"Received {name} ({len(payload) / 2**20:.2f} MB)")

with zipfile.ZipFile(package_path) as archive:
    archive.extractall(INPUT_ROOT)
required = {
    "dbpatchkan_stage7_colab_code.zip",
    "rwth_external_sequences_128.npz",
    "rwth_driver_soh_intervals.csv",
    "STAGE7_UPLOAD_MANIFEST.json",
}
located = {path.name: path for path in INPUT_ROOT.rglob("*") if path.is_file()}
missing = sorted(required - set(located))
assert not missing, f"Package is incomplete: {missing}"
for required_name in required:
    source = located[required_name]
    destination = INPUT_ROOT / required_name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
update("BLOCK 2 complete: all required files found")
"""
        ),
        code_cell(
            """
# BLOCK 3 — Install project and verify A100
update("BLOCK 3 installing local project")
if PROJECT_EXTRACT.exists():
    shutil.rmtree(PROJECT_EXTRACT)
PROJECT_EXTRACT.mkdir(parents=True)
with zipfile.ZipFile(INPUT_ROOT / "dbpatchkan_stage7_colab_code.zip") as archive:
    archive.extractall(PROJECT_EXTRACT)
roots = list(PROJECT_EXTRACT.rglob("pyproject.toml"))
assert len(roots) == 1, f"Expected one project root, found {roots}"
PROJECT_ROOT = roots[0].parent

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "pyarrow>=14", "scikit-learn>=1.4"],
    check=True,
)
source_root = str(PROJECT_ROOT / "src")
if source_root not in sys.path:
    sys.path.insert(0, source_root)
os.environ["PYTHONPATH"] = source_root + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch, dbpatchkan
assert torch.cuda.is_available(), "Enable Runtime > Change runtime type > A100 GPU"
gpu = torch.cuda.get_device_name(0)
update(f"BLOCK 3 complete: project={PROJECT_ROOT}; GPU={gpu}; torch={torch.__version__}")
"""
        ),
        code_cell(
            """
# BLOCK 4 — Validate immutable inputs and locked-test exclusion
import hashlib

update("BLOCK 4 validating package hashes and scope")
manifest = json.loads((INPUT_ROOT / "STAGE7_UPLOAD_MANIFEST.json").read_text())
for filename, expected in manifest["files"].items():
    path = INPUT_ROOT / filename
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    assert value == expected["sha256"], f"Hash mismatch for {filename}"
assert manifest["nissan_locked_test_included"] is False
assert not any("untouched" in path.name.lower() for path in INPUT_ROOT.rglob("*"))

import numpy as np
with np.load(INPUT_ROOT / "rwth_external_sequences_128.npz", allow_pickle=False) as source:
    update(
        f"RWTH tensors: driver={source['driver'].shape}, "
        f"battery={source['battery'].shape}, concepts={source['concept_targets'].shape}"
    )
update("BLOCK 4 complete: hashes valid; Nissan locked test absent")
"""
        ),
        code_cell(
            """
# BLOCK 5 — Download public EVE raw telemetry with resume and progress
update("BLOCK 5 starting EVE public-data download")
from dbpatchkan.data.eve import download_eve_files

eve_paths = download_eve_files(EVE_RAW, reporter=update, retries=4)
total_mb = sum(path.stat().st_size for path in eve_paths.values()) / 2**20
update(f"BLOCK 5 complete: {len(eve_paths)} EVE signals, {total_mb:.1f} MB")
"""
        ),
        code_cell(
            """
# BLOCK 6 — Harmonise EVE and create true 128-step sequence tensors
update("BLOCK 6 starting EVE harmonisation and sequence construction")
import pandas as pd
from dbpatchkan.data.eve import load_eve_telemetry, eve_monthly_checkpoints
from dbpatchkan.data.external_sequences import build_eve_sequences, save_sequence_npz

telemetry, eve_audit = load_eve_telemetry(eve_paths, reporter=update)
update(f"EVE harmonised rows: {len(telemetry):,}")
eve_arrays, eve_trips = build_eve_sequences(telemetry, length=128, reporter=update)
eve_sequence_path = save_sequence_npz(
    EVE_PREPARED / "eve_external_sequences_128.npz", eve_arrays
)
eve_checkpoints, eve_intervals = eve_monthly_checkpoints(eve_trips, eve_paths["soh"])
eve_trips.to_parquet(EVE_PREPARED / "eve_trip_features.parquet", index=False)
eve_checkpoints.to_csv(EVE_PREPARED / "eve_monthly_soh_checkpoints.csv", index=False)
eve_intervals.to_csv(EVE_PREPARED / "eve_monthly_soh_intervals.csv", index=False)
(EVE_PREPARED / "EVE_SEQUENCE_AUDIT.json").write_text(
    json.dumps({
        **eve_audit,
        "sequence_trips": len(eve_trips),
        "sequence_length": 128,
        "health_intervals": len(eve_intervals),
    }, indent=2)
)
del telemetry, eve_arrays
update(
    f"BLOCK 6 complete: {len(eve_trips):,} trip sequences, "
    f"{len(eve_intervals)} monthly intervals"
)
"""
        ),
        code_cell(
            """
# BLOCK 7 — Run full external sequence validation
update("BLOCK 7 starting Stage 7 training and validation")
command = [
    sys.executable, "-u", "-m", "dbpatchkan.analysis.stage7_external_sequence",
    "--rwth-sequences", str(INPUT_ROOT / "rwth_external_sequences_128.npz"),
    "--rwth-intervals", str(INPUT_ROOT / "rwth_driver_soh_intervals.csv"),
    "--eve-sequences", str(eve_sequence_path),
    "--eve-intervals", str(EVE_PREPARED / "eve_monthly_soh_intervals.csv"),
    "--output-dir", str(RUN_ROOT / "stage7_analysis"),
    "--concept-epochs", str(CONCEPT_EPOCHS),
    "--health-epochs", str(HEALTH_EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--seed", str(SEED),
    "--device", "cuda",
]
update("COMMAND: " + " ".join(command))
started = time.time()
process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1, env=os.environ.copy(),
)
for line in process.stdout:
    print(line, end="", flush=True)
return_code = process.wait()
assert return_code == 0, f"Stage 7 failed with return code {return_code}"
update(f"BLOCK 7 complete in {(time.time() - started) / 60:.1f} minutes")
"""
        ),
        code_cell(
            """
# BLOCK 8 — Display health metrics, gates, and statistical comparisons
update("BLOCK 8 loading final results")
analysis_root = RUN_ROOT / "stage7_analysis"
metrics = pd.read_csv(analysis_root / "stage7_external_health_metrics.csv")
bootstrap = pd.read_csv(analysis_root / "stage7_paired_cluster_bootstrap.csv")
gates = pd.read_csv(analysis_root / "stage7_inner_gate_selection.csv")
display(metrics)
display(bootstrap)
display(
    gates[gates.selected.fillna(False)]
    [["outer_fold", "held_out_vehicle", "head", "alpha", "inner_mae", "persistence_inner_mae"]]
)
audit = json.loads((analysis_root / "STAGE7_EXTERNAL_SEQUENCE_AUDIT.json").read_text())
print(json.dumps(audit, indent=2))
update("BLOCK 8 complete")
"""
        ),
        code_cell(
            """
# BLOCK 9 — Create compact result inventory and final scientific verdict
update("BLOCK 9 creating inventory and verdict")
inventory = []
for path in sorted(RUN_ROOT.rglob("*")):
    if path.is_file():
        inventory.append({
            "relative_path": path.relative_to(RUN_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
        })
pd.DataFrame(inventory).to_csv(RUN_ROOT / "artifact_inventory.csv", index=False)

verdict = {
    "full_dual_stream_tested_externally": True,
    "nissan_locked_test_used": False,
    "selection_rule": (
        "The proposed model wins only when its persistence-gated KAN has lower "
        "outer MAE than persistence and the paired cluster-bootstrap CI excludes zero."
    ),
    "important_interpretation": (
        "alpha=0 means external training folds rejected the learned degradation "
        "correction and safely selected persistence."
    ),
}
(RUN_ROOT / "FINAL_INTERPRETATION.json").write_text(json.dumps(verdict, indent=2))
update(f"BLOCK 9 complete: {len(inventory)} artifacts indexed")
"""
        ),
        code_cell(
            """
# BLOCK 10 — ZIP and automatically download every model and result
update("BLOCK 10 packaging all models and results")
from google.colab import files

final_zip = Path("/content/DBPatchKAN_stage7_EXTERNAL_FULL_SEQUENCE_ALL_MODELS_RESULTS.zip")
if final_zip.exists():
    final_zip.unlink()
shutil.make_archive(str(final_zip.with_suffix("")), "zip", RUN_ROOT.parent, RUN_ROOT.name)
update(f"Final ZIP ready: {final_zip.stat().st_size / 2**20:.2f} MB")
files.download(str(final_zip))
update("BLOCK 10 complete: browser download requested")
"""
        ),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)


def main() -> None:
    UPLOAD.mkdir(parents=True, exist_ok=True)
    code_zip = build_code_zip()
    files = {
        code_zip.name: code_zip,
        "rwth_external_sequences_128.npz": UPLOAD / "rwth_external_sequences_128.npz",
        "rwth_driver_soh_intervals.csv": (
            OUTPUT / "stage4_rwth_harmonisation" / "rwth_driver_soh_intervals.csv"
        ),
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Stage 7 package inputs missing: {missing}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "7_external_full_sequence_transfer_and_persistence_gated_health",
        "nissan_locked_test_included": False,
        "rwth_raw_sequences_included": True,
        "sequence_adapter": (
            "one 128-step time-normalized sequence per trip; RWTH source is 1-Hz"
        ),
        "eve_raw_included": False,
        "eve_download_inside_colab": True,
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for name, path in files.items()
        },
    }
    manifest_path = UPLOAD / "STAGE7_UPLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files[manifest_path.name] = manifest_path
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in files.items():
            archive.write(path, name)
    build_notebook()
    print(json.dumps({
        "notebook": str(NOTEBOOK),
        "package": str(PACKAGE),
        "package_mb": PACKAGE.stat().st_size / 2**20,
        "files": list(files),
        "nissan_locked_test_included": False,
    }, indent=2))


if __name__ == "__main__":
    main()
