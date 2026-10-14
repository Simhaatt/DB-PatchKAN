"""Build the Stage 10 normalized cross-dataset transfer notebook and package."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
UPLOAD = OUTPUT / "stage10_normalized_transfer_upload"
NOTEBOOK = ROOT / "notebooks" / "10_colab_a100_stage10_normalized_transfer.ipynb"
PACKAGE = OUTPUT / "DBPatchKAN_Stage10_Normalized_Transfer_Upload.zip"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def build_code_zip() -> Path:
    path = UPLOAD / "dbpatchkan_stage10_colab_code.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in ("pyproject.toml", "README.md"):
            archive.write(ROOT / relative, relative)
        for base in (ROOT / "src", ROOT / "scripts"):
            for file in sorted(base.rglob("*")):
                if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                    archive.write(file, file.relative_to(ROOT).as_posix())
    return path


def input_files(code_zip: Path) -> dict[str, Path]:
    stage9 = OUTPUT / "stage9_cpu_attribution_pack_model"
    stage5 = (
        OUTPUT / "colab_completed" / "stage5_rwth_eve_external" / "results"
        / "stage5_rwth_eve_external_validation"
    )
    return {
        code_zip.name: code_zip,
        "nissan_development_attribution_intervals.csv": stage9 / "nissan_development_attribution_intervals.csv",
        "rwth_capacity_attribution_intervals.csv": stage9 / "rwth_capacity_attribution_intervals.csv",
        "eve_monthly_attribution_intervals.csv": stage9 / "eve_monthly_attribution_intervals.csv",
        "pack_uncertainty_intervals.csv": stage9 / "pack_uncertainty_intervals.csv",
        "nissan_health_windows.csv": OUTPUT / "dbpatchkan" / "health_windows.csv",
        "rwth_capacity_checkpoints.csv": OUTPUT / "stage4_rwth_harmonisation" / "rwth_capacity_checkpoints.csv",
        "eve_monthly_soh_intervals.csv": (
            stage5 / "stage5b_eve_harmonised" / "eve_monthly_soh_intervals.csv"
        ),
        "prior_stage7_external_health_metrics.csv": (
            OUTPUT / "colab_completed" / "stage7_external_full_sequence" / "results"
            / "stage7_external_full_sequence_seed42" / "stage7_analysis"
            / "stage7_external_health_metrics.csv"
        ),
    }


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
            r"""# Stage 10 — Vehicle-relative SOH transfer with domain-calibrated KAN

This notebook tests whether harmonising the **degradation target** improves
cross-dataset transfer without pretending that BMS-reported and
capacity-derived SOH are the same measurement.

The shared target is:

\[
\Delta SOH^{rel}=100\frac{SOH_{anchor}-SOH_{future}}{SOH_{anchor}}
\]

The shared concept KAN produces non-negative latent physical degradation.
Separate Nissan-BMS, EVE-BMS and RWTH-capacity heads map it into each
measurement space. Every imputer, scaler, calibration head, persistence gate
and uncertainty estimate is fitted inside its training fold.

Important scope:

- Nissan uses development intervals only; the sealed 26-trip test is absent.
- Stage 10 reuses harmonised named concepts and Pack-SE uncertainty from Stage
  9. It isolates health-target normalisation and does not retrain the temporal
  encoders.
- RWTH uses leave-one-vehicle-out evaluation; Nissan and EVE use expanding
  chronological folds.
- KAN and a parameter-matched MLP are evaluated with five seeds.

Expected A100 runtime: approximately **15–40 minutes**."""
        ),
        code_cell(
            """
# BLOCK 1 — Configuration
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, time, zipfile

RUN_ID = "stage10_normalized_transfer"
EPOCHS = 600
SEEDS = [42, 52, 62, 72, 82]
INPUT_ROOT = Path("/content/dbpatchkan_stage10_input")
PROJECT_ROOT = Path("/content/dbpatchkan_stage10_project")
RUN_ROOT = Path("/content/dbpatchkan_stage10_runs") / RUN_ID

def update(message):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)

update("BLOCK 1 starting")
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
update(f"BLOCK 1 complete: epochs={EPOCHS}, seeds={SEEDS}")
"""
        ),
        code_cell(
            """
# BLOCK 2 — Upload and extract the single Stage 10 package
from google.colab import files

update("BLOCK 2: upload DBPatchKAN_Stage10_Normalized_Transfer_Upload.zip")
uploaded = files.upload()
zip_items = [(name, payload) for name, payload in uploaded.items() if name.lower().endswith(".zip")]
assert len(zip_items) == 1, f"Upload exactly one ZIP; received {[name for name, _ in zip_items]}"
name, payload = zip_items[0]
package = INPUT_ROOT / "stage10_upload.zip"
package.write_bytes(payload)
update(f"Received {name} ({len(payload) / 2**20:.2f} MB)")

with zipfile.ZipFile(package) as archive:
    archive.extractall(INPUT_ROOT)
required = {
    "dbpatchkan_stage10_colab_code.zip",
    "nissan_development_attribution_intervals.csv",
    "rwth_capacity_attribution_intervals.csv",
    "eve_monthly_attribution_intervals.csv",
    "pack_uncertainty_intervals.csv",
    "nissan_health_windows.csv",
    "rwth_capacity_checkpoints.csv",
    "eve_monthly_soh_intervals.csv",
    "prior_stage7_external_health_metrics.csv",
    "STAGE10_UPLOAD_MANIFEST.json",
}
located = {path.name: path for path in INPUT_ROOT.rglob("*") if path.is_file()}
missing = sorted(required - set(located))
assert not missing, f"Stage 10 package is incomplete: {missing}"
for filename in required:
    source = located[filename]
    destination = INPUT_ROOT / filename
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
update("BLOCK 2 complete: package extracted")
"""
        ),
        code_cell(
            """
# BLOCK 3 — Install code and verify GPU
update("BLOCK 3 installing the project")
if PROJECT_ROOT.exists():
    shutil.rmtree(PROJECT_ROOT)
PROJECT_ROOT.mkdir(parents=True)
with zipfile.ZipFile(INPUT_ROOT / "dbpatchkan_stage10_colab_code.zip") as archive:
    archive.extractall(PROJECT_ROOT)
roots = list(PROJECT_ROOT.rglob("pyproject.toml"))
assert len(roots) == 1, f"Expected one project root, found {roots}"
PROJECT_ROOT = roots[0].parent

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "tabulate>=0.9", "scikit-learn>=1.5"],
    check=True,
)
source_root = str(PROJECT_ROOT / "src")
if source_root not in sys.path:
    sys.path.insert(0, source_root)
os.environ["PYTHONPATH"] = source_root + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch, pandas as pd, numpy as np, dbpatchkan
assert torch.cuda.is_available(), "Enable Runtime > Change runtime type > A100 GPU"
update(f"BLOCK 3 complete: GPU={torch.cuda.get_device_name(0)}, torch={torch.__version__}")
"""
        ),
        code_cell(
            """
# BLOCK 4 — Validate hashes and prove locked-test exclusion
update("BLOCK 4 validating immutable inputs")
manifest = json.loads((INPUT_ROOT / "STAGE10_UPLOAD_MANIFEST.json").read_text())
for filename, metadata in manifest["files"].items():
    value = hashlib.sha256((INPUT_ROOT / filename).read_bytes()).hexdigest()
    assert value == metadata["sha256"], f"Hash mismatch: {filename}"
assert manifest["nissan_locked_test_included"] is False
prohibited = [
    path.name for path in INPUT_ROOT.rglob("*")
    if path.is_file() and ("untouched" in path.name.lower() or "segments_test" in path.name.lower())
]
assert not prohibited, f"Locked/test files are prohibited in Stage 10: {prohibited}"
update("BLOCK 4 complete: hashes valid; Nissan locked test is absent")
"""
        ),
        code_cell(
            """
# BLOCK 5 — Reconstruct and audit vehicle-relative targets before training
update("BLOCK 5 constructing normalized targets")
from dbpatchkan.analysis.stage10_normalized_transfer import build_normalized_table

preview = build_normalized_table(
    INPUT_ROOT / "nissan_development_attribution_intervals.csv",
    INPUT_ROOT / "rwth_capacity_attribution_intervals.csv",
    INPUT_ROOT / "eve_monthly_attribution_intervals.csv",
    INPUT_ROOT / "pack_uncertainty_intervals.csv",
    INPUT_ROOT / "nissan_health_windows.csv",
    INPUT_ROOT / "rwth_capacity_checkpoints.csv",
    INPUT_ROOT / "eve_monthly_soh_intervals.csv",
)
display(preview.groupby(["dataset", "label_type"]).size().rename("intervals").reset_index())
display(preview[[
    "dataset", "vehicle_id", "interval_id", "anchor_soh_pct", "future_soh_pct",
    "signed_loss_pp", "relative_interval_loss_pct"
]].head(12))
assert np.allclose(
    preview.anchor_soh_pct * (1 - preview.relative_interval_loss_pct / 100),
    preview.future_soh_pct,
)
update("BLOCK 5 complete: normalization is exactly reversible")
"""
        ),
        code_cell(
            """
# BLOCK 6 — Run fold-safe normalized KAN/MLP transfer
update("BLOCK 6 starting Stage 10 training and validation")
command = [
    sys.executable, "-u", "-m", "dbpatchkan.analysis.stage10_normalized_transfer",
    "--nissan-intervals", str(INPUT_ROOT / "nissan_development_attribution_intervals.csv"),
    "--rwth-intervals", str(INPUT_ROOT / "rwth_capacity_attribution_intervals.csv"),
    "--eve-intervals", str(INPUT_ROOT / "eve_monthly_attribution_intervals.csv"),
    "--pack-uncertainty", str(INPUT_ROOT / "pack_uncertainty_intervals.csv"),
    "--nissan-health", str(INPUT_ROOT / "nissan_health_windows.csv"),
    "--rwth-checkpoints", str(INPUT_ROOT / "rwth_capacity_checkpoints.csv"),
    "--eve-health", str(INPUT_ROOT / "eve_monthly_soh_intervals.csv"),
    "--prior-stage7-metrics", str(INPUT_ROOT / "prior_stage7_external_health_metrics.csv"),
    "--output-dir", str(RUN_ROOT),
    "--epochs", str(EPOCHS),
    "--seeds", *[str(seed) for seed in SEEDS],
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
assert return_code == 0, f"Stage 10 failed with return code {return_code}"
update(f"BLOCK 6 complete in {(time.time() - started) / 60:.1f} minutes")
"""
        ),
        code_cell(
            """
# BLOCK 7 — Display held-out results and persistence tests
update("BLOCK 7 loading held-out results")
metrics = pd.read_csv(RUN_ROOT / "normalized_transfer_metrics.csv")
bootstrap = pd.read_csv(RUN_ROOT / "paired_persistence_bootstrap.csv")
comparison = pd.read_csv(RUN_ROOT / "stage7_vs_stage10_descriptive.csv")
display(metrics)
display(bootstrap)
display(comparison)
update("BLOCK 7 complete")
"""
        ),
        code_cell(
            """
# BLOCK 8 — Inspect domain calibration, gates, and physical/measurement separation
update("BLOCK 8 inspecting interpretable transfer components")
calibration = pd.read_csv(RUN_ROOT / "domain_calibration_parameters.csv")
gates = pd.read_csv(RUN_ROOT / "inner_persistence_gate_selection.csv")
intervals = pd.read_csv(RUN_ROOT / "normalized_cross_dataset_intervals.csv")
display(calibration.groupby(["calibration_domain", "target", "head"])[
    ["positive_scale", "signed_bias"]
].median().reset_index())
display(gates.groupby(["dataset", "head"])[
    ["alpha", "inner_mae_pp", "inner_persistence_mae_pp", "conformal_radius_pp"]
].mean().reset_index())
display(intervals.groupby("dataset")[[
    "physical_relative_loss_pct", "measurement_recalibration_component_pct"
]].agg(["mean", "median", "std"]))
update("BLOCK 8 complete")
"""
        ),
        code_cell(
            """
# BLOCK 9 — Final decision and artifact inventory
update("BLOCK 9 writing final decision")
wins = bootstrap[
    (bootstrap.challenger == "PersistenceAnchored-Normalized-KAN")
    & bootstrap.significant_win.astype(str).str.lower().eq("true")
]
decision = {
    "proposed_model": "PersistenceAnchored-Normalized-DBPatchKAN",
    "normalization": "vehicle-relative degradation with domain measurement calibration",
    "datasets_with_significant_paired_win_over_persistence": wins.dataset.tolist(),
    "global_win_claim_allowed": len(wins) == metrics.dataset.nunique(),
    "nissan_locked_test_used": False,
    "interpretation": (
        "A dataset is won only when held-out MAE is below persistence and the paired "
        "95% bootstrap CI is entirely below zero. Normalization does not guarantee a win."
    ),
}
(RUN_ROOT / "FINAL_STAGE10_DECISION.json").write_text(json.dumps(decision, indent=2))

inventory = []
for path in sorted(RUN_ROOT.rglob("*")):
    if path.is_file():
        inventory.append({
            "relative_path": path.relative_to(RUN_ROOT).as_posix(),
            "bytes": path.stat().st_size,
        })
pd.DataFrame(inventory).to_csv(RUN_ROOT / "notebook_artifact_inventory.csv", index=False)
print(json.dumps(decision, indent=2))
update(f"BLOCK 9 complete: {len(inventory)} artifacts indexed")
"""
        ),
        code_cell(
            """
# BLOCK 10 — ZIP and automatically download every model and result
update("BLOCK 10 packaging all Stage 10 artifacts")
from google.colab import files

final_zip = Path("/content/DBPatchKAN_stage10_NORMALIZED_TRANSFER_ALL_MODELS_RESULTS.zip")
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
    if UPLOAD.exists():
        shutil.rmtree(UPLOAD)
    UPLOAD.mkdir(parents=True)
    code_zip = build_code_zip()
    files = input_files(code_zip)
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Stage 10 package inputs are missing: {missing}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "10_vehicle_relative_normalized_degradation_transfer",
        "nissan_locked_test_included": False,
        "nissan_development_only": True,
        "label_definitions": {
            "Nissan-development": "BMS-reported SOH trend",
            "EVE": "BMS-reported SOH trend",
            "RWTH": "capacity-derived SOH",
        },
        "normalization": "100*(anchor_soh-future_soh)/anchor_soh",
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for name, path in files.items()
        },
    }
    manifest_path = UPLOAD / "STAGE10_UPLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files[manifest_path.name] = manifest_path
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in files.items():
            archive.write(path, name)
    build_notebook()
    shutil.copy2(NOTEBOOK, UPLOAD / NOTEBOOK.name)
    shutil.copy2(PACKAGE, UPLOAD / PACKAGE.name)
    (UPLOAD / "UPLOAD_INSTRUCTIONS.txt").write_text(
        "Open 10_colab_a100_stage10_normalized_transfer.ipynb in Colab with an A100.\n"
        "Run all cells. In Block 2 upload only DBPatchKAN_Stage10_Normalized_Transfer_Upload.zip.\n"
        "Block 10 automatically downloads DBPatchKAN_stage10_NORMALIZED_TRANSFER_ALL_MODELS_RESULTS.zip.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "notebook": str(NOTEBOOK),
        "upload_package": str(PACKAGE),
        "package_mb": PACKAGE.stat().st_size / 2**20,
        "upload_folder": str(UPLOAD),
        "nissan_locked_test_included": False,
        "files": list(files),
    }, indent=2))


if __name__ == "__main__":
    main()
