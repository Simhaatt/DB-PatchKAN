"""Build the final Stage 8 Colab notebook and direct-upload package."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
UPLOAD = OUTPUT / "stage8_final_upload"
NOTEBOOK = ROOT / "notebooks" / "09_colab_a100_stage8_final_confirmation.ipynb"
PACKAGE = OUTPUT / "DBPatchKAN_Stage8_Final_Confirmation_Upload.zip"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def build_code_zip() -> Path:
    path = UPLOAD / "dbpatchkan_stage8_colab_code.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in ("pyproject.toml", "README.md"):
            archive.write(ROOT / relative, relative)
        for base in (ROOT / "src", ROOT / "scripts"):
            for file in sorted(base.rglob("*")):
                if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                    archive.write(file, file.relative_to(ROOT).as_posix())
    return path


def prior_files() -> dict[str, Path]:
    return {
        "prior_stage4_corrected_ensemble_metrics.csv": (
            OUTPUT / "colab_completed" / "stage4nb_nissan_physics_corrected"
            / "results" / "stage4nb_nissan_physics_corrected" / "corrected_analysis"
            / "corrected_ensemble_metrics.csv"
        ),
        "prior_stage5_external_combined_metrics.csv": (
            OUTPUT / "colab_completed" / "stage5_rwth_eve_external" / "results"
            / "stage5_rwth_eve_external_validation" / "combined_external_report"
            / "external_combined_metrics_LONG.csv"
        ),
        "prior_stage6_model_summary.csv": (
            OUTPUT / "colab_completed" / "stage6_unified_baselines_ablations"
            / "results" / "stage6_unified_baselines_ablations"
            / "model_comparison_summary.csv"
        ),
        "prior_stage6_ablation_bootstrap.csv": (
            OUTPUT / "colab_completed" / "stage6_unified_baselines_ablations"
            / "results" / "stage6_unified_baselines_ablations"
            / "paired_bootstrap_ablation_deltas.csv"
        ),
        "prior_stage7_external_health_metrics.csv": (
            OUTPUT / "colab_completed" / "stage7_external_full_sequence" / "results"
            / "stage7_external_full_sequence_seed42" / "stage7_analysis"
            / "stage7_external_health_metrics.csv"
        ),
        "prior_stage7_external_bootstrap.csv": (
            OUTPUT / "colab_completed" / "stage7_external_full_sequence" / "results"
            / "stage7_external_full_sequence_seed42" / "stage7_analysis"
            / "stage7_paired_cluster_bootstrap.csv"
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
            """# Stage 8 — Final confirmation and one-time Nissan locked test

This is the last GPU experiment. It performs no architecture search.

1. Corrects the Stage 6 initialization confound for six essential ablations.
2. Trains frozen KAN and matched-MLP models with seeds 42, 52, 62, 72 and 82.
3. Writes a frozen-configuration artifact.
4. Only then opens the 26-trip Nissan locked test for final inference.
5. Seals the locked test and prohibits further tuning.

The locked split has 26 current-SOH labels, 8 valid 500-km loss labels, and no
valid 30-day loss labels. Therefore, the 30-day claim remains development-CV
evidence only.

Expected A100 runtime: approximately **25–60 minutes**."""
        ),
        code_cell(
            """
# BLOCK 1 — Configuration
from pathlib import Path
from datetime import datetime, timezone
import json, os, shutil, subprocess, sys, time, zipfile

RUN_ID = "stage8_final_confirmation"
PRETRAIN_EPOCHS = 5
FINETUNE_EPOCHS = 30
SEEDS = [42, 52, 62, 72, 82]
INPUT_ROOT = Path("/content/dbpatchkan_stage8_input")
PROJECT_EXTRACT = Path("/content/dbpatchkan_stage8_project")
RUN_ROOT = Path("/content/dbpatchkan_stage8_runs") / RUN_ID

def update(message):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)

update("BLOCK 1 starting")
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
update(
    f"BLOCK 1 complete: seeds={SEEDS}, "
    f"pretrain={PRETRAIN_EPOCHS}, finetune={FINETUNE_EPOCHS}"
)
"""
        ),
        code_cell(
            """
# BLOCK 2 — Upload the single Stage 8 package
from google.colab import files

update("BLOCK 2: upload DBPatchKAN_Stage8_Final_Confirmation_Upload.zip")
uploaded = files.upload()
zip_items = [
    (name, payload) for name, payload in uploaded.items()
    if name.lower().endswith(".zip")
]
assert len(zip_items) == 1, f"Upload exactly one ZIP; received {[x[0] for x in zip_items]}"
name, payload = zip_items[0]
package = INPUT_ROOT / "stage8_upload.zip"
package.write_bytes(payload)
update(f"Received {name} ({len(payload) / 2**20:.2f} MB)")

with zipfile.ZipFile(package) as archive:
    archive.extractall(INPUT_ROOT)
required = {
    "dbpatchkan_stage8_colab_code.zip",
    "development_101_trips.npz",
    "segments_test_UNTOUCHED.npz",
    "STAGE8_UPLOAD_MANIFEST.json",
}
located = {path.name: path for path in INPUT_ROOT.rglob("*") if path.is_file()}
missing = sorted(required - set(located))
assert not missing, f"Stage 8 package is incomplete: {missing}"
for filename in required:
    source = located[filename]
    destination = INPUT_ROOT / filename
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
update("BLOCK 2 complete: final package extracted")
"""
        ),
        code_cell(
            """
# BLOCK 3 — Install the project and verify A100
update("BLOCK 3 installing project")
if PROJECT_EXTRACT.exists():
    shutil.rmtree(PROJECT_EXTRACT)
PROJECT_EXTRACT.mkdir(parents=True)
with zipfile.ZipFile(INPUT_ROOT / "dbpatchkan_stage8_colab_code.zip") as archive:
    archive.extractall(PROJECT_EXTRACT)
roots = list(PROJECT_EXTRACT.rglob("pyproject.toml"))
assert len(roots) == 1, f"Expected one project root, found {roots}"
PROJECT_ROOT = roots[0].parent

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "xgboost>=2.0"],
    check=True,
)
source_root = str(PROJECT_ROOT / "src")
if source_root not in sys.path:
    sys.path.insert(0, source_root)
os.environ["PYTHONPATH"] = source_root + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch, pandas as pd, dbpatchkan
assert torch.cuda.is_available(), "Enable Runtime > Change runtime type > A100 GPU"
update(
    f"BLOCK 3 complete: GPU={torch.cuda.get_device_name(0)}, "
    f"torch={torch.__version__}"
)
"""
        ),
        code_cell(
            """
# BLOCK 4 — Verify hashes without opening the locked arrays
import hashlib

update("BLOCK 4 validating immutable inputs")
manifest = json.loads((INPUT_ROOT / "STAGE8_UPLOAD_MANIFEST.json").read_text())
for filename, expected in manifest["files"].items():
    path = INPUT_ROOT / filename
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    assert value == expected["sha256"], f"Hash mismatch: {filename}"

assert manifest["locked_test_included"] is True
assert manifest["locked_test_authorized_by_user"] is True
assert manifest["locked_test_label_counts"] == {
    "current_soh": 26,
    "loss_30d": 0,
    "loss_500km": 8,
}

DEVELOPMENT = INPUT_ROOT / "development_101_trips.npz"
LOCKED_TEST = INPUT_ROOT / "segments_test_UNTOUCHED.npz"

# Intentionally do not call np.load on LOCKED_TEST in this block.
update("BLOCK 4 complete: hashes valid; locked arrays remain unopened")
"""
        ),
        code_cell(
            """
# BLOCK 5 — Development-only corrected ablations and five-seed training
update("BLOCK 5 starting development-only final confirmation")
development_command = [
    sys.executable, "-u", "-m", "dbpatchkan.analysis.stage8_final",
    "--phase", "development",
    "--development-data", str(DEVELOPMENT),
    "--output-dir", str(RUN_ROOT),
    "--pretrain-epochs", str(PRETRAIN_EPOCHS),
    "--finetune-epochs", str(FINETUNE_EPOCHS),
]
update("COMMAND: " + " ".join(development_command))
started = time.time()
process = subprocess.Popen(
    development_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=os.environ.copy(),
)
for line in process.stdout:
    print(line, end="", flush=True)
return_code = process.wait()
assert return_code == 0, f"Development phase failed: {return_code}"
update(f"BLOCK 5 complete in {(time.time() - started) / 60:.1f} minutes")
"""
        ),
        code_cell(
            """
# BLOCK 6 — Review development evidence and verify model freeze
update("BLOCK 6 verifying frozen configuration")
ablation_metrics = pd.read_csv(
    RUN_ROOT / "corrected_ablations" / "corrected_ablation_metrics.csv"
)
ablation_bootstrap = pd.read_csv(
    RUN_ROOT / "corrected_ablations" / "corrected_ablation_bootstrap.csv"
)
seed_summary = pd.read_csv(
    RUN_ROOT / "final_models" / "final_multiseed_validation_summary.csv"
)
freeze = json.loads((RUN_ROOT / "FROZEN_CONFIGURATION.json").read_text())

assert freeze["selection_complete"] is True
assert freeze["locked_test_seen"] is False
assert freeze["registered_models"] == 10

display(
    ablation_metrics.groupby(["variant", "target"], as_index=False)
    .agg(mae_mean=("mae", "mean"), folds=("fold", "nunique"))
    .sort_values(["target", "mae_mean"])
)
display(seed_summary)
display(ablation_bootstrap)
print(json.dumps(freeze, indent=2))
update("BLOCK 6 complete: architecture frozen; locked test still unseen")
"""
        ),
        code_cell(
            """
# BLOCK 7 — Open and evaluate the locked Nissan test exactly once
update("BLOCK 7 opening locked test for final inference only")
locked_command = [
    sys.executable, "-u", "-m", "dbpatchkan.analysis.stage8_final",
    "--phase", "locked_test",
    "--development-data", str(DEVELOPMENT),
    "--locked-test-data", str(LOCKED_TEST),
    "--output-dir", str(RUN_ROOT),
]
started = time.time()
process = subprocess.Popen(
    locked_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=os.environ.copy(),
)
for line in process.stdout:
    print(line, end="", flush=True)
return_code = process.wait()
assert return_code == 0, f"Locked-test phase failed: {return_code}"

decision_command = [
    sys.executable, "-u", "-m", "dbpatchkan.analysis.stage8_final",
    "--phase", "decision",
    "--development-data", str(DEVELOPMENT),
    "--output-dir", str(RUN_ROOT),
]
subprocess.run(decision_command, check=True, env=os.environ.copy())
update(f"BLOCK 7 complete and sealed in {(time.time() - started) / 60:.1f} minutes")
"""
        ),
        code_cell(
            """
# BLOCK 8 — Display final locked-test results
update("BLOCK 8 loading final results")
locked_root = RUN_ROOT / "locked_test_once"
metrics = pd.read_csv(locked_root / "locked_test_metrics.csv")
seed_test_summary = pd.read_csv(
    locked_root / "locked_test_multiseed_summary.csv"
)
bootstrap = pd.read_csv(locked_root / "locked_test_paired_bootstrap.csv")
winners = pd.read_csv(RUN_ROOT / "FINAL_LOCKED_TEST_WINNERS.csv")
access = json.loads((locked_root / "LOCKED_TEST_COMPLETE.json").read_text())

display(metrics)
display(seed_test_summary)
display(bootstrap)
display(winners)
print(json.dumps(access, indent=2))
update("BLOCK 8 complete: these results are final and cannot be tuned")
"""
        ),
        code_cell(
            """
# BLOCK 9 — Consolidate prior evidence and create final inventory
update("BLOCK 9 consolidating thesis evidence")
prior_root = RUN_ROOT / "prior_evidence"
prior_root.mkdir(parents=True, exist_ok=True)
for path in INPUT_ROOT.glob("prior_*.csv"):
    shutil.copy2(path, prior_root / path.name)

inventory = []
for path in sorted(RUN_ROOT.rglob("*")):
    if path.is_file():
        inventory.append({
            "relative_path": path.relative_to(RUN_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
pd.DataFrame(inventory).to_csv(RUN_ROOT / "FINAL_ARTIFACT_INVENTORY.csv", index=False)

final_decision = json.loads((RUN_ROOT / "FINAL_STAGE8_DECISION.json").read_text())
assert final_decision["experiments_complete"] is True
assert final_decision["architecture_tuning_closed"] is True
print(json.dumps(final_decision, indent=2))
update(f"BLOCK 9 complete: {len(inventory)} artifacts indexed")
"""
        ),
        code_cell(
            """
# BLOCK 10 — ZIP and automatically download all final models and results
update("BLOCK 10 packaging final Stage 8 artifacts")
from google.colab import files

final_zip = Path("/content/DBPatchKAN_stage8_FINAL_CONFIRMATION_ALL_MODELS_RESULTS.zip")
if final_zip.exists():
    final_zip.unlink()
shutil.make_archive(
    str(final_zip.with_suffix("")),
    "zip",
    RUN_ROOT.parent,
    RUN_ROOT.name,
)
update(f"Final ZIP ready: {final_zip.stat().st_size / 2**20:.2f} MB")
files.download(str(final_zip))
update("BLOCK 10 complete: download requested; experiments are finished")
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
        "development_101_trips.npz": (
            OUTPUT / "colab_completed" / "stage6_unified_baselines_ablations"
            / "results" / "stage6_unified_baselines_ablations"
            / "development_101_trips.npz"
        ),
        "segments_test_UNTOUCHED.npz": (
            OUTPUT / "dbpatchkan" / "segments_test_UNTOUCHED.npz"
        ),
        **prior_files(),
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Stage 8 package inputs are missing: {missing}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "8_final_confirmation_and_one_time_locked_test",
        "locked_test_included": True,
        "locked_test_authorized_by_user": True,
        "locked_test_policy": (
            "hash verification before freeze; arrays opened only after "
            "development models and configuration are frozen"
        ),
        "locked_test_label_counts": {
            "current_soh": 26,
            "loss_30d": 0,
            "loss_500km": 8,
        },
        "frozen_seeds": [42, 52, 62, 72, 82],
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for name, path in files.items()
        },
    }
    manifest_path = UPLOAD / "STAGE8_UPLOAD_MANIFEST.json"
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
        "locked_test_included": True,
        "locked_test_label_counts": manifest["locked_test_label_counts"],
        "files": list(files),
    }, indent=2))


if __name__ == "__main__":
    main()
