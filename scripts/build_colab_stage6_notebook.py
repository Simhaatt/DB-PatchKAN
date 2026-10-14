"""Build the direct-upload Colab A100 Stage 6 notebook and input package."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
NOTEBOOKS = ROOT / "notebooks"
BUILD = OUTPUTS / "stage6_build"
PACKAGE_NAME = "DBPatchKAN_Stage6_Unified_Baselines_Ablations_Upload.zip"


def code_cell(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def build_code_zip(destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in (Path("pyproject.toml"), Path("README.md")):
            archive.write(ROOT / relative, relative.as_posix())
        for path in (ROOT / "src").rglob("*.py"):
            archive.write(path, path.relative_to(ROOT).as_posix())


def build_prior_evidence(destination: Path) -> None:
    wanted = (
        "STAGE1B2_SUMMARY.json", "stage2_head_summary.csv", "stage2_seed_results.csv",
        "STAGE3_SUMMARY.json", "kan_mlp_kanlite_comparison.csv",
        "ACCEPTANCE_DECISION.json", "CORRECTED_NISSAN_SUMMARY.json",
        "corrected_ensemble_metrics.csv", "EXTERNAL_COMBINED_DECISION.json",
        "RWTH_EXTERNAL_VALIDATION_SUMMARY.json", "EVE_EXTERNAL_VALIDATION_SUMMARY.json",
        "LOGROBUST_KAN_SENSITIVITY_DECISION.json", "LOGROBUST_KAN_SENSITIVITY_METRICS.csv",
    )
    zips = list((OUTPUTS / "colab_completed").rglob("*.zip"))
    copied: set[str] = set()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as target:
        for source_path in zips:
            with zipfile.ZipFile(source_path) as source:
                for member in source.infolist():
                    basename = Path(member.filename).name
                    if basename not in wanted or basename in copied:
                        continue
                    target.writestr(f"prior_completed_evidence/{basename}", source.read(member))
                    copied.add(basename)
        target.writestr(
            "prior_completed_evidence/README.json",
            json.dumps({
                "purpose": "Small summaries imported for consolidation only; these experiments are not rerun.",
                "files": sorted(copied),
                "locked_test_used": False,
            }, indent=2),
        )


def build_notebook(destination: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
    }
    nb["cells"] = [
        markdown_cell("""
# Stage 6 — Unified missing baselines and ablations

This notebook uses only the 101 approved Nissan development trips. It does not contain or access the locked 26-trip test split. Completed Stage 1–5 experiments are imported as small evidence summaries and are not repeated.

Expected A100 time: approximately 70–130 minutes. The long ablation block writes partial CSVs and checkpoint markers after every fold, so rerunning that block resumes completed work.
"""),
        code_cell("""
# BLOCK 1 — Configuration and progress helper
from pathlib import Path
from datetime import datetime, timezone
import os, json, shutil, subprocess, sys, time, zipfile

SEED = 42
PRETRAIN_EPOCHS = 3
FINETUNE_EPOCHS = 24
TEMPORAL_EPOCHS = 60
RUN_ID = "stage6_unified_baselines_ablations"
INPUT_ROOT = Path("/content/dbpatchkan_stage6_input")
PROJECT_ROOT = Path("/content/dbpatchkan_stage6_project")
RUN_ROOT = Path("/content/dbpatchkan_stage6_runs") / RUN_ID

def update(message):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] {message}", flush=True)

def run_stream(command, label):
    update(f"START {label}")
    update("COMMAND: " + " ".join(map(str, command)))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(f"{label} failed with exit code {code}")
    update(f"DONE {label}")

update("BLOCK 1 complete")
"""),
        code_cell(f"""
# BLOCK 2 — Upload the single package ZIP (renamed browser copies are accepted)
from google.colab import files

INPUT_ROOT.mkdir(parents=True, exist_ok=True)
update("Upload {PACKAGE_NAME}")
uploaded = files.upload()
zip_uploads = [(name, payload) for name, payload in uploaded.items()
               if name.lower().endswith(".zip")]
assert len(zip_uploads) == 1, f"Upload exactly one ZIP; received: {{list(uploaded)}}"
uploaded_name, payload = zip_uploads[0]
package_path = INPUT_ROOT / "uploaded_stage6_package.zip"
package_path.write_bytes(payload)
with zipfile.ZipFile(package_path) as archive:
    archive.extractall(INPUT_ROOT / "package")

required = {{"dbpatchkan_stage6_code.zip", "segments_train.npz",
            "segments_validation.npz", "stage6_prior_completed_evidence.zip"}}
found = {{path.name: path for path in (INPUT_ROOT / "package").rglob("*") if path.is_file()}}
missing = sorted(required - set(found))
assert not missing, f"Package is incomplete; missing {{missing}}"
FILES = {{name: found[name] for name in required}}
prohibited = [path.name for path in (INPUT_ROOT / "package").rglob("*")
              if path.is_file() and ("untouched" in path.name.lower() or "test" in path.name.lower())]
assert not prohibited, f"Locked/test files are prohibited in Stage 6: {{prohibited}}"
update(f"BLOCK 2 complete; received {{uploaded_name}} ({{len(payload)/2**20:.2f}} MB)")
"""),
        code_cell("""
# BLOCK 3 — Install project and verify A100 environment
if PROJECT_ROOT.exists():
    shutil.rmtree(PROJECT_ROOT)
PROJECT_ROOT.mkdir(parents=True)
with zipfile.ZipFile(FILES["dbpatchkan_stage6_code.zip"]) as archive:
    archive.extractall(PROJECT_ROOT)
run_stream([sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT_ROOT),
            "interpret>=0.6"], "project installation")

import torch, dbpatchkan
assert torch.cuda.is_available(), "Enable Runtime > Change runtime type > A100 GPU"
gpu = torch.cuda.get_device_name(0)
update(f"GPU={gpu}; CUDA={torch.version.cuda}; torch={torch.__version__}")
assert "A100" in gpu, f"A100 requested, but Colab assigned {gpu}"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
(RUN_ROOT / "notebook_settings.json").write_text(json.dumps({
    "seed": SEED, "pretrain_epochs": PRETRAIN_EPOCHS,
    "finetune_epochs": FINETUNE_EPOCHS, "temporal_epochs": TEMPORAL_EPOCHS,
    "locked_test_used": False, "gpu": gpu,
}, indent=2))
update("BLOCK 3 complete")
"""),
        code_cell("""
# BLOCK 4 — Build four chronological development folds
run_stream([
    sys.executable, "-m", "dbpatchkan.analysis.stage6", "--mode", "prepare",
    "--train-data", str(FILES["segments_train.npz"]),
    "--validation-data", str(FILES["segments_validation.npz"]),
    "--output-dir", str(RUN_ROOT), "--seed", str(SEED),
], "Stage 6 fold preparation")
manifest = json.loads((RUN_ROOT / "stage6_fold_manifest.json").read_text())
print(json.dumps({"development_trips": manifest["development_trips"],
                  "fold_counts": [{k:v for k,v in fold.items() if k.startswith("n_")}
                                  for fold in manifest["folds"]],
                  "locked_test_used": manifest["locked_test_used"]}, indent=2))
update("BLOCK 4 complete")
"""),
        code_cell("""
# BLOCK 5 — Missing classical baselines (fold-safe preprocessing)
run_stream([sys.executable, "-m", "dbpatchkan.analysis.stage6", "--mode", "classical",
            "--output-dir", str(RUN_ROOT), "--seed", str(SEED)],
           "classical baseline suite")
import pandas as pd
display(pd.read_csv(RUN_ROOT / "classical_fold_metrics.csv")
        .query("status == 'complete'")
        .groupby(["target", "model"], as_index=False).mae.mean()
        .sort_values(["target", "mae"]))
update("BLOCK 5 complete")
"""),
        code_cell("""
# BLOCK 6 — Missing compact temporal baselines
run_stream([sys.executable, "-m", "dbpatchkan.analysis.stage6", "--mode", "temporal",
            "--output-dir", str(RUN_ROOT), "--seed", str(SEED),
            "--temporal-epochs", str(TEMPORAL_EPOCHS)], "temporal baseline suite")
display(pd.read_csv(RUN_ROOT / "temporal_fold_metrics.csv")
        .groupby(["target", "model"], as_index=False).mae.mean()
        .sort_values(["target", "mae"]))
update("BLOCK 6 complete")
"""),
        code_cell("""
# BLOCK 7 — Missing DB-PatchKAN ablations (long A100 block; resumable)
run_stream([sys.executable, "-m", "dbpatchkan.analysis.stage6", "--mode", "ablations",
            "--output-dir", str(RUN_ROOT), "--seed", str(SEED),
            "--pretrain-epochs", str(PRETRAIN_EPOCHS),
            "--finetune-epochs", str(FINETUNE_EPOCHS)],
           "DB-PatchKAN missing ablations")
display(pd.read_csv(RUN_ROOT / "patchkan_ablation_fold_metrics.csv")
        .groupby(["target", "ablation_family", "variant"], as_index=False).mae.mean()
        .sort_values(["target", "mae"]))
update("BLOCK 7 complete")
"""),
        code_cell("""
# BLOCK 8 — Paired confidence intervals and consolidated decision tables
run_stream([sys.executable, "-m", "dbpatchkan.analysis.stage6", "--mode", "aggregate",
            "--output-dir", str(RUN_ROOT), "--seed", str(SEED)],
           "Stage 6 aggregation")
display(pd.read_csv(RUN_ROOT / "model_comparison_summary.csv")
        .sort_values(["target", "mae_mean"]).head(40))
display(pd.read_csv(RUN_ROOT / "paired_bootstrap_ablation_deltas.csv"))
update("BLOCK 8 complete")
"""),
        code_cell("""
# BLOCK 9 — Add prior completed evidence without rerunning it
prior_root = RUN_ROOT / "prior_completed_evidence"
prior_root.mkdir(exist_ok=True)
with zipfile.ZipFile(FILES["stage6_prior_completed_evidence.zip"]) as archive:
    archive.extractall(RUN_ROOT)
inventory = []
for path in sorted(RUN_ROOT.rglob("*")):
    if path.is_file():
        inventory.append({"relative_path": str(path.relative_to(RUN_ROOT)),
                          "size_bytes": path.stat().st_size})
pd.DataFrame(inventory).to_csv(RUN_ROOT / "artifact_inventory.csv", index=False)
update(f"BLOCK 9 complete; {{len(inventory)}} artifacts inventoried")
"""),
        code_cell("""
# BLOCK 10 — Create and automatically download all Stage 6 models and results
archive_base = Path("/content/DBPatchKAN_stage6_UNIFIED_BASELINES_ABLATIONS_ALL_RESULTS")
archive_path = Path(shutil.make_archive(str(archive_base), "zip", RUN_ROOT.parent, RUN_ROOT.name))
size_mb = archive_path.stat().st_size / 2**20
update(f"Final archive ready: {archive_path.name} ({size_mb:.1f} MB)")
from google.colab import files
files.download(str(archive_path))
update("BLOCK 10 complete; browser download requested")
"""),
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, destination)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    code_zip = BUILD / "dbpatchkan_stage6_code.zip"
    prior_zip = BUILD / "stage6_prior_completed_evidence.zip"
    notebook = NOTEBOOKS / "07_colab_a100_stage6_unified_baselines_ablations.ipynb"
    build_code_zip(code_zip)
    build_prior_evidence(prior_zip)
    build_notebook(notebook)
    package = OUTPUTS / PACKAGE_NAME
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(code_zip, code_zip.name)
        archive.write(prior_zip, prior_zip.name)
        archive.write(OUTPUTS / "dbpatchkan" / "segments_train.npz", "segments_train.npz")
        archive.write(OUTPUTS / "dbpatchkan" / "segments_validation.npz", "segments_validation.npz")
    print(json.dumps({
        "notebook": str(notebook), "package": str(package),
        "package_mb": package.stat().st_size / 2**20,
    }, indent=2))


if __name__ == "__main__":
    main()

