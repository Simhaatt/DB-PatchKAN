"""Build Stage 11 PackSE-residual MIL notebook and direct-upload package."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf

from dbpatchkan.analysis.stage10_normalized_transfer import build_normalized_table
from dbpatchkan.analysis.stage11_packse_mil import (
    build_trip_bags,
    canonical_trip_table,
    save_bags,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
UPLOAD = OUTPUT / "stage11_packse_residual_mil_upload"
NOTEBOOK = ROOT / "notebooks" / "11_colab_a100_stage11_packse_residual_mil.ipynb"
PACKAGE = OUTPUT / "DBPatchKAN_Stage11_PackSE_Residual_MIL_Upload.zip"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def build_code_zip() -> Path:
    path = UPLOAD / "dbpatchkan_stage11_colab_code.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in ("pyproject.toml", "README.md"):
            archive.write(ROOT / relative, relative)
        for base in (ROOT / "src", ROOT / "scripts"):
            for file in sorted(base.rglob("*")):
                if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                    archive.write(file, file.relative_to(ROOT).as_posix())
    return path


def prepare_inputs() -> dict[str, Path]:
    stage9 = OUTPUT / "stage9_cpu_attribution_pack_model"
    eve_root = (
        OUTPUT / "colab_completed" / "stage5_rwth_eve_external" / "results"
        / "stage5_rwth_eve_external_validation"
    )
    table = build_normalized_table(
        stage9 / "nissan_development_attribution_intervals.csv",
        stage9 / "rwth_capacity_attribution_intervals.csv",
        stage9 / "eve_monthly_attribution_intervals.csv",
        stage9 / "pack_uncertainty_intervals.csv",
        OUTPUT / "dbpatchkan" / "health_windows.csv",
        OUTPUT / "stage4_rwth_harmonisation" / "rwth_capacity_checkpoints.csv",
        eve_root / "stage5b_eve_harmonised" / "eve_monthly_soh_intervals.csv",
    )
    table_path = UPLOAD / "stage11_interval_table.csv"
    table.to_csv(table_path, index=False)
    source_paths = {
        "nissan_trips": OUTPUT / "dbpatchkan" / "trip_features.csv",
        "rwth_trips": OUTPUT / "stage4_rwth_harmonisation" / "rwth_trip_physics_features.csv",
        "eve_trips": eve_root / "stage5b_eve_harmonised" / "eve_trip_physics_features.csv",
    }
    trips = canonical_trip_table(*source_paths.values())
    bags = build_trip_bags(table, trips, max_tokens=64)
    bag_path = save_bags(UPLOAD / "stage11_trip_bags_64.npz", bags)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "intervals": len(table), "max_tokens": 64,
        "trip_concepts": 12,
        "source_trip_counts_by_dataset": {
            dataset: int(bags["source_trip_count"][table["dataset"].eq(dataset).to_numpy()].sum())
            for dataset in sorted(table["dataset"].unique())
        },
        "source_files": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
            for name, path in source_paths.items()
        },
        "nissan_locked_test_used": False,
    }
    audit_path = UPLOAD / "STAGE11_BAG_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    prior = (
        OUTPUT / "colab_completed" / "stage10_normalized_transfer" / "results"
        / "stage10_normalized_transfer" / "normalized_transfer_metrics.csv"
    )
    return {
        table_path.name: table_path,
        bag_path.name: bag_path,
        audit_path.name: audit_path,
        "prior_stage10_metrics.csv": prior,
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
            r"""# Stage 11 — PackSE-Residual MIL DB-PatchKAN

This is a development redesign motivated by Stage 10. It combines:

- Pack-SE-NMC as a semi-empirical physical degradation baseline;
- up to 64 chronological trip tokens per health interval;
- Transformer-based multiple-instance attention;
- attention-weighted named driver/battery concepts;
- compact residual KAN and matched MLP;
- out-of-vehicle RF teacher distillation;
- inner-fold constrained ensemble of persistence, MIL, RF and Pack-SE;
- dataset-specific BMS/capacity observation calibration.

The KAN receives exactly 20 named concepts: 12 attention-weighted trip concepts
and 8 interval/Pack-SE concepts. The Transformer is used only to assign trip
importance; it does not replace the explainable concept bottleneck.

Scientific status: RWTH and EVE are now development datasets. A successful
Stage 11 result still requires confirmation on a new external fleet. The Nissan
sealed test is absent and must not be reopened.

Expected A100 runtime: approximately **20–45 minutes**."""
        ),
        code_cell(
            """
# BLOCK 1 — Configuration
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, time, zipfile

RUN_ID = "stage11_packse_residual_mil"
EPOCHS = 220
SEEDS = [42, 52, 62, 72, 82]
INPUT_ROOT = Path("/content/dbpatchkan_stage11_input")
PROJECT_EXTRACT = Path("/content/dbpatchkan_stage11_project")
RUN_ROOT = Path("/content/dbpatchkan_stage11_runs") / RUN_ID

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
# BLOCK 2 — Upload the single Stage 11 package
from google.colab import files

update("BLOCK 2: upload DBPatchKAN_Stage11_PackSE_Residual_MIL_Upload.zip")
uploaded = files.upload()
items = [(name, payload) for name, payload in uploaded.items() if name.lower().endswith(".zip")]
assert len(items) == 1, f"Upload exactly one ZIP; received {[x[0] for x in items]}"
name, payload = items[0]
package = INPUT_ROOT / "stage11_upload.zip"
package.write_bytes(payload)
update(f"Received {name} ({len(payload) / 2**20:.2f} MB)")
with zipfile.ZipFile(package) as archive:
    archive.extractall(INPUT_ROOT)
required = {
    "dbpatchkan_stage11_colab_code.zip", "stage11_interval_table.csv",
    "stage11_trip_bags_64.npz", "STAGE11_BAG_AUDIT.json",
    "prior_stage10_metrics.csv", "STAGE11_UPLOAD_MANIFEST.json",
}
located = {path.name: path for path in INPUT_ROOT.rglob("*") if path.is_file()}
missing = sorted(required - set(located))
assert not missing, f"Package incomplete: {missing}"
for filename in required:
    source, destination = located[filename], INPUT_ROOT / filename
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
update("BLOCK 2 complete")
"""
        ),
        code_cell(
            """
# BLOCK 3 — Install project and verify A100
update("BLOCK 3 installing project")
if PROJECT_EXTRACT.exists():
    shutil.rmtree(PROJECT_EXTRACT)
PROJECT_EXTRACT.mkdir(parents=True)
with zipfile.ZipFile(INPUT_ROOT / "dbpatchkan_stage11_colab_code.zip") as archive:
    archive.extractall(PROJECT_EXTRACT)
roots = list(PROJECT_EXTRACT.rglob("pyproject.toml"))
assert len(roots) == 1
PROJECT_ROOT = roots[0].parent
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn>=1.5"], check=True)
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
# BLOCK 4 — Hash validation and locked-test exclusion
update("BLOCK 4 validating immutable inputs")
manifest = json.loads((INPUT_ROOT / "STAGE11_UPLOAD_MANIFEST.json").read_text())
for filename, metadata in manifest["files"].items():
    assert hashlib.sha256((INPUT_ROOT / filename).read_bytes()).hexdigest() == metadata["sha256"]
assert manifest["nissan_locked_test_included"] is False
assert manifest["rwth_eve_status"] == "development"
prohibited = [p.name for p in INPUT_ROOT.rglob("*") if p.is_file() and (
    "untouched" in p.name.lower() or "segments_test" in p.name.lower()
)]
assert not prohibited, f"Locked/test file detected: {prohibited}"
update("BLOCK 4 complete: Nissan locked test absent")
"""
        ),
        code_cell(
            """
# BLOCK 5 — Inspect interval bags and concept structure
update("BLOCK 5 auditing trip bags")
audit = json.loads((INPUT_ROOT / "STAGE11_BAG_AUDIT.json").read_text())
table = pd.read_csv(INPUT_ROOT / "stage11_interval_table.csv")
with np.load(INPUT_ROOT / "stage11_trip_bags_64.npz", allow_pickle=False) as bags:
    print("tokens", bags["tokens"].shape, "mask", bags["token_mask"].shape)
    display(pd.DataFrame({
        "dataset": table.dataset,
        "source_trips": bags["source_trip_count"],
        "compressed_tokens": bags["compressed_token_count"],
    }).groupby("dataset").agg(["count", "sum", "median", "max"]))
print(json.dumps(audit, indent=2))
update("BLOCK 5 complete")
"""
        ),
        code_cell(
            """
# BLOCK 6 — Run Stage 11 PackSE-residual MIL validation
update("BLOCK 6 starting Stage 11")
command = [
    sys.executable, "-u", "-m", "dbpatchkan.analysis.stage11_packse_mil",
    "--interval-table", str(INPUT_ROOT / "stage11_interval_table.csv"),
    "--trip-bags", str(INPUT_ROOT / "stage11_trip_bags_64.npz"),
    "--prior-stage10-metrics", str(INPUT_ROOT / "prior_stage10_metrics.csv"),
    "--output-dir", str(RUN_ROOT), "--epochs", str(EPOCHS),
    "--seeds", *[str(seed) for seed in SEEDS], "--device", "cuda",
]
update("COMMAND: " + " ".join(command))
started = time.time()
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, bufsize=1, env=os.environ.copy())
for line in process.stdout:
    print(line, end="", flush=True)
return_code = process.wait()
assert return_code == 0, f"Stage 11 failed: {return_code}"
update(f"BLOCK 6 complete in {(time.time() - started) / 60:.1f} minutes")
"""
        ),
        code_cell(
            """
# BLOCK 7 — Held-out metrics, persistence tests, and ensemble weights
update("BLOCK 7 loading results")
metrics = pd.read_csv(RUN_ROOT / "stage11_metrics.csv")
bootstrap = pd.read_csv(RUN_ROOT / "stage11_paired_bootstrap.csv")
weights = pd.read_csv(RUN_ROOT / "inner_constrained_ensemble_weights.csv")
display(metrics)
display(bootstrap)
display(weights.groupby(["dataset", "head"])[
    ["weight_persistence", "weight_mil", "weight_rf", "weight_packse"]
].mean().reset_index())
update("BLOCK 7 complete")
"""
        ),
        code_cell(
            """
# BLOCK 8 — Compare Stage 10 and inspect explanations
update("BLOCK 8 comparing stages and inspecting explanations")
prior = pd.read_csv(RUN_ROOT / "prior_stage10_metrics.csv")
attention = pd.read_csv(RUN_ROOT / "mil_attention_audit.csv")
parameters = pd.read_csv(RUN_ROOT / "domain_physics_calibration_parameters.csv")
display(prior)
display(attention.groupby(["dataset", "head"])[
    ["attention_max", "attention_entropy", "source_trips", "tokens"]
].mean().reset_index())
display(parameters.groupby(["calibration_domain", "head"])[
    ["positive_packse_weight", "positive_measurement_scale", "signed_measurement_bias"]
].median().reset_index())
update("BLOCK 8 complete")
"""
        ),
        code_cell(
            """
# BLOCK 9 — Freeze scientific decision and inventory
update("BLOCK 9 writing decision")
primary = "PackSE-Residual-MIL-KAN"
hybrid = "ConstrainedHybrid-MIL-KAN"
wins = bootstrap[(bootstrap.challenger == primary) & bootstrap.significant_win.astype(str).str.lower().eq("true")]
decision = {
    "proposed_framework": "PackSE-Residual MIL DB-PatchKAN",
    "primary_explainable_model": primary,
    "optional_deployment_hybrid": hybrid,
    "significant_wins_over_persistence": wins.dataset.tolist(),
    "rwth_eve_are_development": True,
    "new_external_confirmation_required": True,
    "nissan_locked_test_used": False,
    "claim_rule": "Lower held-out MAE and paired 95% CI below zero; additionally compare against RF.",
}
(RUN_ROOT / "FINAL_STAGE11_DECISION.json").write_text(json.dumps(decision, indent=2))
inventory = []
for path in sorted(RUN_ROOT.rglob("*")):
    if path.is_file():
        inventory.append({"relative_path": path.relative_to(RUN_ROOT).as_posix(), "bytes": path.stat().st_size})
pd.DataFrame(inventory).to_csv(RUN_ROOT / "notebook_artifact_inventory.csv", index=False)
print(json.dumps(decision, indent=2))
update(f"BLOCK 9 complete: {len(inventory)} artifacts")
"""
        ),
        code_cell(
            """
# BLOCK 10 — ZIP and automatically download all models/results
update("BLOCK 10 packaging Stage 11")
from google.colab import files
final_zip = Path("/content/DBPatchKAN_stage11_PACKSE_RESIDUAL_MIL_ALL_MODELS_RESULTS.zip")
if final_zip.exists():
    final_zip.unlink()
shutil.make_archive(str(final_zip.with_suffix("")), "zip", RUN_ROOT.parent, RUN_ROOT.name)
update(f"Final ZIP: {final_zip.stat().st_size / 2**20:.2f} MB")
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
    prepared = prepare_inputs()
    code_zip = build_code_zip()
    files = {code_zip.name: code_zip, **prepared}
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "11_packse_residual_mil_concept_kan",
        "nissan_locked_test_included": False,
        "rwth_eve_status": "development",
        "new_external_confirmation_required": True,
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for name, path in files.items()
        },
    }
    manifest_path = UPLOAD / "STAGE11_UPLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files[manifest_path.name] = manifest_path
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in files.items():
            archive.write(path, name)
    build_notebook()
    shutil.copy2(NOTEBOOK, UPLOAD / NOTEBOOK.name)
    shutil.copy2(PACKAGE, UPLOAD / PACKAGE.name)
    (UPLOAD / "UPLOAD_INSTRUCTIONS.txt").write_text(
        "Open 11_colab_a100_stage11_packse_residual_mil.ipynb with an A100 and run all.\n"
        "Upload only DBPatchKAN_Stage11_PackSE_Residual_MIL_Upload.zip in Block 2.\n"
        "Block 10 downloads the complete models/results ZIP automatically.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "notebook": str(NOTEBOOK), "package": str(PACKAGE),
        "package_mb": PACKAGE.stat().st_size / 2**20,
        "upload_folder": str(UPLOAD), "nissan_locked_test_included": False,
    }, indent=2))


if __name__ == "__main__":
    main()
