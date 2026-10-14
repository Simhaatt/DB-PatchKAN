"""Generate the Nissan-first PI-DB-PatchKAN A100 notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "04_colab_a100_nissan_physics_hybrid.ipynb"


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip().splitlines(keepends=True)}


def code(source: str) -> dict[str, object]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": dedent(source).strip().splitlines(keepends=True)}


cells = [
    markdown("""
    # Stage 4N — Nissan Physics-Informed DB-PatchKAN

    This notebook completes the proposed hybrid on Nissan development data before
    external validation. It uses dual-stream patch Transformers, a supervised
    twenty-concept physics bottleneck, positive monotonic degradation paths, and
    bounded additive KAN interactions. There is no unrestricted latent-to-SOH shortcut.

    Auxiliary outputs: ambient-adjusted pack-temperature rise, energy/100 km,
    peak C-rate, maximum pack temperature, high-current duration,
    high-SOC/high-temperature duration, and Low/Moderate/High D-BSI stress class.

    The locked final 26-trip tensor is absent and cannot be used by this notebook.
    """),
    markdown("""
    ## Upload one file

    Run Block 2 and select `DBPatchKAN_Nissan_Physics_Hybrid_Upload.zip` without
    extracting it. The package is below 50 MB and contains only train/validation
    tensors, audited physics tables, source code, and five self-supervised checkpoints.
    """),
    code("""
    # BLOCK 1 — Configuration
    from datetime import datetime, timezone

    SEEDS = [42, 52, 62, 72, 82]
    WARMUP_EPOCHS = 20
    END_TO_END_EPOCHS = 20
    BATCH_SIZE = 2
    KAN_GRID = 5
    COVERAGE = 0.90
    RUN_ID = "stage4n_nissan_physics_hybrid"

    print(f"[{datetime.now(timezone.utc).isoformat()}] BLOCK 1 complete")
    print({"seeds": SEEDS, "epochs_per_seed": WARMUP_EPOCHS + END_TO_END_EPOCHS,
           "architecture": "PI-DB-PatchKAN", "locked_test_enabled": False})
    """),
    code("""
    # BLOCK 2 — Upload and robustly extract the one-file package
    import re, shutil, zipfile
    from pathlib import Path
    from datetime import datetime, timezone

    def update(message):
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] {message}", flush=True)

    from google.colab import files
    INPUT_ROOT = Path("/content/dbpatchkan_nissan_physics_input")
    if INPUT_ROOT.exists(): shutil.rmtree(INPUT_ROOT)
    INPUT_ROOT.mkdir(parents=True)
    update("Upload DBPatchKAN_Nissan_Physics_Hybrid_Upload.zip")
    uploaded = files.upload()
    assert uploaded, "No upload received"
    zip_candidates = [(name, payload) for name, payload in uploaded.items() if name.lower().endswith(".zip")]
    assert zip_candidates, "Upload the complete ZIP, not an extracted file"
    uploaded_name, payload = max(zip_candidates, key=lambda item: len(item[1]))
    clean_name = re.sub(r" \\(\\d+\\)(?=\\.[^.]+$)", "", Path(uploaded_name).name)
    package = INPUT_ROOT / clean_name
    package.write_bytes(payload)
    update(f"Received {uploaded_name} ({len(payload)/2**20:.2f} MB)")
    with zipfile.ZipFile(package) as archive: archive.extractall(INPUT_ROOT)

    REQUIRED = {"dbpatchkan_colab_code.zip", "segments_train.npz", "segments_validation.npz",
                "nissan_physics_model_table_DEV.parquet", "UPLOAD_MANIFEST.json",
                *{f"ssl_pretrained_seed_{seed}.pt" for seed in SEEDS}}
    located = {}
    for path in INPUT_ROOT.rglob("*"):
        if path.is_file() and path.name in REQUIRED: located[path.name] = path
    missing = sorted(REQUIRED - set(located))
    assert not missing, f"Incomplete package: {missing}"
    prohibited = [p.name for p in INPUT_ROOT.iterdir() if p.is_file() and
                  ("untouched" in p.name.lower() or "segments_test" in p.name.lower())]
    assert not prohibited, f"Locked-test material is prohibited: {prohibited}"
    RUN_ROOT = Path("/content/dbpatchkan_runs") / RUN_ID
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    update("BLOCK 2 complete; all required development files located")
    """),
    code("""
    # BLOCK 3 — Install and verify project
    import hashlib, json, subprocess, sys, zipfile
    from pathlib import Path

    PROJECT_EXTRACT = Path("/content/dbpatchkan_project")
    if PROJECT_EXTRACT.exists(): shutil.rmtree(PROJECT_EXTRACT)
    PROJECT_EXTRACT.mkdir(parents=True)
    with zipfile.ZipFile(located["dbpatchkan_colab_code.zip"]) as archive:
        archive.extractall(PROJECT_EXTRACT)
    projects = list(PROJECT_EXTRACT.rglob("pyproject.toml"))
    assert len(projects) == 1, projects
    PROJECT_ROOT = projects[0].parent
    update("Installing local project")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT_ROOT)])
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    import torch, dbpatchkan
    assert torch.cuda.is_available(), "Runtime > Change runtime type > A100 GPU"
    gpu = torch.cuda.get_device_name(0)
    manifest = json.loads(located["UPLOAD_MANIFEST.json"].read_text())
    for name, details in manifest["files"].items():
        path = located.get(name, INPUT_ROOT / name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == details["sha256"], f"Hash mismatch: {name}"
    assert manifest["locked_test_included"] is False
    update(f"BLOCK 3 complete; GPU={gpu}; dbpatchkan={dbpatchkan.__version__}")
    """),
    code("""
    # BLOCK 4 — Data/physics audit and architecture smoke test
    import pandas as pd, torch
    from dbpatchkan.models import DBPatchKAN, DBPatchKANConfig, PhysicsInformedDBPatchKAN

    table = pd.read_parquet(located["nissan_physics_model_table_DEV.parquet"])
    assert set(table["split"]) == {"train", "validation"}
    assert len(table) == 101 and table["current_soh_target"].notna().sum() == 101
    assert table["resistance_estimate_ohm"].notna().sum() >= 90
    cfg = DBPatchKANConfig(driver_channels=13, battery_channels=11, segment_length=32,
                           patch_length=8, patch_stride=8, d_model=16,
                           encoder_layers=1, n_heads=4, ff_dim=32, dropout=0.0)
    smoke = PhysicsInformedDBPatchKAN(DBPatchKAN(cfg)).cuda().eval()
    with torch.no_grad():
        out = smoke.forward_trip_bag(torch.zeros(2,1,32,13,device="cuda"),
              torch.zeros(2,1,32,11,device="cuda"),
              segment_mask=torch.ones(2,1,dtype=torch.bool,device="cuda"))
    assert out["physics_concepts"].shape == (2,20) and out["response_predictions"].shape == (2,7)
    assert torch.all(out["loss_500km"] >= 0)
    del smoke, out; torch.cuda.empty_cache()
    update("BLOCK 4 complete: 81 train + 20 validation trips; test absent; hybrid forward valid")
    display(table[["split","current_soh_target","resistance_estimate_ohm",
                   "joule_heat_estimate_kwh","electrothermal_dose_h"]].groupby("split").agg(["count","mean"]))
    """),
    code("""
    # BLOCK 5 — Train five physics-hybrid seeds
    import subprocess, sys, time

    TRAIN = located["segments_train.npz"]
    VALIDATION = located["segments_validation.npz"]
    PHYSICS_TABLE = located["nissan_physics_model_table_DEV.parquet"]
    started_all = time.time()
    for position, seed in enumerate(SEEDS, start=1):
        seed_dir = RUN_ROOT / f"seed_{seed}"
        if (seed_dir / "best.pt").exists():
            update(f"Seed {seed} already complete; skipping")
            continue
        update(f"START seed {seed} ({position}/{len(SEEDS)})")
        command = [sys.executable, "-m", "dbpatchkan.training.physics_hybrid",
            "--train-data", str(TRAIN), "--validation-data", str(VALIDATION),
            "--physics-table", str(PHYSICS_TABLE),
            "--pretrained-checkpoint", str(located[f"ssl_pretrained_seed_{seed}.pt"]),
            "--output-dir", str(seed_dir), "--device", "cuda", "--seed", str(seed),
            "--warmup-epochs", str(WARMUP_EPOCHS),
            "--end-to-end-epochs", str(END_TO_END_EPOCHS),
            "--batch-size", str(BATCH_SIZE), "--num-workers", "2",
            "--kan-grid", str(KAN_GRID), "--resume"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout: print(line, end="", flush=True)
        return_code = process.wait()
        assert return_code == 0, f"Seed {seed} failed with code {return_code}"
        elapsed = (time.time() - started_all) / 60
        update(f"DONE seed {seed}; cumulative elapsed={elapsed:.1f} min")
    update(f"BLOCK 5 complete; total elapsed={(time.time()-started_all)/60:.1f} min")
    """),
    code("""
    # BLOCK 6 — Ensemble, uncertainty, fixed baselines, and mediation
    import subprocess, sys

    ANALYSIS_ROOT = RUN_ROOT / "ensemble_analysis"
    command = [sys.executable, "-m", "dbpatchkan.analysis.nissan_hybrid",
               "--run-root", str(RUN_ROOT), "--physics-table", str(PHYSICS_TABLE),
               "--output-dir", str(ANALYSIS_ROOT), "--coverage", str(COVERAGE)]
    update("START five-seed ensemble and training-only conformal calibration")
    subprocess.check_call(command)
    update("BLOCK 6 complete")
    """),
    code("""
    # BLOCK 7 — Results and thesis figures
    import matplotlib.pyplot as plt
    import numpy as np, pandas as pd

    metrics = pd.read_csv(ANALYSIS_ROOT / "ensemble_metrics.csv")
    baselines = pd.read_csv(ANALYSIS_ROOT / "baseline_metrics.csv")
    intervals = pd.read_csv(ANALYSIS_ROOT / "ensemble_predictions_with_intervals.csv")
    mediation = pd.read_csv(ANALYSIS_ROOT / "driver_stress_soh_mediation.csv")
    display(metrics)
    display(baselines.sort_values(["task","mae"]).groupby("task").head(8))
    display(mediation)

    val = intervals[(intervals.split == "validation") & (intervals.concept_source == "predicted")]
    fig, ax = plt.subplots(figsize=(8,5))
    y = val.current_soh_pred_mean
    lower = y - val.current_soh_interval_low
    upper = val.current_soh_interval_high - y
    ax.errorbar(val.current_soh_true, y, yerr=np.vstack([lower,upper]), fmt="o", alpha=.75, capsize=3)
    limits = [min(val.current_soh_true.min(), y.min())-.5, max(val.current_soh_true.max(), y.max())+.5]
    ax.plot(limits, limits, "k--", label="ideal")
    ax.set(xlabel="Measured BMS SOH (%)", ylabel="Predicted SOH (%)",
           title="PI-DB-PatchKAN validation with 90% conformal intervals")
    ax.legend(); fig.tight_layout()
    fig.savefig(ANALYSIS_ROOT / "validation_soh_intervals.png", dpi=180)
    plt.show()
    update("BLOCK 7 complete; figures saved")
    """),
    code("""
    # BLOCK 8 — Inventory, complete ZIP, and automatic download
    import json, shutil
    from google.colab import files

    settings = {"seeds": SEEDS, "warmup_epochs": WARMUP_EPOCHS,
                "end_to_end_epochs": END_TO_END_EPOCHS, "coverage": COVERAGE,
                "architecture": "PI-DB-PatchKAN", "locked_test_used": False}
    (RUN_ROOT / "notebook_settings.json").write_text(json.dumps(settings, indent=2))
    inventory = []
    for path in sorted(RUN_ROOT.rglob("*")):
        if path.is_file(): inventory.append({"path": str(path.relative_to(RUN_ROOT)), "bytes": path.stat().st_size})
    pd.DataFrame(inventory).to_csv(RUN_ROOT / "artifact_inventory.csv", index=False)
    archive_base = Path("/content/DBPatchKAN_stage4n_Nissan_Physics_Hybrid_ALL_MODELS_RESULTS")
    archive = Path(shutil.make_archive(str(archive_base), "zip", RUN_ROOT.parent, RUN_ROOT.name))
    update(f"ZIP ready: {archive.name} ({archive.stat().st_size/2**20:.2f} MB); downloading")
    files.download(str(archive))
    update("BLOCK 8 complete")
    """),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
