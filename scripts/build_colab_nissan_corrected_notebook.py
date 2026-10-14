"""Generate the corrected Nissan Stage 4N-B Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "05_colab_a100_nissan_physics_corrected.ipynb"


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip().splitlines(keepends=True)}


def code(source: str) -> dict[str, object]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": dedent(source).strip().splitlines(keepends=True)}


cells = [
    markdown("""
    # Stage 4N-B — Corrected Nissan Physics Hybrid

    This corrective experiment was registered after Stage 4N failed. Current SOH is
    anchored to a training-only age/mileage baseline. The KAN predicts a bounded
    physical residual using one representative per degradation mechanism. Future-loss
    heads use recent 30/90-day stress, empirical label scaling, positive monotonic
    weights, and Transformer-predicted electrical/thermal response concepts.

    The locked 26-trip final-test tensor is absent.
    """),
    markdown("""
    ## Upload

    Run Block 2 and upload only `DBPatchKAN_Nissan_Physics_Corrected_Upload.zip`.
    Do not extract it first.
    """),
    code("""
    # BLOCK 1 — Configuration
    from datetime import datetime, timezone
    SEEDS = [42, 52, 62, 72, 82]
    EPOCHS = 400
    BATCH_SIZE = 2
    COVERAGE = 0.90
    RUN_ID = "stage4nb_nissan_physics_corrected"
    print(f"[{datetime.now(timezone.utc).isoformat()}] BLOCK 1 complete")
    print({"seeds": SEEDS, "epochs": EPOCHS, "test_enabled": False,
           "architecture": "PI-DB-PatchKAN-v2-corrected"})
    """),
    code("""
    # BLOCK 2 — Upload and extract robustly
    import re, shutil, zipfile
    from pathlib import Path
    from datetime import datetime, timezone
    from google.colab import files

    def update(message):
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] {message}", flush=True)

    INPUT_ROOT = Path('/content/dbpatchkan_corrected_input')
    if INPUT_ROOT.exists(): shutil.rmtree(INPUT_ROOT)
    INPUT_ROOT.mkdir(parents=True)
    update('Upload DBPatchKAN_Nissan_Physics_Corrected_Upload.zip')
    uploaded = files.upload()
    assert uploaded, 'No file uploaded'
    choices = [(name, payload) for name, payload in uploaded.items() if name.lower().endswith('.zip')]
    assert choices, 'Upload the complete ZIP'
    name, payload = max(choices, key=lambda item: len(item[1]))
    clean = re.sub(r" \\(\\d+\\)(?=\\.[^.]+$)", "", Path(name).name)
    package = INPUT_ROOT / clean
    package.write_bytes(payload)
    with zipfile.ZipFile(package) as archive: archive.extractall(INPUT_ROOT)
    REQUIRED = {"dbpatchkan_colab_code.zip", "segments_train.npz", "segments_validation.npz",
                "nissan_physics_model_table_DEV.parquet", "UPLOAD_MANIFEST.json",
                *{f"stage4n_v1_seed_{seed}.pt" for seed in SEEDS}}
    located = {path.name: path for path in INPUT_ROOT.rglob('*') if path.is_file() and path.name in REQUIRED}
    missing = sorted(REQUIRED - set(located))
    assert not missing, f"Missing: {missing}"
    assert not [path for path in INPUT_ROOT.iterdir()
                if 'untouched' in path.name.lower() or 'segments_test' in path.name.lower()]
    RUN_ROOT = Path('/content/dbpatchkan_runs') / RUN_ID
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    update(f"BLOCK 2 complete; received {len(payload)/2**20:.2f} MB")
    """),
    code("""
    # BLOCK 3 — Install, hashes, and GPU
    import hashlib, json, subprocess, sys, torch
    PROJECT_EXTRACT = Path('/content/dbpatchkan_corrected_project')
    if PROJECT_EXTRACT.exists(): shutil.rmtree(PROJECT_EXTRACT)
    PROJECT_EXTRACT.mkdir(parents=True)
    with zipfile.ZipFile(located['dbpatchkan_colab_code.zip']) as archive:
        archive.extractall(PROJECT_EXTRACT)
    projects = list(PROJECT_EXTRACT.rglob('pyproject.toml'))
    assert len(projects) == 1
    PROJECT_ROOT = projects[0].parent
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT_ROOT)])
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    manifest = json.loads(located['UPLOAD_MANIFEST.json'].read_text())
    for name, meta in manifest['files'].items():
        assert hashlib.sha256(located[name].read_bytes()).hexdigest() == meta['sha256'], name
    assert torch.cuda.is_available() and manifest['locked_test_included'] is False
    update(f"BLOCK 3 complete; GPU={torch.cuda.get_device_name(0)}")
    """),
    code("""
    # BLOCK 4 — Development audit
    import pandas as pd
    table = pd.read_parquet(located['nissan_physics_model_table_DEV.parquet'])
    assert len(table) == 101 and table.split.value_counts().to_dict() == {'train': 81, 'validation': 20}
    print({'model_fit_trips': 65, 'uncertainty_calibration_trips': 16,
           'validation_trips': 20, '30d_labels': int(table.target_loss_30d_pp.notna().sum()),
           '500km_labels': int(table.target_loss_500km_pp.notna().sum()), 'test_rows': 0})
    display(table.groupby('split')[['current_soh_target','target_loss_30d_pp','target_loss_500km_pp']].agg(['count','min','median','max']))
    update('BLOCK 4 complete')
    """),
    code("""
    # BLOCK 5 — Train five corrected seeds
    import subprocess, sys, time
    started = time.time()
    for position, seed in enumerate(SEEDS, 1):
        output = RUN_ROOT / f'seed_{seed}'
        if (output / 'best.pt').exists():
            update(f'seed {seed} already complete')
            continue
        update(f'START corrected seed {seed} ({position}/5)')
        command = [sys.executable, '-m', 'dbpatchkan.training.physics_corrected',
            '--train-data', str(located['segments_train.npz']),
            '--validation-data', str(located['segments_validation.npz']),
            '--physics-table', str(located['nissan_physics_model_table_DEV.parquet']),
            '--v1-checkpoint', str(located[f'stage4n_v1_seed_{seed}.pt']),
            '--output-dir', str(output), '--device', 'cuda', '--seed', str(seed),
            '--epochs', str(EPOCHS), '--batch-size', str(BATCH_SIZE)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout: print(line, end='', flush=True)
        assert process.wait() == 0, f'seed {seed} failed'
        update(f'DONE seed {seed}; cumulative={(time.time()-started)/60:.1f} min')
    update(f'BLOCK 5 complete; total={(time.time()-started)/60:.1f} min')
    """),
    code("""
    # BLOCK 6 — Ensemble, uncertainty, baselines, and associations
    ANALYSIS = RUN_ROOT / 'corrected_analysis'
    command = [sys.executable, '-m', 'dbpatchkan.analysis.nissan_corrected',
        '--run-root', str(RUN_ROOT),
        '--physics-table', str(located['nissan_physics_model_table_DEV.parquet']),
        '--output-dir', str(ANALYSIS), '--coverage', str(COVERAGE)]
    subprocess.check_call(command)
    update('BLOCK 6 complete')
    """),
    code("""
    # BLOCK 7 — Acceptance decision and figure
    import matplotlib.pyplot as plt, numpy as np
    metrics = pd.read_csv(ANALYSIS / 'corrected_ensemble_metrics.csv')
    baselines = pd.read_csv(ANALYSIS / 'baseline_metrics.csv')
    associations = pd.read_csv(ANALYSIS / 'driver_battery_response_associations.csv')
    decision = json.loads((ANALYSIS / 'ACCEPTANCE_DECISION.json').read_text())
    display(metrics)
    display(baselines.sort_values(['task','mae']).groupby('task').head(8))
    display(associations.head(20))
    print(decision)
    predictions = pd.read_csv(ANALYSIS / 'ensemble_predictions_with_intervals.csv')
    validation = predictions[(predictions.split == 'validation') &
                             (predictions.model_mode == 'corrected_hybrid')]
    fig, ax = plt.subplots(figsize=(8,5))
    y = validation.current_soh_pred_mean
    ax.errorbar(validation.current_soh_true, y,
                yerr=np.vstack([y-validation.current_soh_interval_low,
                                validation.current_soh_interval_high-y]), fmt='o', capsize=3)
    limits = [min(validation.current_soh_true.min(), y.min())-.3,
              max(validation.current_soh_true.max(), y.max())+.3]
    ax.plot(limits, limits, 'k--')
    ax.set(xlabel='Measured SOH (%)', ylabel='Corrected predicted SOH (%)',
           title='Corrected PI-DB-PatchKAN validation')
    fig.tight_layout(); fig.savefig(ANALYSIS / 'corrected_soh_intervals.png', dpi=180)
    plt.show(); update('BLOCK 7 complete')
    """),
    code("""
    # BLOCK 8 — Complete ZIP and automatic download
    settings = {'seeds': SEEDS, 'epochs': EPOCHS, 'coverage': COVERAGE,
                'test_used': False, 'architecture': 'PI-DB-PatchKAN-v2-corrected'}
    (RUN_ROOT / 'notebook_settings.json').write_text(json.dumps(settings, indent=2))
    inventory = [{'path': str(path.relative_to(RUN_ROOT)), 'bytes': path.stat().st_size}
                 for path in sorted(RUN_ROOT.rglob('*')) if path.is_file()]
    pd.DataFrame(inventory).to_csv(RUN_ROOT / 'artifact_inventory.csv', index=False)
    archive = Path(shutil.make_archive(
        '/content/DBPatchKAN_stage4nb_Nissan_Physics_Corrected_ALL_RESULTS',
        'zip', RUN_ROOT.parent, RUN_ROOT.name))
    update(f'ZIP ready: {archive.stat().st_size/2**20:.2f} MB; downloading')
    files.download(str(archive))
    update('BLOCK 8 complete')
    """),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
