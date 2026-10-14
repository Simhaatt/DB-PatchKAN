"""Generate the combined RWTH + EVE Stage 5 Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "06_colab_a100_stage5_rwth_eve_external_validation.ipynb"


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    cells = [
        markdown("""# Stage 5 — RWTH external LOOV + EVE measured-thermal replication

This notebook performs the first external validation of the frozen physics-linked
research design. It never loads the Nissan locked test.

- **Stage 5A / RWTH:** 14,167 trips and nine vehicles for leave-one-vehicle-out
  driver→battery-stress validation; 16 capacity checkpoints and eight
  exposure-bearing degradation intervals for conservative health proof-of-concept.
- **Stage 5B / EVE:** public Renault Zoe telemetry downloaded directly from the
  Dataverse API, with measured module temperature, current, voltage, SOC, speed
  and BMS SOH. EVE uses expanding temporal folds because it is one vehicle.

RWTH capacity SOH and EVE/Nissan BMS SOH are kept separate. Negative apparent
capacity loss is retained. KAN is evaluated, not automatically declared the winner.
"""),
        code("""# BLOCK 1 — Configuration
from datetime import datetime, timezone

RUN_ID = "stage5_rwth_eve_external_validation"
KAN_EPOCHS = 300
RUN_EBM = True
RUN_EVE = True
RAW_EVE_EXPECTED_MB = 270

def update(message):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] {message}", flush=True)

update("BLOCK 1 complete")
print({"run_id": RUN_ID, "kan_epochs": KAN_EPOCHS, "run_ebm": RUN_EBM,
       "run_eve": RUN_EVE, "nissan_locked_test_enabled": False})
"""),
        code("""# BLOCK 2 — Upload and robustly extract the one complete package ZIP
import re, shutil, zipfile
from pathlib import Path
from google.colab import files

INPUT_ROOT = Path('/content/dbpatchkan_stage5_input')
if INPUT_ROOT.exists(): shutil.rmtree(INPUT_ROOT)
INPUT_ROOT.mkdir(parents=True)
update('Upload DBPatchKAN_Stage5_RWTH_EVE_External_Validation_Upload.zip')
uploaded = files.upload()
assert uploaded, 'No file was uploaded'
choices = [(name, payload) for name, payload in uploaded.items() if name.lower().endswith('.zip')]
assert choices, 'Upload the complete Stage 5 ZIP'
name, payload = max(choices, key=lambda item: len(item[1]))
clean_name = re.sub(r" \\(\\d+\\)(?=\\.[^.]+$)", "", Path(name).name)
package = INPUT_ROOT / clean_name
package.write_bytes(payload)
with zipfile.ZipFile(package) as archive:
    archive.extractall(INPUT_ROOT)

REQUIRED = {
    'dbpatchkan_stage5_colab_code.zip',
    'rwth_trip_physics_features.parquet',
    'rwth_checkpoint_exposures.csv',
    'rwth_driver_soh_intervals.csv',
    'rwth_capacity_checkpoints.csv',
    'rwth_physics_parameters.json',
    'RWTH_STAGE4_SUMMARY.json',
    'STAGE5_UPLOAD_MANIFEST.json',
}
located = {path.name: path for path in INPUT_ROOT.rglob('*') if path.is_file() and path.name in REQUIRED}
missing = sorted(REQUIRED - set(located))
assert not missing, f'Missing package entries: {missing}'
prohibited = [path.name for path in INPUT_ROOT.rglob('*') if path.is_file()
              and ('segments_test' in path.name.lower() or 'untouched' in path.name.lower())]
assert not prohibited, f'Nissan locked-test material is prohibited: {prohibited}'

RUN_ROOT = Path('/content/dbpatchkan_runs') / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=True)
update(f'BLOCK 2 complete; upload={len(payload)/2**20:.2f} MB; required files={len(located)}')
"""),
        code("""# BLOCK 3 — Verify hashes, install project + EBM, and audit GPU
import hashlib, json, subprocess, sys, torch

manifest = json.loads(located['STAGE5_UPLOAD_MANIFEST.json'].read_text())
assert manifest['nissan_locked_test_included'] is False
for filename, metadata in manifest['files'].items():
    actual = hashlib.sha256(located[filename].read_bytes()).hexdigest()
    assert actual == metadata['sha256'], f'Hash mismatch: {filename}'

PROJECT_EXTRACT = Path('/content/dbpatchkan_stage5_project')
if PROJECT_EXTRACT.exists(): shutil.rmtree(PROJECT_EXTRACT)
PROJECT_EXTRACT.mkdir(parents=True)
with zipfile.ZipFile(located['dbpatchkan_stage5_colab_code.zip']) as archive:
    archive.extractall(PROJECT_EXTRACT)
projects = list(PROJECT_EXTRACT.rglob('pyproject.toml'))
assert len(projects) == 1, projects
PROJECT_ROOT = projects[0].parent
install_target = str(PROJECT_ROOT) + ('[explain]' if RUN_EBM else '')
update('Installing the local project and external-validation dependencies')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-e', install_target])
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
assert torch.cuda.is_available(), 'Enable an A100 GPU runtime before continuing'
update(f'BLOCK 3 complete; GPU={torch.cuda.get_device_name(0)}; project={PROJECT_ROOT}')
"""),
        code("""# BLOCK 4 — Immutable input audit
import pandas as pd

rwth_summary = json.loads(located['RWTH_STAGE4_SUMMARY.json'].read_text())
trips = pd.read_parquet(located['rwth_trip_physics_features.parquet'])
checkpoints = pd.read_csv(located['rwth_checkpoint_exposures.csv'])
intervals = pd.read_csv(located['rwth_driver_soh_intervals.csv'])
assert trips.vehicle_id.nunique() == 9 and len(trips) == 14167
assert int(checkpoints.trip_count.gt(0).sum()) == 16
assert int(intervals.trip_count.gt(0).sum()) == 8
print(rwth_summary)
display(checkpoints.groupby(['vehicle_family','vehicle_id']).agg(
    checkpoints=('checkpoint_id','size'), exposed=('trip_count',lambda x:int((x>0).sum())),
    min_soh=('capacity_soh_pct','min'), max_soh=('capacity_soh_pct','max')))
print({'Nissan rows': 0, 'RWTH trips': len(trips),
       'capacity checkpoints with exposure': int(checkpoints.trip_count.gt(0).sum()),
       'degradation intervals with exposure': int(intervals.trip_count.gt(0).sum())})
update('BLOCK 4 complete')
"""),
        code("""# BLOCK 5 — Stage 5A: full RWTH leave-one-vehicle-out validation
import time

RWTH_OUTPUT = RUN_ROOT / 'stage5a_rwth_external'
command = [sys.executable, '-m', 'dbpatchkan.analysis.rwth_external',
    '--trip-features', str(located['rwth_trip_physics_features.parquet']),
    '--checkpoint-exposures', str(located['rwth_checkpoint_exposures.csv']),
    '--intervals', str(located['rwth_driver_soh_intervals.csv']),
    '--output-dir', str(RWTH_OUTPUT), '--device', 'cuda',
    '--kan-epochs', str(KAN_EPOCHS)]
if not RUN_EBM: command.append('--skip-ebm')
update('START Stage 5A RWTH LOOV (progress prints after every held-out vehicle)')
started = time.time()
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, bufsize=1)
for line in process.stdout: print(line, end='', flush=True)
assert process.wait() == 0, 'RWTH Stage 5A failed'
update(f'BLOCK 5 complete in {(time.time()-started)/60:.1f} min')
"""),
        code("""# BLOCK 6 — Inspect RWTH before downloading the second dataset
rwth_trip_metrics = pd.read_csv(RWTH_OUTPUT / 'rwth_trip_response_loov_metrics.csv')
rwth_increment = pd.read_csv(RWTH_OUTPUT / 'rwth_incremental_driver_value.csv')
rwth_capacity = pd.read_csv(RWTH_OUTPUT / 'rwth_capacity_soh_loov_metrics.csv')
rwth_degradation = pd.read_csv(RWTH_OUTPUT / 'rwth_signed_degradation_loov_metrics.csv')
rwth_decision = json.loads((RWTH_OUTPUT / 'RWTH_EXTERNAL_ACCEPTANCE.json').read_text())
display(rwth_trip_metrics.sort_values(['task','vehicle_macro_mae']))
display(rwth_increment)
display(rwth_capacity)
display(rwth_degradation)
print(rwth_decision)
update('BLOCK 6 complete; negative health results, if any, are retained')
"""),
        code("""# BLOCK 7 — Stage 5B: download and harmonise public EVE Renault-Zoe data
EVE_RAW = Path('/content/eve_pilot_raw')  # raw files stay outside the final result ZIP
EVE_HARMONISED = RUN_ROOT / 'stage5b_eve_harmonised'
if RUN_EVE:
    update(f'START EVE public download (~{RAW_EVE_EXPECTED_MB} MB selected signals)')
    started = time.time()
    command = [sys.executable, str(PROJECT_ROOT / 'scripts' / 'build_eve_stage5.py'),
        '--raw-dir', str(EVE_RAW), '--output-dir', str(EVE_HARMONISED)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout: print(line, end='', flush=True)
    assert process.wait() == 0, 'EVE download/harmonisation failed'
    update(f'BLOCK 7 complete in {(time.time()-started)/60:.1f} min')
else:
    update('BLOCK 7 skipped by RUN_EVE=False')
"""),
        code("""# BLOCK 8 — EVE harmonisation audit
if RUN_EVE:
    eve_summary = json.loads((EVE_HARMONISED / 'EVE_HARMONISATION_SUMMARY.json').read_text())
    eve_trips = pd.read_parquet(EVE_HARMONISED / 'eve_trip_physics_features.parquet')
    eve_checkpoints = pd.read_csv(EVE_HARMONISED / 'eve_monthly_soh_checkpoints.csv')
    print(eve_summary)
    display(eve_trips[['speed_mean_kmh','c_rate_abs_p95','pack_temp_rise_c',
                       'electrothermal_dose_h','dbsi','bms_soh_pct_raw']].describe().T)
    display(eve_checkpoints[['checkpoint_id','bms_soh_pct','bms_soh_trend_pct',
                             'trip_count_cumulative','trip_count_recent_31d']].tail(15))
    assert eve_summary['label_type'].startswith('Renault BMS')
update('BLOCK 8 complete')
"""),
        code("""# BLOCK 9 — EVE expanding-temporal validation with measured temperature
EVE_OUTPUT = RUN_ROOT / 'stage5b_eve_external'
if RUN_EVE:
    command = [sys.executable, '-m', 'dbpatchkan.analysis.eve_external',
        '--trip-features', str(EVE_HARMONISED / 'eve_trip_physics_features.parquet'),
        '--checkpoints', str(EVE_HARMONISED / 'eve_monthly_soh_checkpoints.csv'),
        '--intervals', str(EVE_HARMONISED / 'eve_monthly_soh_intervals.csv'),
        '--output-dir', str(EVE_OUTPUT), '--device', 'cuda',
        '--kan-epochs', str(KAN_EPOCHS)]
    if not RUN_EBM: command.append('--skip-ebm')
    update('START Stage 5B EVE temporal replication')
    started = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout: print(line, end='', flush=True)
    assert process.wait() == 0, 'EVE Stage 5B analysis failed'
    update(f'BLOCK 9 complete in {(time.time()-started)/60:.1f} min')
"""),
        code("""# BLOCK 10 — Combined evidence report; labels remain separate
COMBINED = RUN_ROOT / 'combined_external_report'
if RUN_EVE:
    command = [sys.executable, '-m', 'dbpatchkan.analysis.external_combined',
        '--rwth-dir', str(RWTH_OUTPUT), '--eve-dir', str(EVE_OUTPUT),
        '--output-dir', str(COMBINED)]
    subprocess.check_call(command)
    decision = json.loads((COMBINED / 'EXTERNAL_COMBINED_DECISION.json').read_text())
    evidence = pd.read_csv(COMBINED / 'external_evidence_matrix.csv')
    metrics = pd.read_csv(COMBINED / 'external_combined_metrics_LONG.csv')
    display(evidence)
    display(metrics.sort_values(['dataset','task_family','task','mae']))
    print(decision)
    from IPython.display import Image, display as ipy_display
    ipy_display(Image(filename=str(COMBINED / 'rwth_driver_increment_external.png')))
    ipy_display(Image(filename=str(COMBINED / 'external_health_model_mae.png')))
update('BLOCK 10 complete')
"""),
        code("""# BLOCK 11 — Inventory, complete results ZIP, and automatic download
settings = {
    'run_id': RUN_ID, 'kan_epochs': KAN_EPOCHS, 'run_ebm': RUN_EBM,
    'run_eve': RUN_EVE, 'nissan_locked_test_used': False,
    'rwth_label': 'controlled fixed-voltage-window capacity SOH',
    'eve_label': 'cleaned monthly Renault BMS SOH trend',
    'labels_pooled': False,
}
(RUN_ROOT / 'notebook_settings.json').write_text(json.dumps(settings, indent=2))
inventory = [{'path': str(path.relative_to(RUN_ROOT)), 'bytes': path.stat().st_size}
             for path in sorted(RUN_ROOT.rglob('*')) if path.is_file()]
pd.DataFrame(inventory).to_csv(RUN_ROOT / 'artifact_inventory.csv', index=False)
archive = Path(shutil.make_archive(
    '/content/DBPatchKAN_stage5_RWTH_EVE_EXTERNAL_ALL_RESULTS',
    'zip', RUN_ROOT.parent, RUN_ROOT.name))
update(f'ZIP ready: {archive.stat().st_size/2**20:.2f} MB; automatic download starting')
files.download(str(archive))
update('BLOCK 11 complete')
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
