"""Build Stage 13 persistence-anchored EVBattery forecasting notebook/package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "outputs" / "stage13_persistence_forecast_upload"
TRANSITIONS = BUILD / "evbattery_stage13_transitions.npz"
NOTEBOOK = ROOT / "notebooks" / "13_colab_a100_persistence_anchored_forecast.ipynb"
PACKAGE = ROOT / "outputs" / "DBPatchKAN_Stage13_Persistence_Anchored_Forecast_Upload.zip"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def cell(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_code_zip() -> Path:
    output = BUILD / "dbpatchkan_stage13_code.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in ("pyproject.toml", "README.md"):
            archive.write(ROOT / relative, relative)
        for file in sorted((ROOT / "src").rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                archive.write(file, file.relative_to(ROOT).as_posix())
    return output


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
            r"""# Stage 13 — Persistence-Anchored DB-PatchKAN Forecasting

This is the proper operational formulation:

\[
\widehat{Capacity}_{next}=Capacity_{current}+\widehat{\Delta Capacity}
\]

The current capacity is a strictly earlier checkpoint. The target is the next
distinct-mileage partial-window capacity checkpoint. Same-mileage overlapping
snippets were collapsed before transition creation.

Inputs comprise the current 128-step charging sequence, fourteen independently
calculated charging concepts, current capacity, current mileage, forecast
distance, and non-ordinal manufacturer context. KAN and matched MLP predict only
the signed residual. Persistence, anchored Ridge and anchored Random Forest are
evaluated in the same five vehicle-disjoint folds.

The target remains partial-window capacity in Ah—not absolute SOH—and EVBattery
does not contain driver trajectories.

Expected A100 runtime: **45–90 minutes**. The final block downloads every model
and result automatically."""
        ),
        cell(
            """
# BLOCK 1 — Configuration
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, time, zipfile

FOLDS = 5
SEEDS = [42, 52, 62]
EPOCHS = 60
BATCH_SIZE = 512
INPUT_ROOT = Path('/content/dbpatchkan_stage13_input')
PROJECT_ROOT = Path('/content/dbpatchkan_stage13_project')
RUN_ROOT = Path('/content/dbpatchkan_stage13_runs/stage13_persistence_anchored')

def update(message):
    stamp=datetime.now(timezone.utc).strftime('%H:%M:%S')
    print(f'[{stamp} UTC] {message}',flush=True)

INPUT_ROOT.mkdir(parents=True,exist_ok=True)
RUN_ROOT.mkdir(parents=True,exist_ok=True)
update(f'BLOCK 1 complete: folds={FOLDS}, seeds={SEEDS}, epochs={EPOCHS}, batch={BATCH_SIZE}')
"""
        ),
        cell(
            """
# BLOCK 2 — Upload the single Stage 13 package (about 12 MB)
from google.colab import files
update('BLOCK 2: upload DBPatchKAN_Stage13_Persistence_Anchored_Forecast_Upload.zip')
uploaded=files.upload()
items=[(name,payload) for name,payload in uploaded.items() if name.lower().endswith('.zip')]
assert len(items)==1,f'Upload exactly one ZIP; received {[name for name,_ in items]}'
package=INPUT_ROOT/'stage13_upload.zip'; package.write_bytes(items[0][1])
with zipfile.ZipFile(package) as archive: archive.extractall(INPUT_ROOT)
required={'dbpatchkan_stage13_code.zip','evbattery_stage13_transitions.npz','STAGE13_UPLOAD_MANIFEST.json'}
located={p.name:p for p in INPUT_ROOT.rglob('*') if p.is_file()}
missing=sorted(required-set(located)); assert not missing,f'Package incomplete: {missing}'
for name in required:
    destination=INPUT_ROOT/name
    if located[name].resolve()!=destination.resolve(): shutil.copy2(located[name],destination)
update(f'BLOCK 2 complete: received {len(items[0][1])/2**20:.2f} MB')
"""
        ),
        cell(
            """
# BLOCK 3 — Install project and verify package hashes/A100
update('BLOCK 3 installing project')
if PROJECT_ROOT.exists(): shutil.rmtree(PROJECT_ROOT)
PROJECT_ROOT.mkdir(parents=True)
with zipfile.ZipFile(INPUT_ROOT/'dbpatchkan_stage13_code.zip') as archive: archive.extractall(PROJECT_ROOT)
roots=list(PROJECT_ROOT.rglob('pyproject.toml')); assert len(roots)==1
PROJECT=roots[0].parent
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(PROJECT)],check=True)
source_root=str(PROJECT/'src')
if source_root not in sys.path: sys.path.insert(0,source_root)
os.environ['PYTHONPATH']=source_root+os.pathsep+os.environ.get('PYTHONPATH','')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(2**20),b''): h.update(chunk)
    return h.hexdigest()

manifest=json.loads((INPUT_ROOT/'STAGE13_UPLOAD_MANIFEST.json').read_text())
for name,record in manifest['files'].items():
    observed=sha256(INPUT_ROOT/name)
    assert observed==record['sha256'],f'Hash mismatch for {name}'
    update(f'Verified {name}')
import torch,pandas as pd,numpy as np,dbpatchkan
assert torch.cuda.is_available(),'Enable Runtime > Change runtime type > A100 GPU'
update(f'BLOCK 3 complete: GPU={torch.cuda.get_device_name(0)}, torch={torch.__version__}')
"""
        ),
        cell(
            """
# BLOCK 4 — Audit the transition cohort before opening model training
from dbpatchkan.analysis.stage13_persistence_forecast import load_transition_cache
CACHE=INPUT_ROOT/'evbattery_stage13_transitions.npz'
loaded=load_transition_cache(CACHE); table=loaded['table']
audit=table.groupby('manufacturer_id').agg(transitions=('transition_id','size'),vehicles=('vehicle_key','nunique'),median_horizon_km=('forecast_horizon_km','median'),median_delta_ah=('capacity_delta_ah','median')).reset_index()
display(audit)
update(f"BLOCK 4 complete: {len(table):,} transitions, {table.vehicle_key.nunique()} vehicles, median horizon={pd.to_numeric(table.forecast_horizon_km).median():.1f} km")
assert len(table)==9714 and table.vehicle_key.nunique()==191
"""
        ),
        cell(
            """
# BLOCK 5 — Full five-fold, three-seed A100 validation
update('BLOCK 5 starting; progress prints after every fold/model/10 epochs')
RESULTS=RUN_ROOT/'results'
command=[sys.executable,'-m','dbpatchkan.analysis.stage13_persistence_forecast','validate','--cache',str(CACHE),'--output-dir',str(RESULTS),'--folds',str(FOLDS),'--seeds',*[str(seed) for seed in SEEDS],'--epochs',str(EPOCHS),'--batch-size',str(BATCH_SIZE)]
process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=os.environ.copy())
for line in process.stdout: print(line,end='',flush=True)
return_code=process.wait(); assert return_code==0,f'Stage 13 failed: {return_code}'
update('BLOCK 5 complete')
"""
        ),
        cell(
            """
# BLOCK 6 — Final ensemble, paired and horizon results
summary=pd.read_csv(RESULTS/'stage13_seed_ensemble_metrics.csv').sort_values('vehicle_macro_mae_ah')
paired=pd.read_csv(RESULTS/'stage13_paired_vehicle_bootstrap.csv')
horizon=pd.read_csv(RESULTS/'stage13_horizon_metrics.csv')
display(summary)
display(paired)
display(horizon.pivot(index='model',columns='horizon_band',values='mae_ah'))
winner=summary.iloc[0]
update(f"BLOCK 6 complete: best={winner['model']}, MAE={winner['mae_ah']:.6f} Ah")
"""
        ),
        cell(
            """
# BLOCK 7 — Audit all outputs and save a machine-readable decision
required={'stage13_predictions.csv','stage13_fold_metrics.csv','stage13_model_summary.csv','stage13_seed_ensemble_predictions.csv','stage13_seed_ensemble_metrics.csv','stage13_paired_vehicle_bootstrap.csv','stage13_horizon_metrics.csv','STAGE13_AUDIT.json'}
missing=sorted(name for name in required if not (RESULTS/name).exists())
assert not missing,f'Missing Stage 13 outputs: {missing}'
summary=pd.read_csv(RESULTS/'stage13_seed_ensemble_metrics.csv')
kan=summary.loc[summary.model.eq('PersistenceAnchored_KAN')].iloc[0]
persistence=summary.loc[summary.model.eq('Persistence')].iloc[0]
decision={'persistence_mae_ah':float(persistence.mae_ah),'anchored_kan_mae_ah':float(kan.mae_ah),'kan_minus_persistence_mae_ah':float(kan.mae_ah-persistence.mae_ah),'kan_beats_persistence':bool(kan.mae_ah<persistence.mae_ah),'selection_rule':'paired vehicle interval must also remain below zero','target_is_absolute_soh':False}
(RESULTS/'STAGE13_DECISION.json').write_text(json.dumps(decision,indent=2))
update(f'BLOCK 7 decision: {decision}')
"""
        ),
        cell(
            """
# BLOCK 8 — Automatically download all models and results
from google.colab import files
archive_base=Path('/content/DBPatchKAN_stage13_PERSISTENCE_ANCHORED_FORECAST_ALL_MODELS_RESULTS')
archive_path=Path(shutil.make_archive(str(archive_base),'zip',root_dir=RUN_ROOT))
update(f'BLOCK 8 archive ready: {archive_path.stat().st_size/2**20:.1f} MB')
files.download(str(archive_path))
update('BLOCK 8 complete — browser download triggered; session may now be stopped')
"""
        ),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)


def build_package() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    if not TRANSITIONS.exists():
        raise FileNotFoundError(f"Build transition cache first: {TRANSITIONS}")
    code_zip = build_code_zip()
    files = [code_zip, TRANSITIONS]
    manifest = {
        "stage": 13,
        "purpose": "vehicle-disjoint persistence-anchored next-capacity forecasting",
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in files
        },
        "raw_archives_required_in_colab": False,
        "target": "next_distinct_mileage_partial_window_capacity_ah",
        "target_is_absolute_soh": False,
    }
    manifest_path = BUILD / "STAGE13_UPLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(PACKAGE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in [*files, manifest_path]:
            archive.write(path, path.name)


if __name__ == "__main__":
    build_notebook(); build_package()
    print(NOTEBOOK); print(PACKAGE)
