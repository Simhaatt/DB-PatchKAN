"""Build the Stage 14 A100 robustness notebook and one upload package."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "outputs" / "stage14_hyperparameter_sensitivity_upload"
NOTEBOOK = ROOT / "notebooks" / "16_colab_a100_stage14_hyperparameter_sensitivity.ipynb"
PACKAGE = ROOT / "outputs" / "DBPatchKAN_Stage14_Hyperparameter_Sensitivity_Upload.zip"
STAGE8_PACKAGE = ROOT / "outputs" / "DBPatchKAN_Stage8_Final_Confirmation_Upload.zip"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def code_cell(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_code_zip() -> Path:
    output = BUILD / "dbpatchkan_stage14_code.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in ("pyproject.toml", "README.md"):
            archive.write(ROOT / relative, relative)
        for file in sorted((ROOT / "src").rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                archive.write(file, file.relative_to(ROOT).as_posix())
    return output


def extract_development_source() -> Path:
    destination = BUILD / "development_101_trips.npz"
    if not STAGE8_PACKAGE.exists():
        raise FileNotFoundError(STAGE8_PACKAGE)
    with zipfile.ZipFile(STAGE8_PACKAGE) as archive:
        names = archive.namelist()
        exact = [name for name in names if Path(name).name == destination.name]
        if len(exact) != 1:
            raise RuntimeError(f"Expected one development tensor file, found {exact}")
        destination.write_bytes(archive.read(exact[0]))
    return destination


def build_package() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    code = build_code_zip()
    development = extract_development_source()
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in (code, development)
    }
    manifest = {
        "stage": 14,
        "purpose": "development-only post-locked hyperparameter robustness",
        "files": files,
        "contains_locked_test": False,
        "expected_development_trips": 101,
        "settings": {
            "contexts_seconds": [128, 192, 256],
            "concept_weights": [0.10, 0.20, 0.40],
            "physics_weights": [0.00, 0.05, 0.10, 0.20],
            "unique_settings": 8,
            "seeds": [42, 52, 62],
            "expected_fits": 24,
        },
        "prohibition": (
            "These post-locked development results cannot change the frozen Stage 8 "
            "configuration or be selected using the locked test."
        ),
    }
    manifest_path = BUILD / "STAGE14_UPLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(PACKAGE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(BUILD.iterdir()):
            archive.write(path, path.name)


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
            r"""# Stage 14 — A100 Hyperparameter Robustness

This notebook performs the final **development-only, post-locked sensitivity
analysis** for Persistence-Anchored DB-PatchKAN. It does not reopen the locked
26-trip Nissan test and cannot replace the frozen Stage 8 configuration.

The experiment tests exactly the settings that were not comprehensively swept:

- available sequence context: 128, 192 and 256 seconds;
- concept-supervision weight: 0.10, 0.20 and 0.40;
- physics-consistency weight: 0.00, 0.05, 0.10 and 0.20.

The frozen reference is 256 seconds, concept weight 0.20 and physics weight
0.10. Duplicate reference combinations are run once, giving **eight settings ×
three seeds = 24 KAN fits**. Shared masked pretraining is reused fairly within
each seed/context pair.

The native export contains 256-second windows. A 512-second variant is excluded
because joining overlapping windows would duplicate observations and create a
different sampling process. The shorter contexts use deterministic centred
cropping.

Expected A100 runtime: approximately **25–60 minutes**. Runtime depends on Colab
I/O and early stopping. Every training epoch prints progress. The final block
automatically downloads all models, histories, predictions, tables, figures and
audit files as one ZIP."""
        ),
        code_cell(
            """
# BLOCK 1 — Fixed protocol; these values must not be changed after results are viewed
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, zipfile

SEEDS = [42, 52, 62]
PRETRAIN_EPOCHS = 5
FINETUNE_EPOCHS = 30
EXPECTED_FITS = 24
INPUT_ROOT = Path('/content/dbpatchkan_stage14_input')
PROJECT_ROOT = Path('/content/dbpatchkan_stage14_project')
RUN_ROOT = Path('/content/dbpatchkan_stage14_runs/hyperparameter_robustness')

def update(message):
    stamp=datetime.now(timezone.utc).strftime('%H:%M:%S')
    print(f'[{stamp} UTC] {message}',flush=True)

INPUT_ROOT.mkdir(parents=True,exist_ok=True)
RUN_ROOT.mkdir(parents=True,exist_ok=True)
update(f'BLOCK 1 complete: seeds={SEEDS}, fits={EXPECTED_FITS}, locked_test=False')
"""
        ),
        code_cell(
            """
# BLOCK 2 — Upload the one Stage 14 package; duplicate browser names are supported
from google.colab import files
update('BLOCK 2: upload DBPatchKAN_Stage14_Hyperparameter_Sensitivity_Upload.zip')
uploaded=files.upload()
archives=[(name,payload) for name,payload in uploaded.items() if name.lower().endswith('.zip')]
assert len(archives)==1,f'Upload exactly one ZIP; received {[name for name,_ in archives]}'
package=INPUT_ROOT/'stage14_upload.zip'
package.write_bytes(archives[0][1])
with zipfile.ZipFile(package) as archive:
    unsafe=[name for name in archive.namelist() if name.startswith('/') or Path(name).is_absolute() or '..' in Path(name).parts]
    assert not unsafe,f'Unsafe ZIP members: {unsafe}'
    archive.extractall(INPUT_ROOT)
required={'dbpatchkan_stage14_code.zip','development_101_trips.npz','STAGE14_UPLOAD_MANIFEST.json'}
located={path.name:path for path in INPUT_ROOT.rglob('*') if path.is_file()}
missing=sorted(required-set(located)); assert not missing,f'Package incomplete: {missing}'
for name in required:
    destination=INPUT_ROOT/name
    if located[name].resolve()!=destination.resolve(): shutil.copy2(located[name],destination)
prohibited=[path.name for path in INPUT_ROOT.rglob('*') if path.is_file() and ('test' in path.name.casefold() or 'untouched' in path.name.casefold())]
assert not prohibited,f'Stage 14 refuses test/untouched files: {prohibited}'
update(f'BLOCK 2 complete: received {len(archives[0][1])/2**20:.2f} MB; no locked test present')
"""
        ),
        code_cell(
            """
# BLOCK 3 — Verify immutable inputs, install local project and confirm A100
update('BLOCK 3 starting installation and integrity verification')
if PROJECT_ROOT.exists(): shutil.rmtree(PROJECT_ROOT)
PROJECT_ROOT.mkdir(parents=True)
with zipfile.ZipFile(INPUT_ROOT/'dbpatchkan_stage14_code.zip') as archive: archive.extractall(PROJECT_ROOT)
roots=list(PROJECT_ROOT.rglob('pyproject.toml')); assert len(roots)==1,roots
PROJECT=roots[0].parent
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(PROJECT)],check=True)
source_root=str(PROJECT/'src')
if source_root not in sys.path: sys.path.insert(0,source_root)
os.environ['PYTHONPATH']=source_root+os.pathsep+os.environ.get('PYTHONPATH','')

def sha256(path):
    value=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(2**20),b''): value.update(chunk)
    return value.hexdigest()

manifest=json.loads((INPUT_ROOT/'STAGE14_UPLOAD_MANIFEST.json').read_text())
assert manifest['contains_locked_test'] is False
for name,record in manifest['files'].items():
    observed=sha256(INPUT_ROOT/name)
    assert observed==record['sha256'],f'Hash mismatch for {name}'
    update(f'Verified {name}: {record["bytes"]/2**20:.2f} MB')
import torch, numpy as np, pandas as pd, dbpatchkan
assert torch.cuda.is_available(),'Enable Runtime > Change runtime type > A100 GPU'
assert 'A100' in torch.cuda.get_device_name(0),f'An A100 was requested; found {torch.cuda.get_device_name(0)}'
update(f'BLOCK 3 complete: GPU={torch.cuda.get_device_name(0)}, torch={torch.__version__}')
"""
        ),
        code_cell(
            """
# BLOCK 4 — Audit the development cohort without constructing any test path
from dbpatchkan.analysis.stage6 import trip_order
DEVELOPMENT=INPUT_ROOT/'development_101_trips.npz'
ordered,_=trip_order(DEVELOPMENT)
with np.load(DEVELOPMENT,allow_pickle=False) as data:
    update(f'Development tensor: windows={data["driver"].shape[0]}, seconds={data["driver"].shape[1]}, trips={len(ordered)}')
    assert data['driver'].shape[1]==256
assert len(ordered)==101
assert not any('test' in str(path).casefold() or 'untouched' in str(path).casefold() for path in INPUT_ROOT.rglob('*'))
update('BLOCK 4 complete: chronological split will be 87 training + 14 validation trips; locked test inaccessible')
"""
        ),
        code_cell(
            """
# BLOCK 5 — A100 forward/backward smoke test for the frozen KAN architecture
from dbpatchkan.analysis.stage14_sensitivity import _model_config
from dbpatchkan.models import DBPatchKAN
model=DBPatchKAN(_model_config(128)).cuda()
driver=torch.randn(2,128,13,device='cuda')
battery=torch.randn(2,128,11,device='cuda')
with torch.autocast('cuda',dtype=torch.bfloat16):
    outputs=model(driver,battery,domain='nissan')
    smoke_loss=outputs['current_soh'].float().square().mean()
smoke_loss.backward()
assert torch.isfinite(smoke_loss)
update(f'BLOCK 5 complete: parameters={sum(parameter.numel() for parameter in model.parameters()):,}, loss={smoke_loss.item():.6f}')
del model,driver,battery,outputs,smoke_loss
torch.cuda.empty_cache()
"""
        ),
        code_cell(
            """
# BLOCK 6 — Run 24 development-only robustness fits
update('BLOCK 6 starting; every epoch and every completed fit prints progress')
RESULTS=RUN_ROOT/'results'
command=[sys.executable,'-m','dbpatchkan.analysis.stage14_sensitivity',
         '--development-data',str(DEVELOPMENT),'--output-dir',str(RESULTS),
         '--seeds',*[str(seed) for seed in SEEDS],
         '--pretrain-epochs',str(PRETRAIN_EPOCHS),'--finetune-epochs',str(FINETUNE_EPOCHS)]
update('COMMAND: '+' '.join(command))
process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=os.environ.copy())
for line in process.stdout: print(line,end='',flush=True)
return_code=process.wait(); assert return_code==0,f'Stage 14 failed: {return_code}'
update('BLOCK 6 complete')
"""
        ),
        code_cell(
            """
# BLOCK 7 — Display all seed-level and aggregated sensitivity results
metrics=pd.read_csv(RESULTS/'stage14_seed_metrics.csv')
summary=pd.read_csv(RESULTS/'stage14_sensitivity_summary.csv')
paired=pd.read_csv(RESULTS/'stage14_paired_bootstrap.csv')
robustness=pd.read_csv(RESULTS/'stage14_robustness_table.csv')
for target in ['current_soh','loss_30d','loss_500km']:
    update(f'Results for {target}')
    display(summary[summary.target.eq(target)].sort_values('mae_mean'))
display(paired)
display(robustness[['name','family','target','mae_mean','reference_mae','relative_mae_change_pct','within_5pct_of_reference']])
update('BLOCK 7 complete: lower MAE is better; this table is robustness evidence, not model reselection')
"""
        ),
        code_cell(
            """
# BLOCK 8 — Display publication-ready sensitivity figures and decision
from IPython.display import display, Image
for figure in sorted((RESULTS/'figures').glob('*.png')):
    update(f'Displaying {figure.name}')
    display(Image(filename=str(figure)))
decision=json.loads((RESULTS/'STAGE14_DECISION.json').read_text())
display(decision)
assert decision['frozen_model_changed'] is False
assert decision['locked_test_used'] is False
update('BLOCK 8 complete: frozen Stage 8 model remains unchanged')
"""
        ),
        code_cell(
            """
# BLOCK 9 — Completeness and leakage audit
required={'STAGE14_PROTOCOL.json','stage14_seed_metrics.csv','stage14_validation_predictions.csv',
          'stage14_sensitivity_summary.csv','stage14_paired_bootstrap.csv','stage14_robustness_table.csv',
          'STAGE14_DECISION.json','STAGE14_MODEL_REGISTRY.json','STAGE14_AUDIT.json'}
missing=sorted(name for name in required if not (RESULTS/name).exists()); assert not missing,missing
registry=json.loads((RESULTS/'STAGE14_MODEL_REGISTRY.json').read_text())
audit=json.loads((RESULTS/'STAGE14_AUDIT.json').read_text())
checkpoints=list((RESULTS/'models').rglob('model_lightweight.pt'))
histories=list((RESULTS/'models').rglob('history.json'))
assert len(registry['models'])==EXPECTED_FITS
assert len(checkpoints)==EXPECTED_FITS,(len(checkpoints),EXPECTED_FITS)
assert len(histories)==EXPECTED_FITS,(len(histories),EXPECTED_FITS)
assert audit['locked_test_used'] is False and audit['frozen_configuration_changed'] is False
assert not any('test' in path.name.casefold() or 'untouched' in path.name.casefold() for path in RESULTS.rglob('*'))
update(f'BLOCK 9 complete: models={len(checkpoints)}, histories={len(histories)}, elapsed={audit["elapsed_minutes"]:.1f} min')
"""
        ),
        code_cell(
            """
# BLOCK 10 — Automatically download every model, result, figure and audit file
from google.colab import files
settings_copy=RUN_ROOT/'NOTEBOOK_RUN_SETTINGS.json'
settings_copy.write_text(json.dumps({
    'seeds':SEEDS,'pretrain_epochs':PRETRAIN_EPOCHS,'finetune_epochs':FINETUNE_EPOCHS,
    'expected_fits':EXPECTED_FITS,'development_sha256':sha256(DEVELOPMENT),
    'locked_test_used':False,'post_locked_sensitivity':True,
    'completed_utc':datetime.now(timezone.utc).isoformat()},indent=2))
archive_base=Path('/content/DBPatchKAN_stage14_HYPERPARAMETER_ROBUSTNESS_ALL_MODELS_RESULTS')
archive_path=Path(shutil.make_archive(str(archive_base),'zip',root_dir=RUN_ROOT))
update(f'BLOCK 10 archive ready: {archive_path.stat().st_size/2**20:.1f} MB')
files.download(str(archive_path))
update('BLOCK 10 complete — download triggered; the Colab session may now be stopped')
"""
        ),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)


def main() -> None:
    build_package()
    build_notebook()
    print(NOTEBOOK)
    print(PACKAGE)


if __name__ == "__main__":
    main()
