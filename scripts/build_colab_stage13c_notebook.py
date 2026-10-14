"""Build the Stage 13C post-hoc TimePro notebook and single upload package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "outputs" / "stage13c_timepro_upload"
TRANSITIONS_SOURCE = (
    ROOT / "outputs" / "stage13_persistence_forecast_upload" / "evbattery_stage13_transitions.npz"
)
STAGE13B_RESULTS = (
    ROOT / "outputs" / "colab_completed" / "stage13b_recent_methods"
    / "final_results" / "results"
)
NOTEBOOK = ROOT / "notebooks" / "15_colab_a100_stage13c_timepro_posthoc.ipynb"
PACKAGE = ROOT / "outputs" / "DBPatchKAN_Stage13C_TimePro_Posthoc_Upload.zip"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def cell(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_code_zip() -> Path:
    output = BUILD / "dbpatchkan_stage13c_code.zip"
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
            r"""# Stage 13C — Post-hoc TimePro 2025 Comparison

This supplementary notebook adds the one missing recent architecture family:
**TimePro (ICML 2025)**, a variable- and time-aware state-space/Mamba model.

This comparison was selected after Stage 13B results were inspected. Every
output must therefore be labelled **post-hoc supplementary evidence**.

The official TimePro implementation uses custom CUDA selective-scan and DCNv4
extensions. This notebook uses a pure-PyTorch architecture-faithful adaptation
that preserves:

- independent temporal patching for every measured variable;
- bidirectional selective state-space processing across variable tokens;
- variable/time-aware hyper-state mixing on the patch × variable grid;
- the identical signed persistence-anchored residual target.

The frozen Stage 13B predictions are packaged by hash and are not retrained.
Only 15 TimePro fits are run: five folds × three seeds.

Expected A100 runtime: **10–25 minutes**. The final block automatically
downloads every checkpoint and comparison artifact."""
        ),
        cell(
            """
# BLOCK 1 — Fixed post-hoc configuration
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, zipfile

FOLDS = 5
SEEDS = [42, 52, 62]
EPOCHS = 60
BATCH_SIZE = 512
INPUT_ROOT = Path('/content/dbpatchkan_stage13c_input')
PROJECT_ROOT = Path('/content/dbpatchkan_stage13c_project')
RUN_ROOT = Path('/content/dbpatchkan_stage13c_runs/timepro_posthoc')

def update(message):
    stamp=datetime.now(timezone.utc).strftime('%H:%M:%S')
    print(f'[{stamp} UTC] {message}',flush=True)

INPUT_ROOT.mkdir(parents=True,exist_ok=True)
RUN_ROOT.mkdir(parents=True,exist_ok=True)
update(f'BLOCK 1 complete: folds={FOLDS}, seeds={SEEDS}, epochs={EPOCHS}, batch={BATCH_SIZE}, POSTHOC=True')
"""
        ),
        cell(
            """
# BLOCK 2 — Upload the single Stage 13C package
from google.colab import files
update('BLOCK 2: upload DBPatchKAN_Stage13C_TimePro_Posthoc_Upload.zip')
uploaded=files.upload()
items=[(name,payload) for name,payload in uploaded.items() if name.lower().endswith('.zip')]
assert len(items)==1,f'Upload exactly one ZIP; received {[name for name,_ in items]}'
package=INPUT_ROOT/'stage13c_upload.zip'; package.write_bytes(items[0][1])
with zipfile.ZipFile(package) as archive:
    unsafe=[name for name in archive.namelist() if name.startswith(('/', '\\\\')) or '..' in Path(name).parts]
    assert not unsafe,f'Unsafe archive members: {unsafe}'
    archive.extractall(INPUT_ROOT)
required={'dbpatchkan_stage13c_code.zip','evbattery_stage13_transitions.npz','stage13b_frozen_predictions.csv','stage13b_frozen_metrics.csv','STAGE13C_UPLOAD_MANIFEST.json'}
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
# BLOCK 3 — Install, verify immutable inputs and confirm A100
update('BLOCK 3 installing and verifying')
if PROJECT_ROOT.exists(): shutil.rmtree(PROJECT_ROOT)
PROJECT_ROOT.mkdir(parents=True)
with zipfile.ZipFile(INPUT_ROOT/'dbpatchkan_stage13c_code.zip') as archive: archive.extractall(PROJECT_ROOT)
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

manifest=json.loads((INPUT_ROOT/'STAGE13C_UPLOAD_MANIFEST.json').read_text())
for name,record in manifest['files'].items():
    observed=sha256(INPUT_ROOT/name)
    assert observed==record['sha256'],f'Hash mismatch for {name}'
    update(f'Verified immutable input {name}')
import torch,pandas as pd,numpy as np,dbpatchkan
assert torch.cuda.is_available(),'Enable Runtime > Change runtime type > A100 GPU'
update(f'BLOCK 3 complete: GPU={torch.cuda.get_device_name(0)}, torch={torch.__version__}')
"""
        ),
        cell(
            """
# BLOCK 4 — Audit frozen cohort and Stage 13B comparison
from dbpatchkan.analysis.stage13_persistence_forecast import load_transition_cache
CACHE=INPUT_ROOT/'evbattery_stage13_transitions.npz'
FROZEN=INPUT_ROOT/'frozen_stage13b'
FROZEN.mkdir(exist_ok=True)
shutil.copy2(INPUT_ROOT/'stage13b_frozen_predictions.csv',FROZEN/'stage13b_all_method_ensemble_predictions.csv')
shutil.copy2(INPUT_ROOT/'stage13b_frozen_metrics.csv',FROZEN/'stage13b_all_method_metrics.csv')
loaded=load_transition_cache(CACHE); table=loaded['table']
frozen=pd.read_csv(FROZEN/'stage13b_all_method_ensemble_predictions.csv')
required_models={'Persistence','PersistenceAnchored_KAN','PersistenceAnchored_TSLANet_2024','PersistenceAnchored_ModernTCN_2024','PersistenceAnchored_KALinformer_2025'}
assert len(table)==9714 and table.vehicle_key.nunique()==191
assert required_models.issubset(set(frozen.model))
assert frozen.loc[frozen.model.eq('PersistenceAnchored_KAN')].transition_id.nunique()==9714
display(pd.read_csv(FROZEN/'stage13b_all_method_metrics.csv').sort_values('mae_ah'))
update('BLOCK 4 complete: Stage 13B comparison frozen; no existing model will be retrained')
"""
        ),
        cell(
            """
# BLOCK 5 — TimePro A100 forward/backward smoke test
from dbpatchkan.analysis.stage13c_timepro import MODEL_NAME, PersistenceAnchoredTimePro
from dbpatchkan.analysis.stage13b_recent_methods import parameter_count
model=PersistenceAnchoredTimePro().cuda()
sequence=torch.randn(4,128,8,device='cuda')
concepts=torch.randn(4,20,device='cuda')
with torch.autocast('cuda',dtype=torch.bfloat16): output=model(sequence,concepts)
loss=output.float().square().mean(); loss.backward()
assert output.shape==(4,) and torch.isfinite(output).all()
update(f'BLOCK 5 complete: {MODEL_NAME}, parameters={parameter_count(model):,}')
del model,sequence,concepts,output,loss
torch.cuda.empty_cache()
"""
        ),
        cell(
            """
# BLOCK 6 — Five-fold, three-seed TimePro comparison
update('BLOCK 6 starting; progress prints for every fold, seed and 10 epochs')
RESULTS=RUN_ROOT/'results'
command=[sys.executable,'-m','dbpatchkan.analysis.stage13c_timepro','--cache',str(CACHE),'--frozen-stage13b',str(FROZEN),'--output-dir',str(RESULTS),'--folds',str(FOLDS),'--seeds',*[str(seed) for seed in SEEDS],'--epochs',str(EPOCHS),'--batch-size',str(BATCH_SIZE)]
process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=os.environ.copy())
for line in process.stdout: print(line,end='',flush=True)
return_code=process.wait(); assert return_code==0,f'Stage 13C failed: {return_code}'
update('BLOCK 6 complete')
"""
        ),
        cell(
            """
# BLOCK 7 — Post-hoc results and paired vehicle intervals
summary=pd.read_csv(RESULTS/'stage13c_all_method_metrics.csv').sort_values('mae_ah')
paired=pd.read_csv(RESULTS/'stage13c_paired_bootstrap.csv')
horizon=pd.read_csv(RESULTS/'stage13c_horizon_metrics.csv')
manufacturer=pd.read_csv(RESULTS/'stage13c_manufacturer_metrics.csv')
display(summary)
display(paired)
focus=['Persistence','PersistenceAnchored_KAN','PersistenceAnchored_TSLANet_2024','PersistenceAnchored_TimePro_2025_POSTHOC']
display(horizon[horizon.model.isin(focus)].pivot(index='model',columns='horizon_band',values='mae_ah'))
display(manufacturer[manufacturer.model.isin(focus)].pivot(index='model',columns='manufacturer_id',values='mae_ah'))
update(f"BLOCK 7 complete: lowest overall MAE={summary.iloc[0].model} ({summary.iloc[0].mae_ah:.6f} Ah); POSTHOC=True")
"""
        ),
        cell(
            """
# BLOCK 8 — Audit all 15 new checkpoints and machine-readable decision
required={'stage13c_timepro_predictions.csv','stage13c_timepro_fold_metrics.csv','stage13c_timepro_ensemble_predictions.csv','stage13c_all_method_ensemble_predictions.csv','stage13c_all_method_metrics.csv','stage13c_paired_bootstrap.csv','stage13c_horizon_metrics.csv','stage13c_manufacturer_metrics.csv','STAGE13C_AUDIT.json','STAGE13C_DECISION.json'}
missing=sorted(name for name in required if not (RESULTS/name).exists())
assert not missing,f'Missing outputs: {missing}'
checkpoints=list((RESULTS/'models').rglob('model.pt'))
histories=list((RESULTS/'models').rglob('history.csv'))
assert len(checkpoints)==15 and len(histories)==15,(len(checkpoints),len(histories))
decision=json.loads((RESULTS/'STAGE13C_DECISION.json').read_text())
assert decision['posthoc'] is True
update(f'BLOCK 8 decision: {decision}')
update('All 15 TimePro checkpoints and histories are present')
"""
        ),
        cell(
            """
# BLOCK 9 — Automatically download every model and result
from google.colab import files
archive_base=Path('/content/DBPatchKAN_stage13c_TIMEPRO_2025_POSTHOC_ALL_RESULTS')
archive_path=Path(shutil.make_archive(str(archive_base),'zip',root_dir=RUN_ROOT))
update(f'BLOCK 9 archive ready: {archive_path.stat().st_size/2**20:.1f} MB')
files.download(str(archive_path))
update('BLOCK 9 complete — download triggered; session may now be stopped')
"""
        ),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)


def build_package() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    frozen_predictions = STAGE13B_RESULTS / "stage13b_all_method_ensemble_predictions.csv"
    frozen_metrics = STAGE13B_RESULTS / "stage13b_all_method_metrics.csv"
    for required in (TRANSITIONS_SOURCE, frozen_predictions, frozen_metrics):
        if not required.exists():
            raise FileNotFoundError(required)
    transitions = BUILD / "evbattery_stage13_transitions.npz"
    predictions = BUILD / "stage13b_frozen_predictions.csv"
    metrics = BUILD / "stage13b_frozen_metrics.csv"
    shutil.copy2(TRANSITIONS_SOURCE, transitions)
    shutil.copy2(frozen_predictions, predictions)
    shutil.copy2(frozen_metrics, metrics)
    code_zip = build_code_zip()
    files = [code_zip, transitions, predictions, metrics]
    manifest = {
        "stage": "13C",
        "purpose": "post-hoc TimePro 2025 state-space comparison",
        "posthoc": True,
        "new_method": "PersistenceAnchored_TimePro_2025_POSTHOC",
        "frozen_stage13b_predictions_reused": True,
        "architecture_status": "pure-PyTorch architecture-faithful task adaptation",
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in files
        },
        "target": "next_distinct_mileage_partial_window_capacity_ah",
        "target_is_absolute_soh": False,
    }
    manifest_path = BUILD / "STAGE13C_UPLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(PACKAGE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in [*files, manifest_path]:
            archive.write(path, path.name)


if __name__ == "__main__":
    build_notebook()
    build_package()
    print(NOTEBOOK)
    print(PACKAGE)
