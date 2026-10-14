"""Build the Stage 13B recent-method comparison notebook and upload package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "outputs" / "stage13b_recent_methods_upload"
TRANSITIONS_SOURCE = (
    ROOT
    / "outputs"
    / "stage13_persistence_forecast_upload"
    / "evbattery_stage13_transitions.npz"
)
STAGE13_RESULTS = (
    ROOT
    / "outputs"
    / "colab_completed"
    / "stage13_persistence_anchored_forecast"
    / "final_results"
    / "results"
)
NOTEBOOK = ROOT / "notebooks" / "14_colab_a100_stage13b_recent_methods.ipynb"
PACKAGE = ROOT / "outputs" / "DBPatchKAN_Stage13B_Recent_Methods_Upload.zip"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def cell(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_code_zip() -> Path:
    output = BUILD / "dbpatchkan_stage13b_code.zip"
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
            r"""# Stage 13B — Recent 2024–2025 Method Comparison

This notebook adds only three non-duplicative recent methods:

1. **ModernTCN (ICLR 2024)** — large-kernel pure convolution.
2. **TSLANet (ICML 2024)** — adaptive spectral filtering and interactive convolution.
3. **KAL / KAN-Linformer (Neurocomputing 2025)** — low-rank attention with a
   latent KAN feed-forward block.

Every model receives the same current 128-step charging sequence and the same
20 fold-scaled concepts. Every model predicts a signed residual added to current
capacity. The five vehicle folds, three seeds, target, scaling boundary,
conformal calibration and bootstrap procedure match Stage 13.

The package contains frozen Stage 13 predictions. DB-PatchKAN, matched MLP,
persistence, Ridge and Random Forest are **not retrained**.

These are architecture-faithful task adaptations, not claims of exact paper
reproduction. The KAN-Linformer paper does not provide a task-ready public
implementation for this EVBattery next-checkpoint formulation.

Expected A100 runtime: **45–90 minutes**. The last block automatically downloads
all 45 new model checkpoints and all comparison results."""
        ),
        cell(
            """
# BLOCK 1 — Fixed comparison configuration
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, zipfile

FOLDS = 5
SEEDS = [42, 52, 62]
EPOCHS = 60
BATCH_SIZE = 512
INPUT_ROOT = Path('/content/dbpatchkan_stage13b_input')
PROJECT_ROOT = Path('/content/dbpatchkan_stage13b_project')
RUN_ROOT = Path('/content/dbpatchkan_stage13b_runs/recent_methods')

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
# BLOCK 2 — Upload the single Stage 13B package
from google.colab import files
update('BLOCK 2: upload DBPatchKAN_Stage13B_Recent_Methods_Upload.zip')
uploaded=files.upload()
items=[(name,payload) for name,payload in uploaded.items() if name.lower().endswith('.zip')]
assert len(items)==1,f'Upload exactly one ZIP; received {[name for name,_ in items]}'
package=INPUT_ROOT/'stage13b_upload.zip'; package.write_bytes(items[0][1])
with zipfile.ZipFile(package) as archive:
    unsafe=[name for name in archive.namelist() if name.startswith(('/', '\\\\')) or '..' in Path(name).parts]
    assert not unsafe,f'Unsafe members: {unsafe}'
    archive.extractall(INPUT_ROOT)
required={'dbpatchkan_stage13b_code.zip','evbattery_stage13_transitions.npz','stage13_frozen_predictions.csv','stage13_frozen_metrics.csv','STAGE13B_UPLOAD_MANIFEST.json'}
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
# BLOCK 3 — Install code, verify immutable inputs and confirm A100
update('BLOCK 3 installing and verifying')
if PROJECT_ROOT.exists(): shutil.rmtree(PROJECT_ROOT)
PROJECT_ROOT.mkdir(parents=True)
with zipfile.ZipFile(INPUT_ROOT/'dbpatchkan_stage13b_code.zip') as archive: archive.extractall(PROJECT_ROOT)
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

manifest=json.loads((INPUT_ROOT/'STAGE13B_UPLOAD_MANIFEST.json').read_text())
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
# BLOCK 4 — Audit frozen cohort and frozen Stage 13 predictions
from dbpatchkan.analysis.stage13_persistence_forecast import load_transition_cache
CACHE=INPUT_ROOT/'evbattery_stage13_transitions.npz'
FROZEN=INPUT_ROOT/'frozen_stage13'
FROZEN.mkdir(exist_ok=True)
shutil.copy2(INPUT_ROOT/'stage13_frozen_predictions.csv',FROZEN/'stage13_seed_ensemble_predictions.csv')
shutil.copy2(INPUT_ROOT/'stage13_frozen_metrics.csv',FROZEN/'stage13_seed_ensemble_metrics.csv')
loaded=load_transition_cache(CACHE); table=loaded['table']
frozen=pd.read_csv(FROZEN/'stage13_seed_ensemble_predictions.csv')
required_models={'Persistence','PersistenceAnchored_KAN','PersistenceAnchored_MLP','Anchored_RF','Anchored_Ridge'}
assert len(table)==9714 and table.vehicle_key.nunique()==191
assert required_models.issubset(set(frozen.model))
assert frozen.loc[frozen.model.eq('PersistenceAnchored_KAN')].transition_id.nunique()==9714
display(pd.read_csv(FROZEN/'stage13_seed_ensemble_metrics.csv').sort_values('mae_ah'))
update('BLOCK 4 complete: Stage 13 predictions are frozen and complete')
"""
        ),
        cell(
            """
# BLOCK 5 — A100 forward/backward smoke test for all three new architectures
from dbpatchkan.analysis.stage13b_recent_methods import METHODS, MODEL_NAMES, make_model, parameter_count
for method in METHODS:
    model=make_model(method).cuda()
    sequence=torch.randn(4,128,8,device='cuda')
    concepts=torch.randn(4,20,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16): output=model(sequence,concepts)
    loss=output.float().square().mean()+1e-4*model.regularization_loss()
    loss.backward()
    assert output.shape==(4,) and torch.isfinite(output).all()
    update(f'{MODEL_NAMES[method]} smoke passed; parameters={parameter_count(model):,}')
    del model,sequence,concepts,output,loss
    torch.cuda.empty_cache()
update('BLOCK 5 complete')
"""
        ),
        cell(
            """
# BLOCK 6 — Five-fold, three-seed recent-method benchmark
update('BLOCK 6 starting; progress prints for every fold, model, seed and 10 epochs')
RESULTS=RUN_ROOT/'results'
command=[sys.executable,'-m','dbpatchkan.analysis.stage13b_recent_methods','--cache',str(CACHE),'--stage13-results',str(FROZEN),'--output-dir',str(RESULTS),'--folds',str(FOLDS),'--seeds',*[str(seed) for seed in SEEDS],'--epochs',str(EPOCHS),'--batch-size',str(BATCH_SIZE)]
process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=os.environ.copy())
for line in process.stdout: print(line,end='',flush=True)
return_code=process.wait(); assert return_code==0,f'Stage 13B failed: {return_code}'
update('BLOCK 6 complete')
"""
        ),
        cell(
            """
# BLOCK 7 — Primary results and paired vehicle confidence intervals
summary=pd.read_csv(RESULTS/'stage13b_all_method_metrics.csv').sort_values('mae_ah')
paired=pd.read_csv(RESULTS/'stage13b_proposed_paired_bootstrap.csv')
horizon=pd.read_csv(RESULTS/'stage13b_horizon_metrics.csv')
manufacturer=pd.read_csv(RESULTS/'stage13b_manufacturer_metrics.csv')
display(summary)
display(paired)
display(horizon.pivot(index='model',columns='horizon_band',values='mae_ah'))
display(manufacturer.pivot(index='model',columns='manufacturer_id',values='mae_ah'))
update(f"BLOCK 7 complete: lowest overall MAE={summary.iloc[0].model} ({summary.iloc[0].mae_ah:.6f} Ah)")
"""
        ),
        cell(
            """
# BLOCK 8 — Output audit and final machine-readable decision
required={'stage13b_recent_method_predictions.csv','stage13b_recent_method_fold_metrics.csv','stage13b_all_method_ensemble_predictions.csv','stage13b_all_method_metrics.csv','stage13b_proposed_paired_bootstrap.csv','stage13b_horizon_metrics.csv','stage13b_manufacturer_metrics.csv','STAGE13B_AUDIT.json','STAGE13B_DECISION.json'}
missing=sorted(name for name in required if not (RESULTS/name).exists())
assert not missing,f'Missing outputs: {missing}'
checkpoints=list((RESULTS/'models').rglob('model.pt'))
histories=list((RESULTS/'models').rglob('history.csv'))
assert len(checkpoints)==45 and len(histories)==45,(len(checkpoints),len(histories))
decision=json.loads((RESULTS/'STAGE13B_DECISION.json').read_text())
update(f'BLOCK 8 decision: {decision}')
update('All 45 recent-method checkpoints and histories are present')
"""
        ),
        cell(
            """
# BLOCK 9 — Automatically download every model and result
from google.colab import files
archive_base=Path('/content/DBPatchKAN_stage13b_RECENT_2024_2025_METHODS_ALL_RESULTS')
archive_path=Path(shutil.make_archive(str(archive_base),'zip',root_dir=RUN_ROOT))
update(f'BLOCK 9 archive ready: {archive_path.stat().st_size/2**20:.1f} MB')
files.download(str(archive_path))
update('BLOCK 9 complete — browser download triggered; the session may now be stopped')
"""
        ),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)


def build_package() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    frozen_predictions = STAGE13_RESULTS / "stage13_seed_ensemble_predictions.csv"
    frozen_metrics = STAGE13_RESULTS / "stage13_seed_ensemble_metrics.csv"
    for required in (TRANSITIONS_SOURCE, frozen_predictions, frozen_metrics):
        if not required.exists():
            raise FileNotFoundError(required)
    transitions = BUILD / "evbattery_stage13_transitions.npz"
    predictions = BUILD / "stage13_frozen_predictions.csv"
    metrics = BUILD / "stage13_frozen_metrics.csv"
    shutil.copy2(TRANSITIONS_SOURCE, transitions)
    shutil.copy2(frozen_predictions, predictions)
    shutil.copy2(frozen_metrics, metrics)
    code_zip = build_code_zip()
    files = [code_zip, transitions, predictions, metrics]
    manifest = {
        "stage": "13B",
        "purpose": "recent 2024-2025 persistence-anchored method comparison",
        "new_methods": ["ModernTCN_2024", "TSLANet_2024", "KALinformer_2025"],
        "frozen_stage13_predictions_reused": True,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in files
        },
        "target": "next_distinct_mileage_partial_window_capacity_ah",
        "target_is_absolute_soh": False,
    }
    manifest_path = BUILD / "STAGE13B_UPLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(PACKAGE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in [*files, manifest_path]:
            archive.write(path, path.name)


if __name__ == "__main__":
    build_notebook()
    build_package()
    print(NOTEBOOK)
    print(PACKAGE)
