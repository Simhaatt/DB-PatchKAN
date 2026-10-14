"""Build the Stage 12 EVBattery external-capacity validation notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "12_colab_a100_evbattery_external_capacity.ipynb"
PACKAGE = ROOT / "outputs" / "DBPatchKAN_Stage12_EVBattery_External_Upload.zip"
BUILD_DIR = ROOT / "outputs" / "stage12_evbattery_upload"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_code_zip() -> Path:
    path = BUILD_DIR / "dbpatchkan_stage12_code.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in ("pyproject.toml", "README.md"):
            archive.write(ROOT / relative, relative)
        for base in (ROOT / "src",):
            for file in sorted(base.rglob("*")):
                if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                    archive.write(file, file.relative_to(ROOT).as_posix())
    return path


def build_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "accelerator": "GPU",
        "colab": {"name": NOTEBOOK.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Stage 12 — EVBattery external capacity validation

This notebook performs a new, vehicle-disjoint validation using three real-world
EVBattery manufacturer datasets (464/465 vehicles; over 1.2 million charging
snippets in the complete release).

It evaluates the transferable **battery branch** of DB-PatchKAN:

`charging patches → battery Patch Transformer → named charging concepts → KAN`

The matched MLP, Random Forest, Ridge, mean and warm-start persistence are
evaluated under the same outer vehicle folds. The target is the released
**partial-window capacity in Ah**, not absolute SOH. Driver behaviour is absent
from EVBattery and is not claimed to be externally validated here.

Expected A100 duration: approximately **60–120 minutes**, including one-time
streaming conversion. The cache is saved to Drive and reused after interruption."""
        ),
        code(
            """
# BLOCK 1 — Configuration
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, time, zipfile

DRIVE_DATA_DIR = Path('/content/drive/MyDrive/DBPatchKAN_external/EVBattery')
DRIVE_RUN_DIR = Path('/content/drive/MyDrive/DBPatchKAN_runs/stage12_evbattery')
MAX_MEMBERS_PER_ARCHIVE = 200_000
MAX_RECORDS_PER_VEHICLE = 64
FOLDS = 5
SEEDS = [42, 52, 62]
EPOCHS = 60
BATCH_SIZE = 512

INPUT_ROOT = Path('/content/dbpatchkan_stage12_input')
PROJECT_ROOT = Path('/content/dbpatchkan_stage12_project')
RUN_ROOT = Path('/content/dbpatchkan_stage12_runs/stage12_evbattery')

def update(message):
    stamp = datetime.now(timezone.utc).strftime('%H:%M:%S')
    print(f'[{stamp} UTC] {message}', flush=True)

update('BLOCK 1 complete')
update(f'folds={FOLDS}, seeds={SEEDS}, epochs={EPOCHS}, max members/archive={MAX_MEMBERS_PER_ARCHIVE:,}')
"""
        ),
        code(
            """
# BLOCK 2 — Upload the small Stage 12 code package
from google.colab import files

update('BLOCK 2: upload DBPatchKAN_Stage12_EVBattery_External_Upload.zip')
uploaded = files.upload()
zips = [(name, payload) for name, payload in uploaded.items() if name.lower().endswith('.zip')]
assert len(zips) == 1, f'Upload exactly one Stage 12 ZIP; received {[name for name, _ in zips]}'
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
package = INPUT_ROOT / 'stage12_upload.zip'
package.write_bytes(zips[0][1])
with zipfile.ZipFile(package) as archive:
    archive.extractall(INPUT_ROOT)
required = {'dbpatchkan_stage12_code.zip', 'STAGE12_UPLOAD_MANIFEST.json'}
located = {p.name: p for p in INPUT_ROOT.rglob('*') if p.is_file()}
missing = sorted(required - set(located))
assert not missing, f'Upload package incomplete: {missing}'
for name in required:
    if located[name].resolve() != (INPUT_ROOT / name).resolve():
        shutil.copy2(located[name], INPUT_ROOT / name)
update(f'BLOCK 2 complete: received {len(zips[0][1])/2**20:.2f} MB')
"""
        ),
        code(
            """
# BLOCK 3 — Mount Drive and verify the three immutable source archives
from google.colab import drive
drive.mount('/content/drive')
DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)

expected = {
    'battery_dataset1.tar.gz': 'BF56F38EF1F0C96BB184F085F261F72092D16366C5EFB7264DF55E68FBA16BE5',
    'battery_dataset2.tar.gz': '265E3A7B6CEDCD8A6C1844C20C9E69F0DE7DCDEE07F10816C29479791366107F',
    'battery_dataset3.tar.gz': '0B258606D79D6392C0F8C76E74B1F544650AB7FBF29E758F078C2C6AB74630AD',
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest().upper()

archives=[]
for name, wanted in expected.items():
    path=DRIVE_DATA_DIR/name
    assert path.exists(), f'Missing {path}. Upload all three .tar.gz files to Drive first.'
    update(f'Hashing {name} ({path.stat().st_size/2**30:.2f} GB)')
    observed=sha256(path)
    assert observed == wanted, f'Hash mismatch for {name}: {observed}'
    archives.append(path)
    update(f'Verified {name}')
update('BLOCK 3 complete: all original archives verified')
"""
        ),
        code(
            """
# BLOCK 4 — Install the local project and verify the A100
update('BLOCK 4 installing project')
if PROJECT_ROOT.exists(): shutil.rmtree(PROJECT_ROOT)
PROJECT_ROOT.mkdir(parents=True)
with zipfile.ZipFile(INPUT_ROOT/'dbpatchkan_stage12_code.zip') as archive:
    archive.extractall(PROJECT_ROOT)
roots=list(PROJECT_ROOT.rglob('pyproject.toml'))
assert len(roots)==1
PROJECT=roots[0].parent
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(PROJECT)],check=True)
source_root=str(PROJECT/'src')
if source_root not in sys.path: sys.path.insert(0,source_root)
os.environ['PYTHONPATH']=source_root+os.pathsep+os.environ.get('PYTHONPATH','')
import torch, pandas as pd, numpy as np, dbpatchkan
assert torch.cuda.is_available(), 'Enable Runtime > Change runtime type > A100 GPU'
update(f'BLOCK 4 complete: GPU={torch.cuda.get_device_name(0)}, torch={torch.__version__}')
"""
        ),
        code(
            """
# BLOCK 5 — Stream a balanced labeled sample and cache it on Drive
cache_drive=DRIVE_RUN_DIR/f'evbattery_capacity_m{MAX_MEMBERS_PER_ARCHIVE}_v{MAX_RECORDS_PER_VEHICLE}.npz'
cache_local=RUN_ROOT/cache_drive.name
if cache_drive.exists():
    update(f'BLOCK 5 reusing cached conversion: {cache_drive}')
    shutil.copy2(cache_drive,cache_local)
else:
    update('BLOCK 5 starting one-time direct-from-tar conversion')
    command=[sys.executable,'-m','dbpatchkan.analysis.stage12_evbattery','prepare','--archives',*[str(p) for p in archives],'--output',str(cache_local),'--max-members-per-archive',str(MAX_MEMBERS_PER_ARCHIVE),'--max-records-per-vehicle',str(MAX_RECORDS_PER_VEHICLE),'--seed','42']
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=os.environ.copy())
    for line in process.stdout: print(line,end='',flush=True)
    assert process.wait()==0, 'EVBattery conversion failed'
    shutil.copy2(cache_local,cache_drive)
    update(f'Cached conversion to Drive: {cache_drive}')
update(f'BLOCK 5 complete: cache={cache_local.stat().st_size/2**20:.1f} MB')
"""
        ),
        code(
            """
# BLOCK 6 — Audit the converted cohort before model training
from dbpatchkan.data.evbattery import load_evbattery_sample
loaded=load_evbattery_sample(cache_local)
meta=loaded['concepts']
audit=meta.groupby('manufacturer_id').agg(records=('vehicle_id','size'),vehicles=('vehicle_id','nunique'),capacity_min_ah=('capacity_ah','min'),capacity_max_ah=('capacity_ah','max'),capacity_mean_ah=('capacity_ah','mean')).reset_index()
display(audit)
assert meta[['manufacturer_id','vehicle_id']].drop_duplicates().shape[0] >= 100
assert pd.to_numeric(meta['capacity_ah']).gt(0).all()
update(f"BLOCK 6 complete: {len(meta):,} snippets, {meta[['manufacturer_id','vehicle_id']].drop_duplicates().shape[0]} vehicles")
"""
        ),
        code(
            """
# BLOCK 7 — Five-fold vehicle-disjoint external validation
update('BLOCK 7 starting grouped validation; progress prints every fold/model/10 epochs')
results=RUN_ROOT/'results'
command=[sys.executable,'-m','dbpatchkan.analysis.stage12_evbattery','validate','--cache',str(cache_local),'--output-dir',str(results),'--folds',str(FOLDS),'--seeds',*[str(s) for s in SEEDS],'--epochs',str(EPOCHS),'--batch-size',str(BATCH_SIZE)]
process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=os.environ.copy())
for line in process.stdout: print(line,end='',flush=True)
assert process.wait()==0, 'Stage 12 validation failed'
update('BLOCK 7 complete')
"""
        ),
        code(
            """
# BLOCK 8 — Display the final comparisons and copy results to Drive
summary=pd.read_csv(results/'stage12_model_summary.csv').sort_values('vehicle_macro_mae_ah')
fold_metrics=pd.read_csv(results/'stage12_fold_metrics.csv')
display(summary)
display(fold_metrics.groupby('model').agg(folds=('fold','nunique'),seeds=('seed','nunique'),MAE_Ah=('mae_ah','mean'),Vehicle_macro_MAE_Ah=('vehicle_macro_mae_ah','mean'),R2=('r2','mean')).sort_values('Vehicle_macro_MAE_Ah'))
drive_results=DRIVE_RUN_DIR/'results'
if drive_results.exists(): shutil.rmtree(drive_results)
shutil.copytree(results,drive_results)
update(f'BLOCK 8 complete: results copied to {drive_results}')
"""
        ),
        code(
            """
# BLOCK 9 — Package and automatically download every model/result
from google.colab import files
archive_base=Path('/content/DBPatchKAN_stage12_EVBATTERY_EXTERNAL_ALL_MODELS_RESULTS')
archive_path=Path(shutil.make_archive(str(archive_base),'zip',root_dir=RUN_ROOT))
drive_zip=DRIVE_RUN_DIR/archive_path.name
shutil.copy2(archive_path,drive_zip)
update(f'BLOCK 9 archive ready: {archive_path.stat().st_size/2**20:.1f} MB')
update(f'Drive backup: {drive_zip}')
files.download(str(archive_path))
update('BLOCK 9 complete — browser download triggered')
"""
        ),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK)


def build_package() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    code_zip = build_code_zip()
    manifest = {
        "stage": 12,
        "purpose": "EVBattery vehicle-disjoint partial-window capacity validation",
        "files": {code_zip.name: {"bytes": code_zip.stat().st_size, "sha256": digest(code_zip)}},
        "raw_archives_included": False,
        "raw_archive_location": "/content/drive/MyDrive/DBPatchKAN_external/EVBattery",
    }
    manifest_path = BUILD_DIR / "STAGE12_UPLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(PACKAGE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(code_zip, code_zip.name)
        archive.write(manifest_path, manifest_path.name)


if __name__ == "__main__":
    build_notebook()
    build_package()
    print(NOTEBOOK)
    print(PACKAGE)
