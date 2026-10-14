"""Prepare a Colab delivery with file I/O only. No local analysis or training.

ZIP_STORED avoids CPU-heavy compression while the user's device is occupied.
Compilation checks use the standard-library AST, not the scientific runtime.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/17_colab_a100_stage15_energy_ai_review.ipynb'
PACKAGE=ROOT/'outputs/DBPatchKAN_Stage15_Energy_AI_Review_Upload.zip'
README=ROOT/'docs/STAGE15_COLAB_REMOTE_INSTRUCTIONS.md'
COMPLETED=ROOT/'outputs/colab_completed'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def code(text):
    text=text.strip()
    ast.parse(text)
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text}


def markdown(text):return {'cell_type':'markdown','metadata':{},'source':text.strip()}


def notebook():
    cells=[markdown(r"""# Stage 15 — Energy and AI revision experiments (remote CPU + A100)

All scientific computation runs on **Colab's remote CPU/GPU**. This delivery is
prepared on your device using file I/O and syntax checks only. No scientific
analyses or model fits have been executed locally.

This notebook completes the planned E1–E6 package:

- **E1:** recompute D-BSI on development telemetry; equal weights, single
  components, component removal, threshold/capacity perturbations, 100 bounded
  weight draws, and training-only nonnegative residual weighting.
- **E2:** horizon, anonymous manufacturer and joint subgroups; matched vehicle
  bootstrap intervals, cluster sign-flip tests, within-family Holm correction,
  vehicle-macro MAE and saved-interval coverage.
- **E3:** exact Nissan health-target lineage and shared-checkpoint clustering;
  label plateaus/increases; optional bounded original EVBattery label sample;
  hypothetical label-noise sensitivity; actual logged physics/driver-ID losses.
- **E4:** genuinely future responses at 30/60/120 seconds from the last 120
  seconds, using causal joins and backward derivatives. RWTH leave-one-vehicle-out
  and EVE whole-session chronological splits. Matched persistence,
  constrained longitudinal/electrothermal baseline, physics-feature Ridge,
  context HGB, context+driver HGB and dual-stream
  concept models with KAN or MLP. Three seeds; up to 30 epochs; patience 6.
- **E5:** reuse original fleet KAN checkpoints; choose each switching threshold
  on the original outer-training inner validation vehicles, then evaluate on
  that fold's held-out vehicles.
- **E6:** saved fleet KAN/MLP/TSLANet CPU latency, sizes and sampled process memory;
  concept perturbation stability across three seeds. Cloud hardware is recorded.

E7 needs a new multi-vehicle longitudinal driver+health dataset and is not
included. New analyses are post-hoc strengthening experiments on known datasets;
the frozen Nissan locked-test model stays fixed. Nissan locked-test **saved
predictions and metadata** are audited, without reopening its tensor or retraining.

**Estimated first A100 execution: 2–4 hours**, with an allowance of up to 6 hours
for EVE downloads and remote preparation. Actual time is unknown until this
new protocol runs. Previously completed benchmark training is reused.

Use Drive for this package (>50 MB), or choose browser upload if preferred.
The final block downloads a ZIP with every result, model, history, audit, source,
test log and figure, and saves a copy in Drive when enabled."""),code("""
# BLOCK 1 — Fixed protocol and storage choices
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
INPUT_MODE = 'drive'  # 'drive' or 'upload'
DRIVE_PACKAGE = '/content/drive/MyDrive/DBPatchKAN_stage15/DBPatchKAN_Stage15_Energy_AI_Review_Upload.zip'
DRIVE_RUNS_DIR = '/content/drive/MyDrive/DBPatchKAN_stage15/runs'
RAW_LABEL_DIR = '/content/drive/MyDrive/DBPatchKAN_external'
SAVE_TO_DRIVE = True
RUN_ID = 'stage15_energy_ai_review_v1'
SEEDS = [42, 52, 62]
RESPONSE_EPOCHS = 30
BOOTSTRAP_DRAWS = 10000
MAX_RWTH_EXAMPLES_PER_VEHICLE = 1500
DOWNLOAD_EVE = True
RUN_ORIGINAL_LABEL_SAMPLE = True
MAX_ORIGINAL_LABEL_MEMBERS_PER_ARCHIVE = 10000
INPUT_ROOT = Path('/content/dbpatchkan_stage15_input')
PROJECT_ROOT = Path('/content/dbpatchkan_stage15_project')
def update(message):
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")} UTC] {message}', flush=True)
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
update('BLOCK 1 complete. Computation will run on the remote Colab machine.')
"""),code("""
# BLOCK 2 — Acquire the single package; duplicate upload names are supported
if INPUT_MODE == 'drive' or SAVE_TO_DRIVE or RUN_ORIGINAL_LABEL_SAMPLE:
    from google.colab import drive
    drive.mount('/content/drive')
if INPUT_MODE == 'drive':
    package = Path(DRIVE_PACKAGE)
    assert package.exists(), f'Upload the ZIP to this Drive path first: {package}'
else:
    from google.colab import files
    update('Upload DBPatchKAN_Stage15_Energy_AI_Review_Upload.zip')
    uploaded=files.upload()
    zip_items=[(name,value) for name,value in uploaded.items() if name.lower().endswith('.zip')]
    assert len(zip_items)==1, 'Upload exactly one complete ZIP'
    package=Path('/content/stage15_browser_upload.zip')
    package.write_bytes(zip_items[0][1])
def extract_safely(archive, root):
    for entry in archive.infolist():
        normalized=entry.filename.replace('\\\\','/')
        assert not normalized.startswith('/') and ':' not in normalized.split('/')[0]
        destination=(root/normalized).resolve()
        assert destination==root.resolve() or root.resolve() in destination.parents, entry.filename
        assert (entry.external_attr >> 16) & 0o170000 != 0o120000, 'Symlink ZIP member refused'
    archive.extractall(root)
update(f'BLOCK 2 extracting {package.stat().st_size/2**20:.1f} MB on remote storage')
with zipfile.ZipFile(package) as archive: extract_safely(archive,INPUT_ROOT)
manifest=json.loads((INPUT_ROOT/'STAGE15_UPLOAD_MANIFEST.json').read_text())
PROJECT=INPUT_ROOT/'project'
assert (PROJECT/'src/dbpatchkan/analysis/stage15_review.py').exists()
assert manifest['locked_test_tensor_included'] is False
RUN_ROOT=(Path(DRIVE_RUNS_DIR)/RUN_ID) if SAVE_TO_DRIVE else (Path('/content/dbpatchkan_stage15_runs')/RUN_ID)
RUN_ROOT.mkdir(parents=True,exist_ok=True)
RESULTS=RUN_ROOT/'results';RESULTS.mkdir(exist_ok=True)
PREPARED=RUN_ROOT/'prepared'
update(f'BLOCK 2 complete: inputs={INPUT_ROOT}, outputs={RUN_ROOT}')
"""),code("""
# BLOCK 3 — Remote installation, immutable inputs and runtime verification
update('BLOCK 3 installing the project on Colab')
subprocess.run([sys.executable,'-m','pip','install','-q','numpy>=2.0','pandas>=2.2','pyarrow>=16',
                'scikit-learn>=1.5','scipy','matplotlib','psutil','pytest'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','-e',str(PROJECT)],check=True)
source_root=str(PROJECT/'src')
sys.path.insert(0,source_root)
os.environ['PYTHONPATH']=source_root+os.pathsep+os.environ.get('PYTHONPATH','')
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import importlib
for module_name in list(sys.modules):
    if module_name=='dbpatchkan' or module_name.startswith('dbpatchkan.'):del sys.modules[module_name]
importlib.invalidate_caches()
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
for i,(name,record) in enumerate(manifest['files'].items(),1):
    path=INPUT_ROOT/name
    assert path.exists() and path.stat().st_size==record['bytes'],f'Missing/incomplete {name}'
    assert sha256(path)==record['sha256'],f'Hash mismatch: {name}'
    if i==1 or i%10==0:update(f'Input integrity: {i}/{len(manifest["files"])} files verified')
import torch, numpy as np, pandas as pd, dbpatchkan
assert torch.cuda.is_available(),'Select Runtime > Change runtime type > A100 GPU'
update(f'GPU={torch.cuda.get_device_name(0)}; Python={sys.version.split()[0]}; torch={torch.__version__}')
if 'A100' not in torch.cuda.get_device_name(0):update('A different GPU is active; runtime may be longer.')
protocol={'run_id':RUN_ID,'seeds':SEEDS,'response_epochs':RESPONSE_EPOCHS,
    'bootstrap_draws':BOOTSTRAP_DRAWS,'rwth_cap':MAX_RWTH_EXAMPLES_PER_VEHICLE,'eve':DOWNLOAD_EVE,
    'raw_label_sample':RUN_ORIGINAL_LABEL_SAMPLE,'raw_label_members':MAX_ORIGINAL_LABEL_MEMBERS_PER_ARCHIVE,
    'upload_manifest_sha256':sha256(INPUT_ROOT/'STAGE15_UPLOAD_MANIFEST.json'),
    'locked_test_tensor_opened':False,'source_sha256':sha256(PROJECT/'src/dbpatchkan/analysis/stage15_response.py')}
protocol_path=RUN_ROOT/'STAGE15_RUN_PROTOCOL.json'
if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol,'Resume settings differ; use a new RUN_ID.'
else:protocol_path.write_text(json.dumps(protocol,indent=2))
def run(command,label):
    update(f'START {label}');started=time.time();log=RUN_ROOT/(label.replace(" ","_")+'.log')
    # Stream output from the actual worker, with an elapsed-time heartbeat for long I/O.
    import queue, threading
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=os.environ.copy())
    messages=queue.Queue()
    def reader():
        for line in process.stdout:messages.put(line)
        messages.put(None)
    thread=threading.Thread(target=reader,daemon=True);thread.start()
    with log.open('a',encoding='utf-8') as handle:
        while True:
            try:line=messages.get(timeout=30)
            except queue.Empty:
                update(f'{label}: elapsed={(time.time()-started)/60:.1f} min; worker active');continue
            if line is None:break
            print(line,end='',flush=True);handle.write(line);handle.flush()
    rc=process.wait()
    assert rc==0,f'{label} failed with code {rc}; log={log}. Fix and rerun this block.'
    update(f'DONE {label}: {(time.time()-started)/60:.1f} min')
CPU=[sys.executable,'-u','-m','dbpatchkan.analysis.stage15_review','--input-dir',str(INPUT_ROOT),
     '--output-dir',str(RESULTS),'--bootstrap-draws',str(BOOTSTRAP_DRAWS)]
RESPONSE=[sys.executable,'-u','-m','dbpatchkan.analysis.stage15_response','--input-dir',str(INPUT_ROOT),
          '--output-dir',str(RESULTS),'--prepared-dir',str(PREPARED)]
update('BLOCK 3 complete; source path is explicitly active in notebook and worker imports')
"""),code("""
# BLOCK 4 — Remote scientific integrity tests (no local execution)
run([sys.executable,'-m','pytest','-q',str(PROJECT/'tests/test_stage15_remote.py')], 'remote_integrity_tests')
update('BLOCK 4 complete: boundary invariance, missing temperature, cluster units and neural shapes verified')
"""),code("""
# BLOCK 5 — E3 critical label lineage and actual-loss audit
command=CPU+['--tasks','e3']
raw_path=Path(RAW_LABEL_DIR)
if RUN_ORIGINAL_LABEL_SAMPLE and raw_path.exists():
    command+=['--raw-label-dir',str(raw_path),'--raw-label-members',str(MAX_ORIGINAL_LABEL_MEMBERS_PER_ARCHIVE)]
elif RUN_ORIGINAL_LABEL_SAMPLE:
    update('Original archives were not found in Drive. The report will identify this measurement-noise limitation.')
run(command,'E3_label_and_loss_audit')
display(pd.read_csv(RESULTS/'E3_nissan_independent_counts.csv'))
display(json.loads((RESULTS/'E3_auxiliary_driver_audit.json').read_text()))
update('BLOCK 5 complete: saved locked predictions audited; no locked tensor opened')
"""),code("""
# BLOCK 6 — E1 D-BSI sensitivity and development-only downstream regressions
if not (RESULTS/'E1_COMPLETE.json').exists():run(CPU+['--tasks','e1'],'E1_DBSI_robustness')
else:update('E1 already complete: reusing saved analysis')
display(pd.read_csv(RESULTS/'E1_rank_distribution_robustness.csv').head(20))
update('BLOCK 6 complete')
"""),code("""
# BLOCK 7 — E2 horizon/manufacturer statistics from frozen predictions
if not (RESULTS/'E2_COMPLETE.json').exists():run(CPU+['--tasks','e2'],'E2_subgroup_statistics')
else:update('E2 already complete: reusing saved analysis')
comparison=pd.read_csv(RESULTS/'E2_cluster_comparisons.csv')
display(comparison[comparison.candidate.eq('PersistenceAnchored_KAN')])
update('BLOCK 7 complete')
"""),code("""
# BLOCK 8 — E4 causal native telemetry preparation on the remote CPU
command=RESPONSE+['--task','prepare','--max-per-vehicle',str(MAX_RWTH_EXAMPLES_PER_VEHICLE)]
if not DOWNLOAD_EVE:command+=['--skip-eve']
audit_path=PREPARED/'E4_PREPARATION_AUDIT.json'
ready=audit_path.exists() and (PREPARED/'rwth_future_response.npz').exists()
if ready and DOWNLOAD_EVE:ready=(PREPARED/'eve_future_response.npz').exists()
if not ready:run(command,'E4_native_causal_preparation')
else:update('E4 native preparation already available; reusing cached tensors')
display(json.loads(audit_path.read_text()))
update('BLOCK 8 complete; gaps are not expanded and future samples are excluded from inputs')
"""),code("""
# BLOCK 9 — E4 matched future-response training on A100
command=RESPONSE+['--task','train','--seeds',*[str(x) for x in SEEDS],'--epochs',str(RESPONSE_EPOCHS)]
run(command,'E4_matched_future_response_training')
display(pd.read_csv(RESULTS/'E4_matched_future_response_metrics.csv'))
update('BLOCK 9 complete')
"""),code("""
# BLOCK 10 — E5 selective persistence using the original inner validation sets
if not (RESULTS/'E5_COMPLETE.json').exists():run(CPU+['--tasks','e5'],'E5_inner_selected_persistence_switch')
else:update('E5 already complete: reusing saved thresholds and predictions')
display(pd.read_csv(RESULTS/'E5_inner_selected_thresholds.csv'))
display(pd.read_csv(RESULTS/'E5_comparison_metrics.csv'))
update('BLOCK 10 complete: no threshold was chosen from E2 held-out subgroup results')
"""),code("""
# BLOCK 11 — E6 cloud CPU efficiency and named-concept response stability
run(RESPONSE+['--task','benchmark'],'E6_remote_CPU_efficiency')
display(pd.read_csv(RESULTS/'E6_remote_CPU_latency.csv'))
run(CPU+['--tasks','figures'],'review_figures')
update('BLOCK 11 complete; measured latency belongs to the Colab CPU')
"""),code("""
# BLOCK 12 — Manuscript revision evidence and completeness audit
needed={'E1':'E1_COMPLETE.json','E2':'E2_COMPLETE.json','E3':'E3_COMPLETE.json',
        'E4':'E4_TRAIN_COMPLETE.json','E5':'E5_COMPLETE.json','E6':'E6_remote_CPU_latency.csv'}
status={key:(RESULTS/file).exists() for key,file in needed.items()}
pending=[]
prep=json.loads((PREPARED/'E4_PREPARATION_AUDIT.json').read_text()) if (PREPARED/'E4_PREPARATION_AUDIT.json').exists() else {}
if DOWNLOAD_EVE and prep.get('status',{}).get('EVE')!='prepared':pending.append('EVE native response experiment: data preparation/download incomplete')
scope=json.loads((RESULTS/'E3_SCOPE.json').read_text()) if (RESULTS/'E3_SCOPE.json').exists() else {}
if scope.get('original_label_status','').startswith('unavailable'):pending.append('Original EVBattery repeated-label sample unavailable; hypothetical noise sensitivity is not measurement calibration')
pending.extend(['E7: new multi-vehicle longitudinal driving+health data needed for a general driver-induced degradation claim',
                'Physical stress thresholds remain engineering assumptions; sensitivity does not calibrate pack aging',
                'CPU energy use was not measured with a power meter',
                'Manuscript conclusions require inspection of the returned experiment results'])
audit={'completed':status,'pending_or_scope_limits':pending,'locked_tensor_used':False,
       'local_scientific_computation_used':False,'remote_runtime':True,
       'gpu':torch.cuda.get_device_name(0),'completed_utc':datetime.now(timezone.utc).isoformat(),
       'prior_predictions_sha256':sha256(INPUT_ROOT/'fleet_predictions.csv')}
(RUN_ROOT/'STAGE15_COMPLETENESS.json').write_text(json.dumps(audit,indent=2))
inventory=[{'path':str(p.relative_to(RUN_ROOT)),'bytes':p.stat().st_size} for p in RUN_ROOT.rglob('*') if p.is_file()]
pd.DataFrame(inventory).to_csv(RUN_ROOT/'artifact_inventory.csv',index=False)
lines=['# Stage 15: remote revision experiments','',f'Completion: {status}','',
       '## Evidence to incorporate into the manuscript','',
       '- E1: sensitivity/rank stability and development-only downstream value; no universal calibration claim.',
       '- E2: paired vehicle uncertainty, horizon/manufacturer heterogeneity and actual saved interval coverage.',
       '- E3: independent health transition/destination counts, label variation and active logged training terms.',
       '- E4: fixed-horizon future responses on native telemetry; distinguish measured temperature from missing RWTH thermal labels.',
       '- E5: per-fold inner-selected deployment switch; report performance even when persistence is selected.',
       '- E6: real checkpoint sizes, same-hardware CPU latency, memory and concept perturbation stability.',
       '', '## Remaining limits','']+['- '+x for x in pending]
(RUN_ROOT/'STAGE15_MANUSCRIPT_EVIDENCE.md').write_text('\\n'.join(lines))
display(audit)
update('BLOCK 12 complete. Inspect metrics and limits before revising superiority or causal claims.')
"""),code("""
# BLOCK 13 — Package and automatically download all results and model checkpoints
# This block can also be run after an interrupted stage to recover partial work.
from google.colab import files
shutil.copy2(INPUT_ROOT/'STAGE15_UPLOAD_MANIFEST.json',RUN_ROOT/'STAGE15_UPLOAD_MANIFEST.json')
for folder in ('fleet_checkpoints','benchmark_checkpoints'):
    shutil.copytree(INPUT_ROOT/folder,RUN_ROOT/'reused_model_checkpoints'/folder,dirs_exist_ok=True)
source_archive=RUN_ROOT/'EXECUTED_SOURCE.zip'
with zipfile.ZipFile(source_archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
    for path in PROJECT.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py','.toml','.md','.ipynb'}:
            archive.write(path,path.relative_to(PROJECT))
archive_path=Path('/content/DBPatchKAN_stage15_ENERGY_AI_REVIEW_ALL_MODELS_RESULTS.zip')
update('BLOCK 13 creating the final model/result ZIP on Colab')
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
    for path in RUN_ROOT.rglob('*'):
        if not path.is_file() or 'eve_raw' in path.parts or path.name=='latest.pt':continue
        # Keep final model weights and all results; resume/optimizer state stays in the run folder.
        archive.write(path,path.relative_to(RUN_ROOT))
if SAVE_TO_DRIVE:
    destination=Path(DRIVE_RUNS_DIR)/archive_path.name
    shutil.copy2(archive_path,destination)
    update(f'Drive backup saved: {destination}')
update(f'Final archive: {archive_path.stat().st_size/2**20:.1f} MB')
files.download(str(archive_path))
update('BLOCK 13 complete: download triggered. Check the ZIP/Drive backup before stopping the session.')
""")]
    doc={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU',
        'colab':{'name':NOTEBOOK.name,'provenance':[]},'kernelspec':{'name':'python3','display_name':'Python 3'},
        'language_info':{'name':'python'}},'cells':cells}
    for i,cell in enumerate(doc['cells']):cell['id']=f'stage15-{i:02d}'
    NOTEBOOK.write_text(json.dumps(doc,indent=1),encoding='utf-8')


def package():
    files={}
    def add(source,destination):
        p=Path(source)
        if not p.exists():raise FileNotFoundError(p)
        files[destination]=p
    for p in (ROOT/'src').rglob('*.py'):
        if '__pycache__' not in p.parts:add(p,'project/'+p.relative_to(ROOT).as_posix())
    for name in ('pyproject.toml','README.md','tests/test_stage15_remote.py'):
        add(ROOT/name,'project/'+name)
    add(NOTEBOOK,'project/notebooks/'+NOTEBOOK.name)
    add(README,'project/docs/'+README.name)
    for name in ('stage15_review.py','stage15_response.py'):
        ast.parse((ROOT/'src/dbpatchkan/analysis'/name).read_text(encoding='utf-8'))
    ast.parse((ROOT/'tests/test_stage15_remote.py').read_text(encoding='utf-8'))
    add(ROOT/'outputs/dbpatchkan/telemetry_1hz.parquet','nissan_telemetry_1hz.parquet')
    add(ROOT/'outputs/dbpatchkan/trip_features.csv','nissan_trip_features.csv')
    add(ROOT/'outputs/dbpatchkan/gpu_trip_labels.csv','nissan_trip_labels.csv')
    add(ROOT/'outputs/dbpatchkan/health_windows.csv','nissan_health_windows.csv')
    add(ROOT/'outputs/stage14_hyperparameter_sensitivity_upload/development_101_trips.npz','development_101_trips.npz')
    add(ROOT/'outputs/stage13_persistence_forecast_upload/evbattery_stage13_transitions.npz','evbattery_stage13_transitions.npz')
    fleet=COMPLETED/'stage13c_timepro_posthoc/final_results/results/stage13c_all_method_ensemble_predictions.csv'
    add(fleet,'fleet_predictions.csv')
    add(COMPLETED/'stage8_final_confirmation/results/stage8_final_confirmation/locked_test_once/locked_test_all_predictions.csv','nissan_locked_predictions.csv')
    add(COMPLETED/'stage7_external_full_sequence/results/stage7_external_full_sequence_seed42/stage7_analysis/stage7_external_health_predictions.csv',
        'eve_saved_health_predictions.csv')
    add(COMPLETED/'stage5_rwth_eve_external/results/stage5_rwth_eve_external_validation/stage5b_eve_harmonised/eve_monthly_soh_intervals.csv',
        'eve_original_health_intervals.csv')
    for p in (ROOT/'data/external/rwth/raw/field_test').glob('*.parquet'):add(p,'rwth_native/'+p.name)
    for fold in range(1,6):
        for seed in (42,52,62):
            p=COMPLETED/f'stage13_persistence_anchored_forecast/final_results/results/models/fold_{fold}/kan_seed_{seed}/model.pt'
            add(p,f'fleet_checkpoints/fold_{fold}/kan_seed_{seed}/model.pt')
    add(COMPLETED/'stage13_persistence_anchored_forecast/final_results/results/models/fold_1/mlp_seed_42/model.pt','benchmark_checkpoints/mlp/model.pt')
    add(COMPLETED/'stage13b_recent_methods/final_results/results/models/fold_1/tslanet_seed_42/model.pt','benchmark_checkpoints/tslanet/model.pt')
    for head,model in (('kan','Final_DBSI_Total_KAN'),('mlp','Matched_DBSI_Total_MLP')):
        add(COMPLETED/f'stage8_final_confirmation/results/stage8_final_confirmation/final_models/seed_42/{model}/final_lightweight.pt',
            f'benchmark_checkpoints/nissan_{head}/model.pt')
    training=COMPLETED/'stage8_final_confirmation/results/stage8_final_confirmation/final_models'
    for p in training.rglob('*'):
        if p.name in ('run_config.json','history.json'):
            add(p,'training_evidence/stage8/'+p.relative_to(training).as_posix())
    # Extract ONLY small prior training histories from the Stage14 ZIP, not models.
    extra={}
    stage14=COMPLETED/'stage14_hyperparameter_robustness/DBPatchKAN_stage14_HYPERPARAMETER_ROBUSTNESS_ALL_MODELS_RESULTS.zip'
    with zipfile.ZipFile(stage14) as z:
        for entry in z.infolist():
            if Path(entry.filename).name in ('history.json','run_config.json'):
                extra['training_evidence/stage14/'+entry.filename]=z.read(entry)
    manifest={'stage':15,'locked_test_tensor_included':False,'saved_locked_predictions_for_audit_only':True,
        'local_computation':'file I/O, hashing and syntax validation only; no data analysis/training/benchmarks',
        'compression':'ZIP_STORED to minimize local CPU work','files':{},'missing_experiments':['E7 new data'],
        'original_evbattery_archives':'reuse remote Drive copies if present; not included in this upload'}
    PACKAGE.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(PACKAGE,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
        for i,(name,p) in enumerate(files.items(),1):
            manifest['files'][name]={'bytes':p.stat().st_size,'sha256':digest(p)}
            z.write(p,name)
            if i%25==0:print(f'Packaged {i}/{len(files)} files (I/O only)',flush=True)
        for name,payload in extra.items():
            manifest['files'][name]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
            z.writestr(name,payload)
        z.writestr('STAGE15_UPLOAD_MANIFEST.json',json.dumps(manifest,indent=2))
    print(f'Package: {PACKAGE} ({PACKAGE.stat().st_size/2**20:.1f} MB)',flush=True)


def instructions():
    README.write_text('''# Stage 15: remote Colab execution

Prepared 14 October 2026. Scientific experiments have NOT been run on the local
computer. Local preparation uses file reads, hashes, syntax validation and an
uncompressed ZIP; it does not import the scientific analysis modules.

## Start

1. Open `17_colab_a100_stage15_energy_ai_review.ipynb` in Colab.
2. Select an A100 GPU runtime. Colab supplies both the remote CPU and GPU.
3. Upload `DBPatchKAN_Stage15_Energy_AI_Review_Upload.zip` to Drive inside
   `MyDrive/DBPatchKAN_stage15/`. The package is over 50 MB. Set
   `INPUT_MODE='upload'` for the browser dialog if you prefer direct upload.
4. Leave the original three `battery_dataset*.tar.gz` archives in Drive under
   `MyDrive/DBPatchKAN_external/`. The label audit searches nested folders.
   Original archives are optional; without them the measurement-noise limitation
   is explicitly retained in the results. No full extraction is needed.
5. Run blocks 1–13. Inputs are verified, worker output streams live, and a
   heartbeat prints during long downloads. All computation happens remotely.
6. Download the automatically generated final ZIP. A copy is also saved in
   Drive. Return it for evidence review and manuscript updates.

Expected first run: 2–4 hours on A100, with up to 6 hours allowed for downloads,
remote CPU preparation and early-stopping differences. This is an estimate,
not a measured runtime for this new protocol.

## Experiments and evidence boundaries

E1 uses only development trips for score robustness and downstream fitting.
Future target rows must also be inside development, and fold boundary targets
are purged. Scores are recomputed from the one-second cleaned telemetry;
native-rate original D-BSI values can differ because of resampling. Joint
six-component nonnegative weighting requires at least 24 training transitions
and 12 distinct future checkpoints; it is omitted and flagged otherwise.

E2 reuses all frozen fleet predictions, including TSLANet and the matched MLP.
Anonymous manufacturer IDs are not named car brands. Seed ensembles are treated
as one prediction per transition. Vehicle bootstrap and sign-flip tests avoid
treating overlapping rows or seeds as independent vehicles.

E3 audits saved locked Nissan predictions and metadata, including exact future
observation IDs. It does not load a locked-test tensor or choose a new model.
Original label dispersion is a bounded archive-order descriptive sample, not
manufacturer ground-truth measurement calibration. Hypothetical label-noise
perturbations are explicitly separated from empirical dispersion. Logged
physics/driver-ID losses are inspected alongside actual exported target masks.

E4 is a NEW fixed-horizon future-response experiment. Causal joins, backward
differences, disjoint 120-second histories and 120-second future spans, and
whole-vehicle/session splits are built from native sources on remote CPU.
RWTH provides measured electrical targets without pack temperature. EVE
provides measured module temperatures. Energy and high-current exposure are
derived from measured telemetry and are not independent degradation labels.
The constrained physical baseline fits nonnegative coefficients for
I = a*v/V + b*v^3/V + m_eff*v*accel/V + P_aux/V, an effective nonnegative
resistance in V_future = V_last - R*(I_predicted-I_last), and a local heat balance
deltaT = horizon*(beta*mean_past_I_squared - kappa*(T_last-T_ambient)), beta/kappa
nonnegative. All coefficients use only inner training data. It assumes locally
sustained driver demand and a linear short-window thermal response. Coefficients
are effective pack/fleet fits, not independently measured battery constants.
The additional physics-feature Ridge remains a fitted proxy, not a calibrated
electrochemical pack model. RWTH comparisons cluster by vehicle and EVE
comparisons by seven-day blocks; EVE remains a single-vehicle temporal study.
The 15-concept dual-stream KAN and MLP are trained
on the same inner training/validation sets and evaluated on the same outer rows.
Defaults: 24 observations at 5 seconds; 6-step patches/3-step stride; embedding
96; 2 blocks/stream; 4 heads; FFN192; dropout0.1; AdamW LR3e-4, decay0.01;
concept weight0.2; spline regularization1e-4; gradient clip1; batch256;
max30 epochs, patience6; three seeds42/52/62; KAN grid5; hidden16/8;
MLP hidden32/16. KAN calculations remain float32. These are pre-specified
compact response-experiment settings, not replacements for the frozen Stage8.

E5 reuses original KAN weights and reconstructs each original per-seed inner
validation vehicle set. Threshold choices are 0/500/1000/2000/infinity km;
ties favour persistence. Thresholds never use the outer evaluation labels or
the E2 horizon chart. This remains post-hoc evidence on previously studied data.

E6 uses actual saved fleet checkpoint weights for KAN, MLP and TSLANet, the same
normalized synthetic timing shapes, single-thread float32 CPU, warmup10,
repetitions100, batches1/64. Sizes, buffers, sampled process RSS and latency
are recorded. Full frozen Nissan KAN/MLP architectures are also measured at
256-second segment inputs under their own common profile. That is segment
latency, not the cost of an entire multi-segment trip bag. These measurements describe the cloud CPU; energy in joules
and isolated activation-memory peaks are not claimed. Three-seed median-concept
perturbations are sensitivity diagnostics, not causal feature importance.

E7 cannot be completed without a new multi-vehicle driver+charging+thermal+
reliable repeated-health dataset. No extra dataset is fabricated or substituted.

## Resume and limitations

Drive run folders preserve every training epoch's optimizer/model/RNG state,
best weights and histories. Completed E1/E2/E5 analyses are reused. E4 reuses
completed model fits and resumes interrupted training. Keep protocol settings
fixed for the same RUN_ID. The final downloadable ZIP omits large original EVE
signals and redundant optimizer checkpoints, while keeping final model weights,
prepared tensors, predictions, histories, protocol, test log, source and audits.
Optimizer checkpoints remain in Drive for resume.

Local checks are syntax and file-integrity checks only. The scientific test
suite is included and runs in notebook block4 on Colab. Results are not available
until remote execution. The completeness report distinguishes completed work
from missing EVE downloads, absent repeated labels and E7 limitations.
''',encoding='utf-8')


def main():
    notebook();instructions();package()
    print(f'Notebook: {NOTEBOOK}',flush=True)
    print(f'Instructions: {README}',flush=True)


if __name__=='__main__':main()
