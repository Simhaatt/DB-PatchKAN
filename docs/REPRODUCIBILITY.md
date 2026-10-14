# Reproduction routes

Pin a release/commit before a manuscript reproduction. This initial main branch has no Zenodo DOI or frozen publication tag yet. Hashes of shipped files are in `PUBLIC_FILES_SHA256.json`; source-copy provenance is in `SOURCE_COPY_MANIFEST.json`.

## Route A: inspect saved results and reproduce graphics

No raw dataset or trained model is required for the compact aggregate tables in `results/reference/`. They must be interpreted with the original statistical unit, target definition and scope. This route does not rerun those experiments.

The user-supplied graphics are in `figures/`: draw.io/vector/preview files for Figures 1–9 and supplied high-visibility Figure 8/10 PDFs. Figure 10 is completed by the project's saved CSV curves, high-visibility PNG/SVG and an adapter for the supplied NPZ plotting script. See `docs/FIGURES.md`; no fitted curves are fabricated.

```bash
python -m pip install -e ".[repro]"
python scripts/verify_repository.py
python scripts/export_fig10_saved_npz.py --output outputs/fig10_saved_curves.npz
python figures/scripts/fig10_kan_responses.py outputs/fig10_saved_curves.npz outputs/Fig10
python figures/generate_figures_8_10_visible.py
```

The CSV-to-NPZ operation copies the existing fitted seed grids, mean and saved band without smoothing/re-estimating the band. The alternative renderer generates Figures 8/10 at higher font sizes. Supplied figure files remain unchanged.

## Route B: public EVBattery data → trained fleet forecasts

Use `notebooks/00_public_fleet_reproduction.ipynb` in Colab, or equivalent commands below. No proprietary author-produced upload ZIP is needed for the core public-fleet workflow. The large official source archives are still required. Use A100 for the full multi-fold/multi-seed run and save outputs to Drive.

```bash
python -m dbpatchkan.analysis.stage12_evbattery prepare \
  --archives data/external/evbattery/battery_dataset1.tar.gz data/external/evbattery/battery_dataset2.tar.gz data/external/evbattery/battery_dataset3.tar.gz \
  --output outputs/reproduction/evbattery_stage12_sample.npz \
  --max-members-per-archive 200000 --max-records-per-vehicle 64 --seed 42
python -m dbpatchkan.analysis.stage13_persistence_forecast build \
  --source-cache outputs/reproduction/evbattery_stage12_sample.npz \
  --output outputs/reproduction/evbattery_stage13_transitions.npz
python -m dbpatchkan.analysis.stage13_persistence_forecast validate \
  --cache outputs/reproduction/evbattery_stage13_transitions.npz \
  --output-dir outputs/reproduction/stage13 --folds 5 --seeds 42 52 62 --epochs 60 --batch-size 512
python -m dbpatchkan.analysis.stage13b_recent_methods \
  --cache outputs/reproduction/evbattery_stage13_transitions.npz \
  --stage13-results outputs/reproduction/stage13 \
  --output-dir outputs/reproduction/stage13b --folds 5 --seeds 42 52 62 --epochs 60 --batch-size 512
python -m dbpatchkan.analysis.stage13c_timepro \
  --cache outputs/reproduction/evbattery_stage13_transitions.npz \
  --frozen-stage13b outputs/reproduction/stage13b \
  --output-dir outputs/reproduction/stage13c --folds 5 --seeds 42 52 62 --epochs 60 --batch-size 512
```

Capture the actual commit, Python/PyTorch/CUDA/GPU/library versions and sampling counts. Requirements here specify compatible lower bounds, not reconstructed exact historical versions. Seeds alone do not guarantee bitwise equivalence across software/hardware. Compare cohorts/split lineage before numeric scores. The recent architectures are project adaptations, not claims to reproduce the authors' official benchmark scores. TimePro is explicitly post-hoc.

Stage 15 selective KAN inference additionally needs original Stage 13 fold/seed checkpoints and frozen fleet predictions. The historical Stage 15 builder maps these required inputs. That combined bundle is not yet publicly deposited. Do not derive a switching threshold from outer test/horizon results; recreate its original inner-validation groups.

## Route C: Nissan, RWTH and EVE development/external studies

These routes require the dataset-specific raw inputs, preprocessing tables and frozen input bundles described in `docs/EXPERIMENTS.md`. The historical numbered notebooks expect upload ZIPs created by their paired builders. Those ZIPs are not inside Git. Obtain permitted raw data and reproduce precursor steps, or wait for an approved linked artifact deposit. Do not treat unavailable inputs as included merely because a notebook is present.

For Nissan, set `configs/default.yaml:data.source_csv`, then:

```bash
python -m dbpatchkan.cli preprocess --config configs/default.yaml
python -m dbpatchkan.cli analyze --config configs/default.yaml
python scripts/export_gpu_tensors.py --output-dir outputs/dbpatchkan --seed 42
```

The exporter writes train/validation/test tensors; keep the test excluded from SSL/tuning/scaling and only open it for the locked inference protocol. Historical locked saved-prediction audits are not permission to tune new models on the test.

## Integrity versus scientific verification

`verify_repository.py` uses standard-library parsing and file checks only. CI runs selected synthetic tests remotely. Neither proves regeneration of all paper results. Full scientific reproduction requires correct data versions/cohort, original cached split input ordering, recorded preprocessing, all fold/seed fits and paired statistical comparisons.

Zenodo software and artifact DOIs will be added only after publication. Do not cite a placeholder DOI or assume the repository release automatically includes raw data/checkpoint assets.
