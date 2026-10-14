# Stage and experiment mapping

The source snapshot retains both historical attempts and final comparison modules so negative findings/ablation history are not discarded. Stage numbers are workflow history, not separate independent validation datasets.

| Stage | Main entry point | Inputs and purpose |
| --- | --- | --- |
| Preprocessing | `dbpatchkan.cli`, `scripts/export_gpu_tensors.py` | Permitted Nissan CSV; QC/features/D-BSI; chronological trip-level split |
| 1 / 1B / 2 | `training/train.py`, `training/tune.py`; notebooks 01–02 | Development train/validation tensor bundles; masked pretraining, tuning, KAN/MLP selection |
| 3 | `analysis/deep_explainability.py`; notebook 03 | Fitted development checkpoints and curves; concept response inspection |
| 4N / 4NB | `training/physics_hybrid.py`, `training/physics_corrected.py`; notebooks 04–05 | Nissan hybrid response/health experiments and corrections |
| 5 | `scripts/build_rwth_stage4.py`, `analysis/rwth_external.py`, `analysis/eve_external.py`; notebook 06 | Native RWTH/EVE data; response and sparse health evaluation |
| 6 | `analysis/stage6.py`; notebook 07 | Unified temporal comparators/ablations; subsequent saved-prediction audits |
| 7 | `analysis/stage7_external_sequence.py`; notebook 08 | Full external sequence comparisons; distinguish driver response from health |
| 8 | `analysis/stage8_final.py`; notebook 09 | Locked Nissan inference after development selection; no test tuning |
| 9 | `analysis/stage9_cpu.py` | Predictive attribution and semi-empirical pack-assumption sensitivity; no causal shares |
| 10 | `analysis/stage10_normalized_transfer.py`; notebook 10 | Dataset-specific target normalization/transfer controls |
| 11 | `analysis/stage11_packse_mil.py`; notebook 11 | Pack-SE residual multiple-instance sensitivity study |
| 12 | `analysis/stage12_evbattery.py`; notebook 12 | Secure raw charging archive ingestion; same-record capacity regression |
| 13 | `analysis/stage13_persistence_forecast.py`; notebook 13 | Next-distinct-mileage capacity; persistence, Ridge, RF, matched MLP/KAN; 5 folds × 3 seeds |
| 13B | `analysis/stage13b_recent_methods.py`; notebook 14 | Adapted ModernTCN, TSLANet, KAN-Linformer on frozen same transitions |
| 13C | `analysis/stage13c_timepro.py`; notebook 15 | Pure-PyTorch TimePro adaptation; explicitly post-hoc |
| 14 | `analysis/stage14_sensitivity.py`; notebook 16 | Stress/architecture hyperparameter robustness; not locked-test tuning |
| 15 | `analysis/stage15_review.py`, `analysis/stage15_response.py`; notebook 17 | E1 score robustness; E2 clustered subgroups; E3 label/loss audit; E4 fixed-horizon responses; E5 switching; E6 remote CPU efficiency |

Notebook 00 is the new public-data bootstrap, not a historical experiment. Numbered notebooks are sanitized historical source notebooks with outputs/identifiers removed. Pair `build_colab_stage*` scripts with their corresponding notebooks; inspect required files before attempting to generate an upload package.

## Architecture/configuration separation

- `configs/default.yaml` describes the primary Nissan design (256-s segments, 16-s patches/8-s stride, 128 dimensions, three blocks/stream, four heads). It is not the final configuration for every experiment.
- Actual stage command lines, saved `run_config.json` and checkpoint metadata determine a run. Historical sourcebooks provide an audit trail but must be read with the latest Stage 15 review.
- EVBattery uses charging sequences (128×8) and 20 named physical/context features, not the complete Nissan driver stream. Future capacity is the current-capacity anchor plus a signed residual.
- Stage 15 future-response models use past-only 120-s histories, 5-s cadence, 30/60/120-s targets, 96-dimensional two-block streams and 15 concepts. They are not independent future SOH validation.
- KAN spline operations remain FP32 under mixed precision. MLP heads are matched experimental controls, not another dataset.

## Recent-method provenance

The project's ModernTCN, TSLANet and KAN-Linformer modules adapt their architecture ideas to a shared signed-capacity-residual task. TimePro deliberately uses pure PyTorch instead of official custom CUDA/DCNv4 kernels. Cite the original methods in a manuscript and describe the adaptation; do not label these as official released implementations or exact original benchmark reproductions. No external method repository is vendored in this snapshot.
