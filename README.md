# DB-PatchKAN

Reproducibility companion to **Horizon-Adaptive Battery Capacity Forecasting and Driver-Aware Stress Modelling from Electric-Vehicle Field Data**.

**Authors:** Aswathy B. Raj, T. T. Simhaa, Shihabudheen K. V.

This repository contains research code, clean Colab notebooks, configurations, synthetic tests, selected saved aggregate results, and manuscript figures. It is an initial public source snapshot, not yet a frozen Zenodo release. No DOI has been assigned to this package.

## Start here

1. Read [reproduction instructions](docs/REPRODUCIBILITY.md) and [dataset access](docs/DATASETS.md).
2. For the public charging-capacity workflow, open [the bootstrap notebook](notebooks/00_public_fleet_reproduction.ipynb) in Colab. Use an A100 for training. Download the three EVBattery archives from their original publisher and put them in Drive.
3. For existing numerical evidence, see [selected aggregate tables](results/reference/README.md). These are copies of saved results, not new executions.
4. For manuscript graphics, see [the figure inventory](docs/FIGURES.md). Supplied editable/vector figures are preserved, including both Figure 8 versions and the supplied Figure 10 PDF.
5. Before interpreting any result, read [known limitations](docs/KNOWN_LIMITATIONS.md).

## What is implemented

- Nissan telemetry QC, native/one-second preprocessing, trip/window/health features and leakage-safe tensor exports.
- Exploratory driver archetypes and behaviour classifiers; train-fold feature selection and classical comparators.
- Six-component Driver–Battery Stress Index (D-BSI): rate, thermal, SOC, cycle, transient and interaction.
- Driver/battery patch encoders, bidirectional attention, hierarchical pooling and a named-concept spline-KAN head; matched MLP controls.
- Persistence-anchored Nissan SOH-loss and EVBattery signed next-capacity forecasts, with task-specific label definitions.
- Fleet grouped validation, adapted recent-method controls, physics-assumption sensitivity and remote revision analyses.

The Nissan configuration and the charging-only fleet configuration are **different models**. The fleet does not have a driver-control stream. Refer to [experiment mapping](docs/EXPERIMENTS.md) for the stages and entry points.

## Installation

Use Python 3.12 as the publication-development/CI target. Historical environments are not fully locked; see the reproducibility notes.

```bash
git clone https://github.com/Simhaatt/DB-PatchKAN.git
cd DB-PatchKAN
python -m pip install -e ".[dev,tuning,repro]"
python scripts/verify_repository.py
python -m pytest tests/test_models.py tests/test_evaluation.py tests/test_stage13_persistence_forecast.py
```

Install the appropriate PyTorch CUDA build for your environment before GPU training. On a CPU-only installation, a CPU PyTorch wheel is sufficient for the synthetic tests. Real-data training is not performed by the repository-integrity checker.

## Dataset roles

| Source | Use in this project | Boundary |
| --- | --- | --- |
| Nissan Leaf, one vehicle | Driver/battery development; 127 trips, 56 distinct BMS SOH values | Not fleet health generalization; locked 30-day targets absent |
| RWTH, nine vehicles | Leave-one-vehicle-out driving-response evaluation; sparse controlled capacity labels | Field pack temperature is absent; thermal proxies are not measurements |
| EVE Pilot, one vehicle | Temporal response/health studies from irregular streams | Not independent-vehicle validation; Stage 15 thermal labels are very sparse |
| EVBattery/Tsinghua, 191 labelled vehicles used | 9,714 next-distinct-mileage capacity transitions | Charging-only data; manufacturer partial-window capacity in Ah, not absolute SOH |

Raw third-party datasets, tensor caches, model checkpoints and large result archives are **not in Git**. Source links and known archive hashes are provided. A separately linked Zenodo artifact deposit is planned but has not been published.

## Reported result, with comparison boundary

The saved Stage 15 table reports vehicle-macro MAE of **0.436948 Ah** for selective persistence–KAN versus **0.445858 Ah** for persistence. The paired vehicle-bootstrap difference is −0.008909 Ah, 95% interval [−0.010863, −0.007027]. These are post-hoc results on already studied cohorts. Only KAN received this selective policy in that experiment; the result does not establish KAN-head superiority over equally gated MLP/TSLANet. TSLANet has lower RMSE in the corresponding table. All comparison rows, including negative findings, are retained.

## Scientific boundaries

D-BSI is an engineered operational stress descriptor, not a calibrated degradation law. Driving inputs add predictive information for several immediate electrical responses, but causal long-term degradation shares have not been identified. BMS SOH, controlled fixed-window capacity SOH and released partial-window capacity are not interchangeable targets. The audited historical health physics-loss term was zero; improvements must not be attributed to that inactive term. See the [Stage 15 review](docs/STAGE15_RESULTS_REVIEW.md).

## Citation and licensing

Citation metadata are in [CITATION.cff](CITATION.cff) and [.zenodo.json](.zenodo.json).

Original software is available under [MIT](LICENSE). This does not relicense third-party datasets. Paper figures are included as author-supplied research materials; no separate CC-BY license is asserted for them. Preserve attribution and consult the authors if a separate figure-reuse license is required.

## Verification status

The package was prepared with file operations, source/notebook syntax checks and checksums. No local scientific analyses or model training were rerun during publication preparation. GitHub Actions performs a selected synthetic smoke suite on a remote runner; that is not a reproduction of the paper's numerical results.
