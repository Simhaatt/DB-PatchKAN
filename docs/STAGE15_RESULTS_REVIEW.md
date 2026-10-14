# Stage 15 returned-results review

Reviewed: 14 October 2026.

## Outcome

The returned Colab run finished E1–E6 computationally, including 72 new neural response-model fits and eight passing remote tests. Completion does not mean that every planned scientific question has sufficient evidence. In particular, E3 withheld Nissan uncertainty calculations, the optional original-label audit found no archives, and E4 has insufficient thermal observations.

The strongest new operational result is **Selective Persistence–KAN**: its vehicle-macro MAE is 0.436948 Ah versus 0.445858 Ah for persistence, a 1.998% reduction across 191 vehicles. Its paired vehicle-bootstrap difference is −0.008909 Ah, with a 95% interval [−0.010863, −0.007027] Ah. This is post-hoc evidence on previously studied cohorts, not a new independent confirmation dataset.

Stage 15 does not show universal KAN superiority, calibrated pack-level degradation physics, or a causal percentage of degradation attributable to driving.

## Storage and inspection scope

The original ZIP was moved from Downloads to:

`outputs/colab_completed/DBPatchKAN_stage15_ENERGY_AI_REVIEW_ALL_MODELS_RESULTS.zip`

It is 268,050,354 bytes and contains 1,051 members. The original archive contents were not modified. The archive remains the authoritative record of model weights, prepared tensors, prediction tables, logs and executed source.

Small result tables, audit JSONs and logs were extracted to:

`outputs/colab_completed/stage15_energy_ai_review/`

This review inspected saved output tables and source logic. It did not retrain models, rerun bootstrap experiments, load model checkpoints, or run scientific tests locally. Local activity was limited to file operations and lightweight inspection of existing results. It did not perform a full archive-wide CRC scan or independently reproduce every reported metric from the large per-observation prediction files.

## Run protocol

- Run ID: `stage15_energy_ai_review_v1`.
- GPU: NVIDIA A100-SXM4-40GB.
- Seeds: 42, 52, 62.
- Maximum response training epochs: 30; early stopping applied.
- Bootstrap/randomization draws: 10,000.
- RWTH preparation cap: 1,500 examples per vehicle.
- EVE download enabled and preparation completed.
- Original-label archive audit enabled, but no matching archives were found in remote Drive.
- Locked Nissan test tensors were not opened. Saved predictions and metadata were audited only.
- Remote tests: eight passed.
- Response fits: 9 RWTH vehicle folds and 3 EVE temporal folds, each with KAN/MLP and three seeds; 72 complete fits.
- Completion timestamp: 2026-10-14 12:50:15 UTC.

## E1: D-BSI robustness and downstream value

### Score robustness

The analysis uses 101 Nissan development trips, recomputes the index from one-second telemetry, and does not tune on untouched-test trips.

| Sensitivity | Spearman correlation with reference |
| --- | ---: |
| Equal weights | 0.984368 |
| 100 bounded weight perturbations | 0.973512–0.999802 |
| Rate/slew/cycle thresholds ×0.8 | 0.993733 |
| Rate/slew/cycle thresholds ×1.2 | 0.997659 |
| Alternate capacity assumption | 0.999767 |
| Temperature threshold +5°C | 0.935853 |
| Temperature threshold −5°C | 0.810868 |
| DoD/spread parameters ×0.8 | 0.919907 |
| DoD/spread parameters ×1.2 | 0.976319 |
| Remove SOC component | 0.170961 |
| Remove thermal component | 0.904380 |
| Remove cycle component | 0.884904 |

Interpretation: trip ranking is stable to the tested modest weight perturbations. It is not uniformly insensitive to component removal or thresholds. The SOC component strongly affects this dataset's ranking. This is not evidence that SOC or driving causally dominates degradation.

The reference recomputed score has mean 3.548004, standard deviation 2.164042 and range 0.031653–7.448947. All reference trips fall in the fixed low-stress band. Therefore, absolute Low/Moderate/High thresholds are not externally calibrated by this run. Relative training-fold classes and fixed engineering bands must be described separately.

### Downstream degradation analysis

The development-only residual models use logging age, mileage and lagged health. Logging age is days since data recording began, not known manufacturing age. Charging history is unavailable.

| Target | Evaluated transitions | Reference D-BSI residual-model MAE |
| --- | ---: | ---: |
| 30-day loss | 12 | 0.616022 SOH percentage points |
| 500-km loss | 16 | 0.645607 SOH percentage points |

For the 30-day target, the reference D-BSI correction adds no error reduction relative to the age/mileage/lagged-health control in the reported checkpoint-macro comparison. For 500 km, it slightly worsens that control: difference +0.001401 pp, interval [+0.000313, +0.002659] pp. Both reference residual analyses are worse than persistence on these development evaluation rows.

These are auxiliary residual analyses, not a retraining/evaluation of the frozen full DB-PatchKAN architecture. They do not invalidate every earlier SOH result, but they do not establish additional degradation-prediction value for this D-BSI correction.

Joint six-component NNLS met the supervision threshold only for the last 500-km fold. It was evaluated on four transitions; the fitted coefficients were zero except SOC. Its MAE of 0.446910 pp must not be compared directly with the 16-transition results as if evaluated on the same cohort. These weights are not calibrated physical degradation shares and should not replace the published transparent index.

## E2: Fleet horizon and subgroup statistics

E2 reuses frozen out-of-fold charging-capacity forecasts: 9,714 transitions, 191 vehicles, anonymous released manufacturer groups. The fleet is charging-only evidence, not an external replication of the complete driver–battery pathway.

For the original always-on persistence-anchored KAN:

| Horizon | Vehicles | Vehicle-macro improvement over persistence | 95% interval for KAN minus persistence MAE (Ah) | Holm-adjusted p |
| --- | ---: | ---: | --- | ---: |
| All | 191 | 1.1829% | [−0.007494, −0.003111] | 0.000800 |
| ≤500 km | 191 | −3.2306% | [+0.007072, +0.012633] | 0.003200 |
| 500–1,000 km | 190 | 0.1369% | [−0.003837, +0.002912] | 1.000000 |
| 1,000–2,000 km | 189 | 1.5733% | [−0.011670, −0.003658] | 0.003200 |
| >2,000 km | 186 | 5.1842% | [−0.050394, −0.025615] | 0.003200 |

The >2,000-km row-weighted improvement is 4.3642%; the vehicle-macro improvement is 5.1842%. Both are valid different estimands, not interchangeable percentages.

The always-on correction is harmful on short horizons and helpful at longer horizons. The intermediate 500–1,000-km result is inconclusive. Manufacturer-by-horizon findings are heterogeneous. Small third-manufacturer subgroups have only 6–8 vehicles and require caution. Intervals describe cluster-resampled errors on the known evaluation cohorts; cross-validation training-set overlap is not eliminated by vehicle bootstrapping.

Saved prediction-bound coverage is evaluated, but averaging seed bounds is not a newly calibrated conformal interval.

## E3: Labels, independent observations and actual losses

### Nissan target eligibility: audit defect found

The saved strict-persistence table contains 26 rows for each target, including rows with missing observations. The Stage 15 code compares mapped valid targets against all 26 rows. That prevents interval estimation even when every eligible target maps correctly.

The actual saved-label eligibility is:

- 30-day loss: **zero finite observed targets** in the 26 locked rows. No locked 30-day forecast accuracy can be reported from them.
- 500-km loss: **eight finite observed targets**, eight matched anchor–future pairs, and **four distinct future checkpoints**.
- The eight eligible 500-km reconstructed losses match their saved targets within the audit tolerance.
- The remaining 18 500-km rows have missing observations, rather than failed lineage mappings.
- `E3_checkpoint_cluster_bootstrap.csv` is empty. It contains no new uncertainty result.

Required correction: distinguish total rows, finite observed labels, eligible paired predictions, mapped targets and future checkpoints. Apply candidate/reference finite-label masks before lineage validation. With zero eligible labels, report “not evaluable,” not a mismatch. Compute the eight-transition 500-km interval only after that correction, flagging four destination clusters as very small. Do not train or infer again on locked tensors.

### Label summary

Nissan contains 127 trips, 56 distinct raw BMS SOH values, 71 adjacent raw plateaus, no observed adjacent raw increases, and a minimum positive observed resolution of approximately 0.01 pp. Observed reporting resolution is not a measurement-accuracy certificate. One vehicle remains one independent vehicle regardless of the number of windows.

EVE saved health validation contains 18 held-out interval/checkpoint IDs from 36 source intervals. The reviewed forecast rows have finite anchors and future values; 18 future checkpoint IDs are present. This remains single-vehicle temporal evidence.

### Training objective audit

The exported driver diagnostic target exists, with 5,270 valid entries and codes 0, 1, 2, 4. There are 982 nonzero logged driver-ID loss entries. Therefore, excluding Driver_ID as a predictor did **not** remove its auxiliary training supervision.

There are zero nonzero logged physics-loss entries. No improvement may be attributed to an active soft physics penalty in those audited runs. Physics-derived concepts, output constraints and a physics comparator are separate mechanisms and should be described accurately.

### Measurement uncertainty

The original EVBattery repeated-label archive sample did not run: matching archives were absent from the configured remote Drive search root. The saved perturbation sweep is hypothetical sensitivity to assumed noise, not empirical manufacturer measurement-error calibration. This remains open.

## E4: Genuinely future driver–battery responses

The new tasks predict 30/60/120-second responses from a past-only 120-second history sampled at 5 seconds. Preparation uses backward joins and backward derivatives, with a 240-second stride. Comparisons use the same outer examples and inner training/validation splits.

Targets are mean current (A), 95th-percentile absolute current (A), mean voltage (V), pack-temperature rise (°C), net energy from measured I×V (kWh), and high-current exposure (s). Absolute-current p95 is not a literal maximum or peak C-rate. Derived energy/exposure are not independent health measurements.

RWTH electrical evaluation generally contains 9,672 examples across nine held-out vehicles; 120-second energy/exposure contain 9,582 eligible examples. EVE electrical evaluation contains approximately 2,866–2,868 examples from three chronological folds on one vehicle.

### Predictive contribution of driving inputs

Matched context+driver HGB generally reduces electrical-response error compared with context-only HGB. Examples of vehicle-macro MAE on RWTH:

| Target and horizon | Context-only HGB | Context+driver HGB | Unit |
| --- | ---: | ---: | --- |
| Mean current, 30 s | 9.796941 | 9.410509 | A |
| Mean current, 120 s | 7.011381 | 6.617882 | A |
| Net energy, 120 s | 0.079709 | 0.074858 | kWh |
| Mean voltage, 120 s | 2.184287 | 2.122277 | V |

Corresponding exploratory vehicle-cluster intervals exclude zero for these examples. RWTH has only nine vehicle clusters; EVE uses seven-day temporal blocks, not independent vehicles. E4 intervals are not multiplicity-adjusted. Predictive input-group differences are not causal effects or a decomposition of long-term degradation.

### Architecture comparisons are mixed

| Task | KAN | Matched MLP | Other best reported family | Unit |
| --- | ---: | ---: | --- | --- |
| RWTH absolute-current p95, 30 s | 15.069237 | 15.094036 | KAN has lowest point MAE | A |
| RWTH mean current, 30 s | 9.406268 | 9.434422 | KAN has lowest point MAE; driver HGB 9.410509 | A |
| RWTH mean voltage, 120 s | 2.547574 | 2.764240 | Driver HGB 2.122277 | V |
| EVE mean current, 120 s | 10.875028 | 10.651065 | Matched MLP | A |
| EVE net energy, 120 s | 0.133411 | 0.128750 | Matched MLP | kWh |

These are point estimates, not claims of significant pairwise wins. All point-error rankings and intervals should remain visible in the paper. KAN is not universally the lowest-error response architecture. The longitudinal physical baseline is best for RWTH high-current exposure at 30/60 seconds, and context+driver HGB is often best for voltage and energy.

### Thermal evaluation is not adequately supported

RWTH has no measured pack-temperature targets in this experiment. EVE has no eligible 30-second thermal test row, just **one** eligible 60-second row and **two** eligible 120-second rows. The 60-second KAN point error of 0.325843°C must not be presented as established temperature-prediction superiority. Bootstrap intervals with one or two observations/clusters are not reliable validation evidence. Thermal results should be explicitly marked insufficient, and excluded from aggregated model-win counts.

The model's shared hard output domains are applied to all families before scoring. No active soft physics penalty is used in E4. Effective physical coefficients are fitted only on inner training data; they are not independently measured pack constants or a validated NMC aging law.

## E5: Inner-selected persistence switching

The switch reuses original fleet KAN weights. Each fold/seed chooses among 0/500/1,000/2,000/infinite-km thresholds using its original inner-validation vehicles. Outer labels are not used for threshold choice. All 15 choices select 500 or 1,000 km: seven choose 500 km and eight choose 1,000 km.

| System | Row-weighted MAE (Ah) | Vehicle-macro MAE (Ah) | RMSE (Ah) |
| --- | ---: | ---: | ---: |
| Persistence | 0.443366 | 0.445858 | 0.807626 |
| Always-on anchored KAN | 0.438141 | 0.440584 | 0.791039 |
| Always-on matched MLP | 0.437446 | 0.439536 | 0.791369 |
| Always-on TSLANet | 0.437087 | 0.439296 | 0.787048 |
| Selective Persistence–KAN | **0.434631** | **0.436948** | 0.793079 |

Selective KAN has the lowest reported MAE in this comparison table, but **TSLANet has the lowest RMSE**. The switch should not be called best on every metric.

Paired vehicle-macro MAE differences for selective KAN:

| Reference | Difference (Ah) | 95% vehicle-bootstrap interval (Ah) | Improvement |
| --- | ---: | --- | ---: |
| Persistence | −0.008909 | [−0.010863, −0.007027] | 1.9982% |
| Always-on KAN | −0.003635 | [−0.004663, −0.002656] | 0.8251% |
| Always-on MLP | −0.002587 | [−0.003703, −0.001450] | 0.5886% |
| Always-on TSLANet | −0.002348 | [−0.003568, −0.001134] | 0.5345% |

The vehicle win fraction against persistence is 72.7749%. Row-weighted improvement over persistence is 1.9702%, versus 1.9982% vehicle-macro. The saved switch table has no newly calibrated interval coverage value. Existing always-on coverage cannot be inherited automatically by the switched system.

**Fairness boundary:** only KAN receives the selective gate in this run. Its advantage over always-on MLP/TSLANet is evidence about the combined operational system, not evidence that the KAN head is intrinsically better. A fair architecture comparison requires applying the same inner-selection gate protocol to MLP and TSLANet, with the same thresholds, folds and labels. The present result remains useful if stated within this boundary.

## E6: Efficiency and explanation diagnostics

Benchmarks used actual saved weights on Colab's Intel Xeon 2.20 GHz CPU, one thread and float32. Timing inputs were common zero-normalized synthetic shapes within each task, not live vehicle telemetry.

| Model | Parameters | Batch-1 median latency | Batch-64 median latency |
| --- | ---: | ---: | ---: |
| Fleet KAN | 183,206 | 3.111 ms | 12.100 ms |
| Fleet matched MLP | 180,294 | 1.880 ms | 11.739 ms |
| Fleet TSLANet | 44,067 | 2.495 ms | 23.547 ms |
| Full Nissan DB-PatchKAN | 1,125,412 | 10.678 ms | 115.517 ms |
| Full Nissan matched MLP | 1,122,948 | 7.354 ms | 100.828 ms |

Nissan timings cover one 256-second segment with 24 physical channels, not a complete multi-segment trip. KAN is slower than its matched MLP for these benchmark conditions. Small checkpoint/weight sizes support feasibility, but embedded deployment has not been measured on a vehicle device.

The reported RSS is sampled total process memory, not isolated activation memory. Joule-level energy use was not measured.

The three-seed, 64-example fold-1 concept-replacement diagnostics show notable sensitivity to current capacity, mileage and forecast horizon. Smaller electrical/thermal effects and their signed directions vary across seeds. The manufacturer one-hot replacements show zero effect in this sample, which does not establish fleet-wide irrelevance. Concept replacements may create physically inconsistent inputs and are not causal attribution.

## What the manuscript can say now

1. The engineering D-BSI ranking is stable under the tested bounded weight perturbations, with sensitivity to specific threshold changes and component removal.
2. Driving-history inputs add predictive value for several future electrical responses under matched vehicle/temporal splits.
3. Persistence is a strong baseline, particularly at short horizons; always-on corrections can be harmful.
4. A per-fold inner-selected selective persistence–KAN system improves fleet MAE over persistence in the returned retrospective evaluation.
5. Named concepts support auditable model diagnostics, with measured computational costs and explicit limits.

## Claims not supported by these results

- KAN wins all targets, datasets, horizons or error metrics.
- An improvement of approximately 2% MAE proves clinically/operationally meaningful health precision without application tolerance and measurement uncertainty.
- This charging-only fleet verifies the full driver–battery architecture.
- The Nissan locked set validates 30-day forecasting.
- One or two thermal observations validate a temperature model.
- A nonzero physics-loss improvement was demonstrated in the audited health runs.
- Cell-level semi-empirical NMC laws were experimentally calibrated and validated at pack level.
- A causal driver/calendar/charging/temperature percentage decomposition has been identified.
- Post-hoc experiments on previously studied cohorts are fresh confirmatory external validation.

## Recommended next actions, without repeating completed training

1. Correct the E3 finite-label denominator and completion status in a remote audit-only cell. Use saved predictions and lineage; do not reopen the Nissan test tensor or tune on it. Report the absent 30-day targets and the four destination clusters explicitly.
2. Find the original EVBattery archives in Drive and run only the bounded repeated-label descriptive audit. It will not by itself establish true measurement accuracy.
3. If architecture superiority under switching is a desired claim, run the identical inner-selected switching policy for the saved MLP and TSLANet checkpoints. No new training is required, but inference and clustered comparisons should run remotely.
4. Do not retrain the response suite merely to chase wins. First address the thermal cadence/eligibility problem. Any longer thermal horizon is a new, separately specified experiment with causal inputs and suitable temporal purging, not a relabeling of this completed run.
5. Update manuscript tables and claims to reflect mixed response rankings, small absolute gains, charging-only fleet scope, inactive audited physics loss and the actual target counts.
6. E7 remains conditional on a genuinely new multi-vehicle longitudinal driver/charging/thermal dataset with reliable repeated health measurements.

The completed E1/E2/electrical-E4/E5/E6 training and analyses do not need to be repeated for the audit corrections. No corrective experiments were performed by this review.
