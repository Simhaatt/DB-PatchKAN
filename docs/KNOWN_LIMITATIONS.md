# Known limitations and release boundaries

This initial public snapshot preserves historical numerical results rather than modifying them to favour the proposed model.

## Required before a final manuscript release

- Stage 15 E3 currently includes missing-target rows in its Nissan lineage denominator. The returned bootstrap table is empty. A saved-prediction-only audit correction remains to be run remotely; no locked-test retraining is permitted.
- The locked Nissan table has no finite 30-day labels and eight finite 500-km transitions sharing four destination checkpoints. Missing labels must not be counted as evaluations or mapping failures.
- Stage 15 EVE future thermal evaluation has one eligible 60-second observation and two 120-second observations, with no eligible 30-second thermal test. Do not count these as established thermal-model wins.
- The bounded original EVBattery repeated-label audit did not find its archives. Hypothetical noise sensitivity is not measured label uncertainty.
- Only KAN received the selective-persistence policy in Stage 15 E5. Equally gated MLP/TSLANet comparisons are needed for a head-superiority claim. TSLANet's better always-on RMSE remains visible.
- A new multi-vehicle longitudinal driving/charging/thermal/health cohort is needed for an independently replicated driver-induced long-term degradation claim.
- Historical runtime dependencies are range-constrained, not a verified bitwise-reproduction lock. Pin the published commit and record a complete runtime environment for each reproduction.
- Large input bundles, checkpoint archives and comprehensive split/result lineage are not yet in a published Zenodo artifact record. A code-only clone cannot execute every historical notebook immediately.

## Interpretation

D-BSI threshold sensitivity does not calibrate universal pack aging constants. Pack-SE assumptions are sensitivity descriptors rather than a measured/validated cell-to-pack NMC transfer. Driver input-group comparisons are predictive, not a causal calendar/charging/temperature/driver decomposition. Audited historical physics-loss entries were zero; Driver_ID auxiliary supervision was active even though Driver_ID was excluded as a normal predictor. Report true target definitions, label counts and validation units.

## Scope of verification

Publication assembly uses standard-library syntax/integrity checks only, not local research reruns. CI uses synthetic examples and does not access the raw data. A passing CI job establishes that the selected code path works under that job's environment, not that every reported paper result has been reproduced.
