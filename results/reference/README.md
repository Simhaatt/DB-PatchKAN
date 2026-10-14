# Saved reference tables

These files are copied from completed research runs, not recalculated during repository assembly. No individual vehicle telemetry, row-level forecasts, large tensor caches or model files are shipped in this directory.

`stage15/` contains selected aggregate E1/E2/E4/E5/E6 tables and scope/audit JSONs. The source Stage 15 archive and exact archived member names/hashes appear in `REFERENCE_RESULT_MANIFEST.json`.

Use `E5_comparison_metrics.csv` and `E5_cluster_comparisons.csv` together; note row-weighted versus vehicle-macro estimands, error units, retrospective cohort and the unequal gating comparison. `E2_cluster_comparisons.csv` preserves horizon/manufacturer heterogeneity and multiple-comparison correction. `E4_matched_future_response_metrics.csv` includes the very small thermal target counts; no low-count observation may be used as a reliable model-win claim.

Raw completion JSONs retain the original run's completion flags. They do not override the later scientific review or imply the E3 uncertainty table is populated. See `docs/KNOWN_LIMITATIONS.md` and `docs/STAGE15_RESULTS_REVIEW.md` for the finite-label denominator defect and incomplete archive-label audit.
