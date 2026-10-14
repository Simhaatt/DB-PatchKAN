"""Generate the conditional A100 Stage-1B + Stage-2 Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "02_colab_a100_stage1b_stage2_selection.ipynb"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # DB-PatchKAN — Combined Stage 1B and Stage 2 model selection

        This A100 notebook corrects the absolute-SOH extrapolation problem found in Stage 1 and
        then performs confirmation training without opening the held-out GPU test tensor.

        **Stage 1B** fits a training-only aging baseline, creates a residual-SOH target, compares
        it with the original absolute-SOH formulation, and selects the better formulation.

        **Stage 2** retrains the selected formulation for five seeds using matched KAN and MLP
        heads. It chooses the final deep head from aggregate validation MAE, with a simplicity
        tie-break in favour of MLP. Persistence remains an explicit deployment challenger.

        Runtime is not padded. On an A100 this may finish well under two hours. Every long training
        stage prints epoch records and checkpoints are stored in Drive for automatic resumption.
        """
    ),
    markdown(
        """
        ## Direct notebook upload

        When Block 2 opens the upload dialog, select
        `DBPatchKAN_Stage1B2_Drive_Input.zip`. The bundle contains the code, train/validation
        tensors, manifest, and completed Stage-1 results. The test tensor is deliberately absent
        and is rejected if supplied. Drive mode remains available as an optional setting.
        """
    ),
    code(
        """
        # BLOCK 1 — Configuration
        from datetime import datetime, timezone

        INPUT_MODE = "upload"  # Direct browser upload; change to "drive" only if wanted.
        DRIVE_INPUT_ZIP = (
            "/content/drive/MyDrive/DBPatchKAN/stage1b2_input/"
            "DBPatchKAN_Stage1B2_Drive_Input.zip"
        )
        DRIVE_RUNS_DIR = "/content/drive/MyDrive/DBPatchKAN/runs"
        RUN_ID = "stage1b2_full_selection"

        STAGE1B_PRETRAIN_EPOCHS = 0       # Reuse the completed Stage-1 encoder.
        STAGE1B_FINETUNE_EPOCHS = 40
        STAGE1B_PATIENCE = 7

        STAGE2_SEEDS = [42, 52, 62, 72, 82]
        STAGE2_PRETRAIN_EPOCHS = 60
        STAGE2_FINETUNE_EPOCHS = 80
        STAGE2_PATIENCE = 15

        REFERENCE_SOH = 95.0
        SIMPLICITY_TIE_PP = 0.02
        RUN_STAGE1B = True
        RUN_STAGE2 = True

        assert INPUT_MODE in {"drive", "upload"}
        assert len(STAGE2_SEEDS) == 5 and len(set(STAGE2_SEEDS)) == 5
        print(f"[{datetime.now(timezone.utc).isoformat()}] BLOCK 1 complete")
        print({
            "run_id": RUN_ID,
            "stage2_seeds": STAGE2_SEEDS,
            "stage2_pretrain_epochs": STAGE2_PRETRAIN_EPOCHS,
            "stage2_finetune_epochs": STAGE2_FINETUNE_EPOCHS,
            "test_enabled": False,
        })
        """
    ),
    code(
        """
        # BLOCK 2 — Mount Drive/upload, extract, and verify the input bundle
        import json
        import re
        import shutil
        import zipfile
        from pathlib import Path
        from datetime import datetime, timezone

        def update(message):
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{stamp} UTC] {message}", flush=True)

        INPUT_ROOT = Path("/content/dbpatchkan_stage1b2_input")
        if INPUT_ROOT.exists():
            shutil.rmtree(INPUT_ROOT)
        INPUT_ROOT.mkdir(parents=True)

        update(f"BLOCK 2 starting in {INPUT_MODE!r} mode")
        if INPUT_MODE == "drive":
            from google.colab import drive
            drive.mount("/content/drive")
            source_zip = Path(DRIVE_INPUT_ZIP)
            assert source_zip.is_file(), f"Drive bundle not found: {source_zip}"
            RUN_ROOT = Path(DRIVE_RUNS_DIR) / RUN_ID
        else:
            from google.colab import files
            update("Upload DBPatchKAN_Stage1B2_Drive_Input.zip")
            uploaded = files.upload()
            assert uploaded, "No bundle was uploaded"
            uploaded_name, payload = next(iter(uploaded.items()))
            clean_name = re.sub(r" \\(\\d+\\)(?=\\.[^.]+$)", "", Path(uploaded_name).name)
            source_zip = INPUT_ROOT / clean_name
            source_zip.write_bytes(payload)
            RUN_ROOT = Path("/content/dbpatchkan_runs") / RUN_ID

        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(INPUT_ROOT)
        update(f"Extracted input bundle: {source_zip.name}")

        required_names = {
            "dbpatchkan_colab_code.zip",
            "segments_train.npz",
            "segments_validation.npz",
            "gpu_tensor_split_manifest.json",
            "DBPatchKAN_stage1_reduced_seed42_ALL_MODELS_RESULTS.zip",
        }
        located = {}
        for path in INPUT_ROOT.rglob("*"):
            if path.is_file() and path.name in required_names:
                located[path.name] = path
        missing = sorted(required_names - set(located))
        assert not missing, f"Input bundle is incomplete: {missing}"

        prohibited = [
            path.name for path in INPUT_ROOT.rglob("*")
            if path.is_file() and (
                "untouched" in path.name.casefold()
                or re.search(r"(^|[_\\-.])test([_\\-.]|$)", path.name.casefold())
            )
        ]
        assert not prohibited, f"Stage 1B/2 refuses test files: {prohibited}"

        DATA = located["segments_train.npz"]
        VALIDATION = located["segments_validation.npz"]
        MANIFEST = located["gpu_tensor_split_manifest.json"]

        PROJECT_EXTRACT = Path("/content/dbpatchkan_project")
        if PROJECT_EXTRACT.exists():
            shutil.rmtree(PROJECT_EXTRACT)
        PROJECT_EXTRACT.mkdir(parents=True)
        with zipfile.ZipFile(located["dbpatchkan_colab_code.zip"]) as archive:
            archive.extractall(PROJECT_EXTRACT)
        project_files = list(PROJECT_EXTRACT.rglob("pyproject.toml"))
        assert len(project_files) == 1, f"Expected one project root: {project_files}"
        PROJECT_ROOT = project_files[0].parent

        STAGE1_EXTRACT = Path("/content/dbpatchkan_stage1_results")
        if STAGE1_EXTRACT.exists():
            shutil.rmtree(STAGE1_EXTRACT)
        STAGE1_EXTRACT.mkdir(parents=True)
        with zipfile.ZipFile(located["DBPatchKAN_stage1_reduced_seed42_ALL_MODELS_RESULTS.zip"]) as archive:
            archive.extractall(STAGE1_EXTRACT)
        summaries = list(STAGE1_EXTRACT.rglob("STAGE1_SUMMARY.json"))
        assert len(summaries) == 1, f"Expected one Stage-1 summary: {summaries}"
        STAGE1_ROOT = summaries[0].parent

        settings = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": RUN_ID,
            "stage2_seeds": STAGE2_SEEDS,
            "stage2_pretrain_epochs": STAGE2_PRETRAIN_EPOCHS,
            "stage2_finetune_epochs": STAGE2_FINETUNE_EPOCHS,
            "reference_soh": REFERENCE_SOH,
            "untouched_test_used": False,
        }
        (RUN_ROOT / "notebook_settings.json").write_text(json.dumps(settings, indent=2))
        update(f"BLOCK 2 complete; project={PROJECT_ROOT}; Stage1={STAGE1_ROOT}")
        """
    ),
    code(
        """
        # BLOCK 3 — Install and verify the A100 environment
        import os
        import sys
        import time
        import site
        import hashlib
        import platform
        import subprocess
        import importlib

        update("BLOCK 3 installing the local project")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "PyYAML>=6.0", "joblib>=1.4",
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--no-deps", "--no-build-isolation", "-e", str(PROJECT_ROOT),
        ])

        SRC_ROOT = (PROJECT_ROOT / "src").resolve()
        assert (SRC_ROOT / "dbpatchkan" / "__init__.py").is_file()
        if str(SRC_ROOT) not in sys.path:
            sys.path.insert(0, str(SRC_ROOT))
        site.addsitedir(str(SRC_ROOT))
        os.environ["PYTHONPATH"] = os.pathsep.join([
            str(SRC_ROOT), os.environ.get("PYTHONPATH", "")
        ]).rstrip(os.pathsep)
        importlib.invalidate_caches()

        import numpy as np
        import pandas as pd
        import torch
        import sklearn
        import dbpatchkan

        assert torch.cuda.is_available(), "Enable an A100 GPU runtime"
        GPU_NAME = torch.cuda.get_device_name(0)
        assert "A100" in GPU_NAME.upper(), f"Expected A100; current GPU is {GPU_NAME}"
        assert torch.cuda.is_bf16_supported(), "A100 BF16 support was not detected"

        def sha256(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        environment = {
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "dbpatchkan": str(dbpatchkan.__file__),
            "gpu": GPU_NAME,
            "bf16": torch.cuda.is_bf16_supported(),
        }
        (RUN_ROOT / "environment.json").write_text(json.dumps(environment, indent=2))
        print(json.dumps(environment, indent=2), flush=True)
        update("BLOCK 3 complete")
        """
    ),
    code(
        """
        # BLOCK 4 — Fold-safe aging baseline and residual-SOH tensors
        import joblib
        from sklearn.base import clone
        from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        update("BLOCK 4 auditing targets and fitting training-only aging baselines")
        manifest = json.loads(MANIFEST.read_text())
        for role, path in (("train", DATA), ("validation", VALIDATION)):
            expected = manifest["files"][role]
            assert sha256(path) == expected["sha256"], f"SHA-256 mismatch for {role}"
            update(f"{role}: {expected['trips']} trips; {expected['windows']} windows")

        def load_arrays(path):
            with np.load(path, allow_pickle=False) as source:
                return {name: source[name] for name in source.files}

        def decode_utf8_rows(values):
            return [bytes(row[row != 0]).decode("utf-8") for row in values]

        def trip_table(arrays):
            keys = np.asarray(decode_utf8_rows(arrays["trip_key_utf8"]))
            rows = []
            for key in np.unique(keys):
                positions = np.flatnonzero(keys == key)
                first = positions[np.argmin(arrays["window_start_unix_ms"][positions])]
                concepts = arrays["concept_targets"][positions]
                rows.append({
                    "trip_key": key,
                    "start_ms": int(arrays["window_start_unix_ms"][positions].min()),
                    "soh": float(arrays["current_soh_target"][first]),
                    "calendar_age_days": float(np.nanmedian(concepts[:, 9])),
                    "odometer_km": float(np.nanmedian(concepts[:, 10])),
                    "cumulative_stress_dose": float(np.nanmedian(concepts[:, 11])),
                })
            return pd.DataFrame(rows).sort_values("start_ms").reset_index(drop=True)

        train_arrays = load_arrays(DATA)
        validation_arrays = load_arrays(VALIDATION)
        train_trips = trip_table(train_arrays)
        validation_trips = trip_table(validation_arrays)
        assert len(train_trips) == 81 and len(validation_trips) == 20
        assert train_trips["trip_key"].is_unique and validation_trips["trip_key"].is_unique
        assert set(train_trips["trip_key"]).isdisjoint(validation_trips["trip_key"])

        feature_sets = {
            "age_odometer": ["calendar_age_days", "odometer_km"],
            "age_odometer_dose": [
                "calendar_age_days", "odometer_km", "cumulative_stress_dose"
            ],
        }
        estimators = {
            "linear": make_pipeline(StandardScaler(), LinearRegression()),
            "ridge_0.1": make_pipeline(StandardScaler(), Ridge(alpha=0.1)),
            "ridge_1": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "ridge_10": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            "ridge_100": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
            "huber": make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=1.0)),
        }
        fold_bounds = [(0, 40, 50), (0, 50, 60), (0, 60, 70), (0, 70, 81)]
        candidate_rows = []
        for feature_name, columns in feature_sets.items():
            for estimator_name, estimator in estimators.items():
                fold_mae = []
                for fold, (_, train_end, valid_end) in enumerate(fold_bounds):
                    fitted = clone(estimator).fit(
                        train_trips.loc[:train_end - 1, columns],
                        train_trips.loc[:train_end - 1, "soh"],
                    )
                    prediction = fitted.predict(
                        train_trips.loc[train_end:valid_end - 1, columns]
                    )
                    score = mean_absolute_error(
                        train_trips.loc[train_end:valid_end - 1, "soh"], prediction
                    )
                    fold_mae.append(float(score))
                candidate_rows.append({
                    "feature_set": feature_name,
                    "estimator": estimator_name,
                    "inner_temporal_mae_mean_pp": float(np.mean(fold_mae)),
                    "inner_temporal_mae_std_pp": float(np.std(fold_mae, ddof=1)),
                    "fold_mae_pp": json.dumps(fold_mae),
                })
        baseline_cv = pd.DataFrame(candidate_rows).sort_values(
            ["inner_temporal_mae_mean_pp", "inner_temporal_mae_std_pp"]
        )
        baseline_cv.to_csv(RUN_ROOT / "stage1b_baseline_inner_cv.csv", index=False)
        best_row = baseline_cv.iloc[0]
        selected_columns = feature_sets[best_row["feature_set"]]
        selected_estimator = clone(estimators[best_row["estimator"]])
        selected_estimator.fit(train_trips[selected_columns], train_trips["soh"])
        joblib.dump({
            "model": selected_estimator,
            "features": selected_columns,
            "selection": best_row.to_dict(),
        }, RUN_ROOT / "stage1b_aging_baseline.joblib")

        validation_baseline = selected_estimator.predict(validation_trips[selected_columns])
        persistence_value = float(train_trips.iloc[-1]["soh"])
        persistence_prediction = np.full(len(validation_trips), persistence_value)
        baseline_summary = {
            "selected_estimator": str(best_row["estimator"]),
            "selected_feature_set": str(best_row["feature_set"]),
            "features": selected_columns,
            "inner_temporal_mae_mean_pp": float(best_row["inner_temporal_mae_mean_pp"]),
            "validation_aging_mae_pp": float(mean_absolute_error(
                validation_trips["soh"], validation_baseline
            )),
            "validation_aging_rmse_pp": float(mean_squared_error(
                validation_trips["soh"], validation_baseline
            ) ** 0.5),
            "persistence_anchor_soh_pct": persistence_value,
            "validation_persistence_mae_pp": float(mean_absolute_error(
                validation_trips["soh"], persistence_prediction
            )),
            "validation_persistence_rmse_pp": float(mean_squared_error(
                validation_trips["soh"], persistence_prediction
            ) ** 0.5),
            "train_soh_mean_pct": float(train_trips["soh"].mean()),
            "validation_soh_mean_pct": float(validation_trips["soh"].mean()),
            "validation_soh_std_pp": float(validation_trips["soh"].std(ddof=0)),
            "training_only_fit": True,
        }
        (RUN_ROOT / "stage1b_baseline_summary.json").write_text(
            json.dumps(baseline_summary, indent=2)
        )

        residual_root = RUN_ROOT / "stage1b_residual_tensors"
        residual_root.mkdir(parents=True, exist_ok=True)

        def write_residual_npz(arrays, destination):
            concept_frame = pd.DataFrame({
                "calendar_age_days": arrays["concept_targets"][:, 9],
                "odometer_km": arrays["concept_targets"][:, 10],
                "cumulative_stress_dose": arrays["concept_targets"][:, 11],
            })
            baseline = selected_estimator.predict(concept_frame[selected_columns]).astype(np.float32)
            original = arrays["current_soh_target"].astype(np.float32)
            adjusted = (REFERENCE_SOH + original - baseline).astype(np.float32)
            payload = dict(arrays)
            payload["current_soh_original_target"] = original
            payload["current_soh_baseline_pct"] = baseline
            payload["current_soh_target"] = adjusted
            payload["current_soh_reference_pct"] = np.full(
                len(adjusted), REFERENCE_SOH, dtype=np.float32
            )
            np.savez_compressed(destination, **payload)
            return {
                "rows": len(adjusted),
                "adjusted_mean": float(np.mean(adjusted)),
                "adjusted_std": float(np.std(adjusted)),
                "baseline_mean": float(np.mean(baseline)),
            }

        RESIDUAL_DATA = residual_root / "segments_train_aging_residual.npz"
        RESIDUAL_VALIDATION = residual_root / "segments_validation_aging_residual.npz"
        residual_audit = {
            "train": write_residual_npz(train_arrays, RESIDUAL_DATA),
            "validation": write_residual_npz(validation_arrays, RESIDUAL_VALIDATION),
            "reference_soh": REFERENCE_SOH,
            "metric_identity": (
                "Adjusted-target MAE/RMSE/bias equal rebased physical-SOH errors because "
                "the same baseline is added to prediction and target."
            ),
        }
        (RUN_ROOT / "stage1b_residual_tensor_audit.json").write_text(
            json.dumps(residual_audit, indent=2)
        )
        display(baseline_cv.head(12))
        print(json.dumps(baseline_summary, indent=2), flush=True)
        print(json.dumps(residual_audit, indent=2), flush=True)
        update("BLOCK 4 complete")
        """
    ),
    code(
        """
        # BLOCK 5 — Stage 1B: compare original and aging-residual formulations at seed 42
        from dataclasses import replace
        from dbpatchkan.models import DBPatchKAN, DBPatchKANConfig, EVTripBagDataset
        from dbpatchkan.training.train import TrainingConfig, train
        from dbpatchkan.training.tune import ConceptSubsetDataset

        update("BLOCK 5 starting Stage 1B formulation comparison")
        best_config_path = STAGE1_ROOT / "01_reduced_tuning" / "best_config.json"
        best_payload = json.loads(best_config_path.read_text())
        selected_model_config = DBPatchKANConfig(**best_payload["model_config"])
        selected_mask_ratio = float(best_payload["parameters"]["mask_ratio"])
        shared_checkpoint = STAGE1_ROOT / "02_shared_pretraining" / "pretrain_latest.pt"
        assert shared_checkpoint.is_file(), "Completed Stage-1 shared encoder is missing"
        shared_state = torch.load(shared_checkpoint, map_location="cpu", weights_only=False)["model"]

        residual_train_bags = EVTripBagDataset.from_npz(RESIDUAL_DATA)
        residual_validation_bags = EVTripBagDataset.from_npz(RESIDUAL_VALIDATION)
        residual_train_view = ConceptSubsetDataset(
            residual_train_bags, selected_model_config.n_concepts
        )
        residual_validation_view = ConceptSubsetDataset(
            residual_validation_bags, selected_model_config.n_concepts
        )

        def load_shared_encoder(model, state):
            target = model.state_dict()
            compatible = {
                name: value for name, value in state.items()
                if name in target and target[name].shape == value.shape
                and not name.startswith("fusion_head.")
            }
            model.load_state_dict(compatible, strict=False)
            return len(compatible)

        stage1b_rows = []
        if RUN_STAGE1B:
            for head in ("kan", "mlp"):
                output = RUN_ROOT / "stage1b" / f"aging_residual_{head}_seed42"
                finished = output / "result.json"
                best_checkpoint = output / "best.pt"
                if finished.exists() and best_checkpoint.exists():
                    result = json.loads(finished.read_text())
                    update(f"Reusing completed Stage 1B {head.upper()} run")
                else:
                    model = DBPatchKAN(replace(selected_model_config, fusion_head=head))
                    loaded = load_shared_encoder(model, shared_state)
                    update(f"Stage 1B {head.upper()}: loaded {loaded} shared tensors")
                    config = TrainingConfig(
                        stage="finetune", device="cuda", precision="bf16",
                        pretrain_batch_size=256, finetune_batch_size=2,
                        gradient_accumulation=1, pretrain_epochs=STAGE1B_PRETRAIN_EPOCHS,
                        finetune_epochs=STAGE1B_FINETUNE_EPOCHS,
                        pretrain_lr=3e-4, encoder_lr=1e-5, head_lr=3e-4,
                        weight_decay=0.01, warmup_fraction=0.05, gradient_clip=1.0,
                        patience=STAGE1B_PATIENCE, mask_ratio=selected_mask_ratio,
                        num_workers=4, seed=42, domain="nissan",
                        freeze_encoders=True, trip_bags=True,
                        resume_checkpoint=str(output / "latest.pt")
                        if (output / "latest.pt").exists() else None,
                    )
                    update(f"START Stage 1B aging-residual {head.upper()}")
                    result = train(
                        model, residual_train_view, residual_validation_view,
                        config, output_dir=output,
                    )
                    finished.write_text(json.dumps(result, indent=2))
                    update(f"DONE Stage 1B aging-residual {head.upper()}")
                    del model
                    torch.cuda.empty_cache()
                checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
                stage1b_rows.append({
                    "formulation": "aging_residual",
                    "fusion_head": head,
                    "seed": 42,
                    "best_epoch": checkpoint.get("epoch"),
                    **checkpoint.get("metrics", {}),
                })

        original = pd.read_csv(STAGE1_ROOT / "stage1_model_comparison.csv")
        original.insert(0, "formulation", "original_absolute")
        original["seed"] = 42
        stage1b = pd.concat([original, pd.DataFrame(stage1b_rows)], ignore_index=True)
        stage1b.to_csv(RUN_ROOT / "stage1b_formulation_comparison.csv", index=False)

        deep_rows = stage1b[
            stage1b["formulation"].isin(["original_absolute", "aging_residual"])
        ].copy()
        formulation_summary = deep_rows.groupby("formulation", as_index=False).agg(
            mean_mae_pp=("validation_current_soh_mae_pp", "mean"),
            worst_head_mae_pp=("validation_current_soh_mae_pp", "max"),
            mean_rmse_pp=("validation_current_soh_rmse_pp", "mean"),
        ).sort_values(["mean_mae_pp", "worst_head_mae_pp"])
        SELECTED_FORMULATION = str(formulation_summary.iloc[0]["formulation"])
        selection = {
            "selected_formulation": SELECTED_FORMULATION,
            "selection_rule": (
                "Lowest mean validation current-SOH MAE across matched KAN and MLP heads; "
                "worst-head MAE is the tie-break."
            ),
            "aging_baseline_validation_mae_pp": baseline_summary["validation_aging_mae_pp"],
            "persistence_validation_mae_pp": baseline_summary["validation_persistence_mae_pp"],
            "untouched_test_used": False,
        }
        (RUN_ROOT / "stage1b_formulation_selection.json").write_text(
            json.dumps(selection, indent=2)
        )
        display(stage1b[[
            "formulation", "fusion_head", "seed", "best_epoch",
            "validation_current_soh_mae_pp", "validation_current_soh_rmse_pp",
            "validation_current_soh_bias_pp",
        ]])
        display(formulation_summary)
        print(json.dumps(selection, indent=2), flush=True)
        update(f"BLOCK 5 complete; Stage 2 selected {SELECTED_FORMULATION}")
        """
    ),
    code(
        """
        # BLOCK 6 — Stage 2: full five-seed matched KAN/MLP confirmation
        update(f"BLOCK 6 starting Stage 2 with formulation={SELECTED_FORMULATION}")
        if SELECTED_FORMULATION == "aging_residual":
            stage2_data = RESIDUAL_DATA
            stage2_validation = RESIDUAL_VALIDATION
        else:
            stage2_data = DATA
            stage2_validation = VALIDATION

        stage2_train_bags = EVTripBagDataset.from_npz(stage2_data)
        stage2_validation_bags = EVTripBagDataset.from_npz(stage2_validation)
        stage2_train_view = ConceptSubsetDataset(
            stage2_train_bags, selected_model_config.n_concepts
        )
        stage2_validation_view = ConceptSubsetDataset(
            stage2_validation_bags, selected_model_config.n_concepts
        )

        def completed_epoch(checkpoint_path):
            if not checkpoint_path.exists():
                return 0
            return int(torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            ).get("epoch", 0))

        if RUN_STAGE2:
            for seed_index, seed in enumerate(STAGE2_SEEDS, 1):
                seed_root = RUN_ROOT / "stage2" / SELECTED_FORMULATION / f"seed_{seed}"
                pretrain_root = seed_root / "shared_pretraining"
                pretrain_checkpoint = pretrain_root / "pretrain_latest.pt"
                update(
                    f"Stage 2 seed {seed} ({seed_index}/{len(STAGE2_SEEDS)}): "
                    f"pretraining status {completed_epoch(pretrain_checkpoint)}/"
                    f"{STAGE2_PRETRAIN_EPOCHS}"
                )
                if completed_epoch(pretrain_checkpoint) < STAGE2_PRETRAIN_EPOCHS:
                    pretrain_model = DBPatchKAN(replace(
                        selected_model_config, fusion_head="kan"
                    ))
                    pretrain_config = TrainingConfig(
                        stage="pretrain", device="cuda", precision="bf16",
                        pretrain_batch_size=256, finetune_batch_size=2,
                        gradient_accumulation=1,
                        pretrain_epochs=STAGE2_PRETRAIN_EPOCHS,
                        finetune_epochs=0, pretrain_lr=3e-4,
                        encoder_lr=1e-5, head_lr=3e-4,
                        weight_decay=0.01, warmup_fraction=0.05,
                        gradient_clip=1.0, patience=STAGE2_PATIENCE,
                        mask_ratio=selected_mask_ratio, num_workers=4,
                        seed=seed, domain="nissan", freeze_encoders=True,
                        trip_bags=True,
                        resume_checkpoint=str(pretrain_checkpoint)
                        if pretrain_checkpoint.exists() else None,
                    )
                    update(f"START seed {seed} shared pretraining")
                    result = train(
                        pretrain_model, stage2_train_view, stage2_validation_view,
                        pretrain_config, output_dir=pretrain_root,
                    )
                    (pretrain_root / "result.json").write_text(json.dumps(result, indent=2))
                    update(f"DONE seed {seed} shared pretraining")
                    del pretrain_model
                    torch.cuda.empty_cache()
                assert pretrain_checkpoint.exists()
                seed_shared_state = torch.load(
                    pretrain_checkpoint, map_location="cpu", weights_only=False
                )["model"]

                for head in ("kan", "mlp"):
                    head_root = seed_root / f"finetune_{head}"
                    result_path = head_root / "result.json"
                    best_checkpoint = head_root / "best.pt"
                    if result_path.exists() and best_checkpoint.exists():
                        update(f"Seed {seed} {head.upper()} already complete")
                        continue
                    model = DBPatchKAN(replace(selected_model_config, fusion_head=head))
                    if not (head_root / "latest.pt").exists():
                        loaded = load_shared_encoder(model, seed_shared_state)
                        update(f"Seed {seed} {head.upper()}: loaded {loaded} shared tensors")
                    finetune_config = TrainingConfig(
                        stage="finetune", device="cuda", precision="bf16",
                        pretrain_batch_size=256, finetune_batch_size=2,
                        gradient_accumulation=1, pretrain_epochs=0,
                        finetune_epochs=STAGE2_FINETUNE_EPOCHS,
                        pretrain_lr=3e-4, encoder_lr=1e-5, head_lr=3e-4,
                        weight_decay=0.01, warmup_fraction=0.05,
                        gradient_clip=1.0, patience=STAGE2_PATIENCE,
                        mask_ratio=selected_mask_ratio, num_workers=4,
                        seed=seed, domain="nissan", freeze_encoders=True,
                        trip_bags=True,
                        resume_checkpoint=str(head_root / "latest.pt")
                        if (head_root / "latest.pt").exists() else None,
                    )
                    update(f"START seed {seed} {head.upper()} fine-tuning")
                    result = train(
                        model, stage2_train_view, stage2_validation_view,
                        finetune_config, output_dir=head_root,
                    )
                    result_path.write_text(json.dumps(result, indent=2))
                    update(f"DONE seed {seed} {head.upper()} fine-tuning")
                    del model
                    torch.cuda.empty_cache()
        else:
            update("Stage 2 disabled by configuration")
        update("BLOCK 6 complete")
        """
    ),
    code(
        """
        # BLOCK 7 — Aggregate seeds and automatically choose the final deep head
        update("BLOCK 7 aggregating Stage-2 checkpoints")
        confirmation_rows = []
        for seed in STAGE2_SEEDS:
            for head in ("kan", "mlp"):
                checkpoint_path = (
                    RUN_ROOT / "stage2" / SELECTED_FORMULATION / f"seed_{seed}"
                    / f"finetune_{head}" / "best.pt"
                )
                assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                confirmation_rows.append({
                    "formulation": SELECTED_FORMULATION,
                    "fusion_head": head,
                    "seed": seed,
                    "best_epoch": checkpoint.get("epoch"),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": sha256(checkpoint_path),
                    **checkpoint.get("metrics", {}),
                })
        confirmation = pd.DataFrame(confirmation_rows)
        confirmation.to_csv(RUN_ROOT / "stage2_seed_results.csv", index=False)

        def bootstrap_mean_ci(values, seed=20260722, iterations=20000):
            values = np.asarray(values, dtype=float)
            generator = np.random.default_rng(seed)
            draws = generator.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
            return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

        summary_rows = []
        for head, frame in confirmation.groupby("fusion_head"):
            values = frame["validation_current_soh_mae_pp"].to_numpy(float)
            lower, upper = bootstrap_mean_ci(values)
            summary_rows.append({
                "fusion_head": head,
                "seeds": len(values),
                "mae_mean_pp": float(values.mean()),
                "mae_std_pp": float(values.std(ddof=1)),
                "mae_median_pp": float(np.median(values)),
                "mae_ci95_lower_pp": lower,
                "mae_ci95_upper_pp": upper,
                "rmse_mean_pp": float(
                    frame["validation_current_soh_rmse_pp"].mean()
                ),
                "bias_mean_pp": float(
                    frame["validation_current_soh_bias_pp"].mean()
                ),
            })
        head_summary = pd.DataFrame(summary_rows).sort_values("mae_mean_pp")
        head_summary.to_csv(RUN_ROOT / "stage2_head_summary.csv", index=False)

        pivot = confirmation.pivot(
            index="seed", columns="fusion_head", values="validation_current_soh_mae_pp"
        )
        paired_difference = pivot["kan"] - pivot["mlp"]
        paired_lower, paired_upper = bootstrap_mean_ci(paired_difference.to_numpy())
        kan_mean = float(head_summary.loc[
            head_summary["fusion_head"] == "kan", "mae_mean_pp"
        ].iloc[0])
        mlp_mean = float(head_summary.loc[
            head_summary["fusion_head"] == "mlp", "mae_mean_pp"
        ].iloc[0])
        if kan_mean + SIMPLICITY_TIE_PP < mlp_mean:
            SELECTED_HEAD = "kan"
            head_reason = "KAN mean MAE exceeded the predeclared simplicity margin."
        else:
            SELECTED_HEAD = "mlp"
            head_reason = "MLP won or KAN's advantage was smaller than the simplicity margin."

        selected_mean = kan_mean if SELECTED_HEAD == "kan" else mlp_mean
        with np.load(VALIDATION, allow_pickle=False) as arrays:
            validation_behavior_values = arrays["behavior_target"]
        behavior_counts = {
            str(int(label)): int(count) for label, count in zip(
                *np.unique(validation_behavior_values, return_counts=True)
            )
        }
        selection = {
            "selected_formulation": SELECTED_FORMULATION,
            "selected_deep_head": SELECTED_HEAD,
            "head_selection_reason": head_reason,
            "kan_mean_mae_pp": kan_mean,
            "mlp_mean_mae_pp": mlp_mean,
            "paired_kan_minus_mlp_mean_pp": float(paired_difference.mean()),
            "paired_kan_minus_mlp_ci95_pp": [paired_lower, paired_upper],
            "kan_seed_wins": int((paired_difference < 0).sum()),
            "mlp_seed_wins": int((paired_difference > 0).sum()),
            "aging_baseline_validation_mae_pp": baseline_summary["validation_aging_mae_pp"],
            "persistence_validation_mae_pp": baseline_summary["validation_persistence_mae_pp"],
            "beats_aging_baseline": bool(
                selected_mean < baseline_summary["validation_aging_mae_pp"]
            ),
            "beats_persistence": bool(
                selected_mean < baseline_summary["validation_persistence_mae_pp"]
            ),
            "deployment_recommendation": (
                f"DB-PatchKAN-{SELECTED_HEAD.upper()}"
                if selected_mean < baseline_summary["validation_persistence_mae_pp"]
                else "Persistence for this Nissan validation regime; retain the selected deep "
                     "model for external-vehicle evaluation."
            ),
            "behavior_class_support_by_segment": behavior_counts,
            "behavior_three_class_evaluable": all(
                str(label) in behavior_counts for label in (0, 1, 2)
            ),
            "driver_id_used_for_model_selection": False,
            "untouched_test_used": False,
            "final_test_authorized": False,
        }
        (RUN_ROOT / "STAGE1B2_FINAL_SELECTION.json").write_text(
            json.dumps(selection, indent=2)
        )
        display(confirmation[[
            "fusion_head", "seed", "best_epoch", "validation_current_soh_mae_pp",
            "validation_current_soh_rmse_pp", "validation_current_soh_bias_pp",
        ]])
        display(head_summary)
        print(json.dumps(selection, indent=2), flush=True)
        update(f"BLOCK 7 complete; selected head={SELECTED_HEAD}")
        """
    ),
    code(
        """
        # BLOCK 8 — Inventory, package all models/results, and auto-download
        from google.colab import files

        update("BLOCK 8 building artifact inventory")
        inventory_rows = []
        for path in sorted(RUN_ROOT.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                inventory_rows.append({
                    "relative_path": path.relative_to(RUN_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
        inventory = pd.DataFrame(inventory_rows)
        inventory.to_csv(RUN_ROOT / "artifact_inventory.csv", index=False)

        final_summary = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": RUN_ID,
            "gpu": GPU_NAME,
            "selected_formulation": SELECTED_FORMULATION,
            "selected_head": SELECTED_HEAD,
            "stage2_seeds": STAGE2_SEEDS,
            "untouched_test_used": False,
            "next_step": (
                "Freeze the selected configuration, then decide whether external-vehicle "
                "validation is available before any final Nissan test evaluation."
            ),
        }
        (RUN_ROOT / "STAGE1B2_SUMMARY.json").write_text(
            json.dumps(final_summary, indent=2)
        )
        print(json.dumps(final_summary, indent=2), flush=True)

        local_zip = Path("/content") / f"DBPatchKAN_{RUN_ID}_ALL_MODELS_RESULTS.zip"
        if local_zip.exists():
            local_zip.unlink()
        source_files = [
            path for path in sorted(RUN_ROOT.rglob("*"))
            if path.is_file() and path.suffix != ".zip"
        ]
        total_bytes = sum(path.stat().st_size for path in source_files)
        completed_bytes = 0
        with zipfile.ZipFile(
            local_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for index, path in enumerate(source_files, 1):
                archive.write(
                    path,
                    arcname=(Path(RUN_ID) / path.relative_to(RUN_ROOT)).as_posix(),
                )
                completed_bytes += path.stat().st_size
                if index == 1 or index % 10 == 0 or index == len(source_files):
                    update(
                        f"ZIP {index}/{len(source_files)} files; "
                        f"{completed_bytes / max(total_bytes, 1):.1%} source bytes"
                    )
        update(
            f"ZIP ready: {local_zip.name}; {local_zip.stat().st_size / 2**20:.1f} MB; "
            f"SHA-256={sha256(local_zip)}"
        )
        if INPUT_MODE == "drive":
            drive_copy = RUN_ROOT.parent / local_zip.name
            shutil.copy2(local_zip, drive_copy)
            update(f"Persistent ZIP copied to {drive_copy}")
        update("Starting browser download. The persistent Drive copy is already safe.")
        files.download(str(local_zip))
        """
    ),
    markdown(
        """
        ## Interpretation rule

        The notebook selects a deep formulation and head using validation data only. It can still
        recommend persistence for the Nissan operating regime. A selected deep model should be
        described as proof-of-concept until it is tested on independent vehicles with capacity-based
        SOH labels. Three-class behaviour metrics are not claimed when any intended class is absent.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(OUTPUT)
