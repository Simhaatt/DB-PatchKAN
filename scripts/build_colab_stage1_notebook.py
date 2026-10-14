"""Generate the self-contained reduced A100 Stage-1 Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_colab_a100_reduced_stage1.ipynb"


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
        # DB-PatchKAN — A100 reduced Stage 1

        This notebook performs the first leakage-guarded GPU stage:

        1. acquire the code and **training/validation tensors only**;
        2. verify hashes, tensor shapes, trip counts, and split disjointness;
        3. run KAN and MLP integration smoke tests;
        4. run a resumable 10-trial reduced Optuna/Hyperband search;
        5. pretrain one shared dual-stream encoder;
        6. fine-tune fair KAN and MLP concept heads from that same encoder;
        7. create and automatically download one ZIP containing every Stage-1 model and result.

        Select an **A100 GPU runtime** before starting. The untouched test tensor is intentionally
        prohibited from this notebook. Do not upload the raw 443-MB CSV or the large Parquet files;
        GPU training uses the compact NPZ tensors.
        """
    ),
    markdown(
        """
        ## Files to provide

        Upload these four files either through the Colab upload dialog or place them together in
        `MyDrive/DBPatchKAN/stage1_input/`:

        | File | Approximate size | Required |
        |---|---:|---|
        | `dbpatchkan_colab_code.zip` | <1 MB | Yes |
        | `segments_train.npz` | 6.48 MB | Yes |
        | `segments_validation.npz` | 1.25 MB | Yes |
        | `gpu_tensor_split_manifest.json` | 2 KB | Yes |

        **Do not place `segments_test_UNTOUCHED.npz` in the Stage-1 input folder.** It belongs in a
        separately locked folder and is used only by the final evaluation notebook after every
        decision is frozen.
        """
    ),
    code(
        """
        # BLOCK 1 — User configuration
        from datetime import datetime, timezone

        INPUT_MODE = "upload"  # Change to "drive" to mount Google Drive.
        DRIVE_INPUT_DIR = "/content/drive/MyDrive/DBPatchKAN/stage1_input"
        DRIVE_RUNS_DIR = "/content/drive/MyDrive/DBPatchKAN/runs"
        RUN_ID = "stage1_reduced_seed42"  # Keep unchanged to resume an interrupted run.

        RUN_SMOKE = True
        RUN_TUNING = True
        RUN_SHARED_PRETRAIN = True
        RUN_KAN = True
        RUN_MLP = True

        N_TRIALS = 10
        TUNE_PRETRAIN_EPOCHS = 5
        TUNE_FINETUNE_EPOCHS = 25
        FINAL_PRETRAIN_EPOCHS = 30
        FINAL_FINETUNE_EPOCHS = 40
        FINAL_PATIENCE = 7
        SEED = 42

        assert INPUT_MODE in {"upload", "drive"}
        assert N_TRIALS >= 1
        print(f"[{datetime.now(timezone.utc).isoformat()}] BLOCK 1 complete")
        print({
            "input_mode": INPUT_MODE,
            "run_id": RUN_ID,
            "trials": N_TRIALS,
            "final_pretrain_epochs": FINAL_PRETRAIN_EPOCHS,
            "final_finetune_epochs": FINAL_FINETUNE_EPOCHS,
        })
        """
    ),
    code(
        """
        # BLOCK 2 — Acquire files from Drive or the browser upload dialog
        import json
        import shutil
        import zipfile
        from pathlib import Path
        from datetime import datetime, timezone

        REQUIRED = {
            "dbpatchkan_colab_code.zip",
            "segments_train.npz",
            "segments_validation.npz",
            "gpu_tensor_split_manifest.json",
        }
        INPUT_ROOT = Path("/content/dbpatchkan_stage1_input")
        INPUT_ROOT.mkdir(parents=True, exist_ok=True)

        def update(message):
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] {message}", flush=True)

        update(f"BLOCK 2 starting in {INPUT_MODE!r} mode")
        if INPUT_MODE == "drive":
            from google.colab import drive
            drive.mount("/content/drive")
            source = Path(DRIVE_INPUT_DIR)
            missing = sorted(name for name in REQUIRED if not (source / name).exists())
            assert not missing, f"Missing from Drive input folder: {missing}"
            for name in sorted(REQUIRED):
                shutil.copy2(source / name, INPUT_ROOT / name)
                update(f"Copied {name}")
            RUN_ROOT = Path(DRIVE_RUNS_DIR) / RUN_ID
        else:
            from google.colab import files
            update("Select all four required files in the upload dialog")
            uploaded = files.upload()
            missing = sorted(REQUIRED - set(uploaded))
            assert not missing, f"Upload was incomplete; missing: {missing}"
            for name, payload in uploaded.items():
                (INPUT_ROOT / Path(name).name).write_bytes(payload)
                update(f"Received {name} ({len(payload) / 2**20:.2f} MB)")
            RUN_ROOT = Path("/content/dbpatchkan_runs") / RUN_ID

        prohibited = [
            path.name for path in INPUT_ROOT.iterdir()
            if "untouched" in path.name.casefold() or "test" in path.name.casefold()
        ]
        assert not prohibited, f"Stage 1 refuses test/untouched files: {prohibited}"
        RUN_ROOT.mkdir(parents=True, exist_ok=True)

        EXTRACT_ROOT = Path("/content/dbpatchkan_project")
        if EXTRACT_ROOT.exists():
            shutil.rmtree(EXTRACT_ROOT)
        EXTRACT_ROOT.mkdir(parents=True)
        with zipfile.ZipFile(INPUT_ROOT / "dbpatchkan_colab_code.zip") as archive:
            archive.extractall(EXTRACT_ROOT)
        project_files = list(EXTRACT_ROOT.rglob("pyproject.toml"))
        assert len(project_files) == 1, f"Expected one project root, found: {project_files}"
        PROJECT_ROOT = project_files[0].parent
        DATA = INPUT_ROOT / "segments_train.npz"
        VALIDATION = INPUT_ROOT / "segments_validation.npz"
        MANIFEST = INPUT_ROOT / "gpu_tensor_split_manifest.json"

        (RUN_ROOT / "notebook_settings.json").write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_mode": INPUT_MODE,
            "run_id": RUN_ID,
            "n_trials": N_TRIALS,
            "tune_pretrain_epochs": TUNE_PRETRAIN_EPOCHS,
            "tune_finetune_epochs": TUNE_FINETUNE_EPOCHS,
            "final_pretrain_epochs": FINAL_PRETRAIN_EPOCHS,
            "final_finetune_epochs": FINAL_FINETUNE_EPOCHS,
            "seed": SEED,
            "untouched_test_used": False,
        }, indent=2))
        update(f"BLOCK 2 complete; project={PROJECT_ROOT}, outputs={RUN_ROOT}")
        """
    ),
    code(
        """
        # BLOCK 3 — Install and verify the A100 environment
        import os
        import platform
        import subprocess
        import sys
        import time
        from pathlib import Path
        from datetime import datetime, timezone

        update("BLOCK 3 installing Optuna and the local project")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "optuna>=4.0", "PyYAML>=6.0"
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(PROJECT_ROOT)
        ])

        import numpy as np
        import pandas as pd
        import torch
        import optuna
        import dbpatchkan

        assert torch.cuda.is_available(), "Enable a GPU runtime: Runtime > Change runtime type > A100 GPU"
        gpu_name = torch.cuda.get_device_name(0)
        assert "A100" in gpu_name.upper(), f"Expected A100; current GPU is {gpu_name}"
        assert torch.cuda.is_bf16_supported(), "This notebook expects A100 BF16 support"

        def gpu_status():
            allocated = torch.cuda.memory_allocated() / 2**30
            reserved = torch.cuda.memory_reserved() / 2**30
            return f"GPU={gpu_name}; allocated={allocated:.2f} GB; reserved={reserved:.2f} GB"

        def run_with_updates(command, label, output_dir=None, interval=30):
            output_dir = Path(output_dir) if output_dir else None
            update(f"START {label}")
            print("COMMAND:", " ".join(map(str, command)), flush=True)
            with (RUN_ROOT / "commands.log").open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now(timezone.utc).isoformat()} | {label} | {' '.join(map(str, command))}\\n")
            started = time.time()
            process = subprocess.Popen([str(item) for item in command], cwd=PROJECT_ROOT)
            while process.poll() is None:
                time.sleep(interval)
                checkpoint = ""
                if output_dir and output_dir.exists():
                    candidates = sorted(output_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime)
                    if candidates:
                        latest = candidates[-1]
                        checkpoint = f"; latest={latest.name} ({latest.stat().st_size / 2**20:.1f} MB)"
                update(f"{label}: elapsed={(time.time()-started)/60:.1f} min; {gpu_status()}{checkpoint}")
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
            update(f"DONE {label}: elapsed={(time.time()-started)/60:.1f} min")

        environment = {
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "optuna": optuna.__version__,
            "gpu": gpu_name,
            "bf16": torch.cuda.is_bf16_supported(),
        }
        (RUN_ROOT / "environment.json").write_text(json.dumps(environment, indent=2))
        print(json.dumps(environment, indent=2))
        update("BLOCK 3 complete")
        """
    ),
    code(
        """
        # BLOCK 4 — Verify hashes, arrays, trip bags, and train/validation separation
        import hashlib
        import numpy as np
        from dbpatchkan.models import EVTripBagDataset, assert_disjoint_trip_bags

        def sha256(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        update("BLOCK 4 validating manifest hashes")
        manifest = json.loads(MANIFEST.read_text())
        for role, path in (("train", DATA), ("validation", VALIDATION)):
            expected = manifest["files"][role]
            actual_hash = sha256(path)
            assert actual_hash == expected["sha256"], f"SHA-256 mismatch for {role}"
            with np.load(path, allow_pickle=False) as arrays:
                shapes = {name: list(value.shape) for name, value in arrays.items()}
                print(f"{role}: {expected['trips']} trips, {expected['windows']} windows, {path.stat().st_size/2**20:.2f} MB")
                print("  principal arrays:", {
                    name: shapes[name] for name in ("driver", "battery", "concept_targets")
                    if name in shapes
                })

        train_bags = EVTripBagDataset.from_npz(DATA)
        validation_bags = EVTripBagDataset.from_npz(VALIDATION)
        assert_disjoint_trip_bags(train_bags, validation_bags)
        train_sizes = np.asarray([
            int(train_bags[index]["driver"].shape[0]) for index in range(len(train_bags))
        ])
        validation_sizes = np.asarray([
            int(validation_bags[index]["driver"].shape[0])
            for index in range(len(validation_bags))
        ])
        audit = {
            "train_trip_bags": len(train_bags),
            "validation_trip_bags": len(validation_bags),
            "train_segments_min_median_max": [
                int(train_sizes.min()), float(np.median(train_sizes)), int(train_sizes.max())
            ],
            "validation_segments_min_median_max": [
                int(validation_sizes.min()), float(np.median(validation_sizes)), int(validation_sizes.max())
            ],
            "trip_overlap": 0,
            "untouched_test_loaded": False,
        }
        (RUN_ROOT / "data_audit.json").write_text(json.dumps(audit, indent=2))
        print(json.dumps(audit, indent=2))
        update("BLOCK 4 complete")
        """
    ),
    code(
        """
        # BLOCK 5 — One-epoch KAN and MLP smoke tests on the real train/validation split
        if RUN_SMOKE:
            for head in ("kan", "mlp"):
                smoke_output = RUN_ROOT / f"00_smoke_{head}"
                if (smoke_output / "best.pt").exists():
                    update(f"BLOCK 5 skipping completed {head} smoke test")
                    continue
                command = [
                    sys.executable, "-m", "dbpatchkan.training.train",
                    "--data", str(DATA), "--validation-data", str(VALIDATION), "--trip-bags",
                    "--output-dir", str(smoke_output), "--stage", "both",
                    "--device", "cuda", "--precision", "bf16", "--domain", "nissan",
                    "--fusion-head", head, "--segment-length", "256",
                    "--patch-length", "16", "--patch-stride", "8", "--d-model", "128",
                    "--encoder-layers", "3", "--heads", "4", "--ff-dim", "256",
                    "--dropout", "0.1", "--kan-grid", "5",
                    "--pretrain-batch-size", "256", "--finetune-batch-size", "2",
                    "--pretrain-epochs", "1", "--epochs", "1", "--patience", "1",
                    "--pretrain-lr", "3e-4", "--encoder-lr", "1e-5", "--head-lr", "3e-4",
                    "--weight-decay", "0.01", "--warmup-fraction", "0.05",
                    "--gradient-clip", "1.0", "--mask-ratio", "0.4",
                    "--gradient-accumulation", "1", "--num-workers", "4",
                    "--seed", str(SEED), "--freeze-encoders",
                ]
                run_with_updates(command, f"real-data {head.upper()} smoke", smoke_output)
        else:
            update("BLOCK 5 skipped by configuration")
        update("BLOCK 5 complete")
        """
    ),
    markdown(
        """
        ## Reduced validation-only tuning

        This uses ten resumable trials. Expensive encoder settings are fixed at patch length 16,
        embedding width 128, and three encoder blocks. The search varies dropout, concept count,
        KAN grid, and KAN hidden width. Each screening trial uses short pretraining; the selected
        encoder is subsequently pretrained once for 30 epochs.
        """
    ),
    code(
        """
        # BLOCK 6 — Resumable 10-trial reduced Optuna/Hyperband search
        TUNING_OUTPUT = RUN_ROOT / "01_reduced_tuning"
        BEST_CONFIG = TUNING_OUTPUT / "best_config.json"
        if RUN_TUNING:
            command = [
                sys.executable, "-m", "dbpatchkan.training.tune",
                "--train-data", str(DATA), "--validation-data", str(VALIDATION), "--trip-bags",
                "--output-dir", str(TUNING_OUTPUT),
                "--study-db", str(TUNING_OUTPUT / "optuna_study.db"),
                "--study-name", "dbpatchkan_a100_reduced",
                "--search-profile", "reduced", "--trials", str(N_TRIALS),
                "--device", "cuda", "--precision", "bf16", "--domain", "nissan",
                "--pretrain-epochs", str(TUNE_PRETRAIN_EPOCHS),
                "--finetune-epochs", str(TUNE_FINETUNE_EPOCHS),
                "--pretrain-batch-size", "256", "--finetune-batch-size", "2",
                "--gradient-accumulation", "1", "--pretrain-lr", "3e-4",
                "--encoder-lr", "1e-5", "--head-lr", "3e-4",
                "--weight-decay", "0.01", "--gradient-clip", "1.0",
                "--num-workers", "4", "--seed", str(SEED), "--freeze-encoders",
            ]
            run_with_updates(command, "reduced Optuna search", TUNING_OUTPUT)
        else:
            update("BLOCK 6 tuning skipped by configuration")

        if BEST_CONFIG.exists():
            best_payload = json.loads(BEST_CONFIG.read_text())
            print(json.dumps(best_payload, indent=2))
            trials = pd.read_csv(TUNING_OUTPUT / "trials.csv")
            display(trials.sort_values("value").head(10))
        else:
            best_payload = None
            update("No best_config.json exists; the next block will use registered defaults")
        update("BLOCK 6 complete")
        """
    ),
    code(
        """
        # BLOCK 7 — Pretrain one shared encoder, then fine-tune fair KAN and MLP heads
        from dataclasses import replace
        from dbpatchkan.models import DBPatchKAN, DBPatchKANConfig
        from dbpatchkan.training.train import TrainingConfig, train
        from dbpatchkan.training.tune import ConceptSubsetDataset

        if best_payload and best_payload.get("model_config"):
            selected_model_config = DBPatchKANConfig(**best_payload["model_config"])
            selected_mask_ratio = float(best_payload["parameters"]["mask_ratio"])
            update(f"Using Optuna trial {best_payload['best_trial']} configuration")
        else:
            selected_model_config = DBPatchKANConfig()
            selected_mask_ratio = 0.4
            update("Using registered default model configuration")

        train_view = ConceptSubsetDataset(train_bags, selected_model_config.n_concepts)
        validation_view = ConceptSubsetDataset(validation_bags, selected_model_config.n_concepts)
        shared_output = RUN_ROOT / "02_shared_pretraining"
        shared_checkpoint = shared_output / "pretrain_latest.pt"

        if RUN_SHARED_PRETRAIN:
            completed_pretrain_epochs = 0
            if shared_checkpoint.exists():
                completed_pretrain_epochs = int(torch.load(
                    shared_checkpoint, map_location="cpu", weights_only=False
                ).get("epoch", 0))
            if completed_pretrain_epochs >= FINAL_PRETRAIN_EPOCHS:
                update(f"Shared pretraining already complete at epoch {completed_pretrain_epochs}")
            else:
                resume = str(shared_checkpoint) if shared_checkpoint.exists() else None
                pretrain_config = TrainingConfig(
                    stage="pretrain", device="cuda", precision="bf16",
                    pretrain_batch_size=256, finetune_batch_size=2,
                    gradient_accumulation=1, pretrain_epochs=FINAL_PRETRAIN_EPOCHS,
                    finetune_epochs=0, pretrain_lr=3e-4, encoder_lr=1e-5, head_lr=3e-4,
                    weight_decay=0.01, warmup_fraction=0.05, gradient_clip=1.0,
                    patience=FINAL_PATIENCE, mask_ratio=selected_mask_ratio,
                    num_workers=4, seed=SEED, domain="nissan", freeze_encoders=True,
                    trip_bags=True, resume_checkpoint=resume,
                )
                update(f"START shared pretraining; resume={bool(resume)}")
                result = train(
                    DBPatchKAN(replace(selected_model_config, fusion_head="kan")),
                    train_view, validation_view, pretrain_config, output_dir=shared_output,
                )
                (shared_output / "result.json").write_text(json.dumps(result, indent=2))
                update("DONE shared pretraining")

        assert shared_checkpoint.exists(), "Shared pretraining checkpoint is missing"
        shared_state = torch.load(shared_checkpoint, map_location="cpu", weights_only=False)["model"]
        comparison_rows = []
        for head, enabled in (("kan", RUN_KAN), ("mlp", RUN_MLP)):
            if not enabled:
                update(f"Skipping {head.upper()} by configuration")
                continue
            head_output = RUN_ROOT / f"03_finetune_{head}_seed{SEED}"
            latest = head_output / "latest.pt"
            model = DBPatchKAN(replace(selected_model_config, fusion_head=head))
            finished_result = head_output / "result.json"
            best_checkpoint = head_output / "best.pt"
            if finished_result.exists() and best_checkpoint.exists():
                update(f"{head.upper()} fine-tuning already completed; reusing saved result")
                result = json.loads(finished_result.read_text())
            else:
                if not latest.exists():
                    target_state = model.state_dict()
                    compatible = {
                        name: value for name, value in shared_state.items()
                        if name in target_state
                        and target_state[name].shape == value.shape
                        and not name.startswith("fusion_head.")
                    }
                    incompatible = model.load_state_dict(compatible, strict=False)
                    update(
                        f"Loaded {len(compatible)} shared tensors into {head.upper()}; "
                        f"new/unloaded tensors={len(incompatible.missing_keys)}"
                    )
                training_config = TrainingConfig(
                    stage="finetune", device="cuda", precision="bf16",
                    pretrain_batch_size=256, finetune_batch_size=2,
                    gradient_accumulation=1, pretrain_epochs=0,
                    finetune_epochs=FINAL_FINETUNE_EPOCHS,
                    pretrain_lr=3e-4, encoder_lr=1e-5, head_lr=3e-4,
                    weight_decay=0.01, warmup_fraction=0.05, gradient_clip=1.0,
                    patience=FINAL_PATIENCE, mask_ratio=selected_mask_ratio,
                    num_workers=4, seed=SEED, domain="nissan", freeze_encoders=True,
                    trip_bags=True,
                    resume_checkpoint=str(latest) if latest.exists() else None,
                )
                update(f"START {head.upper()} fine-tuning; resume={latest.exists()}")
                result = train(
                    model, train_view, validation_view, training_config, output_dir=head_output
                )
                finished_result.write_text(json.dumps(result, indent=2))
                update(f"DONE {head.upper()} fine-tuning")
            assert best_checkpoint.exists(), f"No validation-selected checkpoint for {head}"
            best = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
            metrics = best.get("metrics", {})
            comparison_rows.append({
                "fusion_head": head,
                "best_epoch": best.get("epoch"),
                "parameter_count": result.get("parameter_count"),
                "peak_vram_gb": result.get("peak_vram_gb"),
                **metrics,
            })

        comparison = pd.DataFrame(comparison_rows)
        comparison.to_csv(RUN_ROOT / "stage1_model_comparison.csv", index=False)
        display(comparison)
        update("BLOCK 7 complete")
        """
    ),
    code(
        """
        # BLOCK 8 — Final Stage-1 audit and human-readable summary
        import platform

        update("BLOCK 8 building result inventory")
        inventory = []
        for path in sorted(RUN_ROOT.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                inventory.append({
                    "relative_path": path.relative_to(RUN_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
        inventory_frame = pd.DataFrame(inventory)
        inventory_frame.to_csv(RUN_ROOT / "artifact_inventory.csv", index=False)
        summary = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": RUN_ID,
            "gpu": gpu_name,
            "training_trips": len(train_bags),
            "validation_trips": len(validation_bags),
            "tuning_trials_requested": N_TRIALS if RUN_TUNING else 0,
            "shared_pretrain_epochs": FINAL_PRETRAIN_EPOCHS,
            "head_finetune_epochs_max": FINAL_FINETUNE_EPOCHS,
            "produced_heads": comparison["fusion_head"].tolist() if len(comparison) else [],
            "untouched_test_used": False,
            "next_step": "Review validation results before multi-seed confirmation; do not open the test set.",
        }
        (RUN_ROOT / "STAGE1_SUMMARY.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        display(inventory_frame[["relative_path", "bytes"]])
        update("BLOCK 8 complete")
        """
    ),
    code(
        """
        # BLOCK 9 — Package every model/result and automatically download one ZIP
        import zipfile
        from google.colab import files

        update("BLOCK 9 creating the complete Stage-1 ZIP")
        local_zip = Path("/content") / f"DBPatchKAN_{RUN_ID}_ALL_MODELS_RESULTS.zip"
        if local_zip.exists():
            local_zip.unlink()
        source_files = [
            path for path in sorted(RUN_ROOT.rglob("*"))
            if path.is_file() and path.suffix != ".zip"
        ]
        total_bytes = sum(path.stat().st_size for path in source_files)
        completed_bytes = 0
        with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for index, path in enumerate(source_files, 1):
                archive.write(path, arcname=(Path(RUN_ID) / path.relative_to(RUN_ROOT)).as_posix())
                completed_bytes += path.stat().st_size
                if index == 1 or index % 10 == 0 or index == len(source_files):
                    update(
                        f"ZIP progress: {index}/{len(source_files)} files; "
                        f"{completed_bytes / max(total_bytes, 1):.1%} of source bytes"
                    )

        zip_hash = sha256(local_zip)
        update(f"ZIP ready: {local_zip.name}, {local_zip.stat().st_size/2**20:.1f} MB")
        print("SHA-256:", zip_hash)
        if INPUT_MODE == "drive":
            drive_copy = RUN_ROOT.parent / local_zip.name
            shutil.copy2(local_zip, drive_copy)
            update(f"Persistent copy saved to {drive_copy}")
        update("Starting browser download now. Keep this tab open until it begins.")
        files.download(str(local_zip))
        """
    ),
    markdown(
        """
        ## After Notebook 1

        Keep the downloaded ZIP and, if Drive mode was used, the persistent run directory. Review
        `stage1_model_comparison.csv` and `01_reduced_tuning/best_config.json`. Notebook 2 should run
        the selected KAN, MLP, and core deep baselines across confirmation folds/seeds. The untouched
        test set remains locked until the final configuration is selected.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(OUTPUT)
