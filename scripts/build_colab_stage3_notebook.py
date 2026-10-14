"""Generate the validation-only DB-PatchKAN Stage-3 explainability notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "03_colab_a100_kan_explainability.ipynb"


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
        # DB-PatchKAN Stage 3 — KAN explainability and stability

        This notebook explains the five frozen validation-selected KAN checkpoints from Stage 2.
        It does not load the locked Nissan test tensor and does not retune the main KAN.

        Outputs include cross-seed KAN response curves, graph-edge and path importance,
        integrated gradients, concept ablation, channel/time-block occlusion, pruning sensitivity,
        a direct additive KAN-Lite ablation, figures, an Excel workbook, and a complete ZIP.

        Channel and temporal occlusion use the actual end-to-end predicted-concept pathway.
        Concept ablation overrides one predicted concept at its training centre after encoding.
        Attention weights are not treated as feature importance.
        """
    ),
    markdown(
        """
        ## File to upload

        Run Block 2 and select `DBPatchKAN_Stage3_Upload_Input.zip`. Do not unzip it. The bundle
        contains the current source code and the completed Stage-1B/2 results archive. It contains
        no final-test tensor.
        """
    ),
    code(
        """
        # BLOCK 1 — Configuration
        from datetime import datetime, timezone

        INPUT_MODE = "upload"
        RUN_STRUCTURAL_EXPLANATIONS = True
        RUN_OCCLUSION = True
        RUN_PRUNING_SENSITIVITY = True
        RUN_KAN_LITE = True

        SEEDS = [42, 52, 62, 72, 82]
        RESPONSE_POINTS = 81
        IG_STEPS = 32
        OCCLUSION_BATCH_SIZE = 2
        PRUNING_QUANTILES = [0.10, 0.20, 0.30]
        KAN_LITE_EPOCHS = 80
        KAN_LITE_PATIENCE = 15
        KAN_LITE_NONINFERIORITY_MARGIN_PP = 0.05
        RUN_ID = "stage3_kan_explainability"

        assert INPUT_MODE in {"upload", "drive"}
        assert len(SEEDS) == 5 and len(set(SEEDS)) == 5
        print(f"[{datetime.now(timezone.utc).isoformat()}] BLOCK 1 complete")
        print({
            "seeds": SEEDS,
            "response_points": RESPONSE_POINTS,
            "integrated_gradient_steps": IG_STEPS,
            "run_kan_lite": RUN_KAN_LITE,
            "test_enabled": False,
        })
        """
    ),
    code(
        """
        # BLOCK 2 — Acquire and extract the Stage-3 bundle
        import json
        import re
        import shutil
        import zipfile
        from pathlib import Path
        from datetime import datetime, timezone

        def update(message):
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{stamp} UTC] {message}", flush=True)

        INPUT_ROOT = Path("/content/dbpatchkan_stage3_input")
        if INPUT_ROOT.exists():
            shutil.rmtree(INPUT_ROOT)
        INPUT_ROOT.mkdir(parents=True)

        update(f"BLOCK 2 starting in {INPUT_MODE!r} mode")
        if INPUT_MODE == "drive":
            from google.colab import drive
            drive.mount("/content/drive")
            source_zip = Path(
                "/content/drive/MyDrive/DBPatchKAN/stage3_input/"
                "DBPatchKAN_Stage3_Upload_Input.zip"
            )
            assert source_zip.is_file(), f"Bundle not found: {source_zip}"
            RUN_ROOT = Path("/content/drive/MyDrive/DBPatchKAN/runs") / RUN_ID
        else:
            from google.colab import files
            update("Upload DBPatchKAN_Stage3_Upload_Input.zip")
            uploaded = files.upload()
            assert uploaded, "No Stage-3 bundle was uploaded"
            uploaded_name, payload = next(iter(uploaded.items()))
            clean_name = re.sub(r" \\(\\d+\\)(?=\\.[^.]+$)", "", Path(uploaded_name).name)
            source_zip = INPUT_ROOT / clean_name
            source_zip.write_bytes(payload)
            RUN_ROOT = Path("/content/dbpatchkan_runs") / RUN_ID

        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(INPUT_ROOT)
        required = {
            "dbpatchkan_colab_code.zip",
            "DBPatchKAN_stage1b2_full_selection_ALL_MODELS_RESULTS.zip",
        }
        located = {}
        for path in INPUT_ROOT.rglob("*"):
            if path.is_file() and path.name in required:
                located[path.name] = path
        missing = sorted(required - set(located))
        assert not missing, f"Stage-3 input bundle is incomplete: {missing}"

        prohibited = [
            path.name for path in INPUT_ROOT.rglob("*")
            if path.is_file() and (
                "untouched" in path.name.casefold()
                or re.search(r"(^|[_\\-.])test([_\\-.]|$)", path.name.casefold())
            )
        ]
        assert not prohibited, f"Stage 3 refuses test files: {prohibited}"

        PROJECT_EXTRACT = Path("/content/dbpatchkan_project")
        if PROJECT_EXTRACT.exists():
            shutil.rmtree(PROJECT_EXTRACT)
        PROJECT_EXTRACT.mkdir(parents=True)
        with zipfile.ZipFile(located["dbpatchkan_colab_code.zip"]) as archive:
            archive.extractall(PROJECT_EXTRACT)
        project_files = list(PROJECT_EXTRACT.rglob("pyproject.toml"))
        assert len(project_files) == 1, f"Expected one project root: {project_files}"
        PROJECT_ROOT = project_files[0].parent

        STAGE2_EXTRACT = Path("/content/dbpatchkan_stage2_results")
        if STAGE2_EXTRACT.exists():
            shutil.rmtree(STAGE2_EXTRACT)
        STAGE2_EXTRACT.mkdir(parents=True)
        with zipfile.ZipFile(
            located["DBPatchKAN_stage1b2_full_selection_ALL_MODELS_RESULTS.zip"]
        ) as archive:
            archive.extractall(STAGE2_EXTRACT)
        summaries = list(STAGE2_EXTRACT.rglob("STAGE1B2_SUMMARY.json"))
        assert len(summaries) == 1, f"Expected one Stage-2 summary: {summaries}"
        STAGE2_ROOT = summaries[0].parent

        (RUN_ROOT / "notebook_settings.json").write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": RUN_ID,
            "seeds": SEEDS,
            "response_points": RESPONSE_POINTS,
            "ig_steps": IG_STEPS,
            "untouched_test_used": False,
        }, indent=2))
        update(f"BLOCK 2 complete; project={PROJECT_ROOT}; Stage2={STAGE2_ROOT}")
        """
    ),
    code(
        """
        # BLOCK 3 — Install and verify the explanation environment
        import os
        import sys
        import site
        import hashlib
        import platform
        import subprocess
        import importlib

        update("BLOCK 3 installing the local project and plotting dependencies")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "PyYAML>=6.0", "openpyxl>=3.1", "seaborn>=0.13",
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--no-deps", "--no-build-isolation", "-e", str(PROJECT_ROOT),
        ])

        SRC_ROOT = (PROJECT_ROOT / "src").resolve()
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
        import matplotlib.pyplot as plt
        import seaborn as sns
        import dbpatchkan

        assert torch.cuda.is_available(), "Enable an A100 GPU runtime"
        GPU_NAME = torch.cuda.get_device_name(0)
        assert "A100" in GPU_NAME.upper(), f"Expected A100; current GPU is {GPU_NAME}"
        DEVICE = torch.device("cuda")
        sns.set_theme(style="whitegrid", context="notebook")

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
        # BLOCK 4 — Verify frozen checkpoints and prepare trip-level concepts
        from dbpatchkan.models import EVTripBagDataset
        from dbpatchkan.training.evaluate import load_checkpoint_model, state_fingerprint

        stage2_summary = json.loads((STAGE2_ROOT / "STAGE1B2_SUMMARY.json").read_text())
        stage2_selection = json.loads(
            (STAGE2_ROOT / "STAGE1B2_FINAL_SELECTION.json").read_text()
        )
        assert stage2_summary["untouched_test_used"] is False
        SELECTED_FORMULATION = stage2_summary["selected_formulation"]
        assert SELECTED_FORMULATION == "aging_residual"

        TRAIN_DATA = (
            STAGE2_ROOT / "stage1b_residual_tensors" /
            "segments_train_aging_residual.npz"
        )
        VALIDATION_DATA = (
            STAGE2_ROOT / "stage1b_residual_tensors" /
            "segments_validation_aging_residual.npz"
        )
        assert TRAIN_DATA.is_file() and VALIDATION_DATA.is_file()
        train_bags = EVTripBagDataset.from_npz(TRAIN_DATA)
        validation_bags = EVTripBagDataset.from_npz(VALIDATION_DATA)
        assert len(train_bags) == 81 and len(validation_bags) == 20

        def checkpoint_path(seed, head="kan"):
            path = (
                STAGE2_ROOT / "stage2" / SELECTED_FORMULATION / f"seed_{seed}"
                / f"finetune_{head}" / "best.pt"
            )
            assert path.is_file(), f"Missing frozen checkpoint: {path}"
            return path

        first_model, first_checkpoint = load_checkpoint_model(
            checkpoint_path(SEEDS[0]), device=DEVICE
        )
        CONCEPT_NAMES = tuple(first_model.config.concept_names)
        N_CONCEPTS = len(CONCEPT_NAMES)

        def trip_concept_matrix(dataset):
            values, masks = [], []
            for index in range(len(dataset)):
                item = dataset[index]
                values.append(item["concept_targets"][:N_CONCEPTS].numpy())
                masks.append(item["concept_mask"][:N_CONCEPTS].numpy())
            return np.stack(values), np.stack(masks)

        train_concepts, train_concept_mask = trip_concept_matrix(train_bags)
        validation_concepts, validation_concept_mask = trip_concept_matrix(validation_bags)
        concept_ranges = {}
        for index, name in enumerate(CONCEPT_NAMES):
            valid = train_concepts[train_concept_mask[:, index], index]
            valid = valid[np.isfinite(valid)]
            if len(valid):
                lower, upper = np.quantile(valid, [0.01, 0.99])
            else:
                lower = float(first_model.concept_center[index] - first_model.concept_scale[index])
                upper = float(first_model.concept_center[index] + first_model.concept_scale[index])
            if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
                center = float(first_model.concept_center[index])
                width = max(abs(center) * 0.05, 0.01)
                lower, upper = center - width, center + width
                if name.startswith("bsi_") or name.startswith("behavior_share_"):
                    lower, upper = max(0.0, lower), min(1.0, upper)
                    if upper <= lower:
                        upper = min(1.0, lower + 0.01)
            concept_ranges[name] = (float(lower), float(upper))

        checkpoint_audit = []
        for seed in SEEDS:
            model, checkpoint = load_checkpoint_model(checkpoint_path(seed), device=DEVICE)
            checkpoint_audit.append({
                "Seed": seed,
                "Checkpoint": str(checkpoint_path(seed)),
                "SHA256": sha256(checkpoint_path(seed)),
                "State_Fingerprint": state_fingerprint(model),
                "Best_Epoch": checkpoint.get("epoch"),
                "Validation_MAE_pp": checkpoint.get("metrics", {}).get(
                    "validation_current_soh_mae_pp"
                ),
                "Fusion_Head": model.config.fusion_head,
                "Concept_Count": model.config.n_concepts,
            })
            del model
            torch.cuda.empty_cache()
        checkpoint_audit = pd.DataFrame(checkpoint_audit)
        checkpoint_audit.to_csv(RUN_ROOT / "checkpoint_audit.csv", index=False)
        (RUN_ROOT / "concept_ranges.json").write_text(json.dumps(concept_ranges, indent=2))
        display(checkpoint_audit)
        print(json.dumps({
            "training_trips": len(train_bags),
            "validation_trips": len(validation_bags),
            "concepts": CONCEPT_NAMES,
            "untouched_test_used": False,
        }, indent=2), flush=True)
        del first_model
        torch.cuda.empty_cache()
        update("BLOCK 4 complete")
        """
    ),
    code(
        """
        # BLOCK 5 — Cross-seed KAN structure, curves, gates, and integrated gradients
        from dbpatchkan.analysis.deep_explainability import (
            aggregate_response_curves,
            channel_gate_table,
            concept_integrated_gradients,
            concept_response_curves,
            kan_edge_strength_table,
            kan_path_importance_table,
        )

        TABLE_ROOT = RUN_ROOT / "tables"
        FIGURE_ROOT = RUN_ROOT / "figures"
        TABLE_ROOT.mkdir(parents=True, exist_ok=True)
        FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

        edge_frames, path_frames, curve_frames, ig_frames, gate_frames = [], [], [], [], []
        if RUN_STRUCTURAL_EXPLANATIONS:
            for seed_index, seed in enumerate(SEEDS, 1):
                update(f"Structural explanations seed {seed} ({seed_index}/{len(SEEDS)})")
                model, _ = load_checkpoint_model(checkpoint_path(seed), device=DEVICE)
                edge_frames.append(kan_edge_strength_table(model, seed=seed))
                path_frames.append(kan_path_importance_table(model, seed=seed))
                gate_frames.append(channel_gate_table(model, seed=seed))
                curve_frames.append(concept_response_curves(
                    model, concept_ranges, points=RESPONSE_POINTS, seed=seed
                ))
                ig_frames.append(concept_integrated_gradients(
                    model,
                    validation_concepts,
                    validation_concept_mask,
                    output_name="current_soh",
                    steps=IG_STEPS,
                    seed=seed,
                ))
                del model
                torch.cuda.empty_cache()

        edges = pd.concat(edge_frames, ignore_index=True)
        paths = pd.concat(path_frames, ignore_index=True)
        curves = pd.concat(curve_frames, ignore_index=True)
        integrated_gradients = pd.concat(ig_frames, ignore_index=True)
        gates = pd.concat(gate_frames, ignore_index=True)
        curve_aggregate, curve_stability = aggregate_response_curves(curves)

        for name, frame in {
            "kan_edge_strengths.csv": edges,
            "kan_path_importance.csv": paths,
            "kan_response_curves_by_seed.csv": curves,
            "kan_response_curves_aggregate.csv": curve_aggregate,
            "kan_response_stability.csv": curve_stability,
            "kan_concept_integrated_gradients.csv": integrated_gradients,
            "channel_gate_stability.csv": gates,
        }.items():
            frame.to_csv(TABLE_ROOT / name, index=False)

        current_curves = curve_aggregate[
            curve_aggregate["Output_Name"] == "current_soh"
        ]
        fig, axes = plt.subplots(4, 4, figsize=(18, 15), constrained_layout=True)
        for axis, concept in zip(axes.flat, CONCEPT_NAMES):
            subset = current_curves[current_curves["Concept_Name"] == concept]
            x = subset["Concept_Value"].to_numpy(float)
            mean = subset["Response_Mean"].to_numpy(float)
            lower = subset["Response_Lower"].to_numpy(float)
            upper = subset["Response_Upper"].to_numpy(float)
            axis.plot(x, mean, color="#1f77b4", linewidth=2)
            axis.fill_between(x, lower, upper, color="#1f77b4", alpha=0.20)
            axis.set_title(concept)
            axis.set_xlabel("Physical concept value")
            axis.set_ylabel("Residual-scale SOH output")
        fig.suptitle("DB-PatchKAN concept response curves across five seeds", fontsize=18)
        fig.savefig(FIGURE_ROOT / "kan_current_soh_response_curves.png", dpi=180)
        plt.close(fig)

        path_summary = paths.groupby(
            ["Output_Name", "Concept_Name"], as_index=False
        )["Normalized_Path_Importance"].mean()
        heatmap = path_summary.pivot(
            index="Concept_Name", columns="Output_Name", values="Normalized_Path_Importance"
        ).reindex(CONCEPT_NAMES)
        fig, axis = plt.subplots(figsize=(9, 9))
        sns.heatmap(heatmap, cmap="viridis", annot=True, fmt=".3f", ax=axis)
        axis.set_title("Mean KAN path importance across five seeds")
        fig.tight_layout()
        fig.savefig(FIGURE_ROOT / "kan_path_importance_heatmap.png", dpi=180)
        plt.close(fig)

        ig_summary = integrated_gradients.groupby("Concept_Name", as_index=False).agg(
            Mean_Absolute_IG=("IG_Mean_Absolute", "mean"),
            Std_Absolute_IG=("IG_Mean_Absolute", "std"),
        ).sort_values("Mean_Absolute_IG")
        fig, axis = plt.subplots(figsize=(10, 8))
        axis.barh(
            ig_summary["Concept_Name"], ig_summary["Mean_Absolute_IG"],
            xerr=ig_summary["Std_Absolute_IG"], color="#4c78a8", alpha=0.9
        )
        axis.set_xlabel("Mean absolute integrated gradient (SOH pp)")
        axis.set_title("Concept integrated gradients across five KAN seeds")
        fig.tight_layout()
        fig.savefig(FIGURE_ROOT / "kan_integrated_gradients.png", dpi=180)
        plt.close(fig)
        update("BLOCK 5 complete")
        """
    ),
    code(
        """
        # BLOCK 6 — Predicted-concept ablation and end-to-end channel/time-block occlusion
        from dbpatchkan.analysis.deep_explainability import validation_occlusion_tables

        concept_occlusion_frames, channel_occlusion_frames, patch_occlusion_frames = [], [], []
        if RUN_OCCLUSION:
            for seed_index, seed in enumerate(SEEDS, 1):
                update(f"START occlusion seed {seed} ({seed_index}/{len(SEEDS)})")
                model, _ = load_checkpoint_model(checkpoint_path(seed), device=DEVICE)
                tables = validation_occlusion_tables(
                    model,
                    validation_bags,
                    seed=seed,
                    device=DEVICE,
                    batch_size=OCCLUSION_BATCH_SIZE,
                    patch_length=model.config.patch_length,
                    progress=lambda message, seed=seed: update(f"seed {seed}: {message}"),
                )
                concept_occlusion_frames.append(tables["concept"])
                channel_occlusion_frames.append(tables["channel"])
                patch_occlusion_frames.append(tables["patch"])
                del model
                torch.cuda.empty_cache()
                update(f"DONE occlusion seed {seed}")

        concept_occlusion = pd.concat(concept_occlusion_frames, ignore_index=True)
        channel_occlusion = pd.concat(channel_occlusion_frames, ignore_index=True)
        patch_occlusion = pd.concat(patch_occlusion_frames, ignore_index=True)
        concept_occlusion.to_csv(TABLE_ROOT / "concept_ablation_by_seed.csv", index=False)
        channel_occlusion.to_csv(TABLE_ROOT / "channel_occlusion_by_seed.csv", index=False)
        patch_occlusion.to_csv(TABLE_ROOT / "temporal_block_occlusion_by_seed.csv", index=False)

        concept_summary = concept_occlusion.groupby("Concept_Name", as_index=False).agg(
            Mean_Ablation_Delta_pp=("Ablation_Delta_pp", "mean"),
            Std_Ablation_Delta_pp=("Ablation_Delta_pp", "std"),
            Positive_Frequency=("Ablation_Delta_pp", lambda x: float((x > 0).mean())),
        ).sort_values("Mean_Ablation_Delta_pp")
        channel_summary = channel_occlusion.groupby(
            ["Stream", "Channel_Name"], as_index=False
        ).agg(
            Mean_Occlusion_Delta_pp=("Channel_Occlusion_Delta_pp", "mean"),
            Std_Occlusion_Delta_pp=("Channel_Occlusion_Delta_pp", "std"),
        ).sort_values("Mean_Occlusion_Delta_pp")
        patch_summary = patch_occlusion.groupby(["Start_s", "Stop_s"], as_index=False).agg(
            Mean_Occlusion_Delta_pp=("Patch_Occlusion_Delta_pp", "mean"),
            Std_Occlusion_Delta_pp=("Patch_Occlusion_Delta_pp", "std"),
        )
        concept_summary.to_csv(TABLE_ROOT / "concept_ablation_summary.csv", index=False)
        channel_summary.to_csv(TABLE_ROOT / "channel_occlusion_summary.csv", index=False)
        patch_summary.to_csv(TABLE_ROOT / "temporal_block_occlusion_summary.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
        axes[0].barh(
            concept_summary["Concept_Name"], concept_summary["Mean_Ablation_Delta_pp"],
            xerr=concept_summary["Std_Ablation_Delta_pp"], color="#f58518"
        )
        axes[0].axvline(0, color="black", linewidth=1)
        axes[0].set_title("Predicted-concept centre ablation")
        axes[0].set_xlabel("Validation MAE increase (pp)")
        labels = channel_summary["Stream"] + ": " + channel_summary["Channel_Name"]
        axes[1].barh(
            labels, channel_summary["Mean_Occlusion_Delta_pp"],
            xerr=channel_summary["Std_Occlusion_Delta_pp"], color="#54a24b"
        )
        axes[1].axvline(0, color="black", linewidth=1)
        axes[1].set_title("End-to-end raw-channel occlusion")
        axes[1].set_xlabel("Validation MAE increase (pp)")
        fig.savefig(FIGURE_ROOT / "concept_and_channel_occlusion.png", dpi=180)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(11, 5))
        centers = (patch_summary["Start_s"] + patch_summary["Stop_s"]) / 2
        axis.errorbar(
            centers, patch_summary["Mean_Occlusion_Delta_pp"],
            yerr=patch_summary["Std_Occlusion_Delta_pp"], marker="o", capsize=3
        )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xlabel("Occluded time-block centre (s)")
        axis.set_ylabel("End-to-end validation MAE increase (pp)")
        axis.set_title("Temporal-block occlusion across five seeds")
        fig.tight_layout()
        fig.savefig(FIGURE_ROOT / "temporal_block_occlusion.png", dpi=180)
        plt.close(fig)
        update("BLOCK 6 complete")
        """
    ),
    code(
        """
        # BLOCK 7 — Frozen-checkpoint pruning sensitivity (no main-model mutation)
        import copy
        from dbpatchkan.analysis.deep_explainability import trip_predictions

        pruning_rows = []
        if RUN_PRUNING_SENSITIVITY:
            for seed_index, seed in enumerate(SEEDS, 1):
                update(f"Pruning sensitivity seed {seed} ({seed_index}/{len(SEEDS)})")
                model, _ = load_checkpoint_model(checkpoint_path(seed), device=DEVICE)
                target, prediction = trip_predictions(
                    model, validation_bags, device=DEVICE, batch_size=2
                )
                baseline_mae = float(np.mean(np.abs(prediction - target)))
                strengths = []
                for layer in model.fusion_head.layers:
                    strength = (
                        layer.base_weight.detach().abs()
                        + layer.spline_weight.detach().abs().mean(dim=-1)
                    )[layer.edge_mask.bool()]
                    strengths.extend(strength.cpu().tolist())
                for quantile in PRUNING_QUANTILES:
                    threshold = float(np.quantile(strengths, quantile))
                    candidate = copy.deepcopy(model)
                    before = sum(
                        int(layer.edge_mask.sum().item())
                        for layer in candidate.fusion_head.layers
                    )
                    removed = candidate.prune_kan_edges(threshold)
                    actual, estimate = trip_predictions(
                        candidate, validation_bags, device=DEVICE, batch_size=2
                    )
                    mae = float(np.mean(np.abs(estimate - actual)))
                    pruning_rows.append({
                        "Seed": seed,
                        "Quantile": quantile,
                        "Threshold": threshold,
                        "Active_Edges_Before": before,
                        "Edges_Removed": removed,
                        "Removed_Fraction": removed / max(before, 1),
                        "Baseline_MAE_pp": baseline_mae,
                        "Pruned_MAE_pp": mae,
                        "MAE_Delta_pp": mae - baseline_mae,
                        "Main_Checkpoint_Modified": False,
                    })
                    del candidate
                    torch.cuda.empty_cache()
                    update(
                        f"seed {seed}, q={quantile:.2f}: removed={removed}, "
                        f"delta={mae - baseline_mae:+.4f} pp"
                    )
                del model
                torch.cuda.empty_cache()
        pruning = pd.DataFrame(pruning_rows)
        pruning.to_csv(TABLE_ROOT / "kan_pruning_sensitivity.csv", index=False)
        pruning_summary = pruning.groupby("Quantile", as_index=False).agg(
            Mean_Removed_Fraction=("Removed_Fraction", "mean"),
            Mean_MAE_Delta_pp=("MAE_Delta_pp", "mean"),
            Worst_MAE_Delta_pp=("MAE_Delta_pp", "max"),
            NonWorse_Seed_Frequency=("MAE_Delta_pp", lambda x: float((x <= 0).mean())),
        )
        pruning_summary.to_csv(TABLE_ROOT / "kan_pruning_summary.csv", index=False)
        display(pruning_summary)
        update("BLOCK 7 complete")
        """
    ),
    code(
        """
        # BLOCK 8 — Five-seed direct additive KAN-Lite ablation
        from dataclasses import replace
        from dbpatchkan.models import DBPatchKAN
        from dbpatchkan.training.train import TrainingConfig, train
        from dbpatchkan.training.tune import ConceptSubsetDataset

        kan_lite_rows = []
        train_view = ConceptSubsetDataset(train_bags, N_CONCEPTS)
        validation_view = ConceptSubsetDataset(validation_bags, N_CONCEPTS)

        def load_shared_nonfusion(model, shared_state):
            target = model.state_dict()
            compatible = {
                name: value for name, value in shared_state.items()
                if name in target and target[name].shape == value.shape
                and not name.startswith("fusion_head.")
            }
            model.load_state_dict(compatible, strict=False)
            return len(compatible)

        if RUN_KAN_LITE:
            for seed_index, seed in enumerate(SEEDS, 1):
                output = RUN_ROOT / "kan_lite" / f"seed_{seed}"
                best_path = output / "best.pt"
                result_path = output / "result.json"
                main_model, _ = load_checkpoint_model(checkpoint_path(seed), device=DEVICE)
                lite_config = replace(main_model.config, fusion_head="kan", kan_hidden=())
                del main_model
                torch.cuda.empty_cache()
                if not (best_path.exists() and result_path.exists()):
                    shared_path = (
                        STAGE2_ROOT / "stage2" / SELECTED_FORMULATION / f"seed_{seed}"
                        / "shared_pretraining" / "pretrain_latest.pt"
                    )
                    shared_state = torch.load(
                        shared_path, map_location="cpu", weights_only=False
                    )["model"]
                    model = DBPatchKAN(lite_config)
                    if not (output / "latest.pt").exists():
                        loaded = load_shared_nonfusion(model, shared_state)
                        update(f"KAN-Lite seed {seed}: loaded {loaded} shared tensors")
                    config = TrainingConfig(
                        stage="finetune", device="cuda", precision="bf16",
                        pretrain_batch_size=256, finetune_batch_size=2,
                        gradient_accumulation=1, pretrain_epochs=0,
                        finetune_epochs=KAN_LITE_EPOCHS,
                        pretrain_lr=3e-4, encoder_lr=1e-5, head_lr=3e-4,
                        weight_decay=0.01, warmup_fraction=0.05,
                        gradient_clip=1.0, patience=KAN_LITE_PATIENCE,
                        mask_ratio=0.4, num_workers=4, seed=seed,
                        domain="nissan", freeze_encoders=True, trip_bags=True,
                        resume_checkpoint=str(output / "latest.pt")
                        if (output / "latest.pt").exists() else None,
                    )
                    update(f"START KAN-Lite seed {seed} ({seed_index}/{len(SEEDS)})")
                    result = train(
                        model, train_view, validation_view, config, output_dir=output
                    )
                    result_path.write_text(json.dumps(result, indent=2))
                    update(f"DONE KAN-Lite seed {seed}")
                    del model
                    torch.cuda.empty_cache()
                checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
                kan_lite_rows.append({
                    "Model": "KAN-Lite",
                    "Seed": seed,
                    "Best_Epoch": checkpoint.get("epoch"),
                    "Checkpoint": str(best_path),
                    "SHA256": sha256(best_path),
                    **checkpoint.get("metrics", {}),
                })

        kan_lite_results = pd.DataFrame(kan_lite_rows)
        kan_lite_results.to_csv(TABLE_ROOT / "kan_lite_seed_results.csv", index=False)
        stage2_results = pd.read_csv(STAGE2_ROOT / "stage2_seed_results.csv")
        comparison_rows = []
        for name, frame in (
            ("KAN", stage2_results[stage2_results["fusion_head"] == "kan"]),
            ("MLP", stage2_results[stage2_results["fusion_head"] == "mlp"]),
            ("KAN-Lite", kan_lite_results),
        ):
            values = frame["validation_current_soh_mae_pp"].to_numpy(float)
            comparison_rows.append({
                "Model": name,
                "Seeds": len(values),
                "MAE_Mean_pp": float(values.mean()),
                "MAE_Std_pp": float(values.std(ddof=1)),
                "MAE_Median_pp": float(np.median(values)),
                "RMSE_Mean_pp": float(
                    frame["validation_current_soh_rmse_pp"].mean()
                ),
                "Bias_Mean_pp": float(
                    frame["validation_current_soh_bias_pp"].mean()
                ),
            })
        architecture_comparison = pd.DataFrame(comparison_rows).sort_values("MAE_Mean_pp")
        architecture_comparison.to_csv(
            TABLE_ROOT / "kan_mlp_kanlite_comparison.csv", index=False
        )
        main_kan_mean = float(architecture_comparison.loc[
            architecture_comparison["Model"] == "KAN", "MAE_Mean_pp"
        ].iloc[0])
        lite_mean = float(architecture_comparison.loc[
            architecture_comparison["Model"] == "KAN-Lite", "MAE_Mean_pp"
        ].iloc[0])
        KAN_LITE_NONINFERIOR = bool(
            lite_mean <= main_kan_mean + KAN_LITE_NONINFERIORITY_MARGIN_PP
        )
        display(architecture_comparison)
        update(
            f"BLOCK 8 complete; KAN-Lite noninferior={KAN_LITE_NONINFERIOR} "
            f"under +{KAN_LITE_NONINFERIORITY_MARGIN_PP:.3f} pp margin"
        )
        """
    ),
    code(
        """
        # BLOCK 9 — Robust-concept synthesis, figures, and Stage-3 workbook
        current_paths = paths[paths["Output_Name"] == "current_soh"].copy()
        current_paths["Top_Half"] = current_paths.groupby("Seed")[
            "Normalized_Path_Importance"
        ].rank(ascending=False, method="min") <= (N_CONCEPTS / 2)
        path_stability = current_paths.groupby("Concept_Name", as_index=False).agg(
            Mean_Path_Importance=("Normalized_Path_Importance", "mean"),
            Path_Top_Half_Frequency=("Top_Half", "mean"),
        )
        ig_stability = integrated_gradients.groupby("Concept_Name", as_index=False).agg(
            Mean_Absolute_IG=("IG_Mean_Absolute", "mean"),
            Mean_Signed_IG=("IG_Mean_Signed", "mean"),
        )
        response_current = curve_stability[
            curve_stability["Output_Name"] == "current_soh"
        ][[
            "Concept_Name", "Majority_Direction", "Direction_Agreement",
            "Median_Spearman", "Median_Absolute_Amplitude", "Stable_Response",
        ]]
        robust = path_stability.merge(ig_stability, on="Concept_Name").merge(
            concept_summary, on="Concept_Name"
        ).merge(response_current, on="Concept_Name")
        robust["Robustly_Important"] = (
            (robust["Path_Top_Half_Frequency"] >= 0.70)
            & (robust["Positive_Frequency"] >= 0.70)
            & robust["Stable_Response"]
        )
        robust = robust.sort_values(
            ["Robustly_Important", "Mean_Path_Importance"], ascending=[False, False]
        )
        robust.to_csv(TABLE_ROOT / "robust_concept_summary.csv", index=False)
        dbsi_names = list(CONCEPT_NAMES[:6])
        dbsi_summary = robust[robust["Concept_Name"].isin(dbsi_names)].copy()
        dbsi_summary.to_csv(TABLE_ROOT / "dbsi_kan_explanation_summary.csv", index=False)

        fig, axis = plt.subplots(figsize=(11, 8))
        plot_frame = robust.sort_values("Mean_Path_Importance")
        colors = np.where(plot_frame["Robustly_Important"], "#e45756", "#72b7b2")
        axis.barh(plot_frame["Concept_Name"], plot_frame["Mean_Path_Importance"], color=colors)
        axis.set_xlabel("Mean normalized KAN path importance")
        axis.set_title("Robust concept importance across five KAN seeds")
        fig.tight_layout()
        fig.savefig(FIGURE_ROOT / "robust_concept_importance.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        for axis, concept in zip(axes.flat, dbsi_names):
            subset = current_curves[current_curves["Concept_Name"] == concept]
            x = subset["Concept_Value"].to_numpy(float)
            mean = subset["Response_Mean"].to_numpy(float)
            lower = subset["Response_Lower"].to_numpy(float)
            upper = subset["Response_Upper"].to_numpy(float)
            axis.plot(x, mean, color="#e45756", linewidth=2)
            axis.fill_between(x, lower, upper, color="#e45756", alpha=0.2)
            axis.set_title(concept)
            axis.set_xlabel("D-BSI component")
            axis.set_ylabel("Residual-scale SOH")
        fig.suptitle("D-BSI component response stability", fontsize=17)
        fig.savefig(FIGURE_ROOT / "dbsi_component_response_curves.png", dpi=180)
        plt.close(fig)

        stage3_summary = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "primary_explainable_model": "DB-PatchKAN",
            "main_kan_frozen": True,
            "kan_lite_role": (
                "interpretable_additive_candidate"
                if KAN_LITE_NONINFERIOR else "interpretability_ablation_only"
            ),
            "kan_lite_noninferiority_margin_pp": KAN_LITE_NONINFERIORITY_MARGIN_PP,
            "robust_concepts": robust.loc[
                robust["Robustly_Important"], "Concept_Name"
            ].tolist(),
            "stable_dbsi_components": dbsi_summary.loc[
                dbsi_summary["Stable_Response"], "Concept_Name"
            ].tolist(),
            "attention_used_as_importance": False,
            "channel_occlusion_pathway": "end_to_end_predicted_concept_pathway",
            "deployment_model_for_nissan": "persistence",
            "untouched_test_used": False,
            "next_stage": "External multi-vehicle data harmonisation and validation",
        }
        (RUN_ROOT / "STAGE3_SUMMARY.json").write_text(json.dumps(stage3_summary, indent=2))

        workbook_path = RUN_ROOT / "DBPatchKAN_Stage3_Explainability.xlsx"
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            checkpoint_audit.to_excel(writer, sheet_name="CHECKPOINTS", index=False)
            robust.to_excel(writer, sheet_name="ROBUST_CONCEPTS", index=False)
            dbsi_summary.to_excel(writer, sheet_name="DBSI_SUMMARY", index=False)
            curve_stability.to_excel(writer, sheet_name="RESPONSE_STABILITY", index=False)
            paths.to_excel(writer, sheet_name="KAN_PATHS", index=False)
            integrated_gradients.to_excel(writer, sheet_name="INTEGRATED_GRADIENTS", index=False)
            concept_summary.to_excel(writer, sheet_name="CONCEPT_ABLATION", index=False)
            channel_summary.to_excel(writer, sheet_name="CHANNEL_OCCLUSION", index=False)
            patch_summary.to_excel(writer, sheet_name="PATCH_OCCLUSION", index=False)
            pruning_summary.to_excel(writer, sheet_name="PRUNING", index=False)
            architecture_comparison.to_excel(writer, sheet_name="KAN_LITE", index=False)
        print(json.dumps(stage3_summary, indent=2), flush=True)
        display(robust)
        update("BLOCK 9 complete")
        """
    ),
    code(
        """
        # BLOCK 10 — Inventory, package, and automatically download all Stage-3 artifacts
        from google.colab import files

        update("BLOCK 10 building artifact inventory")
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

        local_zip = Path("/content") / f"DBPatchKAN_{RUN_ID}_ALL_RESULTS.zip"
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
        update("Starting browser download now")
        files.download(str(local_zip))
        """
    ),
    markdown(
        """
        ## After Stage 3

        Preserve the downloaded ZIP. The next scientifically necessary phase is external
        multi-vehicle harmonisation and leave-one-vehicle-out validation. The locked Nissan test
        remains unopened until all external-data and final-model decisions are frozen.
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
