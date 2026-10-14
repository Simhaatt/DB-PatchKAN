"""Command-line orchestration for the reproducible DB-PatchKAN workflow."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .analysis import run_classical_analysis
from .config import PROJECT_ROOT, load_config, stable_hash
from .data import run_nissan_pipeline
from .training import run_synthetic_smoke


DL_COLUMNS = [
    "Run_ID", "Timestamp_UTC", "Environment", "GPU", "Dataset", "Outer_Fold",
    "Inner_Fold", "Seed", "Model_Variant", "Feature_Set", "Config_Hash",
    "Patch_Length", "Patch_Stride", "Embedding_Dim", "Blocks", "Heads",
    "KAN_Grid", "KAN_Shape", "Batch_Size", "Precision", "Best_Epoch",
    "Parameter_Count", "FLOPs", "Train_Time_Min", "Peak_VRAM_GB", "MAE_pp",
    "RMSE_pp", "R2", "Macro_F1", "Interval_Coverage", "Checkpoint_Path",
    "Status", "Notes",
]


def _output_dir(config: dict[str, Any], override: str | Path | None) -> Path:
    return Path(override or config["project"]["output_dir"]).resolve()


def preprocess_command(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Path]:
    output = _output_dir(config, args.output_dir)
    source = Path(args.source or config["data"]["source_csv"]).resolve()
    paths = run_nissan_pipeline(source, output, config=config, quick=args.quick)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return paths


def analyze_command(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Path]:
    output = _output_dir(config, args.output_dir)
    paths = run_classical_analysis(
        output / "trip_features.csv",
        output / "health_windows.csv",
        output,
        random_state=int(config["project"].get("seed", 42)),
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return paths


def smoke_command(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    output = _output_dir(config, args.output_dir)
    seed = int(args.seed if args.seed is not None else config["project"].get("seed", 42))
    run_id = f"smoke-{args.fusion_head}-{seed}"
    smoke_dir = output / "model_smoke" / run_id
    started = time.perf_counter()
    result = run_synthetic_smoke(
        device=args.device,
        output_dir=smoke_dir,
        fusion_head=args.fusion_head,
        seed=seed,
    )
    elapsed = time.perf_counter() - started
    record = {column: None for column in DL_COLUMNS}
    record.update(
        {
            "Run_ID": run_id,
            "Timestamp_UTC": datetime.now(timezone.utc).isoformat(),
            "Environment": f"local-{platform.system().lower()}",
            "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "Dataset": "deterministic synthetic tensors",
            "Seed": seed,
            "Model_Variant": f"DB-PatchKAN-{args.fusion_head.upper()} smoke",
            "Feature_Set": "13 driver + 11 battery channels; 20 named concepts",
            "Config_Hash": stable_hash(config),
            "Patch_Length": 8,
            "Patch_Stride": 4,
            "Embedding_Dim": 32,
            "Blocks": 1,
            "Heads": 4,
            "KAN_Grid": 5 if args.fusion_head == "kan" else None,
            "KAN_Shape": "[20,16,8,3]" if args.fusion_head == "kan" else "[20,32,16,3]",
            "Batch_Size": 4,
            "Precision": result.get("precision", "fp32"),
            "Best_Epoch": result.get("epochs_completed"),
            "Parameter_Count": result.get("parameter_count"),
            "Train_Time_Min": elapsed / 60.0,
            "Peak_VRAM_GB": result.get("peak_vram_gb", 0.0),
            "Checkpoint_Path": str((smoke_dir / "best.pt").resolve()),
            "Status": "passed",
            "Notes": "One-batch pretraining + fine-tuning integration test; not a scientific result.",
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "deep_experiments.csv"
    new = pd.DataFrame([record], columns=DL_COLUMNS)
    if ledger_path.exists():
        previous = pd.read_csv(ledger_path, low_memory=False)
        new = pd.concat([previous, new], ignore_index=True).drop_duplicates(
            subset=["Run_ID"], keep="last"
        )
    new.to_csv(ledger_path, index=False)
    result["ledger_path"] = str(ledger_path)
    result["elapsed_seconds"] = elapsed
    print(json.dumps(result, indent=2, default=str))
    return result


def workbook_command(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    output = _output_dir(config, args.output_dir)
    final = Path(args.workbook or output / "DB_PatchKAN_Research_Workbook.xlsx").resolve()
    source = Path(args.source or config["data"]["source_csv"]).resolve()
    script = PROJECT_ROOT / "scripts" / "build_workbook.py"
    command = [
        sys.executable,
        str(script),
        "--input-dir", str(output),
        "--output", str(final),
        "--config", config["_meta"]["config_path"],
        "--source", str(source),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    print(final)
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Run Nissan QC/features/D-BSI")
    preprocess.add_argument("--config", dest="sub_config")
    preprocess.add_argument("--source")
    preprocess.add_argument("--output-dir")
    preprocess.add_argument("--quick", action="store_true")

    analyze = subparsers.add_parser("analyze", help="Run clustering, selection, and baselines")
    analyze.add_argument("--config", dest="sub_config")
    analyze.add_argument("--output-dir")

    smoke = subparsers.add_parser("smoke-model", help="Run one-batch DB-PatchKAN integration test")
    smoke.add_argument("--config", dest="sub_config")
    smoke.add_argument("--output-dir")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--fusion-head", choices=("kan", "mlp"), default="kan")
    smoke.add_argument("--seed", type=int)

    workbook = subparsers.add_parser("build-workbook", help="Build the 19-sheet Excel artifact")
    workbook.add_argument("--config", dest="sub_config")
    workbook.add_argument("--output-dir")
    workbook.add_argument("--source")
    workbook.add_argument("--workbook")

    all_parser = subparsers.add_parser("run-all", help="Run CPU pipeline, analysis, smoke, and workbook")
    all_parser.add_argument("--config", dest="sub_config")
    all_parser.add_argument("--source")
    all_parser.add_argument("--output-dir")
    all_parser.add_argument("--quick", action="store_true")
    all_parser.add_argument("--device", default="cpu")
    all_parser.add_argument("--fusion-head", choices=("kan", "mlp"), default="kan")
    all_parser.add_argument("--seed", type=int)
    all_parser.add_argument("--workbook")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(getattr(args, "sub_config", None) or args.config)
    if args.command == "preprocess":
        preprocess_command(args, config)
    elif args.command == "analyze":
        analyze_command(args, config)
    elif args.command == "smoke-model":
        smoke_command(args, config)
    elif args.command == "build-workbook":
        workbook_command(args, config)
    elif args.command == "run-all":
        preprocess_command(args, config)
        analyze_command(args, config)
        smoke_command(args, config)
        workbook_command(args, config)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
