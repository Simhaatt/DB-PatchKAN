"""Configuration and reproducibility helpers for DB-PatchKAN."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load YAML configuration and resolve project-relative paths.

    Paths remain serialisable strings in the returned mapping.  The source and
    output locations are absolute so notebooks and command-line runs behave the
    same regardless of their current directory.
    """

    config_path = Path(path or PROJECT_ROOT / "configs" / "default.yaml").resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = deepcopy(config)
    base_dir = config_path.parent.parent
    source = Path(config["data"]["source_csv"])
    output = Path(config["project"]["output_dir"])
    config["data"]["source_csv"] = str(source if source.is_absolute() else base_dir / source)
    config["project"]["output_dir"] = str(
        output if output.is_absolute() else base_dir / output
    )
    config["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(base_dir),
    }
    return config


def stable_hash(value: Any, length: int = 12) -> str:
    """Return a stable compact SHA-256 hash for JSON-compatible content."""

    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def source_manifest(
    source_path: str | os.PathLike[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an immutable source/config/software manifest."""

    source = Path(source_path).resolve()
    project_root = Path(config.get("_meta", {}).get("project_root", PROJECT_ROOT))
    stat = source.stat()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "source_size_bytes": stat.st_size,
        "source_modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "source_sha256": sha256_file(source),
        "config_hash": stable_hash(config),
        "git_revision": _git_revision(project_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


__all__ = ["PROJECT_ROOT", "load_config", "sha256_file", "source_manifest", "stable_hash"]
