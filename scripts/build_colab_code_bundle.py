"""Create a small cross-platform source bundle for the Colab notebooks."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


INCLUDE_ROOTS = ("src", "configs")
INCLUDE_FILES = ("pyproject.toml", "README.md", "requirements-gpu.txt")


def build_bundle(project_root: Path, output: Path) -> Path:
    project_root = project_root.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [project_root / name for name in INCLUDE_FILES]
    for directory in INCLUDE_ROOTS:
        files.extend(path for path in (project_root / directory).rglob("*") if path.is_file())
    files = sorted(
        (
            path
            for path in files
            if "__pycache__" not in path.parts and path.suffix.casefold() not in {".pyc", ".pyo"}
        ),
        key=lambda path: path.relative_to(project_root).as_posix(),
    )
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(project_root)
            archive.write(path, arcname=(Path("dbpatchkan_project") / relative).as_posix())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_bundle(args.project_root, args.output))


if __name__ == "__main__":
    main()
