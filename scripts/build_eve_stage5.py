"""Download and harmonise the public EVE Pilot signals for Stage 5B."""

from __future__ import annotations

import argparse
from pathlib import Path

from dbpatchkan.data.eve import build_eve_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    summary = build_eve_stage(
        args.raw_dir.resolve(),
        args.output_dir.resolve(),
        download=not args.no_download,
    )
    print(summary)


if __name__ == "__main__":
    main()
