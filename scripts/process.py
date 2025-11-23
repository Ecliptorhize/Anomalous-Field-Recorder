"""CLI wrapper to process acquired datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from anomalous_field_recorder import process_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Process simulated anomalous field dataset")
    parser.add_argument("raw_dir", type=Path, help="Directory containing raw acquisition artifacts")
    parser.add_argument("processed_dir", type=Path, help="Destination directory for processed output")
    args = parser.parse_args()

    summary = process_dataset(args.raw_dir, args.processed_dir)
    print(f"Processed {summary['records']} records from {args.raw_dir}")


if __name__ == "__main__":
    main()
