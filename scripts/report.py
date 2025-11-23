"""CLI wrapper to generate human-readable reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from anomalous_field_recorder import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown report from processed data")
    parser.add_argument("processed_dir", type=Path, help="Directory containing processed outputs")
    parser.add_argument("--output", type=Path, help="Optional path for the generated report")
    args = parser.parse_args()

    report_path = generate_report(args.processed_dir, args.output)
    print(f"Generated report at {report_path}")


if __name__ == "__main__":
    main()
