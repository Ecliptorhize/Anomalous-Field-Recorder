"""Command line interface for Anomalous Field Recorder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import load_experiment_config
from .pipeline import process_dataset, simulate_acquisition
from .reporting import generate_report


def _print_result(result: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afr",
        description="Run acquisition, processing, and reporting for multi-discipline data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire = subparsers.add_parser("acquire", help="Simulate acquisition from a config file")
    acquire.add_argument("config", type=Path, help="Path to experiment configuration (YAML or JSON)")
    acquire.add_argument("output", type=Path, help="Directory to store simulated metadata")
    acquire.add_argument("--json", action="store_true", help="Emit metadata as JSON to stdout")

    process = subparsers.add_parser("process", help="Process a raw dataset into a summary")
    process.add_argument("raw_dir", type=Path, help="Directory containing raw acquisition artifacts")
    process.add_argument("processed_dir", type=Path, help="Destination directory for processed output")
    process.add_argument("--json", action="store_true", help="Emit summary as JSON to stdout")

    report = subparsers.add_parser("report", help="Generate a Markdown report from processed data")
    report.add_argument("processed_dir", type=Path, help="Directory containing processed outputs")
    report.add_argument("--output", type=Path, help="Optional path for the generated report")
    report.add_argument("--json", action="store_true", help="Emit report path as JSON to stdout")

    describe = subparsers.add_parser("describe", help="Inspect a configuration without running")
    describe.add_argument("config", type=Path, help="Path to experiment configuration (YAML or JSON)")
    describe.add_argument("--json", action="store_true", help="Emit configuration as JSON to stdout")

    subparsers.add_parser("version", help="Display the installed version")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "acquire":
        metadata = simulate_acquisition(args.config, args.output)
        if args.json:
            _print_result(metadata, as_json=True)
        else:
            print(
                f"Wrote metadata with {len(metadata.get('config', {}))} entries "
                f"to {args.output} (domain={metadata.get('domain_profile', {}).get('domain', 'unknown')})"
            )
    elif args.command == "process":
        summary = process_dataset(args.raw_dir, args.processed_dir)
        if args.json:
            _print_result(summary, as_json=True)
        else:
            print(
                f"Processed {summary['records']} records from {args.raw_dir} "
                f"(domain={summary.get('domain', 'unknown')})"
            )
    elif args.command == "report":
        report_path = generate_report(args.processed_dir, args.output)
        if args.json:
            _print_result({"report": str(report_path)}, as_json=True)
        else:
            print(f"Generated report at {report_path}")
    elif args.command == "describe":
        config = load_experiment_config(args.config)
        if args.json:
            _print_result(config, as_json=True)
        else:
            print(f"Config keys: {', '.join(sorted(config.keys())) or '(empty config)'}")
    elif args.command == "version":
        print(__version__)


if __name__ == "__main__":
    main()

