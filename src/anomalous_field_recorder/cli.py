"""Command line interface for Anomalous Field Recorder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import load_experiment_config, validate_config, validate_config_file
from .logging_utils import configure_logging
from .pipeline import process_dataset, simulate_acquisition
from .registry import list_runs
from .reporting import generate_report
from .service import create_app
from .signals import (
    apply_filters,
    compute_bandpower,
    compute_signal_metrics,
    compute_spectral_entropy,
    compute_spectral_metrics,
    generate_multichannel_eeg,
    generate_synthetic_series,
    ingest_samples,
    score_anomalies,
)


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
    parser.add_argument("--registry", type=Path, help="Path to registry database (optional)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    parser.add_argument("--json-logs", action="store_true", help="Emit JSON logs")

    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire = subparsers.add_parser("acquire", help="Simulate acquisition from a config file")
    acquire.add_argument("config", type=Path, help="Path to experiment configuration (YAML or JSON)")
    acquire.add_argument("output", type=Path, help="Directory to store simulated metadata")
    acquire.add_argument("--duration-s", type=float, default=1.0, help="Synthetic capture duration")
    acquire.add_argument("--no-samples", action="store_true", help="Skip synthetic sample generation")
    acquire.add_argument("--json", action="store_true", help="Emit metadata as JSON to stdout")

    process = subparsers.add_parser("process", help="Process a raw dataset into a summary")
    process.add_argument("raw_dir", type=Path, help="Directory containing raw acquisition artifacts")
    process.add_argument("processed_dir", type=Path, help="Destination directory for processed output")
    process.add_argument("--band", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Bandpass filter range")
    process.add_argument("--notch", type=float, help="Notch filter frequency")
    process.add_argument("--json", action="store_true", help="Emit summary as JSON to stdout")

    report = subparsers.add_parser("report", help="Generate a Markdown report from processed data")
    report.add_argument("processed_dir", type=Path, help="Directory containing processed outputs")
    report.add_argument("--output", type=Path, help="Optional path for the generated report")
    report.add_argument("--json", action="store_true", help="Emit report path as JSON to stdout")

    describe = subparsers.add_parser("describe", help="Inspect a configuration without running")
    describe.add_argument("config", type=Path, help="Path to experiment configuration (YAML or JSON)")
    describe.add_argument("--json", action="store_true", help="Emit configuration as JSON to stdout")
    describe.add_argument("--validate", action="store_true", help="Validate the configuration")

    validate = subparsers.add_parser("validate", help="Validate a configuration file")
    validate.add_argument("config", type=Path, help="Path to configuration file")
    validate.add_argument("--json", action="store_true", help="Emit validation result as JSON")

    analyze = subparsers.add_parser("analyze", help="Analyze a sample file for metrics and anomalies")
    analyze.add_argument("samples", type=Path, help="Path to CSV/JSON/JSONL samples")
    analyze.add_argument("--sample-rate", type=float, default=1000.0, help="Sample rate of the data")
    analyze.add_argument("--value-column", type=str, help="Column name for CSV inputs")
    analyze.add_argument("--band", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Bandpass filter range")
    analyze.add_argument("--notch", type=float, help="Notch filter frequency")
    analyze.add_argument("--anomaly-threshold", type=float, default=3.5, help="Z-score threshold for anomalies")
    analyze.add_argument("--output", type=Path, help="Optional path to write analysis JSON")
    analyze.add_argument("--json", action="store_true", help="Emit analysis as JSON")

    synth = subparsers.add_parser("synth", help="Generate synthetic samples for testing")
    synth.add_argument("--duration-s", type=float, default=1.0, help="Duration of synthetic capture")
    synth.add_argument("--sample-rate", type=float, default=1000.0, help="Sample rate for synthetic capture")
    synth.add_argument(
        "--component",
        action="append",
        nargs=2,
        metavar=("FREQ", "AMPLITUDE"),
        type=float,
        help="Sinusoid component (freq Hz, amplitude). Repeat to add multiple components.",
    )
    synth.add_argument("--noise-std", type=float, default=0.02, help="Noise standard deviation")
    synth.add_argument(
        "--channels",
        type=int,
        default=0,
        help="Number of EEG-like channels to generate (0 for single-channel)",
    )
    synth.add_argument("--output", type=Path, help="Optional path to write synthetic samples (JSON)")
    synth.add_argument("--json", action="store_true", help="Emit generated samples as JSON to stdout")

    normalize = subparsers.add_parser("normalize", help="Normalize a config and apply defaults")
    normalize.add_argument("config", type=Path, help="Path to experiment configuration (YAML or JSON)")
    normalize.add_argument("--output", type=Path, help="Optional path to write normalized config (JSON)")
    normalize.add_argument(
        "--include-messages",
        action="store_true",
        help="Include validation errors and warnings alongside normalized config",
    )
    normalize.add_argument("--json", action="store_true", help="Emit normalized config as JSON")

    serve = subparsers.add_parser("serve", help="Run FastAPI service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    runs = subparsers.add_parser("runs", help="List recent runs from the registry")
    runs.add_argument("--limit", type=int, default=20)
    runs.add_argument("--json", action="store_true", help="Emit runs as JSON")

    subparsers.add_parser("version", help="Display the installed version")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level, json_logs=args.json_logs)

    if args.command == "acquire":
        metadata = simulate_acquisition(
            args.config,
            args.output,
            generate_samples=not args.no_samples,
            duration_s=args.duration_s,
            registry_path=args.registry,
        )
        if args.json:
            _print_result(metadata, as_json=True)
        else:
            print(
                f"Wrote metadata with {len(metadata.get('config', {}))} entries "
                f"to {args.output} (domain={metadata.get('domain_profile', {}).get('domain', 'unknown')})"
            )
    elif args.command == "process":
        summary = process_dataset(
            args.raw_dir,
            args.processed_dir,
            band=tuple(args.band) if args.band else None,
            notch=args.notch,
            registry_path=args.registry,
        )
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
        if args.validate:
            validation = validate_config(config)
            config = {"config": config, "validation": validation.as_dict()}
        if args.json:
            _print_result(config, as_json=True)
        else:
            keys = config["config"].keys() if isinstance(config, dict) and "config" in config else config.keys()
            print(f"Config keys: {', '.join(sorted(keys)) or '(empty config)'}")
    elif args.command == "validate":
        result = validate_config_file(args.config)
        _print_result(result.as_dict(), as_json=args.json)
    elif args.command == "analyze":
        samples = ingest_samples(args.samples, value_column=args.value_column)
        band = tuple(args.band) if args.band else None
        filtered = apply_filters(samples, args.sample_rate, band=band, notch=args.notch) if (band or args.notch) else samples
        metrics = compute_signal_metrics(filtered)
        spectral = compute_spectral_metrics(filtered, args.sample_rate)
        anomalies = score_anomalies(filtered, z_threshold=args.anomaly_threshold)
        bandpower = compute_bandpower(filtered, args.sample_rate)
        entropy = compute_spectral_entropy(filtered, args.sample_rate)
        analysis = {
            "metrics": metrics,
            "spectral": spectral,
            "bandpower": bandpower,
            "spectral_entropy": entropy,
            "anomalies": anomalies,
            "filters": {"band": args.band, "notch": args.notch},
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
            print(f"Wrote analysis to {args.output}")
        else:
            _print_result(analysis, as_json=args.json)
    elif args.command == "synth":
        if args.channels < 0:
            raise SystemExit("Number of channels cannot be negative.")
        components = [{"freq": freq, "amplitude": amp} for freq, amp in args.component] if args.component else None
        if args.channels:
            channels, events = generate_multichannel_eeg(
                num_channels=args.channels,
                duration_s=args.duration_s,
                sample_rate=args.sample_rate,
                base_components=components,
                noise_std=args.noise_std,
            )
            payload: Any = {
                "channels": channels,
                "events_s": events,
                "sample_rate": args.sample_rate,
                "duration_s": args.duration_s,
            }
            summary_msg = f"{len(channels)} channels with {len(events)} events"
        else:
            samples = generate_synthetic_series(
                duration_s=args.duration_s,
                sample_rate=args.sample_rate,
                components=components,
                noise_std=args.noise_std,
            )
            payload = {"samples": samples, "sample_rate": args.sample_rate, "duration_s": args.duration_s}
            summary_msg = f"{len(samples)} samples"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Wrote synthetic data to {args.output} ({summary_msg})")
        else:
            _print_result(payload, as_json=args.json)
    elif args.command == "normalize":
        config = load_experiment_config(args.config)
        validation = validate_config(config)
        payload = validation.normalized
        if args.include_messages:
            payload = {
                "normalized": validation.normalized,
                "errors": validation.errors,
                "warnings": validation.warnings,
                "domain": validation.domain,
            }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Wrote normalized config to {args.output}")
        else:
            _print_result(payload, as_json=args.json)
    elif args.command == "serve":
        app = create_app(registry_path=args.registry)
        try:
            import uvicorn
        except ModuleNotFoundError:
            raise SystemExit("uvicorn is required to run the service. Install with `pip install uvicorn`.")
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "runs":
        if not args.registry:
            raise SystemExit("Specify --registry to read runs.")
        runs = [r.__dict__ for r in list_runs(args.registry, limit=args.limit)]
        _print_result({"runs": runs}, as_json=args.json)
    elif args.command == "version":
        print(__version__)


if __name__ == "__main__":
    main()
