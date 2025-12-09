"""Command-line interface for AFR."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .pipeline import PipelineConfig, run_pipeline
from .sensors import SimulatedSensor
from .export import export_run


def _parse_duration(text: str) -> float:
    text = text.strip().lower()
    if text.endswith("ms"):
        return float(text[:-2]) / 1000.0
    if text.endswith("s"):
        return float(text[:-1])
    if text.endswith("m"):
        return float(text[:-1]) * 60.0
    return float(text)


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping/object")
    return data


def cmd_record(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    duration = _parse_duration(cfg.get("duration", args.duration))
    sample_rate = float(cfg.get("sample_rate", args.sample_rate))
    sensor = SimulatedSensor(
        sample_rate=sample_rate,
        baseline=float(cfg.get("baseline", 0.0)),
        noise=float(cfg.get("noise", 0.2)),
        anomaly_magnitude=float(cfg.get("anomaly_magnitude", 4.0)),
        anomaly_chance=float(cfg.get("anomaly_chance", 0.02)),
    )
    df = sensor.to_dataframe(duration_s=duration)
    output = Path(args.output or cfg.get("output") or "datasets/recording.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"[record] wrote {len(df)} samples to {output}")


def cmd_process(args: argparse.Namespace) -> None:
    cfg = PipelineConfig.from_file(args.config) if args.config else PipelineConfig()
    if args.sample_rate:
        cfg.sample_rate = float(args.sample_rate)
    output_dir = Path(args.output or "data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(args.dataset, output_dir, cfg)
    print(f"[process] anomalies={len(result['anomalies'])} json={result['report']['output_json']} pdf={result['report']['output_pdf']}")


def cmd_simulate(args: argparse.Namespace) -> None:
    duration = _parse_duration(args.duration)
    sensor = SimulatedSensor(
        sample_rate=float(args.sample_rate),
        anomaly_chance=float(args.anomaly_chance),
        anomaly_magnitude=float(args.anomaly_magnitude),
        noise=float(args.noise),
    )
    df = sensor.to_dataframe(duration_s=duration)
    output = Path(args.output or f"datasets/simulated_{int(time.time())}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"[simulate] saved {len(df)} samples to {output}")
    if args.plot:
        png_path = sensor.visualize(duration_s=duration, output_path=args.plot)
        print(f"[simulate] plot saved to {png_path}")


def cmd_report(args: argparse.Namespace) -> None:
    result_path = Path(args.result)
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    anomalies = payload.get("anomalies", [])
    report = payload.get("report", {})
    print(f"[report] source={payload.get('metadata', {}).get('source', 'unknown')} samples={payload.get('n_samples')} anomalies={len(anomalies)}")
    print(f"[report] markdown={report.get('output_markdown')} pdf={report.get('output_pdf')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anomalous Field Recorder CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="Record (simulated) data from a config file")
    record.add_argument("--config", required=False, help="YAML config describing the sensor simulation")
    record.add_argument("--output", required=False, help="Where to store the captured CSV")
    record.add_argument("--duration", default="10s", help="Duration (e.g. 10s, 2m)")
    record.add_argument("--sample-rate", default=200.0, type=float, help="Override sample rate for simulation")
    record.set_defaults(func=cmd_record)

    process = sub.add_parser("process", help="Run the offline processing pipeline on a dataset")
    process.add_argument("dataset", help="Path to the dataset CSV")
    process.add_argument("--config", help="Pipeline config (YAML or JSON)")
    process.add_argument("--output", help="Output directory for reports")
    process.add_argument("--sample-rate", type=float, help="Override sample rate")
    process.set_defaults(func=cmd_process)

    simulate = sub.add_parser("simulate", help="Generate synthetic data with anomalies")
    simulate.add_argument("--duration", default="5s", help="Duration (e.g. 30s)")
    simulate.add_argument("--sample-rate", default=200.0, type=float, help="Sample rate in Hz")
    simulate.add_argument("--noise", default=0.2, type=float, help="Noise standard deviation")
    simulate.add_argument("--anomaly-chance", default=0.02, type=float, help="Chance per sample of anomaly injection")
    simulate.add_argument("--anomaly-magnitude", default=4.0, type=float, help="Magnitude of anomalies")
    simulate.add_argument("--output", help="Output CSV path")
    simulate.add_argument("--plot", help="Save a quick-look plot to this path")
    simulate.set_defaults(func=cmd_simulate)

    report = sub.add_parser("report", help="Summarize an existing pipeline result")
    report.add_argument("result", help="Path to result JSON generated by afr process")
    report.set_defaults(func=cmd_report)

    export = sub.add_parser("export-run", help="Package a processed run directory into a shareable archive")
    export.add_argument("run_dir", help="Path to processed run directory containing result.json")
    export.add_argument("--output", help="Destination .zip (default: run_dir/export.zip)")
    export.add_argument(
        "--formats",
        nargs="+",
        help="Dataset export formats (csv, parquet, hdf5). Defaults to csv.",
    )
    export.set_defaults(
        func=lambda args: print(
            f"[export] archive={export_run(args.run_dir, args.output or Path(args.run_dir) / 'export.zip', args.formats)}"
        )
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
