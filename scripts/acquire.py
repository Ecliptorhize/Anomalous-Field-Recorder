"""CLI wrapper to simulate field data acquisition."""

from __future__ import annotations

import argparse
from pathlib import Path

from anomalous_field_recorder import simulate_acquisition


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate anomalous field acquisition")
    parser.add_argument("config", type=Path, help="Path to experiment configuration (YAML or JSON)")
    parser.add_argument("output", type=Path, help="Directory where simulated metadata will be stored")
    args = parser.parse_args()

    metadata = simulate_acquisition(args.config, args.output)
    print(f"Wrote metadata with {len(metadata.get('config', {}))} config entries to {args.output}")


if __name__ == "__main__":
    main()
