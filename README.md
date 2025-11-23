# Anomalous Field Recorder

Anomalous Field Recorder is a research-oriented toolkit for logging, analyzing, and visualizing anomalous electromagnetic field measurements. The repository provides the scaffolding required to reproduce experiments, process incoming sensor data, and publish results in a transparent and reproducible manner.

## Scientific Setup

The project assumes access to a field acquisition rig equipped with:

- **Sensor Array** – Three-axis magnetometers capable of sampling at 1 kHz or higher, a broadband electric-field probe, and environmental baselines (temperature, humidity).
- **Acquisition Hardware** – An embedded computer (e.g., Raspberry Pi 4) connected via USB or SPI to the sensors.
- **Timing Reference** – GPS-disciplined clock or NTP-synchronized time source to align field recordings across deployments.
- **Power Supply** – Regulated 5V/12V supply with surge protection.
 
### Field Deployment Checklist

1. Verify sensor calibration using the calibration script in `tools/calibration/`.
2. Confirm accurate timestamp synchronization with the GPS/NTP service.
3. Run a 5-minute baseline capture to ensure noise levels fall within acceptable limits.
4. Store raw data to redundant storage (local SSD + remote S3 bucket).

### Laboratory Processing Environment

- Python 3.11+
- Poetry for dependency management (`pip install poetry`)
- Optional GPU for accelerated spectral analysis (CUDA 12 or ROCm 5)

Set up the environment:

```bash
poetry install
poetry run pytest
```

Use the provided Jupyter notebooks in `notebooks/` to explore recorded sessions. Each notebook is tagged with metadata describing the sensor configuration used.

## Repository Structure

```
.
├── data/                 # Raw and processed field captures (git-ignored)
├── docs/                 # Project documentation and experiment logs
├── notebooks/            # Exploratory analysis and visualization
├── scripts/              # Command-line utilities for acquisition and processing
├── src/                  # Library code for signal processing, filtering, and storage
└── tests/                # Automated test suite
```

> **Note:** Some directories are placeholders until the corresponding modules are implemented. Ensure `.gitignore` rules prevent accidental commits of large raw datasets.

## Usage Overview

1. Configure experiment metadata in `docs/experiments/<experiment-id>.yml`.
2. Launch the acquisition script:
   ```bash
   poetry run python scripts/acquire.py --config docs/experiments/test-field.yml
   ```
3. Process the resulting dataset:
   ```bash
   poetry run python scripts/process.py data/raw/test-field
   ```
4. Generate plots and reports from notebooks or run:
   ```bash
   poetry run python scripts/report.py --input data/processed/test-field
   ```

## Continuous Integration

CI should run the following jobs:

- Linting: `poetry run ruff check .`
- Static typing: `poetry run mypy src`
- Testing: `poetry run pytest`

## Community Health

- Review the [Code of Conduct](CODE_OF_CONDUCT.md).
- Follow the [Contributing Guide](CONTRIBUTING.md) for development workflows.
- Report security issues according to the [Security Policy](SECURITY.md).

## License

This project is licensed under the terms of the [MIT License](LICENSE).

