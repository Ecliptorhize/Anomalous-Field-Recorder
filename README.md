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

Notebook references:

- `notebooks/01_quickstart.ipynb` – end-to-end acquisition → processing → report walkthrough.
- `notebooks/02_neuro_bandpower.ipynb` – EEG/LFP-style synthetic signal generation and bandpower analysis.
- `notebooks/03_neuro_connectivity.ipynb` – coherence, phase-locking value, and event-locked peaks for synthetic EEG.

### CLI Quickstart

The new `afr` CLI wraps the end-to-end pipeline (acquire → preprocess → filter → detect → report):

```bash
# Generate synthetic data (noise + injected anomalies)
afr simulate --duration 30s --sample-rate 200 --output datasets/simulated.csv --plot data/plots/simulated.png

# Record (simulated) from a YAML config describing the sensor parameters
afr record --config docs/configs/simulated.yaml --output data/raw/run01.csv

# Run the offline pipeline and emit JSON + Markdown + PDF reports
afr process data/raw/run01.csv --config docs/configs/pipeline.yaml --output data/reports/run01

# Summarize an existing result.json
afr report data/reports/run01/result.json
```

## Processing Pipeline (Acquire → Preprocess → Filter → Detect → Report)

- **Pipeline core** – `src/afr/pipeline.py` loads CSV datasets, interpolates gaps, applies a configurable filter chain, runs multiple detectors, and writes `result.json`, `report.md`, and a plotted `report.pdf`.
- **Filters** – Butterworth (low/high/band), notch, bandpass, and smoothing filters live in `src/afr/filters/`. Compose them via YAML/JSON config and reuse them in the real-time stack.
- **Validated models** – Pydantic-backed `Recording`, `RecordingMetadata`, and `AnomalyEvent` models enforce clean inputs and make JSON outputs predictable.
- **Detectors** – Statistical: z-score, rolling mean/variance drift, MAD. ML: IsolationForest, One-Class SVM, PyTorch autoencoder. All are available through config or directly via the anomaly engine registry.
- **Config** – Use `PipelineConfig` to set `sample_rate`, `window_size`, `stride`, `filters`, and `detectors`. See `docs/configs/pipeline.yaml` (add your own) for a ready-to-run template.
- **Reports** – Markdown + PDF plots (raw/filtered traces with anomaly markers) plus structured JSON for downstream tooling.

## Virtual Sensors and Sample Data

- Use `afr simulate` or `afr record` to generate realistic streams without hardware. The simulator supports adjustable sample rates, noise, and anomaly injection, and can export quick-look PNGs.
- The `datasets/` folder ships with `synthetic_1.csv` and `synthetic_2.csv` so contributors can run the pipeline immediately.
- For notebooks or batch jobs, import `SimulatedSensor` from `afr.sensors` to produce DataFrames and live streams programmatically.

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

## Cross-Domain Profiles

Sample configurations covering field engineering, medical imaging, clinical lab, and chemistry lab pipelines live in `docs/experiments/`:

- `field-baseline.yml` – tri-axis magnetometer baseline capture
- `medical-imaging.yml` – MRI metadata stub
- `clinical-lab.yml` – CBC/hematology analyzer run
- `chemistry-lab.yml` – LC-MS pesticide screen
- `neuro-eeg.yml` – computational neuroscience EEG session

Processing will report domain and instrument details along with quality flags highlighting missing required or suggested metadata.

## Advanced Capabilities

- **Validation** – Domain-aware schema validation with defaults via `afr validate <config>`.
- **Signal analysis** – Synthetic sample generation during acquisition, band/notch filters, FFT-derived spectral summaries, anomaly scoring, and rich Markdown reports.
- **Neuroscience bandpower** – Welch PSD-derived bandpower (delta/theta/alpha/beta/gamma) for EEG/LFP datasets is included in processed summaries and reports.
- **Registry** – SQLite-backed run registry (`--registry path/to/db.sqlite`) plus `afr runs` to inspect history.
- **API** – FastAPI service `afr serve --host 0.0.0.0 --port 8000` exposing `/health`, `/version`, `/validate`, `/normalize`, `/analyze`, `/synth`, `/acquire`, `/process`, `/report`, and `/runs`.
- **Structured logging** – Enable JSON logs with `AFR_JSON_LOGS=true` or `--json-logs` on CLI.

## Real-Time Anomaly Platform

AFR now ships with a modular, streaming-oriented architecture:

- **Streaming daemon** – `afr stream --config docs/configs/realtime-sqlite.yaml` spins up `StreamingService`, which buffers live samples, applies a `RealTimeFilterChain`, and forwards windows into the anomaly engine.
- **Pluggable detectors** – Unified `AnomalyEngine` accepts detectors declared in YAML (z-score, spectral bandpower, CUSUM change-point, PyTorch autoencoder). Add your own by registering factories on `AnomalyEngine`.
- **Storage abstraction** – `SQLiteBackend` for lightweight local storage and a `TimescaleBackend` (optional `psycopg2-binary`) for high-throughput deployments. Events are also forwarded through registry plugins for traceability.
- **Dashboard/API** – FastAPI dashboard (`afr.dashboard.create_dashboard_app`) now serves a lightweight ECharts view over WebSockets (live charts + anomaly flags), dataset browser, and run history endpoints. Embed alongside the main service for quick visibility.
- **Config-first** – See `docs/configs/realtime-sqlite.yaml` and `docs/configs/realtime-timescale.yaml` for end-to-end YAML examples covering filters, detectors, and storage.
- **Extras** – Install optional integrations with `pip install .[timescale,torch]` to enable the TimescaleDB backend and the PyTorch autoencoder detector.

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

## Container Usage

Build and run with Docker:

```bash
docker compose build
docker compose run --rm afr afr acquire docs/experiments/field-baseline.yml /app/data/raw/field-baseline
docker compose run --rm afr afr process /app/data/raw/field-baseline /app/data/processed/field-baseline
docker compose run --rm afr afr report /app/data/processed/field-baseline
```

The compose file mounts `./data` and `./docs` into the container for sharing configs and artifacts.

## Community Health

- Review the [Code of Conduct](CODE_OF_CONDUCT.md).
- Follow the [Contributing Guide](CONTRIBUTING.md) for development workflows.
- Report security issues according to the [Security Policy](SECURITY.md).

## License

This project is licensed under the terms of the [MIT License](LICENSE).
