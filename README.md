# Data Center Vision-Stream Impact Classification Pipeline

PoC pipeline for classifying data center environmental impact tiers from building records and multi-spectral satellite imagery, with Google Earth Engine extraction, GCS storage, and Vertex AI training.

## Prerequisites

- Conda with the `acm-summer` environment
- Google Cloud project `datacenter-summer-poc` with Earth Engine, Cloud Storage, and Vertex AI APIs enabled
- Application Default Credentials configured for GCP

## Setup

```bash
conda activate acm-summer
pip install -r requirements.txt
gcloud auth application-default login
```

Verify the GCP environment:

```bash
python verify_setup.py
```

## Pipeline Execution

Open and run [`pipeline.ipynb`](pipeline.ipynb) top to bottom:

1. **Environment check** — confirm imports, Earth Engine init, and package versions
2. **Tabular preprocessing** — parse `data/buildings.csv`, upload `parsed_manifest.csv` to GCS
3. **Sentinel-2 extraction** — fetch tiles via Earth Engine and upload directly to GCS (`image_tiles/`, `raw_satellite/`)
4. **GCS alignment QA** — verify tiles exist in the bucket
5. **Tile visualization** — inspect sample tiles per impact tier from GCS
6. **Vertex AI training** — submit Custom Training job (reads from GCS fuse mount)
7. **Post-training evaluation** — download artifacts and review loss curve / confusion matrix

For offline development without Earth Engine, set `USE_MOCK = True` in Section 3.

### Local Training Dry-Run

Download training data from GCS, then run the entrypoint locally:

```bash
conda activate acm-summer
python -c "from src.gcs import download_training_data; download_training_data('/tmp/training')"
python src/vertex_entrypoint.py \
  --epochs 2 \
  --batch-size 4 \
  --training /tmp/training \
  --model-dir /tmp/model
```

## GCS Layout (bucket root)

| Path | Contents |
|------|----------|
| `gs://datacenter-summer-poc-data/parsed_manifest.csv` | Training manifest |
| `gs://datacenter-summer-poc-data/image_tiles/` | `.npy` tensor files |
| `gs://datacenter-summer-poc-data/raw_satellite/` | PNG previews and metadata |
| `gs://datacenter-summer-poc-data/output/models/{run_id}/` | Model weights and eval artifacts |

### Training Artifacts (per run)

| File | Description |
|------|-------------|
| `model.pth` | Trained ResNet-34 weights |
| `metrics.json` | Hyperparameters, class distribution, epoch history |
| `metrics.csv` | Tabular epoch log |
| `loss_curve.png` | Train/val loss plot |
| `confusion_matrix.png` | Validation confusion matrix |

## Vertex AI Configuration

| Parameter | Value |
|-----------|-------|
| Project | `datacenter-summer-poc` |
| Region | `us-central1` |
| Entry point | `src/vertex_entrypoint.py` |
| GCS fuse root | `/gcs/datacenter-summer-poc-data` |
| Machine | `n1-standard-4` + `NVIDIA_TESLA_T4` |
| Container | `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0.py310:latest` |
| Hyperparameters | `epochs=10`, `batch-size=8` |

## Code Quality

```bash
conda activate acm-summer
ruff check . --fix
ruff format .
ty check .
```

## Repository Layout

```text
├── pipeline.ipynb          # Orchestrator notebook
├── verify_setup.py         # GCP environment validation
├── data/
│   ├── buildings.csv       # Source building records
│   └── projects.csv        # Supporting campus records
├── src/
│   ├── model_def.py
│   ├── vertex_entrypoint.py
│   ├── gcs.py
│   ├── eval.py
│   ├── preprocessing.py
│   ├── satellite.py
│   ├── eda.py
│   └── logging_config.py
└── artifacts/              # Downloaded training runs (gitignored)
```

See [`spec.md`](spec.md) for the full system specification.
