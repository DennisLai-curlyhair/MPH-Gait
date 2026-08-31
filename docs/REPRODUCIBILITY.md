# Reproducibility

## Data

```bash
bash dataset/scripts/link_local_data.sh /path/to/pointcloud /path/to/projection
```

Public archives should follow `dataset/metadata/dataset_schema.yaml`; update
`dataset/download_links.yaml` with URLs and SHA-256 values.

## Formal Runs

```bash
WANDB=off bash projection/scripts/run_len15_5split_3seed.sh
WANDB=off bash pc_v1/scripts/run_len15_5split_3seed.sh
WANDB=off bash mph_gait/scripts/run_len15_5split_3seed.sh

bash lidargaitpp/scripts/fetch_opengait.sh
WANDB=off bash lidargaitpp/scripts/run_len15_5split_3seed.sh
```

Restrict a smoke run with environment variables:

```bash
SEEDS=0 SPLITS=0 DRY_RUN=1 WANDB=off \
  bash mph_gait/scripts/run_len15_5split_3seed.sh
```

## Checkpoints

Each method's `checkpoints/checkpoint_manifest.csv` defines the external release
filename, seed, split, expected bytes, and SHA-256. Binary weights are omitted
from Git history and should be placed according to `release_filename` after
download.

## Integrity

```bash
conda run -n pytorch python docs/tools/validate_release.py
```

The source lineage is in `docs/provenance/source_lineage.json` and the complete
release file manifest is in `docs/provenance/release_manifest.json`.
