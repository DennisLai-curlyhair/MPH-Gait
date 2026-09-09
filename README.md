# MPH-Gait

Official research code for **MPH-Gait: Lightweight Multi-Scale Point-Hierarchy
Gait Identification from Raw Point Clouds**.

The repository provides the proposed MPH-Gait encoder, the PointNet-TMax
baseline, three cylindrical projection baselines, an official-code adapted
LidarGait++ comparison, the fixed-special5 subject-disjoint protocol, and
reproducible experiment launchers.

## Research Reproducibility Notice

This repository is published to support academic evaluation and reproduction
of the reported experiments. Third-party projects, model implementations,
names, and trademarks remain the property of their respective owners and are
governed by their original terms. This project does not claim ownership of or
grant additional rights to those materials. See
[`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) for source and version details.

## Method Overview

| Method | Input | Parameters | Descriptor |
|---|---|---:|---:|
| PointNet-TMax | axis-aligned XYZ point sequence | 313,664 | 256 |
| MPH-Gait | axis-aligned XYZ point sequence | 405,954 | 256 |
| Adapted LidarGait++ | official OpenGait point representation | approximately 2.447M | 7,936 |
| Projection baselines | RGB-depth, gray-depth, or silhouette sequence | approximately 7.12M | 4,096 |

MPH-Gait reuses PointNet-TMax per-point features to form 16 fine and four
coarse spatial tokens per frame. One within-frame Transformer layer models
multi-scale local geometry. Norm-matched bounded residual fusion adds the
local correction to the global PointNet-TMax descriptor without introducing a
second point backbone.

## Repository Layout

```text
MPH_Gait/
├── pointnet-tmax/          # lightweight direct point-cloud baseline
├── mph-gait/               # proposed multi-scale point-hierarchy method
├── lidargaitpp/            # OpenGait adapter, pinned-source fetcher, patch
├── projection-baselines/   # RGB-depth, gray-depth, silhouette baselines
├── shared/                 # protocol, datasets, trainer, retrieval metrics
├── dataset/                # schema, split manifest, and download metadata
├── final-training/         # 24-identity release-weight training protocol
├── docs/                   # architecture, protocol, experiments, results
├── scripts/                # repository validation and aggregate launchers
├── environment.yml
└── requirements.txt
```

Method-specific source is isolated under each method's `code/` directory.
All methods share the same protocol and retrieval endpoints through
`shared/gait_core/`.

## Installation

Only load trusted datasets and checkpoints. See [Security and data handling](SECURITY.md)
for checkpoint-loading risks, dependency advisories, and sensitive-file handling.

Linux with an NVIDIA GPU is recommended. LidarGait++ training requires CUDA.

```bash
conda env create -f environment.yml
conda activate mph-gait

# Install a PyTorch build compatible with the local CUDA driver.
# Example for CUDA 12.1:
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

python scripts/validate_release.py
```

A pip workflow is also supported after installing a compatible PyTorch wheel:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python scripts/validate_release.py
```

For the LidarGait++ comparison, fetch the pinned official OpenGait source and
apply the included runtime compatibility patch:

```bash
bash lidargaitpp/scripts/fetch_opengait.sh
```

The fetcher checks out commit
`f754f6f3831e9f83bb28f4e2f63dd43d8bcf9dc4` and verifies the official
LidarGait++ source hashes.

## Dataset

The dataset is intentionally not included in Git because of its size. Place or
link the released data under:

```text
dataset/local/pointcloud/
dataset/local/projection/
```

Download `dataset.tar.gz`, `dataset_proj.tar.gz`, and `SHA256SUMS` from the
[Google Drive dataset folder](https://drive.google.com/drive/folders/1Ec63IgVLfJkezbmzXADklmkVndpL73V2?usp=drive_link).
Verify the checksums and extract the archives outside the repository, then
run `dataset/scripts/link_local_data.sh` as documented in
[`dataset/README.md`](dataset/README.md). Archive metadata are recorded in
[`dataset/download_links.yaml`](dataset/download_links.yaml), and the
fixed-special5 split definition is provided in `dataset/metadata/`.
Access/consent terms: **TBD**.

For access to original acquisition files (raw data, RGB images, and depth
images), please contact [dennis.y.lai@gmail.com](mailto:dennis.y.lai@gmail.com).

No launcher uses train, validation, or test identities outside the committed
protocol manifest. The fixed special-clothing identities are P003, P007, P018,
P019, and P026.

## Main Experiments

Run commands from the repository root. Weights & Biases logging is disabled in
the examples.

### T=15, Five Splits, Three Matched Seeds

```bash
WANDB=off bash pointnet-tmax/scripts/run_len15_5split_3seed.sh
WANDB=off bash mph-gait/scripts/run_len15_5split_3seed.sh

bash lidargaitpp/scripts/fetch_opengait.sh
WANDB=off bash lidargaitpp/scripts/run_len15_5split_3seed.sh

WANDB=off bash projection-baselines/scripts/run_len15_5split_3seed.sh
```

Each run stores a config snapshot, protocol manifest, checkpoints, per-split
retrieval metrics, and cross-seed summaries under the method's ignored
`outputs/` directory.

### Clip-Length Experiments

```bash
SUITES=clip_length LENGTHS="1 4 8 30" SEEDS="0 1 2" \
WANDB=off bash pointnet-tmax/scripts/run_followup_5split_3seed.sh

SUITES=clip_length LENGTHS="1 4 8 30" SEEDS="0 1 2" \
WANDB=off bash mph-gait/scripts/run_followup_5split_3seed.sh

SUITES=clip_length LENGTHS="1 4 8 30" SEEDS="0 1 2" \
WANDB=off bash lidargaitpp/scripts/run_followup_5split_3seed.sh

SUITES=clip_length LENGTHS="1 4 8 30" SEEDS="0 1 2" \
WANDB=off bash projection-baselines/scripts/run_followup_5split_3seed.sh
```

The formal matched-seed operating points use `T=15` and `T=30`. Shorter
lengths may be run as seed-0 diagnostics by replacing the launcher suffix with
`run_followup_5split_seed0.sh`.

### Training-Clothing Experiments

```bash
SUITES=train_clothing TRAIN_VARIANTS="c2 c3" SEEDS="0 1 2" \
WANDB=off bash pointnet-tmax/scripts/run_followup_5split_3seed.sh
```

Use the same environment variables with the MPH-Gait, LidarGait++, or
projection launcher. `c2c3_full` is the T=15 main setting and is not repeated
by default.

### Training-Video-Count Experiments

```bash
SUITES=train_video_count MAX_VIDEO_IDS="1 2 4 8" SEEDS="0 1 2" \
WANDB=off bash mph-gait/scripts/run_followup_5split_3seed.sh
```

`MAX_VIDEO_IDS=1 2 4 8` corresponds to 2, 4, 8, and 16 training videos per
identity across C2 and C3. The full T=15 main setting uses 32 videos per
identity.

Use `DRY_RUN=1 SPLITS=0 SEEDS=0` with any launcher to generate configs and
inspect the command matrix without training.

## Final 24-Identity Training

The five fixed special-clothing identities remain evaluation-only. The other
24 identities are used to train release checkpoints under a fixed budget.

```bash
WANDB=off bash final-training/scripts/run_pointnet_tmax_final24_3seed.sh
WANDB=off bash final-training/scripts/run_mph_gait_final24_3seed.sh
bash lidargaitpp/scripts/fetch_opengait.sh
bash final-training/scripts/run_lidargaitpp_final24_3seed.sh

# Sequential aggregate launcher and comparison report
WANDB=off bash final-training/scripts/run_all_final24_3seed.sh
```

## Results and Reproducibility

Compact, auditable results are stored under each method's `results/` folder
and in [`docs/results/main_results.csv`](docs/results/main_results.csv).
Binary checkpoints remain outside Git history. The formal T=15 release is
packaged as six method-specific archives containing 90 original checkpoints
and matching run metadata. See [Checkpoint distribution](docs/CHECKPOINTS.md)
for filenames, verification, model loading, and the separate Final-24
deployment bundles. Download the archives from the
[checkpoint folder](https://drive.google.com/drive/folders/1LXsMvUE-496sVhIrpKOs6q_Fyo0gRbsW?usp=sharing)
under `checkpoints/`.

Protocol details, metric definitions, experiment matrices, and architecture
diagrams are available in:

- [Protocol](docs/PROTOCOL.md)
- [Experiments](docs/EXPERIMENTS.md)
- [Architectures](docs/ARCHITECTURES.md)
- [Results](docs/RESULTS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Third-party components](docs/THIRD_PARTY.md)

## Citation and License

Publication citation metadata: **TBD** (`CITATION.cff.template`).

Project license: **TBD** (`LICENSE_PENDING.md`).
