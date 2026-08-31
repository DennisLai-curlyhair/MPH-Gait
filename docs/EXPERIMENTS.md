# Experiment Matrix

All experiments use the committed
`fixed_special5_subject_holdout_5split_v1` identity protocol and the same
gallery/probe endpoints. Run commands from the repository root.

## Methods and Launchers

| Method | T=15 matched seeds | Follow-up launcher |
|---|---|---|
| PointNet-TMax | `pointnet-tmax/scripts/run_len15_5split_3seed.sh` | `pointnet-tmax/scripts/run_followup_5split_3seed.sh` |
| MPH-Gait | `mph-gait/scripts/run_len15_5split_3seed.sh` | `mph-gait/scripts/run_followup_5split_3seed.sh` |
| Adapted LidarGait++ | `lidargaitpp/scripts/run_len15_5split_3seed.sh` | `lidargaitpp/scripts/run_followup_5split_3seed.sh` |
| Projection baselines | `projection-baselines/scripts/run_len15_5split_3seed.sh` | `projection-baselines/scripts/run_followup_5split_3seed.sh` |

LidarGait++ requires `bash lidargaitpp/scripts/fetch_opengait.sh` before its
first run.

## Shared Runtime Controls

The shell launchers accept the following environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SEEDS` | `0 1 2` for 3-seed launchers | matched random seeds |
| `SPLITS` | `0 1 2 3 4` | subject-disjoint development splits |
| `WANDB` | `off` | optional experiment logging |
| `DRY_RUN` | `0` | generate/print settings without training |
| `SKIP_COMPLETED` | `1` | preserve completed split results on restart |
| `EXTRA_ARGS` | empty | additional trainer arguments |
| `RUN_STAMP` | current timestamp | deterministic output group name when supplied |

Projection launchers additionally accept `METHODS`, whose values are
`rgb_depth`, `gray_depth`, and `silhouette`.

## Main Operating Point

The main model comparison uses:

```text
clip length:               15
points per frame:          1024
training clothing:         C2 + C3 normal
training videos/identity:  32 (16 C2 + 16 C3)
splits x seeds:            5 x 3
checkpoint selection:      validation C1 query-micro mAP
```

Example:

```bash
WANDB=off bash mph-gait/scripts/run_len15_5split_3seed.sh
```

## Clip Length

```bash
SUITES=clip_length LENGTHS="1 4 8 30" SEEDS="0 1 2" \
WANDB=off bash mph-gait/scripts/run_followup_5split_3seed.sh
```

The T=15 main run is already produced by the main launcher. Include 15 in
`LENGTHS` only when an independently named repeated run is required.

For a lower-cost diagnostic:

```bash
SUITES=clip_length LENGTHS="1 4 8 30" \
WANDB=off bash mph-gait/scripts/run_followup_5split_seed0.sh
```

## Training Clothing

```bash
SUITES=train_clothing TRAIN_VARIANTS="c2 c3" SEEDS="0 1 2" \
WANDB=off bash mph-gait/scripts/run_followup_5split_3seed.sh
```

| Variant | Training samples |
|---|---|
| `c2` | C2 normal, V001-V016 |
| `c3` | C3 normal, V001-V016 |
| `c2c3_full` | C2+C3 normal, V001-V016; main setting |
| `c2c3_balanced` | C2+C3 normal, V001-V008; equivalent to the 16-video setting |

## Training Videos per Identity

```bash
SUITES=train_video_count MAX_VIDEO_IDS="1 2 4 8" SEEDS="0 1 2" \
WANDB=off bash pointnet-tmax/scripts/run_followup_5split_3seed.sh
```

The limit is applied independently to C2 and C3:

| `MAX_VIDEO_IDS` | C2 videos | C3 videos | Total/identity |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 2 |
| 2 | 2 | 2 | 4 |
| 4 | 4 | 4 | 8 |
| 8 | 8 | 8 | 16 |
| 16 | 16 | 16 | 32 (main setting) |

## Matched-Seed Completion

To run clothing and video-count cells for seeds 1 and 2 after a seed-0
diagnostic:

```bash
SEEDS="1 2" SUITES="train_clothing train_video_count" \
WANDB=off bash mph-gait/scripts/run_followup_5split_3seed.sh
```

The same interface applies to PointNet-TMax, LidarGait++, and projections.
For all methods sequentially:

```bash
SEEDS="1 2" WANDB=off \
bash final-training/scripts/run_all_missing_clothing_video_seeds_1_2.sh
```

## Dry Run and Partial Restart

```bash
DRY_RUN=1 SEEDS=0 SPLITS=0 SUITES=clip_length LENGTHS=15 \
WANDB=off bash mph-gait/scripts/run_followup_5split_3seed.sh

SPLITS="3 4" SEEDS=2 SKIP_COMPLETED=1 \
WANDB=off bash pointnet-tmax/scripts/run_followup_5split_3seed.sh
```

Generated configs and all runtime outputs are stored below the corresponding
method's `outputs/` directory and are excluded from version control.

## Result Interpretation

Report both query-micro and subject-macro metrics:

- **Query-micro mAP** gives equal weight to each probe sequence.
- **Subject-macro mAP** first averages probes per identity and then gives each
  identity equal weight.
- **Rank-1** is the primary deployed identification outcome.
- **mAP** measures the full ranked list and is not interchangeable with Rank-1.

The fixed-five cohort has unequal probe counts, so subject-macro mAP is the
primary person-balanced special-clothing ranking metric.
