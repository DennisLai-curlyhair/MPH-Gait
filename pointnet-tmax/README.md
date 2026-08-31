# PointNet-TMax

PointNet-TMax is the lightweight direct point-cloud baseline used by MPH-Gait.
Each frame is encoded by a shared point-wise MLP, followed by symmetric point
max pooling and temporal max pooling. A 256-dimensional BNNeck descriptor is
used for cosine retrieval. The model has 313,664 trainable parameters.

The public folder name and reported method name are `pointnet-tmax`. The
internal Python package remains `pc_v1` to preserve checkpoint state-dict and
runtime compatibility.

Run commands from the repository root:

```bash
WANDB=off bash pointnet-tmax/scripts/run_len15_5split_3seed.sh
WANDB=off bash pointnet-tmax/scripts/run_followup_5split_seed0.sh
WANDB=off bash pointnet-tmax/scripts/run_followup_5split_3seed.sh
```

The follow-up launcher accepts `SUITES`, `LENGTHS`, `TRAIN_VARIANTS`,
`MAX_VIDEO_IDS`, `SEEDS`, and `SPLITS`. See
[`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) for the complete matrix.
