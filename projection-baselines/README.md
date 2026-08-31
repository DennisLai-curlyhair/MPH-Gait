# Projection Baselines

This package contains RGB-depth, gray-depth, and silhouette gait baselines.
All representations use the same ResNet9-style frame encoder, temporal max
pooling, horizontal pyramid pooling, part-wise heads, and fixed-special5
gallery/probe evaluator. Only the projected representation and input channel
count differ.

Run commands from the repository root:

```bash
# T=15, all three representations, 5 splits x 3 seeds
WANDB=off bash projection-baselines/scripts/run_len15_5split_3seed.sh

# Seed-0 diagnostic curves
WANDB=off bash projection-baselines/scripts/run_followup_5split_seed0.sh

# Complete matched-seed follow-up matrix
WANDB=off bash projection-baselines/scripts/run_followup_5split_3seed.sh
```

Select representations or experiment suites with environment variables:

```bash
METHODS="rgb_depth silhouette" \
SUITES="clip_length" LENGTHS="1 4 8 30" \
WANDB=off bash projection-baselines/scripts/run_followup_5split_3seed.sh

METHODS="rgb_depth gray_depth silhouette" \
SUITES="train_clothing" TRAIN_VARIANTS="c2 c3" \
WANDB=off bash projection-baselines/scripts/run_followup_5split_3seed.sh

METHODS="rgb_depth gray_depth silhouette" \
SUITES="train_video_count" MAX_VIDEO_IDS="1 2 4 8" \
WANDB=off bash projection-baselines/scripts/run_followup_5split_3seed.sh
```

See [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) for protocol semantics and
the mapping from `MAX_VIDEO_IDS` to videos per identity.
