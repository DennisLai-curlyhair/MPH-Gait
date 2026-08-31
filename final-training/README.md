# Final 24-Identity Training

This protocol trains release checkpoints on all 24 non-fixed identities while
keeping P003, P007, P018, P019, and P026 for C1 and C4/C5/C7 evaluation.

Unlike the five-split development protocol, final training does not use a
held-out validation fold for checkpoint selection. Each method uses a fixed
training budget and reports the final checkpoint:

- PointNet-TMax: 50 epochs
- MPH-Gait: 50 epochs
- Adapted LidarGait++: 10,000 iterations

Run three seeds:

```bash
WANDB=off bash final-training/scripts/run_pointnet_tmax_final24_3seed.sh
WANDB=off bash final-training/scripts/run_mph_gait_final24_3seed.sh

bash lidargaitpp/scripts/fetch_opengait.sh
bash final-training/scripts/run_lidargaitpp_final24_3seed.sh
```

Run all methods sequentially and create a comparison report:

```bash
WANDB=off bash final-training/scripts/run_all_final24_3seed.sh
```

Outputs are written to `outputs/final-training/`. Curated summaries from the
completed study are retained under `results/completed_e3_5split_3seed/`.
