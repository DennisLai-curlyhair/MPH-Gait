# MPH-Gait

MPH-Gait augments the PointNet-TMax global path with a multi-scale local point
hierarchy. Shared point features are grouped into 16 fine and four coarse
tokens per frame, processed by one within-frame Transformer layer, and fused
with the global descriptor through norm-matched bounded residual fusion. The
model has 405,954 trainable parameters and retains a 256-dimensional retrieval
descriptor.

`code/mph_gait/` contains the method-specific tokenizer, model, base config,
and training entry point. Dataset parsing, protocol construction, retrieval
metrics, and common training logic are provided by `shared/gait_core/`.

Run commands from the repository root:

```bash
WANDB=off bash mph-gait/scripts/run_len15_5split_3seed.sh
WANDB=off bash mph-gait/scripts/run_followup_5split_seed0.sh
WANDB=off bash mph-gait/scripts/run_followup_5split_3seed.sh
```

See [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) for clip-length, training
clothing, training-video-count, and matched-seed commands.
