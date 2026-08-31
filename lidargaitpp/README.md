# LidarGait++

This directory contains the project adapter for the official OpenGait
LidarGait++ model. It preserves the official model source and principal recipe,
while adapting Kinect coordinates, Fixed-Special5 identities, validation-only
checkpoint selection, and the common retrieval report.

`code/lidargaitpp_adapter/` contains only this project's OpenGait adapter. The
official checkout is fetched to `third_party/OpenGait`, while the common split
and evaluator are imported from `shared/gait_core/`.

```bash
bash lidargaitpp/scripts/fetch_opengait.sh
WANDB=off bash lidargaitpp/scripts/run_len15_5split_3seed.sh
WANDB=off bash lidargaitpp/scripts/run_followup_5split_seed0.sh
WANDB=off bash lidargaitpp/scripts/run_followup_5split_3seed.sh
```

See `docs/THIRD_PARTY.md` for the pinned commit and verified source hashes.
The follow-up adapter supports variable clip length, clothing filters, and
training-video limits without modifying the pinned official model.
