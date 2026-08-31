# Shared experiment infrastructure

`gait_core/` contains infrastructure used by more than one method:

- point-cloud dataset discovery and loading;
- the Fixed-Special5 subject-disjoint protocol;
- retrieval metrics and checkpoint utilities;
- the common point-cloud training loop.

No model architecture is defined here. Method-specific models remain under
`pointnet-tmax/code/pc_v1`, `mph-gait/code/mph_gait`, `projection-baselines/code/projection_baseline`,
and `lidargaitpp/code/lidargaitpp_adapter`.

`tools/make_followup_config.py` and `scripts/run_pointcloud_followup.sh` provide
the shared Fixed-Special5 ablation orchestration used by PointNet-TMax and MPH-Gait.
They contain no model implementation.
