# Code Structure and Ownership

The release uses one-way dependencies:

```text
dataset metadata
      |
      v
shared/gait_core
  - pointcloud_dataset.py
  - protocol.py
  - utils.py (retrieval/checkpoint utilities)
  - pointcloud_train.py
      |
      +--> pointnet-tmax/code/pc_v1
      +--> mph-gait/code/mph_gait
      +--> projection-baselines/code/projection_baseline
      +--> lidargaitpp/code/lidargaitpp_adapter
```

## Ownership rule

| Directory | Allowed method-specific content |
|---|---|
| `pointnet-tmax/code/pc_v1/` | PointNet-TMax model, config, thin train entry |
| `mph-gait/code/mph_gait/` | MPH-Gait tokenizer, model, config, thin train entry |
| `projection-baselines/code/projection_baseline/` | Projection dataset, model, configs, trainer, preparation scripts |
| `lidargaitpp/code/lidargaitpp_adapter/` | OpenGait dataset adapter, evaluator, checkpoint selection |
| `shared/gait_core/` | Method-agnostic infrastructure only; no model architecture |

`pc_v1`, `mph_gait`, `projection_baseline`, and `lidargaitpp_adapter` never
import one another. They may import `gait_core`. This keeps comparisons on one
protocol without disguising another method's source as a local dependency.

The public launchers set `PYTHONPATH` to the selected method's `code/` and the
single `shared/` directory. The release validator enforces these ownership
boundaries and fails if an unrelated method package appears inside a method's
`code/` directory.

