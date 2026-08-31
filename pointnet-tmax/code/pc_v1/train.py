from __future__ import annotations

from pathlib import Path

from gait_core.pointcloud_train import main
from pc_v1.model import build_model


PACKAGE_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    main(
        model_builder=build_model,
        default_config="pc_v1/configs/pc_v1_len15.yaml",
        method_name="PointNet-TMax",
        provenance_sources=(PACKAGE_ROOT / "model.py", Path(__file__)),
    )

