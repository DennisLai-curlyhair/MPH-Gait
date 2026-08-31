from __future__ import annotations

from pathlib import Path

from gait_core.pointcloud_train import main
from mph_gait.model import build_model


PACKAGE_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    main(
        model_builder=build_model,
        default_config="mph_gait/configs/mph_gait_len15.yaml",
        method_name="MPH-Gait",
        provenance_sources=(
            PACKAGE_ROOT / "model.py",
            PACKAGE_ROOT / "tokenizer.py",
            Path(__file__),
        ),
    )

