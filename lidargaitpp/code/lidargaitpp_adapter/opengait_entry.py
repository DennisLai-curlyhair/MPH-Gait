#!/usr/bin/env python3
"""Launch the pinned OpenGait entry point without editing the vendor tree."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import torch

_TORCHVISION_LIBRARY = None


def ensure_optional_import_stubs() -> None:
    global _TORCHVISION_LIBRARY
    try:
        _TORCHVISION_LIBRARY = torch.library.Library("torchvision", "DEF")
        _TORCHVISION_LIBRARY.define(
            "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
        )
    except RuntimeError as exc:
        if "already" not in str(exc).lower():
            raise

    if "imageio" not in sys.modules:
        imageio = types.ModuleType("imageio")

        def unavailable(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("Unused OpenGait imageio stub was called")

        imageio.imwrite = unavailable
        imageio.mimsave = unavailable
        sys.modules["imageio"] = imageio


def split_launch_args(argv: list[str]) -> tuple[list[str], list[str]]:
    launch_args: list[str] = []
    remaining = list(argv)
    while remaining and (
        remaining[0].startswith("--local-rank")
        or remaining[0].startswith("--local_rank")
    ):
        launch_args.append(remaining.pop(0))
        if launch_args[-1] in {"--local-rank", "--local_rank"} and remaining:
            launch_args.append(remaining.pop(0))
    return launch_args, remaining


def main() -> None:
    launch_args, remaining = split_launch_args(sys.argv[1:])
    if not remaining:
        raise SystemExit(
            "Usage: opengait_entry.py [--local-rank N] <opengait/main.py> [args...]"
        )
    main_path = Path(remaining[0]).resolve()
    sys.path.insert(0, str(main_path.parent))
    sys.argv = [str(main_path), *launch_args, *remaining[1:]]
    ensure_optional_import_stubs()
    runpy.run_path(str(main_path), run_name="__main__")


if __name__ == "__main__":
    main()
