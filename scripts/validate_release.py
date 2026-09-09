#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TEXT = (
    "Point_Cloud_Gait_Recognition_V2/",
    "benchmark_baseline_v2/",
    "MPH_Gait_Release/",
    "experiment_propose/",
)
LOCAL_HOME_PATH = re.compile(
    r"""(?:/(?:home|Users)/[^/\s"'<>]+/|[A-Za-z]:\\+Users\\+[^\\\s"'<>]+\\+)"""
)



def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def required_files() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "environment.yml",
        ROOT / "requirements.txt",
        ROOT / "dataset/metadata/fixed_special5_protocol.json",
        ROOT / "pointnet-tmax/code/pc_v1/model.py",
        ROOT / "pointnet-tmax/code/pc_v1/configs/pc_v1_len15.yaml",
        ROOT / "mph-gait/code/mph_gait/model.py",
        ROOT / "mph-gait/code/mph_gait/tokenizer.py",
        ROOT / "mph-gait/code/mph_gait/configs/mph_gait_len15.yaml",
        ROOT / "projection-baselines/code/projection_baseline/model.py",
        ROOT / "projection-baselines/code/projection_baseline/configs/rgb_depth_len15.yaml",
        ROOT / "lidargaitpp/scripts/fetch_opengait.sh",
        ROOT / "lidargaitpp/patches/opengait_runtime_compat.patch",
    ]


def validate_protocol() -> dict[str, Any]:
    manifest = json.loads(
        (ROOT / "dataset/metadata/fixed_special5_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    fixed_ids = manifest["fixed_special_test_ids"]
    groups = manifest["development_groups"]
    config_paths = [
        ROOT / "pointnet-tmax/code/pc_v1/configs/pc_v1_len15.yaml",
        ROOT / "mph-gait/code/mph_gait/configs/mph_gait_len15.yaml",
        ROOT / "projection-baselines/code/projection_baseline/configs/rgb_depth_len15.yaml",
        ROOT / "projection-baselines/code/projection_baseline/configs/gray_depth_len15.yaml",
        ROOT / "projection-baselines/code/projection_baseline/configs/silhouette_len15.yaml",
    ]
    for path in config_paths:
        protocol = load_yaml(path)["protocol"]
        if protocol["fixed_special_test_ids"] != fixed_ids:
            raise RuntimeError(f"Fixed-special IDs differ in {path}")
        if protocol["development_groups"] != groups:
            raise RuntimeError(f"Development groups differ in {path}")
    return {
        "protocol": manifest["name"],
        "fixed_special_test_ids": fixed_ids,
        "development_group_sizes": [len(group) for group in groups],
        "validated_configs": len(config_paths),
    }


def validate_text_paths() -> list[str]:
    failures: list[str] = []
    suffixes = {".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".txt"}
    for path in ROOT.rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if "results" in path.parts or "provenance" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if LOCAL_HOME_PATH.search(text):
            failures.append(f"{path.relative_to(ROOT)}: local home-directory path")
        for marker in FORBIDDEN_TEXT:
            if marker in text:
                failures.append(f"{path.relative_to(ROOT)}: {marker}")
    return failures


def parameter_counts() -> dict[str, int]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return {}

    sys.path.insert(0, str(ROOT / "shared"))
    sys.path.insert(0, str(ROOT / "pointnet-tmax/code"))
    from pc_v1.model import build_model as build_pointnet

    point_cfg = load_yaml(
        ROOT / "pointnet-tmax/code/pc_v1/configs/pc_v1_len15.yaml"
    )
    pointnet = build_pointnet(point_cfg["model"])

    sys.path.insert(0, str(ROOT / "mph-gait/code"))
    from mph_gait.model import build_model as build_mph

    mph_cfg = load_yaml(ROOT / "mph-gait/code/mph_gait/configs/mph_gait_len15.yaml")
    mph = build_mph(mph_cfg["model"])

    sys.path.insert(0, str(ROOT / "projection-baselines/code"))
    from projection_baseline.model import build_model as build_projection

    projection_cfg = load_yaml(
        ROOT
        / "projection-baselines/code/projection_baseline/configs/rgb_depth_len15.yaml"
    )
    projection = build_projection(projection_cfg["model"])

    counts = {
        "pointnet_tmax": sum(p.numel() for p in pointnet.parameters()),
        "mph_gait": sum(p.numel() for p in mph.parameters()),
        "projection_rgb_depth": sum(p.numel() for p in projection.parameters()),
    }
    expected = {
        "pointnet_tmax": 313_664,
        "mph_gait": 405_954,
        "projection_rgb_depth": 7_122_240,
    }
    for key, expected_count in expected.items():
        if counts[key] != expected_count:
            raise RuntimeError(
                f"Parameter count mismatch for {key}: "
                f"{counts[key]} != {expected_count}"
            )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the MPH-Gait release.")
    parser.add_argument(
        "--skip-model-imports",
        action="store_true",
        help="Skip PyTorch model construction and parameter-count checks.",
    )
    args = parser.parse_args()

    missing = [str(path.relative_to(ROOT)) for path in required_files() if not path.is_file()]
    path_failures = validate_text_paths()
    if missing or path_failures:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "missing": missing,
                    "workspace_path_references": path_failures,
                },
                indent=2,
            )
        )
        return 1

    report: dict[str, Any] = {
        "status": "ready",
        "root": str(ROOT),
        "protocol": validate_protocol(),
        "dataset_payload_included": any(
            path.is_file() and path.name != ".gitkeep"
            for base in (
                ROOT / "dataset/local/pointcloud",
                ROOT / "dataset/local/projection",
            )
            for path in base.rglob("*")
        ),
    }
    if not args.skip_model_imports:
        report["parameter_counts"] = parameter_counts()
        if not report["parameter_counts"]:
            report["model_imports"] = "skipped: torch is not installed"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
