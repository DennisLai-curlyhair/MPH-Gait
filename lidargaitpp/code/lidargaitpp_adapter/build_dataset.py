#!/usr/bin/env python3
"""Prepare axis-aligned Kinect point clouds for official LidarGait++.

This v2 adapter does not modify the pinned official model source. It maps Azure
Kinect camera coordinates to the metric LiDAR convention expected by the official
size-aware pipeline, writes subject-disjoint manifests, and creates OpenGait configs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(__file__).resolve().parents[3]

from gait_core.pointcloud_dataset import collect_pointcloud_videos
from gait_core.protocol import build_protocol


COORDINATE_ADAPTER = "kinect_xyz_mm_to_lidar_forward_lateral_height_m_v1"
INPUT_AXES = ["camera_x_horizontal", "camera_y_vertical_down", "camera_z_forward"]
OUTPUT_AXES = ["forward", "lateral", "height_up"]


def normalize_experiment_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    if not tag:
        raise ValueError("experiment tag must contain an alphanumeric character")
    return tag


@dataclass
class CoordinateAudit:
    """Streaming audit for the semantic axis and unit conversion."""

    unit_scale: float
    frames: int = 0
    empty_frames: int = 0
    points: int = 0
    raw_min: np.ndarray = field(
        default_factory=lambda: np.full(3, np.inf, dtype=np.float64)
    )
    raw_max: np.ndarray = field(
        default_factory=lambda: np.full(3, -np.inf, dtype=np.float64)
    )
    output_min: np.ndarray = field(
        default_factory=lambda: np.full(3, np.inf, dtype=np.float64)
    )
    output_max: np.ndarray = field(
        default_factory=lambda: np.full(3, -np.inf, dtype=np.float64)
    )
    height_span_sum: float = 0.0
    height_span_min: float = float("inf")
    height_span_max: float = 0.0
    height_over_2m_points: int = 0

    def update(self, raw_points: np.ndarray, output_points: np.ndarray) -> None:
        self.frames += 1
        if len(raw_points) == 0:
            self.empty_frames += 1
            return
        self.points += int(len(raw_points))
        self.raw_min = np.minimum(self.raw_min, raw_points.min(axis=0))
        self.raw_max = np.maximum(self.raw_max, raw_points.max(axis=0))
        self.output_min = np.minimum(self.output_min, output_points.min(axis=0))
        self.output_max = np.maximum(self.output_max, output_points.max(axis=0))

        ground_height = output_points[:, 2] - float(output_points[:, 2].min())
        height_span = float(ground_height.max())
        self.height_span_sum += height_span
        self.height_span_min = min(self.height_span_min, height_span)
        self.height_span_max = max(self.height_span_max, height_span)
        self.height_over_2m_points += int(np.count_nonzero(ground_height > 2.0))

    def as_dict(self) -> dict[str, Any]:
        nonempty_frames = self.frames - self.empty_frames
        finite = self.points > 0
        return {
            "adapter": COORDINATE_ADAPTER,
            "input_axes": INPUT_AXES,
            "input_unit": "millimeter",
            "output_axes": OUTPUT_AXES,
            "output_unit": "meter",
            "mapping": "[forward, lateral, height_up] = [Z, X, -Y] * unit_scale",
            "unit_scale": float(self.unit_scale),
            "frames": int(self.frames),
            "empty_frames": int(self.empty_frames),
            "points": int(self.points),
            "raw_axis_min": self.raw_min.tolist() if finite else None,
            "raw_axis_max": self.raw_max.tolist() if finite else None,
            "output_axis_min": self.output_min.tolist() if finite else None,
            "output_axis_max": self.output_max.tolist() if finite else None,
            "ground_relative_height_span_m": {
                "min": float(self.height_span_min) if nonempty_frames else None,
                "mean": (
                    float(self.height_span_sum / nonempty_frames)
                    if nonempty_frames
                    else None
                ),
                "max": float(self.height_span_max) if nonempty_frames else None,
            },
            "ground_relative_height_over_2m_fraction": (
                float(self.height_over_2m_points / self.points)
                if self.points
                else None
            ),
            "official_hap_height_range_m": [0.0, 2.0],
        }

    def validate(self) -> None:
        if self.points == 0:
            raise RuntimeError("Coordinate audit found no finite point-cloud points")
        nonempty_frames = self.frames - self.empty_frames
        mean_height = self.height_span_sum / max(1, nonempty_frames)
        over_2m_fraction = self.height_over_2m_points / self.points
        if not 0.5 <= mean_height <= 2.5:
            raise RuntimeError(
                "Axis/unit conversion produced an implausible mean body-height span: "
                f"{mean_height:.4f} m"
            )
        if over_2m_fraction > 0.10:
            raise RuntimeError(
                "More than 10% of ground-relative heights exceed the official 2 m "
                f"HAP range: {over_2m_fraction:.4%}"
            )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sample_id(sample: dict[str, Any]) -> str:
    condition = str(sample["condition_tag"])
    condition_suffix = "" if condition == "normal" else f"_{condition}"
    return (
        f"P{int(sample['person_id']):03d}_"
        f"C{int(sample['clothes_id'])}{condition_suffix}_"
        f"V{int(sample['video_id']):03d}"
    )


def serializable_sample(sample: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "id": sample_id(sample),
        "role": role,
        "path": str(sample["video_dir"]),
        "person_id": int(sample["person_id"]),
        "label": int(sample["label"]),
        "clothes_id": int(sample["clothes_id"]),
        "condition_tag": str(sample["condition_tag"]),
        "video_id": int(sample["video_id"]),
        "protocol_split": str(sample["protocol_split"]),
    }


def sample_matches(sample: dict[str, Any], filters: dict[str, Any]) -> bool:
    clothes_ids = filters.get("clothes_ids")
    if clothes_ids is not None and int(sample["clothes_id"]) not in {
        int(value) for value in clothes_ids
    }:
        return False
    excluded = filters.get("exclude_clothes_ids")
    if excluded is not None and int(sample["clothes_id"]) in {
        int(value) for value in excluded
    }:
        return False
    condition_tags = filters.get("condition_tags")
    if condition_tags is not None and str(sample["condition_tag"]) not in {
        str(value) for value in condition_tags
    }:
        return False
    video_ids = filters.get("video_ids")
    if video_ids is not None and int(sample["video_id"]) not in {
        int(value) for value in video_ids
    }:
        return False
    min_video_id = filters.get("min_video_id")
    if min_video_id is not None and int(sample["video_id"]) < int(min_video_id):
        return False
    max_video_id = filters.get("max_video_id")
    if max_video_id is not None and int(sample["video_id"]) > int(max_video_id):
        return False
    protocol_splits = filters.get("protocol_splits")
    if protocol_splits is not None and str(sample["protocol_split"]) not in {
        str(value) for value in protocol_splits
    }:
        return False
    return True


def ids_from_source(source: str | list[int], protocol: dict[str, Any]) -> set[int]:
    if isinstance(source, list):
        return {int(value) for value in source}
    if source == "all":
        return {int(value) for value in protocol["all_ids"]}
    if source not in protocol:
        raise KeyError(f"Unknown protocol ID source: {source}")
    return {int(value) for value in protocol[source]}


def side_filters(pair_cfg: dict[str, Any], side: str) -> dict[str, Any]:
    filters = dict(pair_cfg.get(f"{side}_filters", {}) or {})
    for key in (
        "clothes_ids",
        "exclude_clothes_ids",
        "condition_tags",
        "video_ids",
        "min_video_id",
        "max_video_id",
        "protocol_splits",
    ):
        legacy_key = f"{side}_{key}"
        if legacy_key in pair_cfg and key not in filters:
            filters[key] = pair_cfg[legacy_key]
    return filters


def build_pairs(
    samples: list[dict[str, Any]],
    protocol: dict[str, Any],
    retrieval_cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for name, pair_cfg in retrieval_cfg.get("pairs", {}).items():
        default_source = "val_ids" if name.startswith("val") else "mixed_test_ids"
        shared_source = pair_cfg.get("id_source", default_source)
        gallery_source = pair_cfg.get("gallery_id_source", shared_source)
        probe_source = pair_cfg.get("probe_id_source", shared_source)
        gallery_ids = ids_from_source(gallery_source, protocol)
        probe_ids = ids_from_source(probe_source, protocol)

        gallery_splits = pair_cfg.get(
            "gallery_splits", pair_cfg.get("gallery_split")
        )
        probe_splits = pair_cfg.get("probe_splits", pair_cfg.get("probe_split"))
        gallery_split_set = {
            str(value) for value in ([gallery_splits] if isinstance(gallery_splits, str) else gallery_splits)
        }
        probe_split_set = {
            str(value) for value in ([probe_splits] if isinstance(probe_splits, str) else probe_splits)
        }
        gallery_filters = side_filters(pair_cfg, "gallery")
        probe_filters = side_filters(pair_cfg, "probe")

        gallery = [
            sample_id(sample)
            for sample in samples
            if int(sample["person_id"]) in gallery_ids
            and str(sample["protocol_split"]) in gallery_split_set
            and sample_matches(sample, gallery_filters)
        ]
        probe = [
            sample_id(sample)
            for sample in samples
            if int(sample["person_id"]) in probe_ids
            and str(sample["protocol_split"]) in probe_split_set
            and sample_matches(sample, probe_filters)
        ]
        if not gallery or not probe:
            raise RuntimeError(
                f"Pair {name!r} is empty: gallery={len(gallery)}, probe={len(probe)}"
            )
        gallery_people = {
            int(sample["person_id"])
            for sample in samples
            if sample_id(sample) in set(gallery)
        }
        probe_people = {
            int(sample["person_id"])
            for sample in samples
            if sample_id(sample) in set(probe)
        }
        missing = sorted(probe_people - gallery_people)
        if missing:
            raise RuntimeError(f"Pair {name!r} lacks gallery IDs for probes: {missing}")
        pairs[name] = {
            "gallery": sorted(gallery),
            "probe": sorted(probe),
            "subset_gallery_to_probe_ids": bool(
                pair_cfg.get("subset_gallery_to_probe_ids", False)
            ),
            "gallery_id_source": gallery_source,
            "probe_id_source": probe_source,
        }
    return pairs


def kinect_xyz_mm_to_lidar_xyz_m(
    points: np.ndarray,
    unit_scale: float,
) -> np.ndarray:
    """Map Kinect camera XYZ to forward/lateral/up LiDAR axes in meters."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point array [N, 3], got {points.shape}")
    if not np.isclose(unit_scale, 0.001):
        raise ValueError(
            "Axis-aligned v2 is intentionally fixed to millimeter-to-meter "
            f"conversion (0.001), got {unit_scale}"
        )
    output = np.stack(
        [points[:, 2], points[:, 0], -points[:, 1]],
        axis=1,
    )
    return np.asarray(output * unit_scale, dtype=np.float32)


def load_frame(
    path: Path,
    unit_scale: float,
    coordinate_audit: CoordinateAudit,
) -> np.ndarray:
    raw_points = np.asarray(np.load(path)[:, :3], dtype=np.float32)
    raw_points = raw_points[np.isfinite(raw_points).all(axis=1)]
    points = kinect_xyz_mm_to_lidar_xyz_m(raw_points, unit_scale)
    coordinate_audit.update(raw_points, points)
    return points


def usable_paths(sample: dict[str, Any], drop_first_frames: int) -> list[Path]:
    paths = list(sample["frame_paths"])
    usable = paths[max(0, int(drop_first_frames)) :]
    return usable if usable else paths


def uniform_eval_paths(paths: list[Path], clip_len: int) -> list[Path]:
    if not paths:
        return []
    if len(paths) >= clip_len:
        indices = np.linspace(0, len(paths) - 1, clip_len).round().astype(np.int64)
        return [paths[int(index)] for index in indices]
    return [*paths, *([paths[-1]] * (clip_len - len(paths)))]


def deterministic_eval_points(points: np.ndarray, points_num: int) -> np.ndarray:
    """Select a repeatable fixed-size point set for protocol evaluation."""
    if points_num <= 0:
        raise ValueError(f"points_num must be positive, got {points_num}")
    if len(points) == 0:
        return np.zeros((points_num, 3), dtype=np.float32)
    if len(points) >= points_num:
        indices = np.linspace(0, len(points) - 1, points_num).astype(np.int64)
    else:
        indices = np.arange(points_num, dtype=np.int64) % len(points)
    return np.asarray(points[indices], dtype=np.float32)


def sequence_frames(
    sample: dict[str, Any],
    drop_first_frames: int,
    eval_clip_len: int | None,
    points_num: int,
    unit_scale: float,
    coordinate_audit: CoordinateAudit,
) -> list[np.ndarray]:
    paths = usable_paths(sample, drop_first_frames)
    if eval_clip_len is not None:
        paths = uniform_eval_paths(paths, eval_clip_len)
    frames = [
        load_frame(path, unit_scale, coordinate_audit) for path in paths
    ]
    if eval_clip_len is not None:
        frames = [
            deterministic_eval_points(frame, points_num) for frame in frames
        ]
    return frames if frames else [np.zeros((0, 3), dtype=np.float32)]


def opengait_type(sample: dict[str, Any], role: str) -> str:
    clothes_id = int(sample["clothes_id"])
    if role == "train":
        return f"train-C{clothes_id}"
    if clothes_id == 1:
        return "probe-C1-personal"
    if clothes_id in {4, 5, 7}:
        return f"probe-C{clothes_id}-{sample['condition_tag']}"
    return f"gallery-C{clothes_id}"


def write_sequence(
    data_root: Path,
    sample: dict[str, Any],
    role: str,
    drop_first_frames: int,
    clip_len: int,
    points_num: int,
    unit_scale: float,
    coordinate_audit: CoordinateAudit,
) -> Path:
    label = f"{int(sample['person_id']):03d}"
    output = (
        data_root
        / label
        / opengait_type(sample, role)
        / sample_id(sample)
        / "00-data.pkl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = sequence_frames(
        sample,
        drop_first_frames,
        eval_clip_len=None if role == "train" else clip_len,
        points_num=points_num,
        unit_scale=unit_scale,
        coordinate_audit=coordinate_audit,
    )
    with output.open("wb") as handle:
        pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output


def write_partition(path: Path, train_ids: list[int], test_ids: list[int]) -> None:
    save_json(
        path,
        {
            "TRAIN_SET": [f"{person_id:03d}" for person_id in train_ids],
            "TEST_SET": [f"{person_id:03d}" for person_id in test_ids],
        },
    )


def opengait_config(
    *,
    dataset_name: str,
    data_root: Path,
    partition: Path,
    save_name: str,
    class_num: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "data_cfg": {
            "dataset_name": dataset_name,
            "dataset_root": str(data_root.resolve()),
            "dataset_partition": str(partition.resolve()),
            "num_workers": int(args.num_workers),
            "cache": False,
            "data_in_use": [True] + [False] * 15,
            "remove_no_gallery": False,
            "test_dataset_name": "SUSTech1K",
        },
        "evaluator_cfg": {
            "enable_float16": False,
            "restore_ckpt_strict": True,
            "restore_hint": 0,
            "save_name": save_name,
            "eval_func": "evaluate_indoor_dataset",
            "sampler": {
                "batch_shuffle": False,
                "points_in_use": {
                    "pointcloud_index": 0,
                    # Evaluation PKLs contain deterministic fixed-size points.
                    "points_num": None,
                },
                "batch_size": int(args.eval_batch_size),
                "frames_num_fixed": int(args.clip_len),
                "frames_skip_num": 0,
                "sample_type": "all_ordered",
                "frames_all_limit": int(args.clip_len),
                "type": "InferenceSampler",
            },
            "metric": "euc",
            "transform": [
                {
                    "type": "PointCloudsTransform",
                    "xyz_only": True,
                    "scale_aware": True,
                }
            ],
        },
        "loss_cfg": [
            {
                "loss_term_weight": 1.0,
                "margin": 0.2,
                "type": "TripletLoss",
                "log_prefix": "triplet",
                "lazy": False,
            },
            {
                "loss_term_weight": 0.1,
                "scale": 31,
                "type": "CrossEntropyLoss",
                "log_prefix": "softmax",
                "log_accuracy": True,
            },
        ],
        "model_cfg": {
            "model": "LidarGaitPlusPlus",
            "pool": "PPP_HAP",
            "sampling": "knn",
            "channel": 16,
            "npoints": [512, 256, 128],
            "nsample": 32,
            "scale_aware": True,
            "normalize_dp": True,
            "SeparateFCs": {
                "in_channels": 256,
                "out_channels": 256,
                "parts_num": 31,
            },
            "SeparateBNNecks": {
                "class_num": int(class_num),
                "in_channels": 256,
                "parts_num": 31,
            },
            "scale": [1, 2, 4, 8, 16],
        },
        "optimizer_cfg": {
            "lr": float(args.learning_rate),
            "momentum": 0.9,
            "solver": "SGD",
            "weight_decay": 0.0005,
        },
        "scheduler_cfg": {
            "T_max": int(args.total_iter),
            "eta_min": 0.0001,
            "scheduler": "CosineAnnealingLR",
        },
        "trainer_cfg": {
            "enable_float16": False,
            "fix_BN": False,
            "with_test": False,
            "log_iter": int(args.log_iter),
            "restore_ckpt_strict": True,
            "restore_hint": 0,
            "save_iter": int(args.save_iter),
            "save_name": save_name,
            "sync_BN": False,
            "find_unused_parameters": False,
            "total_iter": int(args.total_iter),
            "sampler": {
                "batch_shuffle": True,
                "batch_size": [int(args.batch_id), int(args.batch_seq)],
                "frames_num_fixed": int(args.clip_len),
                "sample_type": "fixed_unordered",
                "type": "TripletSampler",
                "points_in_use": {
                    "pointcloud_index": 0,
                    "points_num": int(args.points_num),
                },
            },
            "transform": [
                {
                    "type": "PointCloudsTransform",
                    "xyz_only": True,
                    "scale_aware": True,
                    "scale_prob": 1,
                    "flip_prob": 0.15,
                }
            ],
        },
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> None:
    split_root = Path(args.split_root).resolve()
    if split_root.exists() and args.force:
        shutil.rmtree(split_root)
    split_root.mkdir(parents=True, exist_ok=True)

    base_config_path = Path(args.base_config).resolve()
    base_config = load_yaml(base_config_path)
    protocol_config = dict(base_config)
    protocol_config["protocol"] = dict(base_config.get("protocol", {}))
    protocol_config["protocol"]["split_index"] = int(args.split_index)

    samples = collect_pointcloud_videos(
        args.data_root,
        dataset_name=args.dataset_name,
    )
    available_ids = sorted({int(sample["person_id"]) for sample in samples})
    protocol = build_protocol(protocol_config, available_ids)
    fixed_budget = str(protocol.get("training_mode", "")).lower() == "fixed_budget"

    train_ids = {int(value) for value in protocol["train_ids"]}
    eval_ids = {
        *[int(value) for value in protocol["val_ids"]],
        *[int(value) for value in protocol["mixed_test_ids"]],
    }
    train_filters = dict(base_config.get("data", {}).get("train_filters", {}))
    train_filters["max_video_id"] = int(args.max_train_video_id)
    train_samples = [
        sample
        for sample in samples
        if int(sample["person_id"]) in train_ids
        and str(sample["protocol_split"]) == "train"
        and sample_matches(sample, train_filters)
    ]
    eval_samples = [
        sample
        for sample in samples
        if int(sample["person_id"]) in eval_ids
        and str(sample["protocol_split"])
        in {"val_gallery", "val_probe", "gallery", "testA_probe", "testB_probe"}
    ]
    pairs = build_pairs(samples, protocol, base_config.get("retrieval", {}))
    if args.inspect_only:
        print(
            json.dumps(
                {
                    "status": "inspect_only",
                    "protocol": protocol,
                    "train_samples": len(train_samples),
                    "eval_samples": len(eval_samples),
                    "pairs": {
                        name: {
                            "gallery": len(pair["gallery"]),
                            "probe": len(pair["probe"]),
                        }
                        for name, pair in pairs.items()
                    },
                },
                indent=2,
            )
        )
        return

    data_root = split_root / "prepared" / "opengait_data"
    coordinate_audit = CoordinateAudit(unit_scale=float(args.unit_scale))
    for sample in train_samples:
        write_sequence(
            data_root,
            sample,
            "train",
            args.drop_first_frames,
            args.clip_len,
            args.points_num,
            args.unit_scale,
            coordinate_audit,
        )
    for sample in eval_samples:
        write_sequence(
            data_root,
            sample,
            "eval",
            args.drop_first_frames,
            args.clip_len,
            args.points_num,
            args.unit_scale,
            coordinate_audit,
        )

    coordinate_audit.validate()
    coordinate_audit_path = split_root / "prepared" / "coordinate_audit.json"
    coordinate_audit_data = coordinate_audit.as_dict()
    save_json(coordinate_audit_path, coordinate_audit_data)

    config_root = split_root / "configs"
    partition_root = split_root / "prepared" / "partitions"
    val_ids = sorted(int(value) for value in protocol["val_ids"])
    all_eval_ids = sorted(eval_ids)
    val_partition = partition_root / "validation.json"
    full_partition = partition_root / "full_eval.json"
    write_partition(val_partition, protocol["train_ids"], val_ids)
    write_partition(full_partition, protocol["train_ids"], all_eval_ids)

    experiment_tag = normalize_experiment_tag(args.experiment_tag)
    protocol_label = "Final24" if fixed_budget else "FixedSpecial5"
    dataset_name = (
        f"{protocol_label}_LidarGaitPP_{experiment_tag}_split{args.split_index}"
    )
    save_name = f"lidargaitpp_{experiment_tag}_split{args.split_index}"
    train_config = opengait_config(
        dataset_name=dataset_name,
        data_root=data_root,
        partition=val_partition,
        save_name=save_name,
        class_num=len(protocol["train_ids"]),
        args=args,
    )
    val_config = opengait_config(
        dataset_name=dataset_name,
        data_root=data_root,
        partition=val_partition,
        save_name=save_name,
        class_num=len(protocol["train_ids"]),
        args=args,
    )
    final_config = opengait_config(
        dataset_name=dataset_name,
        data_root=data_root,
        partition=full_partition,
        save_name=save_name,
        class_num=len(protocol["train_ids"]),
        args=args,
    )
    write_config(config_root / "train.yaml", train_config)
    write_config(config_root / "validation_eval.yaml", val_config)
    write_config(config_root / "final_eval.yaml", final_config)

    manifest = {
        "method": "official_code_lidargaitpp",
        "method_role": "external_sota_adaptation",
        "randomness": {
            "training_seed": int(args.training_seed),
            "evaluation_seed": int(args.evaluation_seed),
            "seed_changes_protocol_split": False,
        },
        "protocol_name": protocol["name"],
        "protocol_version": protocol["version"],
        "experiment_tag": experiment_tag,
        "split_index": int(args.split_index),
        "clip_len": int(args.clip_len),
        "drop_first_frames": int(args.drop_first_frames),
        "num_points": int(args.points_num),
        "coordinate_adapter": coordinate_audit_data,
        "coordinate_adapter_is_model_input_only": True,
        "selection_pair": None if fixed_budget else "val",
        "selection_metric": None if fixed_budget else "query_micro_mAP",
        "checkpoint_policy": (
            "last_fixed_budget" if fixed_budget else "best_validation_mAP"
        ),
        "selection_uses_fixed_special": False,
        "protocol": protocol,
        "train_filters": train_filters,
        "counts": {
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples),
            "pairs": {
                name: {
                    "gallery": len(pair["gallery"]),
                    "probe": len(pair["probe"]),
                }
                for name, pair in pairs.items()
            },
        },
        "train_samples": [
            serializable_sample(sample, "train") for sample in train_samples
        ],
        "eval_samples": [
            serializable_sample(sample, "eval") for sample in eval_samples
        ],
        "pairs": pairs,
        "paths": {
            "data_root": str(data_root),
            "coordinate_audit": str(coordinate_audit_path),
            "train_config": str(config_root / "train.yaml"),
            "validation_config": str(config_root / "validation_eval.yaml"),
            "final_config": str(config_root / "final_eval.yaml"),
            "validation_partition": str(val_partition),
            "full_partition": str(full_partition),
        },
    }
    save_json(split_root / "manifest.json", manifest)
    save_json(split_root / "protocol.json", protocol)

    official_repo = (CODE_ROOT / args.opengait_repo).resolve()
    official_model = (
        official_repo / "opengait" / "modeling" / "models" / "lidargaitv2.py"
    )
    official_utils = (
        official_repo
        / "opengait"
        / "modeling"
        / "models"
        / "lidargaitv2_utils.py"
    )
    provenance = {
        "method": "official_code_lidargaitpp",
        "randomness": {
            "training_seed": int(args.training_seed),
            "evaluation_seed": int(args.evaluation_seed),
            "opengait_seed_environment": "OPENGAIT_SEED_OFFSET",
        },
        "official_repo": str(official_repo),
        "official_commit": repo_commit(official_repo),
        "official_model": str(official_model),
        "official_model_sha256": source_sha256(official_model),
        "official_utils": str(official_utils),
        "official_utils_sha256": source_sha256(official_utils),
        "base_protocol_config": str(base_config_path),
        "base_protocol_config_sha256": source_sha256(base_config_path),
        "adapter": str(Path(__file__).resolve()),
        "adapter_sha256": source_sha256(Path(__file__).resolve()),
        "coordinate_adapter": coordinate_audit_data,
        "recipe": {
            "experiment_tag": experiment_tag,
            "model": "LidarGaitPlusPlus",
            "transform": "PointCloudsTransform(scale_aware=True)",
            "sampler": f"TripletSampler(P={args.batch_id}, K={args.batch_seq})",
            "loss": "TripletLoss(weight=1.0)+CrossEntropyLoss(weight=0.1)",
            "optimizer": f"SGD(lr={args.learning_rate}, momentum=0.9, weight_decay=0.0005)",
            "total_iter": int(args.total_iter),
            "save_iter": int(args.save_iter),
            "adaptations": [
                "Azure Kinect [X,Y,Z] millimeters mapped to [Z,X,-Y] meters",
                "official height-aware channel now receives vertical height in meters",
                "Fixed-Special5 subject split and shared retrieval evaluator",
                (
                    "fixed training budget and final-iteration checkpoint"
                    if fixed_budget
                    else "validation-C1-only checkpoint selection"
                ),
                "P/K batch reduced for single-GPU memory",
                "training iterations reduced from official 40000 to 10000 by default",
                "evaluation sequences uniformly fixed to protocol clip_len",
                "evaluation points deterministically fixed before OpenGait collation",
                "deterministic farthest-point sampling during evaluation only",
            ],
        },
    }
    save_json(split_root / "provenance.json", provenance)

    print(
        json.dumps(
            {
                "split_root": str(split_root),
                "split_index": int(args.split_index),
                "training_seed": int(args.training_seed),
                "evaluation_seed": int(args.evaluation_seed),
                "train_ids": protocol["train_ids"],
                "val_ids": protocol["val_ids"],
                "general_test_ids": protocol["general_test_ids"],
                "fixed_special_test_ids": protocol["fixed_special_test_ids"],
                "counts": manifest["counts"],
                "coordinate_audit": coordinate_audit_data,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--split-index", type=int, required=True)
    parser.add_argument(
        "--base-config",
        default=str(
            RELEASE_ROOT
            / "shared/configs/fixed_special5_pointcloud_len15.yaml"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=str(RELEASE_ROOT / "dataset/local/pointcloud"),
    )
    parser.add_argument("--dataset-name", default="PersonRecognitionWalking_29")
    parser.add_argument(
        "--opengait-repo",
        default=str(METHOD_ROOT / "third_party/OpenGait"),
    )
    parser.add_argument("--clip-len", type=int, default=15)
    parser.add_argument("--experiment-tag", default="len15_c2c3_full")
    parser.add_argument("--drop-first-frames", type=int, default=30)
    parser.add_argument("--max-train-video-id", type=int, default=16)
    parser.add_argument("--points-num", type=int, default=1024)
    parser.add_argument("--unit-scale", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--batch-id", type=int, default=4)
    parser.add_argument("--batch-seq", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--total-iter", type=int, default=10000)
    parser.add_argument("--save-iter", type=int, default=1000)
    parser.add_argument("--log-iter", type=int, default=100)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
