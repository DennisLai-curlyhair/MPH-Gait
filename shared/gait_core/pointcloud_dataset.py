#!/usr/bin/env python3
"""Protocol-aware point-cloud dataset for gait recognition.

This loader keeps the raw point-cloud pipeline separate from the projected-image
pipeline. Labels are based on the canonical ``person_xxx`` directory instead of
only the ``Pxx`` token in video folder names, because a few folders in this data
collection have mismatched ``Pxx`` names.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_NAME = "PersonRecognitionWalking_29"
VIDEO_RE = re.compile(r"P(\d+)_C(\d+)(?:_(.+))?_V(\d+)")
PERSON_RE = re.compile(r"person_(\d+)")
POINT_FRAME_RE = re.compile(r"clear_data_(\d+)\.npy")

SENSOR_NATIVE_COORDINATES = "kinect_camera_xyz_mm_v1"
AXIS_ALIGNED_METRIC_COORDINATES = (
    "kinect_xyz_mm_to_forward_lateral_height_m_v1"
)
SUPPORTED_COORDINATE_ADAPTERS = {
    SENSOR_NATIVE_COORDINATES,
    AXIS_ALIGNED_METRIC_COORDINATES,
}

SPLIT_ALIASES = {
    "train": "train",
    "val_gallery": "val_gallery",
    "valg": "val_gallery",
    "val_probe": "val_probe",
    "val": "val_probe",
    "valp": "val_probe",
    "gallery": "gallery",
    "enroll": "gallery",
    "testa": "testA_probe",
    "testa_probe": "testA_probe",
    "test1": "testA_probe",
    "test1_probe": "testA_probe",
    "testb": "testB_probe",
    "testb_probe": "testB_probe",
    "test2": "testB_probe",
    "test2_probe": "testB_probe",
    "extra": "extra",
    "unused": "extra",
    "all": "all",
}

ORDERED_SPLITS = (
    "train",
    "val_gallery",
    "val_probe",
    "gallery",
    "testA_probe",
    "testB_probe",
    "extra",
)


def canonical_split(split: str) -> str:
    key = split.replace("-", "_").lower()
    if key not in SPLIT_ALIASES:
        valid = ", ".join(ORDERED_SPLITS + ("all",))
        raise ValueError(f"Unknown split '{split}'. Valid splits: {valid}")
    return SPLIT_ALIASES[key]


def resolve_pointcloud_root(root: str | Path, dataset_name: str = DATASET_NAME) -> Path:
    root = Path(root)
    if (root / dataset_name).is_dir():
        return root / dataset_name
    if any(root.glob("person_*")):
        return root
    raise FileNotFoundError(
        f"Cannot find point-cloud dataset under {root}. Expected either "
        f"{root / dataset_name} or person_* folders directly under root."
    )


def parse_person_label(person_dir_name: str) -> tuple[int, int] | None:
    match = PERSON_RE.fullmatch(person_dir_name)
    if match is None:
        return None
    person_id = int(match.group(1))
    return person_id - 1, person_id


def parse_video_folder(video_dir_name: str) -> dict[str, int | str] | None:
    match = VIDEO_RE.fullmatch(video_dir_name)
    if match is None:
        return None
    return {
        "video_person_id": int(match.group(1)),
        "clothes_id": int(match.group(2)),
        "condition_tag": match.group(3) or "normal",
        "video_id": int(match.group(4)),
    }


def frame_number(path: Path) -> int:
    match = POINT_FRAME_RE.fullmatch(path.name)
    if match is None:
        return math.inf
    return int(match.group(1))


def assign_protocol_split(clothes_id: int, video_id: int) -> str:
    if clothes_id in (2, 3) and 1 <= video_id <= 16:
        return "train"
    if clothes_id in (2, 3) and video_id == 17:
        return "val_gallery"
    if clothes_id in (2, 3) and video_id == 18:
        return "val_probe"
    if clothes_id in (2, 3) and video_id in (19, 20):
        return "gallery"
    if clothes_id == 1:
        return "testA_probe"
    if clothes_id in (4, 5, 6, 7):
        return "testB_probe"
    return "extra"


def collect_pointcloud_videos(root: str | Path, dataset_name: str = DATASET_NAME) -> list[dict]:
    data_root = resolve_pointcloud_root(root, dataset_name=dataset_name)
    samples: list[dict] = []

    for person_dir in sorted(data_root.glob("person_*")):
        if not person_dir.is_dir():
            continue
        parsed_person = parse_person_label(person_dir.name)
        if parsed_person is None:
            continue
        label, person_id = parsed_person

        for video_dir in sorted(person_dir.glob("P*_C*_V*")):
            if not video_dir.is_dir():
                continue
            info = parse_video_folder(video_dir.name)
            if info is None:
                continue

            frame_paths = sorted(video_dir.glob("clear_data_*.npy"), key=frame_number)
            if not frame_paths:
                continue

            protocol_split = assign_protocol_split(
                int(info["clothes_id"]), int(info["video_id"])
            )
            samples.append(
                {
                    "video_dir": video_dir,
                    "frame_paths": frame_paths,
                    "label": label,
                    "person_id": person_id,
                    "video_person_id": info["video_person_id"],
                    "clothes_id": info["clothes_id"],
                    "condition_tag": info["condition_tag"],
                    "video_id": info["video_id"],
                    "protocol_split": protocol_split,
                }
            )

    return samples


def adapt_point_coordinates(
    points: np.ndarray,
    coordinate_adapter: str = SENSOR_NATIVE_COORDINATES,
) -> np.ndarray:
    """Apply an explicit camera-to-model coordinate convention.

    Azure Kinect point clouds are stored as camera [X, Y, Z] in
    millimeters, with positive Y pointing down and positive Z pointing
    forward. The axis-aligned adapter returns
    [forward, lateral, height_up] = [Z, X, -Y] * 0.001 in meters.
    """

    points = np.asarray(points[:, :3], dtype=np.float32)
    if coordinate_adapter == SENSOR_NATIVE_COORDINATES:
        return points
    if coordinate_adapter == AXIS_ALIGNED_METRIC_COORDINATES:
        return np.stack(
            (
                points[:, 2],
                points[:, 0],
                -points[:, 1],
            ),
            axis=1,
        ).astype(np.float32, copy=False) * np.float32(0.001)
    valid = ", ".join(sorted(SUPPORTED_COORDINATE_ADAPTERS))
    raise ValueError(
        f"Unknown coordinate_adapter {coordinate_adapter!r}. Valid: {valid}"
    )


class PointCloudGaitProtocolDataset(Dataset):
    """Raw point-cloud gait dataset using a gallery/probe protocol split.

    Parameters
    ----------
    root:
        Usually ``dataset``. Passing ``dataset/PersonRecognitionWalking_29`` is
        also supported.
    split:
        One of ``train``, ``val_gallery``, ``val_probe``, ``gallery``,
        ``testA_probe``, ``testB_probe``, ``extra`` or ``all``. Aliases such as
        ``test1`` and ``test2`` are accepted for backward compatibility.
    clip_len:
        Fixed number of frames to return. Set to ``None`` to return the whole
        sequence, but then default PyTorch collation cannot batch variable T.
    drop_first_frames:
        Point-cloud version defaults to 30 to preserve the old dataset.py
        behavior. The projected-image version defaults to 0.
    """

    def __init__(
        self,
        root: str | Path = "dataset",
        split: str = "train",
        dataset_name: str = DATASET_NAME,
        num_points: int = 1024,
        clip_len: int | None = 30,
        sampling: str = "auto",
        drop_first_frames: int = 30,
        random_point_sample_train: bool = True,
        pad_short: bool = True,
        return_path: bool = True,
        coordinate_adapter: str = SENSOR_NATIVE_COORDINATES,
    ) -> None:
        self.data_root = resolve_pointcloud_root(root, dataset_name=dataset_name)
        self.split = canonical_split(split)
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.clip_len = clip_len
        self.sampling = self._resolve_sampling(sampling)
        self.drop_first_frames = max(0, int(drop_first_frames))
        self.random_point_sample_train = random_point_sample_train
        self.pad_short = pad_short
        self.return_path = return_path
        if coordinate_adapter not in SUPPORTED_COORDINATE_ADAPTERS:
            valid = ", ".join(sorted(SUPPORTED_COORDINATE_ADAPTERS))
            raise ValueError(
                f"Unknown coordinate_adapter {coordinate_adapter!r}. Valid: {valid}"
            )
        self.coordinate_adapter = str(coordinate_adapter)

        all_samples = collect_pointcloud_videos(self.data_root, dataset_name=dataset_name)
        if self.split == "all":
            self.samples = all_samples
        else:
            self.samples = [s for s in all_samples if s["protocol_split"] == self.split]

    def _resolve_sampling(self, sampling: str) -> str:
        if sampling == "auto":
            return "random" if canonical_split(self.split) == "train" else "uniform"
        if sampling not in {"random", "uniform", "center", "all"}:
            raise ValueError("sampling must be one of: auto, random, uniform, center, all")
        return sampling

    @property
    def is_gallery(self) -> bool:
        return self.split in {"val_gallery", "gallery"}

    @property
    def is_probe(self) -> bool:
        return self.split in {"val_probe", "testA_probe", "testB_probe"}

    def __len__(self) -> int:
        return len(self.samples)

    def _usable_frame_paths(self, frame_paths: list[Path]) -> list[Path]:
        usable = frame_paths[self.drop_first_frames :]
        return usable if usable else frame_paths

    def _select_indices(self, n_frames: int) -> np.ndarray:
        if n_frames <= 0:
            raise ValueError("Cannot sample from an empty frame list")
        if self.clip_len is None or self.sampling == "all":
            return np.arange(n_frames, dtype=np.int64)

        clip_len = int(self.clip_len)
        if clip_len <= 0:
            raise ValueError("clip_len must be positive or None")

        if n_frames >= clip_len:
            if self.sampling == "random":
                start = np.random.randint(0, n_frames - clip_len + 1)
                return np.arange(start, start + clip_len, dtype=np.int64)
            if self.sampling == "center":
                start = (n_frames - clip_len) // 2
                return np.arange(start, start + clip_len, dtype=np.int64)
            return np.linspace(0, n_frames - 1, clip_len).round().astype(np.int64)

        if not self.pad_short:
            raise ValueError(f"Sequence has only {n_frames} frames, shorter than clip_len={clip_len}")

        indices = np.arange(n_frames, dtype=np.int64)
        pad = np.full(clip_len - n_frames, n_frames - 1, dtype=np.int64)
        return np.concatenate([indices, pad], axis=0)

    def _sample_points(self, points: np.ndarray, deterministic: bool) -> torch.Tensor:
        points = adapt_point_coordinates(points, self.coordinate_adapter)
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]

        if len(points) == 0:
            return torch.zeros(self.num_points, 3, dtype=torch.float32)

        n_raw = points.shape[0]
        if n_raw >= self.num_points:
            if deterministic:
                idx = np.linspace(0, n_raw - 1, self.num_points).round().astype(np.int64)
            else:
                idx = np.random.choice(n_raw, self.num_points, replace=False)
        else:
            if deterministic:
                repeats = int(math.ceil(self.num_points / n_raw))
                idx = np.tile(np.arange(n_raw, dtype=np.int64), repeats)[: self.num_points]
            else:
                idx = np.random.choice(n_raw, self.num_points, replace=True)

        return torch.from_numpy(points[idx].copy()).float()

    def _load_video(self, frame_paths: list[Path]) -> torch.Tensor:
        usable_paths = self._usable_frame_paths(frame_paths)
        selected = self._select_indices(len(usable_paths))
        deterministic = not (
            self.split == "train" and self.random_point_sample_train and self.sampling == "random"
        )

        frames = []
        for frame_idx in selected:
            points = np.load(usable_paths[int(frame_idx)])
            frames.append(self._sample_points(points, deterministic=deterministic))
        return torch.stack(frames, dim=0)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        item = {
            "points": self._load_video(sample["frame_paths"]),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "person_id": sample["person_id"],
            "video_person_id": sample["video_person_id"],
            "clothes_id": sample["clothes_id"],
            "condition_tag": sample["condition_tag"],
            "video_id": sample["video_id"],
            "protocol_split": sample["protocol_split"],
            "is_gallery": sample["protocol_split"] in {"val_gallery", "gallery"},
            "is_probe": sample["protocol_split"] in {"val_probe", "testA_probe", "testB_probe"},
        }
        if self.return_path:
            item["path"] = str(sample["video_dir"])
        return item

    def labels(self) -> list[int]:
        return [int(s["label"]) for s in self.samples]

    def person_ids(self) -> list[int]:
        return [int(s["person_id"]) for s in self.samples]

    def summary(self) -> dict[str, dict | int]:
        clothes_count: dict[str, int] = {}
        condition_count: dict[str, int] = {}
        label_set = set()
        for sample in self.samples:
            label_set.add(sample["label"])
            clothes_key = f"C{sample['clothes_id']}"
            condition_key = f"C{sample['clothes_id']}_{sample['condition_tag']}"
            clothes_count[clothes_key] = clothes_count.get(clothes_key, 0) + 1
            condition_count[condition_key] = condition_count.get(condition_key, 0) + 1
        return {
            "split": self.split,
            "videos": len(self.samples),
            "identities": len(label_set),
            "clothes": dict(sorted(clothes_count.items())),
            "conditions": dict(sorted(condition_count.items())),
        }


def build_all_split_summary(root: str | Path = "dataset") -> dict[str, dict[str, dict | int]]:
    return {
        split: PointCloudGaitProtocolDataset(root=root, split=split, clip_len=None).summary()
        for split in ORDERED_SPLITS
    }


if __name__ == "__main__":
    import pprint

    pprint.pp(build_all_split_summary("dataset"))
