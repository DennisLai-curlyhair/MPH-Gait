#!/usr/bin/env python3
"""Protocol-aware projected-image dataset for gait recognition.

Expected layout after projection_baseline/build_dataset.py:

    dataset_proj/PersonRecognitionWalking_29/lidargait_rgb/person_001/P1_C1_V001/*.png

The split protocol mirrors `gait_core.pointcloud_dataset`, but this loader keeps all
projected frames by default and does not drop the first 30 frames.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DATASET_NAME = "PersonRecognitionWalking_29"
DEFAULT_MODE = "lidargait_rgb"
VALID_MODES = {"lidargait_rgb", "lidargait_gray", "lidargait_silhouette", "pseudo_rgb"}
VIDEO_RE = re.compile(r"P(\d+)_C(\d+)(?:_(.+))?_V(\d+)")
PERSON_RE = re.compile(r"person_(\d+)")
IMAGE_FRAME_RE = re.compile(r".*_(\d+)\.png")

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


def resolve_projected_root(
    root: str | Path,
    mode: str = DEFAULT_MODE,
    dataset_name: str = DATASET_NAME,
) -> Path:
    if mode not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"Unknown projection mode '{mode}'. Valid modes: {valid}")

    root = Path(root)
    candidates = [
        root / dataset_name / mode,
        root / mode,
        root,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("person_*")):
            return candidate

    raise FileNotFoundError(
        f"Cannot find projected dataset mode '{mode}' under {root}. Expected "
        f"{root / dataset_name / mode}, {root / mode}, or person_* folders directly under root."
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


def image_frame_number(path: Path) -> int:
    match = IMAGE_FRAME_RE.fullmatch(path.name)
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


def collect_projected_videos(
    root: str | Path,
    mode: str = DEFAULT_MODE,
    dataset_name: str = DATASET_NAME,
) -> list[dict]:
    mode_root = resolve_projected_root(root, mode=mode, dataset_name=dataset_name)
    samples: list[dict] = []

    for person_dir in sorted(mode_root.glob("person_*")):
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

            frame_paths = sorted(video_dir.glob(f"{mode}_*.png"), key=image_frame_number)
            if not frame_paths:
                frame_paths = sorted(video_dir.glob("*.png"), key=image_frame_number)
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
                    "mode": mode,
                }
            )

    return samples


class ProjectedGaitProtocolDataset(Dataset):
    """Projected-image gait dataset using the same gallery/probe protocol.

    Parameters
    ----------
    root:
        Usually ``dataset_proj``. Passing ``dataset_proj/PersonRecognitionWalking_29``
        or the concrete mode folder is also supported.
    mode:
        ``lidargait_rgb``, ``lidargait_gray``, ``lidargait_silhouette`` or
        ``pseudo_rgb``.
    channels:
        ``None`` keeps the natural mode channel count, ``1`` forces grayscale,
        and ``3`` forces RGB.
    clip_len:
        Fixed number of frames to return. Set to ``None`` to return the whole
        sequence.
    drop_first_frames:
        Defaults to 30 to remove the idle/pre-walking frames and keep protocol
        behavior consistent with the point-cloud loader. Set to 0 for ablation.
    """

    def __init__(
        self,
        root: str | Path = "dataset_proj",
        split: str = "train",
        mode: str = DEFAULT_MODE,
        dataset_name: str = DATASET_NAME,
        clip_len: int | None = 30,
        sampling: str = "auto",
        channels: int | None = None,
        image_size: int | tuple[int, int] | None = None,
        drop_first_frames: int = 30,
        pad_short: bool = True,
        return_path: bool = True,
    ) -> None:
        self.mode = mode
        self.mode_root = resolve_projected_root(root, mode=mode, dataset_name=dataset_name)
        self.split = canonical_split(split)
        self.dataset_name = dataset_name
        self.clip_len = clip_len
        self.sampling = self._resolve_sampling(sampling)
        self.channels = channels
        self.image_size = self._normalize_image_size(image_size)
        self.drop_first_frames = max(0, int(drop_first_frames))
        self.pad_short = pad_short
        self.return_path = return_path

        if self.channels not in {None, 1, 3}:
            raise ValueError("channels must be None, 1, or 3")

        all_samples = collect_projected_videos(self.mode_root, mode=mode, dataset_name=dataset_name)
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

    def _normalize_image_size(self, image_size: int | tuple[int, int] | None) -> tuple[int, int] | None:
        if image_size is None:
            return None
        if isinstance(image_size, int):
            return (image_size, image_size)
        if len(image_size) != 2:
            raise ValueError("image_size tuple must be (height, width)")
        return int(image_size[0]), int(image_size[1])

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

    def _target_pil_mode(self) -> str:
        if self.channels == 1:
            return "L"
        if self.channels == 3:
            return "RGB"
        if self.mode in {"lidargait_gray", "lidargait_silhouette"}:
            return "L"
        return "RGB"

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert(self._target_pil_mode())
            if self.image_size is not None:
                height, width = self.image_size
                image = image.resize((width, height), Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0

        if array.ndim == 2:
            array = array[None, :, :]
        else:
            array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array.copy()).float()

    def _load_video(self, frame_paths: list[Path]) -> torch.Tensor:
        usable_paths = self._usable_frame_paths(frame_paths)
        selected = self._select_indices(len(usable_paths))
        frames = [self._load_image(usable_paths[int(frame_idx)]) for frame_idx in selected]
        return torch.stack(frames, dim=0)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        item = {
            "images": self._load_video(sample["frame_paths"]),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "person_id": sample["person_id"],
            "video_person_id": sample["video_person_id"],
            "clothes_id": sample["clothes_id"],
            "condition_tag": sample["condition_tag"],
            "video_id": sample["video_id"],
            "protocol_split": sample["protocol_split"],
            "mode": sample["mode"],
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

    def summary(self) -> dict[str, dict | int | str]:
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
            "mode": self.mode,
            "split": self.split,
            "videos": len(self.samples),
            "identities": len(label_set),
            "clothes": dict(sorted(clothes_count.items())),
            "conditions": dict(sorted(condition_count.items())),
        }


def build_all_split_summary(
    root: str | Path = "dataset_proj",
    mode: str = DEFAULT_MODE,
) -> dict[str, dict[str, dict | int | str]]:
    return {
        split: ProjectedGaitProtocolDataset(root=root, mode=mode, split=split, clip_len=None).summary()
        for split in ORDERED_SPLITS
    }


if __name__ == "__main__":
    import pprint

    pprint.pp(build_all_split_summary("dataset_proj", mode=DEFAULT_MODE))
