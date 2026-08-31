#!/usr/bin/env python3
"""Generate one Fixed-Special5 follow-up configuration from a base YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


TRAIN_VARIANTS = {
    "c2": {
        "suffix": "train_c2_only",
        "filters": {
            "clothes_ids": [2],
            "condition_tags": ["normal"],
        },
    },
    "c3": {
        "suffix": "train_c3_only",
        "filters": {
            "clothes_ids": [3],
            "condition_tags": ["normal"],
        },
    },
    "c2c3_balanced": {
        "suffix": "train_c2c3_balanced",
        "filters": {
            "clothes_ids": [2, 3],
            "condition_tags": ["normal"],
            "max_video_id": 8,
        },
    },
    "c2c3_full": {
        "suffix": "train_c2c3_full",
        "filters": {
            "clothes_ids": [2, 3],
            "condition_tags": ["normal"],
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clip-len", type=int)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--train-variant", choices=sorted(TRAIN_VARIANTS))
    choice.add_argument("--max-video-id", type=int)
    parser.add_argument("--experiment-root")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def method_id(config: dict[str, Any]) -> str:
    return str(
        config.get("method", {}).get(
            "id",
            config.get("model", {}).get("name", "method"),
        )
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.base.read_text(encoding="utf-8")) or {}
    data = config.setdefault("data", {})
    suffix = "train_c2c3_full"

    if args.clip_len is not None:
        if args.clip_len <= 0:
            raise ValueError("--clip-len must be positive")
        data["clip_len"] = int(args.clip_len)

    if args.train_variant is not None:
        variant = TRAIN_VARIANTS[args.train_variant]
        data["train_filters"] = dict(variant["filters"])
        suffix = str(variant["suffix"])

    if args.max_video_id is not None:
        if not 1 <= args.max_video_id <= 16:
            raise ValueError("--max-video-id must be in [1, 16]")
        data["train_filters"] = {
            "clothes_ids": [2, 3],
            "condition_tags": ["normal"],
            "max_video_id": int(args.max_video_id),
        }
        suffix = f"train_{args.max_video_id * 2}videos_per_id"

    if args.experiment_root is not None:
        config.setdefault("experiment", {})["root"] = args.experiment_root
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("--seed must be non-negative")
        config.setdefault("experiment", {})["seed"] = int(args.seed)
        data["loader_seed"] = int(args.seed)
        data.pop("train_loader_seed", None)
        data.pop("eval_loader_seed", None)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        data["num_workers"] = int(args.num_workers)

    clip_len = int(data.get("clip_len", 15))
    config.setdefault("experiment", {})["name"] = (
        f"{method_id(config)}_len{clip_len}_{suffix}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
