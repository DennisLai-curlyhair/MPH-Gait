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
    parser = argparse.ArgumentParser(
        description="Generate a public projection-baseline config."
    )
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clip-len", type=int, default=None)
    parser.add_argument(
        "--train-variant",
        choices=sorted(TRAIN_VARIANTS),
        default=None,
    )
    parser.add_argument("--max-video-id", type=int, default=None)
    parser.add_argument("--name-suffix", default=None)
    parser.add_argument("--experiment-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def update_experiment_name(
    cfg: dict[str, Any],
    suffix: str,
) -> None:
    method_id = str(
        cfg.get("method", {}).get(
            "id",
            cfg.get("model", {}).get("name", "projection"),
        )
    )
    clip_len = cfg.get("data", {}).get("clip_len", "na")
    cfg.setdefault("experiment", {})["name"] = (
        f"{method_id}_len{clip_len}_{suffix}"
    )


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(
        args.base.read_text(encoding="utf-8")
    ) or {}
    data_cfg = cfg.setdefault("data", {})
    suffix = args.name_suffix or "train_c2c3_full"

    if args.clip_len is not None:
        if args.clip_len <= 0:
            raise ValueError("--clip-len must be positive")
        data_cfg["clip_len"] = int(args.clip_len)

    if args.train_variant is not None:
        variant = TRAIN_VARIANTS[args.train_variant]
        data_cfg["train_filters"] = dict(variant["filters"])
        suffix = str(variant["suffix"])

    if args.max_video_id is not None:
        if args.max_video_id <= 0 or args.max_video_id > 16:
            raise ValueError("--max-video-id must be in [1, 16]")
        data_cfg["train_filters"] = {
            "clothes_ids": [2, 3],
            "condition_tags": ["normal"],
            "max_video_id": int(args.max_video_id),
        }
        suffix = f"train_{args.max_video_id * 2}videos_per_id"

    if args.experiment_root is not None:
        cfg.setdefault("experiment", {})["root"] = str(
            args.experiment_root
        )
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("--seed must be non-negative")
        cfg.setdefault("experiment", {})["seed"] = int(args.seed)
        data_cfg["loader_seed"] = int(args.seed)
        data_cfg.pop("train_loader_seed", None)
        data_cfg.pop("eval_loader_seed", None)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.num_workers is not None:
        data_cfg["num_workers"] = int(args.num_workers)

    update_experiment_name(cfg, suffix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(
            cfg,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

