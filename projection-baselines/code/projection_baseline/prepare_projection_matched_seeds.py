from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RESULT_ROOT = Path("../outputs/formal_len15_5split_3seed")
METHOD_SPECS = {
    "rgb_depth": {
        "config": Path("projection_baseline/configs/rgb_depth_len15.yaml"),
        "root_name": "rgb_depth",
        "run_tag": "projection_rgb_depth_len15",
    },
    "gray_depth": {
        "config": Path("projection_baseline/configs/gray_depth_len15.yaml"),
        "root_name": "gray_depth",
        "run_tag": "projection_gray_depth_len15",
    },
    "silhouette": {
        "config": Path("projection_baseline/configs/silhouette_len15.yaml"),
        "root_name": "silhouette",
        "run_tag": "projection_silhouette_len15",
    },
}
REQUIRED_EQUAL_SECTIONS = (
    "protocol",
    "loss",
    "train",
    "optimizer",
    "scheduler",
    "retrieval",
)
REQUIRED_EQUAL_DATA_KEYS = (
    "root",
    "dataset_name",
    "clip_len",
    "drop_first_frames",
    "pad_short",
    "train_split",
    "val_split",
    "batch_size",
    "num_workers",
    "train_sampler",
    "train_filters",
    "val_filters",
)


def parse_int_list(value: str) -> list[int]:
    return [
        int(item)
        for item in re.split(r"[\s,]+", value.strip())
        if item
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare isolated RGB-depth, gray-depth, and silhouette "
            "projection configs for matched seed/split evaluation."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--methods", default="rgb_depth gray_depth silhouette")
    parser.add_argument("--seeds", default="0 1 2")
    parser.add_argument("--splits", default="0 1 2 3 4")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_yaml_compatible(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if load_yaml(path) != value:
            raise RuntimeError(
                f"Refusing to overwrite a different generated config: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_json_compatible(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != value:
            raise RuntimeError(
                f"Refusing to overwrite a different manifest: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_inputs(
    configs: dict[str, dict[str, Any]],
    methods: list[str],
    seeds: list[int],
    splits: list[int],
) -> None:
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("Methods must be a non-empty unique list")
    unknown = sorted(set(methods) - set(METHOD_SPECS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be a non-empty unique list of non-negative integers")
    if not splits or len(splits) != len(set(splits)):
        raise ValueError("Splits must be a non-empty unique list")

    reference = configs[methods[0]]
    num_splits = len(
        reference.get("protocol", {}).get("development_groups", [])
    )
    if num_splits <= 0:
        raise RuntimeError("Protocol has no development groups")
    if any(split < 0 or split >= num_splits for split in splits):
        raise ValueError(f"Splits must be in [0, {num_splits - 1}]")

    for method in methods:
        cfg = configs[method]
        if int(cfg.get("data", {}).get("clip_len", -1)) != 15:
            raise RuntimeError(f"{method} config is not clip_len=15")
        if cfg.get("train", {}).get("selection_pair") != "val":
            raise RuntimeError(f"{method} must select checkpoints using val")
        if not bool(cfg.get("train", {}).get("deterministic", False)):
            raise RuntimeError(f"{method} must use deterministic training")
        for section in REQUIRED_EQUAL_SECTIONS:
            if cfg.get(section) != reference.get(section):
                raise RuntimeError(
                    f"Projection configs disagree in required section: {section}"
                )
        for key in REQUIRED_EQUAL_DATA_KEYS:
            if cfg.get("data", {}).get(key) != reference.get("data", {}).get(key):
                raise RuntimeError(
                    f"Projection configs disagree in required data key: {key}"
                )


def seeded_config(
    source: dict[str, Any],
    *,
    seed: int,
    experiment_root: Path,
    run_tag: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(source)
    experiment = cfg.setdefault("experiment", {})
    data = cfg.setdefault("data", {})
    experiment["root"] = str(experiment_root)
    experiment["seed"] = int(seed)
    experiment["name"] = f"{run_tag}_matched_seed_{seed}"
    data["loader_seed"] = int(seed)
    data.pop("train_loader_seed", None)
    data.pop("eval_loader_seed", None)
    return cfg


def main() -> None:
    args = parse_args()
    methods = [
        item for item in re.split(r"[\s,]+", args.methods.strip()) if item
    ]
    seeds = parse_int_list(args.seeds)
    splits = parse_int_list(args.splits)
    configs = {
        method: load_yaml(METHOD_SPECS[method]["config"])
        for method in methods
    }
    validate_inputs(configs, methods, seeds, splits)

    generated_root = args.result_root / "generated_configs" / args.run_stamp
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "Projection len15 5-split matched-seed evaluation",
        "run_stamp": args.run_stamp,
        "result_root": str(args.result_root),
        "methods_order": methods,
        "seeds": seeds,
        "splits": splits,
        "protocol": configs[methods[0]].get("protocol", {}).get("name"),
        "clip_len": 15,
        "selection_pair": "val",
        "selection_uses_fixed_special": False,
        "seed_contract": {
            "model_seed": "seed",
            "loader_seed": "seed",
            "train_loader_seed": "seed + split_index * 1009",
            "eval_loader_seed": "seed + split_index * 1009 + 100000",
        },
        "methods": {},
    }

    for method in methods:
        source_path = METHOD_SPECS[method]["config"]
        source = configs[method]
        experiment_root = args.result_root / METHOD_SPECS[method]["root_name"]
        method_manifest: dict[str, Any] = {
            "method_id": source.get("method", {}).get("id"),
            "representation": source.get("method", {}).get("representation"),
            "source_config": {
                "path": str(source_path),
                "sha256": sha256(source_path),
            },
            "experiment_root": str(experiment_root),
            "configs": {},
            "run_groups": {},
        }
        run_tag = str(METHOD_SPECS[method]["run_tag"])
        for seed in seeds:
            config_path = generated_root / f"{method}_seed{seed}.yaml"
            run_name = f"{args.run_stamp}_{run_tag}_seed{seed}"
            cfg = seeded_config(
                source,
                seed=seed,
                experiment_root=experiment_root,
                run_tag=run_tag,
            )
            write_yaml_compatible(config_path, cfg)
            method_manifest["configs"][str(seed)] = str(config_path)
            method_manifest["run_groups"][str(seed)] = str(
                experiment_root / run_name
            )
        manifest["methods"][method] = method_manifest

    manifest_path = args.result_root / "manifests" / f"{args.run_stamp}.json"
    write_json_compatible(manifest_path, manifest)
    print(f"manifest: {manifest_path}")
    print(f"generated configs: {generated_root}")


if __name__ == "__main__":
    main()
