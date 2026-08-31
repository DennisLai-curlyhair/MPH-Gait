#!/usr/bin/env python3
"""Extract official LidarGait++ embeddings and run Fixed-Special5 retrieval."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import socket
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_TORCHVISION_LIBRARY = None
METHOD_ROOT = Path(__file__).resolve().parents[2]

from gait_core.utils import evaluate_retrieval


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        def unavailable(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("Unused OpenGait imageio stub was called")

        imageio.imwrite = unavailable
        imageio.mimsave = unavailable
        sys.modules["imageio"] = imageio


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def init_single_process_dist() -> None:
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{free_port()}",
            rank=0,
            world_size=1,
        )


def enable_deterministic_fps() -> None:
    """Make official PointNet++ sampling repeatable during evaluation only."""
    from modeling.models import lidargaitv2_utils  # type: ignore

    def deterministic_farthest_point_sample(
        xyz: torch.Tensor, npoint: int
    ) -> torch.Tensor:
        device = xyz.device
        batch_size, point_count, channels = xyz.shape
        centroids = torch.zeros(
            batch_size, npoint, dtype=torch.long, device=device
        )
        distance = torch.full(
            (batch_size, point_count), 1e10, device=device
        )
        coordinates = xyz[:, :, :3]
        farthest = coordinates.square().sum(dim=-1).max(dim=-1)[1]
        batch_indices = torch.arange(
            batch_size, dtype=torch.long, device=device
        )
        for index in range(npoint):
            centroids[:, index] = farthest
            centroid = coordinates[batch_indices, farthest].view(
                batch_size, 1, channels
            )
            squared_distance = (coordinates - centroid).square().sum(dim=-1)
            mask = squared_distance < distance
            distance[mask] = squared_distance[mask]
            farthest = distance.max(dim=-1)[1]
        return centroids

    lidargaitv2_utils.farthest_point_sample = (
        deterministic_farthest_point_sample
    )


def extract_embeddings(
    config_path: Path,
    checkpoint_path: Path,
    opengait_repo: Path,
    log_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("Official OpenGait LidarGait++ evaluation requires CUDA")

    ensure_optional_import_stubs()
    opengait_root = opengait_repo / "opengait"
    sys.path.insert(0, str(opengait_root))

    with pushd(opengait_repo):
        from modeling import models  # type: ignore
        from utils import config_loader, get_msg_mgr  # type: ignore

        enable_deterministic_fps()
        init_single_process_dist()
        set_seed(seed)
        cfg = copy.deepcopy(config_loader(str(config_path)))
        cfg["evaluator_cfg"]["restore_hint"] = str(checkpoint_path)
        cfg["evaluator_cfg"]["restore_ckpt_strict"] = True
        cfg["evaluator_cfg"]["enable_float16"] = False
        cfg["evaluator_cfg"]["sampler"]["batch_size"] = int(batch_size)
        cfg["data_cfg"]["num_workers"] = int(num_workers)

        log_dir.mkdir(parents=True, exist_ok=True)
        get_msg_mgr().init_logger(str(log_dir), log_to_file=False)
        model_cls = getattr(models, cfg["model_cfg"]["model"])
        model = model_cls(cfg, training=False)
        model.eval()
        with torch.no_grad():
            info = model.inference(torch.distributed.get_rank())

    embeddings = torch.from_numpy(info["embeddings"]).detach().float().cpu()
    if embeddings.ndim > 2:
        embeddings = embeddings.flatten(start_dim=1)
    finite = torch.isfinite(embeddings).all(dim=1)
    if not bool(finite.all()):
        raise RuntimeError(
            f"Official LidarGait++ produced {(~finite).sum().item()} non-finite embeddings"
        )
    embeddings = F.normalize(embeddings, dim=1)
    views = [str(value) for value in model.test_loader.dataset.views_list]
    if len(views) != embeddings.shape[0]:
        raise RuntimeError(
            f"Embedding/view mismatch: embeddings={embeddings.shape[0]}, views={len(views)}"
        )
    return {view: embeddings[index] for index, view in enumerate(views)}


def sample_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(sample["id"]): sample for sample in manifest.get("eval_samples", [])
    }


def bundle(
    sample_ids: list[str],
    embeddings: dict[str, torch.Tensor],
    samples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing = [
        sample_id
        for sample_id in sample_ids
        if sample_id not in embeddings or sample_id not in samples
    ]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} requested embeddings; first={missing[:5]}"
        )
    selected = [embeddings[sample_id] for sample_id in sample_ids]
    metadata_keys = (
        "person_id",
        "clothes_id",
        "condition_tag",
        "video_id",
        "path",
    )
    return {
        "embeddings": torch.stack(selected),
        "labels": torch.tensor(
            [int(samples[sample_id]["person_id"]) for sample_id in sample_ids],
            dtype=torch.long,
        ),
        "meta": {
            key: [samples[sample_id][key] for sample_id in sample_ids]
            for key in metadata_keys
        }
        | {"sample_id": list(sample_ids)},
    }


def subset_bundle(features: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    return {
        "embeddings": features["embeddings"][tensor_indices],
        "labels": features["labels"][tensor_indices],
        "meta": {
            key: [values[index] for index in indices]
            for key, values in features.get("meta", {}).items()
        },
    }


def evaluate_subject_macro(
    gallery: dict[str, Any],
    probe: dict[str, Any],
    subset_gallery_to_probe_ids: bool,
) -> dict[str, Any]:
    metrics = evaluate_retrieval(
        gallery,
        probe,
        metric="cosine",
        ranks=(1, 5),
        subset_gallery_to_probe_ids=subset_gallery_to_probe_ids,
    )
    person_ids = [int(value) for value in probe["meta"]["person_id"]]
    subject_results: dict[str, dict[str, Any]] = {}
    for person_id in sorted(set(person_ids)):
        indices = [
            index for index, value in enumerate(person_ids) if value == person_id
        ]
        subject_results[f"P{person_id:03d}"] = evaluate_retrieval(
            gallery,
            subset_bundle(probe, indices),
            metric="cosine",
            ranks=(1, 5),
            subset_gallery_to_probe_ids=subset_gallery_to_probe_ids,
        )
    macro: dict[str, Any] = {"num_subjects": len(subject_results)}
    for metric_name in ("rank1", "rank5", "mAP"):
        values = [
            float(result.get(metric_name, float("nan")))
            for result in subject_results.values()
        ]
        values = [value for value in values if math.isfinite(value)]
        macro[metric_name] = float(np.mean(values)) if values else 0.0
    metrics["subject"] = subject_results
    metrics["subject_macro"] = macro
    return metrics


def embedding_diagnostics(embeddings: dict[str, torch.Tensor]) -> dict[str, Any]:
    matrix = torch.stack(list(embeddings.values()))
    sample_count = int(matrix.shape[0])
    if sample_count > 1:
        gram = matrix.matmul(matrix.T)
        mask = ~torch.eye(sample_count, dtype=torch.bool)
        off_diagonal = gram[mask]
        cosine_mean = float(off_diagonal.mean())
        cosine_std = float(off_diagonal.std(unbiased=False))
    else:
        cosine_mean = 1.0
        cosine_std = 0.0
    return {
        "num_embeddings": sample_count,
        "embedding_dim": int(matrix.shape[1]),
        "all_finite": bool(torch.isfinite(matrix).all()),
        "feature_std_mean": float(matrix.std(dim=0, unbiased=False).mean()),
        "sample_norm_mean": float(matrix.norm(dim=1).mean()),
        "sample_norm_std": float(matrix.norm(dim=1).std(unbiased=False)),
        "off_diagonal_cosine_mean": cosine_mean,
        "off_diagonal_cosine_std": cosine_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opengait-repo",
        default=str(METHOD_ROOT / "third_party/OpenGait"),
    )
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError(
            "LidarGaitPlusPlus evaluation requires batch_size=1 because the "
            "official forward path does not split concatenated all_ordered sequences."
        )

    config_path = Path(args.config).resolve()
    manifest_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    opengait_repo = Path(args.opengait_repo).resolve()
    manifest = load_json(manifest_path)

    pair_names = [value for value in args.pairs.split(",") if value]
    if not pair_names:
        pair_names = sorted(manifest["pairs"])
    unknown = sorted(set(pair_names) - set(manifest["pairs"]))
    if unknown:
        raise KeyError(f"Unknown pair names: {unknown}")

    embeddings = extract_embeddings(
        config_path,
        checkpoint_path,
        opengait_repo,
        output_dir / "opengait_logs",
        args.batch_size,
        args.num_workers,
        args.seed,
    )
    samples = sample_map(manifest)
    required_ids = {
        sample_id
        for pair_name in pair_names
        for side in ("gallery", "probe")
        for sample_id in manifest["pairs"][pair_name][side]
    }
    missing = sorted(required_ids - set(embeddings))
    if missing:
        raise RuntimeError(
            f"Inference partition omitted {len(missing)} required samples: {missing[:5]}"
        )

    retrieval: dict[str, Any] = {}
    for pair_name in pair_names:
        pair = manifest["pairs"][pair_name]
        retrieval[pair_name] = evaluate_subject_macro(
            bundle(pair["gallery"], embeddings, samples),
            bundle(pair["probe"], embeddings, samples),
            bool(pair.get("subset_gallery_to_probe_ids", False)),
        )

    diagnostics = embedding_diagnostics(
        {sample_id: embeddings[sample_id] for sample_id in sorted(required_ids)}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "retrieval_metrics.json", retrieval)
    save_json(output_dir / "embedding_diagnostics.json", diagnostics)
    save_json(
        output_dir / "summary.json",
        {
            "method": manifest.get(
                "method",
                "official_code_lidargaitpp",
            ),
            "split_index": manifest["protocol"]["split_index"],
            "protocol_name": manifest["protocol"]["name"],
            "selection_uses_fixed_special": False,
            "deterministic_evaluation_fps": True,
            "evaluation_seed": int(args.seed),
            "config": str(config_path),
            "manifest": str(manifest_path),
            "checkpoint": str(checkpoint_path),
            "pairs": pair_names,
            "embedding_diagnostics": diagnostics,
            "retrieval_metrics": retrieval,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint": str(checkpoint_path),
                "pairs": pair_names,
                "diagnostics": diagnostics,
            },
            indent=2,
        )
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
