from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; using CPU instead.")
        device_name = "cpu"
    return torch.device(device_name)


def make_run_dir(cfg: dict[str, Any], seed: int, run_name: str | None = None) -> Path:
    exp_cfg = cfg.get("experiment", {})
    root = Path(exp_cfg.get("root", "experiments/benchmark_baseline"))
    name = run_name or exp_cfg.get("name", "projection_gait_baseline")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / name / f"{timestamp}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_loader(dataset, cfg: dict[str, Any], train: bool = False) -> DataLoader:
    data_cfg = cfg.get("data", {})
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=train,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=1)
    dist = torch.cdist(embeddings, embeddings, p=2)
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(labels.numel(), device=labels.device, dtype=torch.bool)
    positive_mask = same & ~eye
    negative_mask = ~same

    if not positive_mask.any() or not negative_mask.any():
        return embeddings.new_tensor(0.0)

    hardest_positive = dist.masked_fill(~positive_mask, -1.0).max(dim=1)[0]
    hardest_negative = dist.masked_fill(~negative_mask, float("inf")).min(dim=1)[0]
    valid = (hardest_positive >= 0.0) & torch.isfinite(hardest_negative)
    if not valid.any():
        return embeddings.new_tensor(0.0)
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


@torch.no_grad()
def evaluate_classification(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            outputs = model(images)
            loss = loss_fn(outputs["part_logits"], labels)
        preds = outputs["logits"].argmax(dim=1)
        batch_size = int(labels.numel())
        loss_sum += float(loss.item()) * batch_size
        correct += int((preds == labels).sum().item())
        count += batch_size
    return {
        "loss": loss_sum / max(count, 1),
        "acc": correct / max(count, 1),
    }


def _as_python_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    normalize: bool = True,
) -> dict[str, Any]:
    model.eval()
    embeddings = []
    labels = []
    meta: dict[str, list[Any]] = defaultdict(list)

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            outputs = model(images)
        emb = outputs["embedding"].detach().float().cpu()
        if normalize:
            emb = F.normalize(emb, dim=1)
        embeddings.append(emb)
        labels.extend(_as_python_list(batch["label"]))

        for key in ("person_id", "video_person_id", "clothes_id", "condition_tag", "video_id", "path"):
            if key in batch:
                meta[key].extend(_as_python_list(batch[key]))

    return {
        "embeddings": torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 1),
        "labels": torch.tensor(labels, dtype=torch.long),
        "meta": dict(meta),
    }


def _average_precision(sorted_matches: torch.Tensor) -> float:
    total_relevant = int(sorted_matches.sum().item())
    if total_relevant == 0:
        return 0.0
    precision = sorted_matches.float().cumsum(dim=0) / torch.arange(
        1, sorted_matches.numel() + 1, dtype=torch.float32
    )
    return float((precision * sorted_matches.float()).sum().item() / total_relevant)


def evaluate_retrieval(
    gallery: dict[str, Any],
    probe: dict[str, Any],
    metric: str = "cosine",
    ranks: tuple[int, ...] = (1, 5),
    subset_gallery_to_probe_ids: bool = False,
) -> dict[str, Any]:
    gallery_embeddings = gallery["embeddings"]
    gallery_labels = gallery["labels"]
    probe_embeddings = probe["embeddings"]
    probe_labels = probe["labels"]

    empty_metrics: dict[str, Any] = {
        "num_gallery": int(gallery_labels.numel()),
        "num_probe": int(probe_labels.numel()),
        "mAP": 0.0,
        "condition": {},
        "skipped": True,
    }
    for rank in ranks:
        empty_metrics[f"rank{rank}"] = 0.0
    if gallery_labels.numel() == 0 or probe_labels.numel() == 0:
        return empty_metrics

    if subset_gallery_to_probe_ids:
        keep_ids = set(int(label) for label in probe_labels.tolist())
        keep_mask = torch.tensor([int(label) in keep_ids for label in gallery_labels.tolist()], dtype=torch.bool)
        gallery_embeddings = gallery_embeddings[keep_mask]
        gallery_labels = gallery_labels[keep_mask]
        if gallery_labels.numel() == 0:
            empty_metrics["num_gallery"] = 0
            return empty_metrics

    if metric == "cosine":
        distance = 1.0 - probe_embeddings.matmul(gallery_embeddings.T)
    elif metric == "euclidean":
        distance = torch.cdist(probe_embeddings, gallery_embeddings, p=2)
    else:
        raise ValueError("metric must be cosine or euclidean")

    rank_hits = {rank: [] for rank in ranks}
    average_precisions = []
    per_query = []
    probe_meta = probe.get("meta", {})
    condition_values = probe_meta.get("condition_tag", ["unknown"] * len(probe_labels))
    clothes_values = probe_meta.get("clothes_id")
    if clothes_values is not None:
        condition_values = [
            f"C{clothes_values[idx]}_{condition_values[idx]}"
            if idx < len(clothes_values) and idx < len(condition_values)
            else "unknown"
            for idx in range(len(probe_labels))
        ]

    for query_idx in range(probe_embeddings.size(0)):
        order = torch.argsort(distance[query_idx])
        sorted_labels = gallery_labels[order]
        matches = sorted_labels.eq(probe_labels[query_idx])
        for rank in ranks:
            rank_hits[rank].append(float(matches[:rank].any().item()))
        ap = _average_precision(matches)
        average_precisions.append(ap)
        per_query.append(
            {
                "label": int(probe_labels[query_idx].item()),
                "condition": str(condition_values[query_idx]) if query_idx < len(condition_values) else "unknown",
                "rank1": float(matches[:1].any().item()),
                "ap": ap,
            }
        )

    metrics: dict[str, Any] = {
        "num_gallery": int(gallery_labels.numel()),
        "num_probe": int(probe_labels.numel()),
        "mAP": float(np.mean(average_precisions)) if average_precisions else 0.0,
        "skipped": False,
    }
    for rank, values in rank_hits.items():
        metrics[f"rank{rank}"] = float(np.mean(values)) if values else 0.0

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item in per_query:
        grouped[item["condition"]].append(item)
    metrics["condition"] = {
        name: {
            "count": len(items),
            "rank1": float(np.mean([item["rank1"] for item in items])),
            "mAP": float(np.mean([item["ap"] for item in items])),
        }
        for name, items in sorted(grouped.items())
    }
    return metrics


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            flat[f"{prefix}/{key}"] = float(value)
    return flat


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    cfg: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
        },
        path,
    )


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def init_wandb(cfg: dict[str, Any], run_dir: Path, override: str | None = None):
    wandb_cfg = cfg.get("wandb", {})
    enabled = bool(wandb_cfg.get("enabled", False))
    if override == "on":
        enabled = True
    if override == "off":
        enabled = False
    if not enabled:
        return None

    import wandb

    return wandb.init(
        project=wandb_cfg.get("project", "point-cloud-gait"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name") or run_dir.name,
        dir=str(run_dir),
        config=cfg,
        mode=wandb_cfg.get("mode", "online"),
        tags=wandb_cfg.get("tags", ["benchmark-baseline"]),
    )
