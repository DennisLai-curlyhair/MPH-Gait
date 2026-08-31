from __future__ import annotations

import argparse
import copy
import hashlib
import math
import platform
import random
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from gait_core.utils import (
    append_jsonl,
    autocast_context,
    batch_hard_triplet_loss,
    evaluate_retrieval,
    flatten_metrics,
    init_wandb,
    load_yaml,
    make_run_dir,
    make_scaler,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)
from gait_core.pointcloud_dataset import PointCloudGaitProtocolDataset, canonical_split
from gait_core.protocol import build_protocol


RELEASE_ROOT = Path(__file__).resolve().parents[2]
ModelBuilder = Callable[[dict[str, Any]], torch.nn.Module]


class IdentityBalancedBatchSampler(Sampler[list[int]]):
    """Build deterministic P x K batches from the unchanged training pool."""

    def __init__(
        self,
        dataset: PointCloudGaitProtocolDataset,
        identities_per_batch: int,
        instances_per_identity: int,
        seed: int,
    ) -> None:
        self.identities_per_batch = int(identities_per_batch)
        self.instances_per_identity = int(instances_per_identity)
        if self.identities_per_batch <= 1 or self.instances_per_identity <= 1:
            raise ValueError("identity-balanced sampling requires P > 1 and K > 1")

        self.label_to_indices: dict[int, list[int]] = {}
        for index, sample in enumerate(dataset.samples):
            label = int(sample["label"])
            self.label_to_indices.setdefault(label, []).append(index)
        if len(self.label_to_indices) < 2:
            raise ValueError("identity-balanced sampling requires at least two identities")

        batch_size = self.identities_per_batch * self.instances_per_identity
        self.num_batches = max(1, len(dataset) // batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        labels = list(self.label_to_indices)
        for _ in range(self.num_batches):
            if len(labels) >= self.identities_per_batch:
                selected_labels = rng.sample(labels, self.identities_per_batch)
            else:
                selected_labels = rng.choices(labels, k=self.identities_per_batch)

            batch: list[int] = []
            for label in selected_labels:
                candidates = self.label_to_indices[label]
                if len(candidates) >= self.instances_per_identity:
                    selected = rng.sample(candidates, self.instances_per_identity)
                else:
                    selected = rng.choices(candidates, k=self.instances_per_identity)
                batch.extend(selected)
            rng.shuffle(batch)
            yield batch


_DATASET_TEMPLATE_CACHE: dict[tuple[Any, ...], PointCloudGaitProtocolDataset] = {}


def seed_loader_worker(_worker_id: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def loader_seed(cfg: dict[str, Any], train: bool) -> int:
    data_cfg = cfg.get("data", {})
    key = "train_loader_seed" if train else "eval_loader_seed"
    if key in data_cfg:
        return int(data_cfg[key])
    base_seed = int(
        data_cfg.get(
            "loader_seed",
            cfg.get("experiment", {}).get("seed", 0),
        )
    )
    split_index = int(cfg.get("protocol", {}).get("split_index", 0))
    return base_seed + split_index * 1009 + (0 if train else 100_000)


def make_loader(dataset, cfg: dict[str, Any], train: bool = False) -> DataLoader:
    data_cfg = cfg.get("data", {})
    seed = loader_seed(cfg, train=train)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler_name = str(data_cfg.get("train_sampler", "random")).lower()
    if train and sampler_name == "identity_balanced":
        identities_per_batch = int(data_cfg.get("identities_per_batch", 8))
        instances_per_identity = int(data_cfg.get("instances_per_identity", 2))
        expected_batch_size = identities_per_batch * instances_per_identity
        configured_batch_size = int(data_cfg.get("batch_size", expected_batch_size))
        if configured_batch_size != expected_batch_size:
            raise ValueError(
                "For identity_balanced sampling, batch_size must equal "
                "identities_per_batch * instances_per_identity"
            )
        batch_sampler = IdentityBalancedBatchSampler(
            dataset,
            identities_per_batch=identities_per_batch,
            instances_per_identity=instances_per_identity,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=int(data_cfg.get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_loader_worker,
            generator=generator,
        )
    if train and sampler_name not in {"random", "shuffle"}:
        raise ValueError(f"Unknown train_sampler: {sampler_name}")
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=train,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        worker_init_fn=seed_loader_worker,
        generator=generator,
    )


def normalize_splits(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def parse_filter_list(value: Any, cast) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        return [cast(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [cast(item) for item in value]
    return [cast(value)]


def sample_matches_filters(
    sample: dict[str, Any],
    filters: dict[str, Any],
) -> bool:
    clothes_ids = parse_filter_list(filters.get("clothes_ids"), int)
    if clothes_ids is not None and int(sample["clothes_id"]) not in set(clothes_ids):
        return False

    excluded = parse_filter_list(filters.get("exclude_clothes_ids"), int)
    if excluded is not None and int(sample["clothes_id"]) in set(excluded):
        return False

    conditions = parse_filter_list(filters.get("condition_tags"), str)
    if conditions is not None and str(sample["condition_tag"]) not in set(conditions):
        return False

    video_ids = parse_filter_list(filters.get("video_ids"), int)
    if video_ids is not None and int(sample["video_id"]) not in set(video_ids):
        return False

    min_video_id = filters.get("min_video_id")
    if min_video_id is not None and int(sample["video_id"]) < int(min_video_id):
        return False

    max_video_id = filters.get("max_video_id")
    if max_video_id is not None and int(sample["video_id"]) > int(max_video_id):
        return False

    protocol_splits = parse_filter_list(filters.get("protocol_splits"), str)
    if (
        protocol_splits is not None
        and str(sample["protocol_split"]) not in set(protocol_splits)
    ):
        return False
    return True


def build_dataset(
    cfg: dict[str, Any],
    split: str | list[str] | tuple[str, ...],
    train: bool,
    person_ids: list[int] | set[int] | None = None,
    filters: dict[str, Any] | None = None,
) -> PointCloudGaitProtocolDataset:
    data_cfg = cfg["data"]
    cache_key = (
        str(Path(data_cfg.get("root", "dataset")).resolve()),
        str(data_cfg.get("dataset_name", "PersonRecognitionWalking_29")),
        int(data_cfg.get("num_points", 1024)),
        data_cfg.get("clip_len", 15),
        int(data_cfg.get("drop_first_frames", 30)),
        bool(data_cfg.get("random_point_sample_train", True)),
        bool(data_cfg.get("pad_short", True)),
        str(data_cfg.get("coordinate_adapter", "kinect_camera_xyz_mm_v1")),
    )
    template = _DATASET_TEMPLATE_CACHE.get(cache_key)
    if template is None:
        template = PointCloudGaitProtocolDataset(
            root=data_cfg.get("root", "dataset"),
            split="all",
            dataset_name=data_cfg.get(
                "dataset_name",
                "PersonRecognitionWalking_29",
            ),
            num_points=int(data_cfg.get("num_points", 1024)),
            clip_len=data_cfg.get("clip_len", 15),
            sampling="uniform",
            drop_first_frames=int(data_cfg.get("drop_first_frames", 30)),
            random_point_sample_train=bool(
                data_cfg.get("random_point_sample_train", True)
            ),
            pad_short=bool(data_cfg.get("pad_short", True)),
            return_path=True,
            coordinate_adapter=str(
                data_cfg.get(
                    "coordinate_adapter",
                    "kinect_camera_xyz_mm_v1",
                )
            ),
        )
        _DATASET_TEMPLATE_CACHE[cache_key] = template

    datasets = []
    for split_name in normalize_splits(split):
        dataset_item = copy.copy(template)
        dataset_item.split = canonical_split(split_name)
        dataset_item.sampling = "random" if train else "uniform"
        if dataset_item.split == "all":
            dataset_item.samples = list(template.samples)
        else:
            dataset_item.samples = [
                sample
                for sample in template.samples
                if sample["protocol_split"] == dataset_item.split
            ]
        datasets.append(dataset_item)
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset.samples = [
            sample
            for split_dataset in datasets
            for sample in split_dataset.samples
        ]

    if person_ids is not None:
        keep_ids = {int(person_id) for person_id in person_ids}
        dataset.samples = [
            sample
            for sample in dataset.samples
            if int(sample["person_id"]) in keep_ids
        ]
    if filters:
        dataset.samples = [
            sample
            for sample in dataset.samples
            if sample_matches_filters(sample, filters)
        ]
    return dataset


def collect_subject_ids(cfg: dict[str, Any]) -> list[int]:
    all_set = build_dataset(cfg, "all", train=False)
    return sorted({int(sample["person_id"]) for sample in all_set.samples})


def side_filters(
    pair_cfg: dict[str, Any],
    side: str,
) -> dict[str, Any]:
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


def ids_from_source(
    source: str | list[int] | tuple[int, ...],
    protocol: dict[str, Any],
    pair_name: str,
    side: str,
) -> list[int] | None:
    if isinstance(source, (list, tuple)):
        return [int(person_id) for person_id in source]
    if source == "all":
        return None
    ids = protocol.get(str(source))
    if ids is None:
        raise ValueError(
            f"Unknown {side}_id_source '{source}' for retrieval pair '{pair_name}'"
        )
    return [int(person_id) for person_id in ids]


def build_eval_pairs(
    cfg: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for name, pair_cfg in cfg.get("retrieval", {}).get("pairs", {}).items():
        shared_source = pair_cfg.get(
            "id_source",
            "val_ids" if name.startswith("val") else "mixed_test_ids",
        )
        gallery_source = pair_cfg.get("gallery_id_source", shared_source)
        probe_source = pair_cfg.get("probe_id_source", shared_source)
        gallery_ids = ids_from_source(
            gallery_source,
            protocol,
            name,
            "gallery",
        )
        probe_ids = ids_from_source(
            probe_source,
            protocol,
            name,
            "probe",
        )
        gallery_split = pair_cfg.get(
            "gallery_splits",
            pair_cfg.get("gallery_split"),
        )
        probe_split = pair_cfg.get(
            "probe_splits",
            pair_cfg.get("probe_split"),
        )
        gallery_filters = side_filters(pair_cfg, "gallery")
        probe_filters = side_filters(pair_cfg, "probe")
        gallery = build_dataset(
            cfg,
            gallery_split,
            train=False,
            person_ids=gallery_ids,
            filters=gallery_filters,
        )
        probe = build_dataset(
            cfg,
            probe_split,
            train=False,
            person_ids=probe_ids,
            filters=probe_filters,
        )
        if len(gallery) == 0:
            raise RuntimeError(f"Retrieval pair '{name}' has an empty gallery")
        if len(probe) == 0:
            raise RuntimeError(f"Retrieval pair '{name}' has an empty probe")

        gallery_labels = {int(sample["label"]) for sample in gallery.samples}
        probe_labels = {int(sample["label"]) for sample in probe.samples}
        missing_labels = sorted(probe_labels - gallery_labels)
        if missing_labels:
            missing_person_ids = [label + 1 for label in missing_labels]
            raise RuntimeError(
                f"Retrieval pair '{name}' has probe identities absent from gallery: "
                f"{missing_person_ids}"
            )

        pairs[name] = {
            "gallery": gallery,
            "probe": probe,
            "gallery_loader": make_loader(gallery, cfg, train=False),
            "probe_loader": make_loader(probe, cfg, train=False),
            "subset_gallery_to_probe_ids": bool(
                pair_cfg.get("subset_gallery_to_probe_ids", False)
            ),
            "gallery_id_source": gallery_source,
            "probe_id_source": probe_source,
            "gallery_filters": gallery_filters,
            "probe_filters": probe_filters,
        }
    return pairs


def build_optimizer(
    model: torch.nn.Module,
    cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    optimizer_cfg = cfg.get("optimizer", {})
    name = str(optimizer_cfg.get("name", "adamw")).lower()
    lr = float(optimizer_cfg.get("lr", 0.001))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.01))
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=float(optimizer_cfg.get("momentum", 0.9)),
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, cfg: dict[str, Any]):
    scheduler_cfg = cfg.get("scheduler", {})
    name = str(scheduler_cfg.get("name", "cosine")).lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.get("train", {}).get("epochs", 50)),
        )
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(scheduler_cfg.get("milestones", [30, 40])),
        gamma=float(scheduler_cfg.get("gamma", 0.1)),
    )


def finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def tensors_are_finite(*tensors: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def classification_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
    scale = float(cfg.get("loss", {}).get("ce_scale", 1.0))
    return F.cross_entropy(outputs["logits"] * scale, labels)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    cfg: dict[str, Any],
    amp_enabled: bool,
    epoch: int,
) -> dict[str, float]:
    model.train()
    loss_cfg = cfg.get("loss", {})
    ce_weight = float(loss_cfg.get("ce_weight", 1.0))
    triplet_weight = float(loss_cfg.get("triplet_weight", 0.0))
    triplet_margin = float(loss_cfg.get("triplet_margin", 0.2))
    train_cfg = cfg.get("train", {})
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    grad_clip_norm = (
        None if grad_clip_norm is None else float(grad_clip_norm)
    )
    nonfinite_policy = str(
        train_cfg.get("nonfinite_policy", "raise")
    ).lower()
    max_nonfinite = int(train_cfg.get("max_nonfinite_batches", 0))

    loss_sum = 0.0
    ce_sum = 0.0
    triplet_sum = 0.0
    correct = 0
    count = 0
    valid_batches = 0
    nonfinite_batches = 0
    residual_sum = 0.0
    residual_count = 0
    temporal_association_sum = 0.0
    temporal_association_count = 0

    for step, batch in enumerate(loader, start=1):
        points = batch["points"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            outputs = model(points)
            ce_loss = classification_loss(outputs, labels, cfg)
            if triplet_weight > 0.0:
                triplet_feature_name = str(
                    loss_cfg.get("triplet_feature", "embedding")
                )
                if triplet_feature_name not in outputs:
                    raise KeyError(
                        f"Model output has no triplet feature {triplet_feature_name!r}"
                    )
                triplet_loss = batch_hard_triplet_loss(
                    outputs[triplet_feature_name],
                    labels,
                    margin=triplet_margin,
                )
            else:
                triplet_loss = ce_loss.new_zeros(())
            loss = ce_weight * ce_loss + triplet_weight * triplet_loss

        if not tensors_are_finite(
            loss,
            ce_loss,
            triplet_loss,
            outputs["logits"],
            outputs["embedding"],
        ):
            nonfinite_batches += 1
            message = (
                f"epoch {epoch:03d} step {step:04d}/{len(loader):04d} "
                "encountered non-finite loss/logits/embedding"
            )
            if nonfinite_policy == "raise" or nonfinite_batches > max_nonfinite:
                raise FloatingPointError(message)
            print(f"[warning] {message}; skip batch")
            continue

        scaler.scale(loss).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip_norm,
            )
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(labels.numel())
        loss_sum += float(loss.item()) * batch_size
        ce_sum += float(ce_loss.item()) * batch_size
        triplet_sum += float(triplet_loss.item()) * batch_size
        correct += int(
            (outputs["logits"].argmax(dim=1) == labels).sum().item()
        )
        count += batch_size
        valid_batches += 1
        if "residual_weight" in outputs:
            values = outputs["residual_weight"].detach().float()
            residual_sum += float(values.sum().item())
            residual_count += int(values.numel())
        if "temporal_association_weight" in outputs:
            values = outputs["temporal_association_weight"].detach().float()
            temporal_association_sum += float(values.sum().item())
            temporal_association_count += int(values.numel())

        log_interval = int(train_cfg.get("log_interval", 20))
        if log_interval > 0 and step % log_interval == 0:
            print(
                f"epoch {epoch:03d} step {step:04d}/{len(loader):04d} "
                f"loss {loss_sum / max(count, 1):.4f} "
                f"acc {correct / max(count, 1):.4f}"
            )

    metrics = {
        "loss": loss_sum / count if count else 1e12,
        "ce_loss": ce_sum / count if count else 1e12,
        "triplet_loss": triplet_sum / count if count else 1e12,
        "acc": correct / max(count, 1),
        "valid_batches": float(valid_batches),
        "nonfinite_batches": float(nonfinite_batches),
    }
    if residual_count:
        metrics["residual_weight"] = residual_sum / residual_count
    if temporal_association_count:
        metrics["temporal_association_weight"] = (
            temporal_association_sum / temporal_association_count
        )
    return metrics


@torch.no_grad()
def evaluate_classification(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    nonfinite_batches = 0
    for batch in loader:
        points = batch["points"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            outputs = model(points)
            loss = classification_loss(outputs, labels, cfg)
        if not tensors_are_finite(
            loss,
            outputs["logits"],
            outputs["embedding"],
        ):
            nonfinite_batches += 1
            continue
        batch_size = int(labels.numel())
        loss_sum += float(loss.item()) * batch_size
        correct += int(
            (outputs["logits"].argmax(dim=1) == labels).sum().item()
        )
        count += batch_size
    return {
        "loss": loss_sum / count if count else 1e12,
        "acc": correct / max(count, 1),
        "nonfinite_batches": float(nonfinite_batches),
    }


def _as_python_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    normalize: bool,
) -> dict[str, Any]:
    model.eval()
    embeddings = []
    labels = []
    metadata: dict[str, list[Any]] = {
        "person_id": [],
        "video_person_id": [],
        "clothes_id": [],
        "condition_tag": [],
        "video_id": [],
        "path": [],
    }
    dropped_nonfinite = 0

    for batch in loader:
        points = batch["points"].to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            outputs = model(points)
        embedding = outputs["embedding"].detach().float().cpu()
        if normalize:
            embedding = F.normalize(embedding, dim=1)

        batch_labels = _as_python_list(batch["label"])
        batch_metadata = {
            key: _as_python_list(batch[key])
            for key in metadata
            if key in batch
        }
        valid = torch.isfinite(embedding).all(dim=1)
        if not bool(valid.all().item()):
            keep = valid.tolist()
            dropped_nonfinite += int((~valid).sum().item())
            embedding = embedding[valid]
            batch_labels = [
                value
                for value, is_valid in zip(batch_labels, keep)
                if is_valid
            ]
            batch_metadata = {
                key: [
                    value
                    for value, is_valid in zip(values, keep)
                    if is_valid
                ]
                for key, values in batch_metadata.items()
            }

        embeddings.append(embedding)
        labels.extend(batch_labels)
        for key, values in batch_metadata.items():
            metadata[key].extend(values)

    return {
        "embeddings": (
            torch.cat(embeddings, dim=0)
            if embeddings
            else torch.empty(0, 1)
        ),
        "labels": torch.tensor(labels, dtype=torch.long),
        "meta": metadata,
        "dropped_nonfinite": dropped_nonfinite,
    }


def subset_features(
    features: dict[str, Any],
    indices: list[int],
) -> dict[str, Any]:
    index_tensor = torch.tensor(indices, dtype=torch.long)
    metadata = features.get("meta", {})
    return {
        "embeddings": features["embeddings"][index_tensor],
        "labels": features["labels"][index_tensor],
        "meta": {
            key: [values[index] for index in indices]
            for key, values in metadata.items()
        },
    }


def evaluate_with_subject_macro(
    gallery: dict[str, Any],
    probe: dict[str, Any],
    metric: str,
    ranks: tuple[int, ...],
    subset_gallery_to_probe_ids: bool,
) -> dict[str, Any]:
    metrics = evaluate_retrieval(
        gallery,
        probe,
        metric=metric,
        ranks=ranks,
        subset_gallery_to_probe_ids=subset_gallery_to_probe_ids,
    )
    person_values = [
        int(person_id)
        for person_id in probe.get("meta", {}).get("person_id", [])
    ]
    subject_metrics: dict[str, dict[str, Any]] = {}
    for person_id in sorted(set(person_values)):
        indices = [
            index
            for index, value in enumerate(person_values)
            if value == person_id
        ]
        subject_result = evaluate_retrieval(
            gallery,
            subset_features(probe, indices),
            metric=metric,
            ranks=ranks,
            subset_gallery_to_probe_ids=subset_gallery_to_probe_ids,
        )
        subject_metrics[f"P{person_id:03d}"] = subject_result

    macro: dict[str, float] = {
        "num_subjects": float(len(subject_metrics)),
    }
    for metric_name in ("mAP", *(f"rank{rank}" for rank in ranks)):
        values = [
            finite_float(result.get(metric_name), float("nan"))
            for result in subject_metrics.values()
        ]
        values = [value for value in values if math.isfinite(value)]
        macro[metric_name] = float(np.mean(values)) if values else 0.0

    metrics["subject"] = subject_metrics
    metrics["subject_macro"] = macro
    return metrics


def run_retrieval_eval(
    model: torch.nn.Module,
    eval_pairs: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    retrieval_cfg = cfg.get("retrieval", {})
    metric = str(retrieval_cfg.get("metric", "cosine"))
    ranks = tuple(int(rank) for rank in retrieval_cfg.get("ranks", [1, 5]))
    normalize = bool(
        retrieval_cfg.get("normalize_embeddings", True)
    )
    results: dict[str, Any] = {}
    for name, pair in eval_pairs.items():
        gallery = extract_embeddings(
            model,
            pair["gallery_loader"],
            device,
            amp_enabled,
            normalize,
        )
        probe = extract_embeddings(
            model,
            pair["probe_loader"],
            device,
            amp_enabled,
            normalize,
        )
        result = evaluate_with_subject_macro(
            gallery,
            probe,
            metric,
            ranks,
            pair["subset_gallery_to_probe_ids"],
        )
        result["gallery_dropped_nonfinite"] = int(
            gallery["dropped_nonfinite"]
        )
        result["probe_dropped_nonfinite"] = int(
            probe["dropped_nonfinite"]
        )
        results[name] = result
    return results


def compute_task_aggregates(
    retrieval_results: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    aggregate_results: dict[str, Any] = {}
    definitions = cfg.get("retrieval", {}).get("task_macro_aggregates", {})
    for name, pair_names in definitions.items():
        rows = [
            retrieval_results[pair_name]["subject_macro"]
            for pair_name in pair_names
            if pair_name in retrieval_results
        ]
        if not rows:
            continue
        aggregate_results[name] = {
            "pairs": [str(pair_name) for pair_name in pair_names],
            "rank1": float(np.mean([row.get("rank1", 0.0) for row in rows])),
            "rank5": float(np.mean([row.get("rank5", 0.0) for row in rows])),
            "mAP": float(np.mean([row.get("mAP", 0.0) for row in rows])),
        }
    return aggregate_results


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_provenance(
    run_dir: Path,
    config_path: Path,
    method_name: str,
    provenance_sources: Sequence[Path],
) -> None:
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "pointcloud_dataset.py",
        Path(__file__).resolve().parent / "protocol.py",
        Path(__file__).resolve().parent / "utils.py",
        *[Path(path).resolve() for path in provenance_sources],
    ]
    save_json(
        run_dir / "provenance.json",
        {
            "method": method_name,
            "config_source": str(config_path.resolve()),
            "sources": {
                str(path.relative_to(RELEASE_ROOT)): source_sha256(path)
                for path in source_paths
            },
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    )


def selection_value(
    retrieval_metrics: dict[str, Any],
) -> float:
    return finite_float(
        retrieval_metrics.get("val", {}).get("mAP"),
        -1.0,
    )


def parse_args(default_config: str, method_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train {method_name} under the Fixed-Special5 protocol."
    )
    parser.add_argument(
        "--config",
        default=default_config,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--split-index", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Use this exact output directory instead of a timestamped run directory.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", choices=["on", "off"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Build protocol and datasets without loading a training batch.",
    )
    return parser.parse_args()


def main(
    *,
    model_builder: ModelBuilder,
    default_config: str,
    method_name: str,
    provenance_sources: Sequence[Path] = (),
) -> None:
    args = parse_args(default_config, method_name)
    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    seed = int(
        args.seed
        if args.seed is not None
        else cfg.get("experiment", {}).get("seed", 0)
    )
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.device is not None:
        cfg.setdefault("train", {})["device"] = str(args.device)
    if args.split_index is not None:
        cfg.setdefault("protocol", {})["split_index"] = int(
            args.split_index
        )

    set_seed(seed)
    train_cfg = cfg.setdefault("train", {})
    deterministic = bool(train_cfg.get("deterministic", True))
    torch.backends.cudnn.benchmark = bool(
        train_cfg.get("cudnn_benchmark", False)
    )
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(cfg, seed=seed, run_name=args.run_name)
    cfg.setdefault("experiment", {})["seed"] = seed
    cfg["run_dir"] = str(run_dir)

    available_ids = collect_subject_ids(cfg)
    protocol = build_protocol(cfg, available_ids)
    fixed_budget = str(protocol.get("training_mode", "")).lower() == "fixed_budget"
    cfg["protocol_runtime"] = protocol
    data_cfg = cfg.setdefault("data", {})
    split_index = int(protocol["split_index"])
    loader_seed_base = int(data_cfg.get("loader_seed", seed))
    data_cfg["loader_seed"] = loader_seed_base
    data_cfg["train_loader_seed"] = (
        loader_seed_base + split_index * 1009
    )
    data_cfg["eval_loader_seed"] = (
        loader_seed_base + split_index * 1009 + 100_000
    )
    save_json(run_dir / "config.json", cfg)
    save_json(run_dir / "protocol.json", protocol)
    save_provenance(
        run_dir,
        config_path,
        method_name=method_name,
        provenance_sources=provenance_sources,
    )

    train_set = build_dataset(
        cfg,
        data_cfg.get("train_split", "train"),
        train=True,
        person_ids=protocol["train_ids"],
        filters=data_cfg.get("train_filters"),
    )
    val_set = None
    if not fixed_budget:
        val_set = build_dataset(
            cfg,
            data_cfg.get("val_split", "testA_probe"),
            train=False,
            person_ids=protocol["val_ids"],
            filters=data_cfg.get("val_filters"),
        )
    if len(train_set) == 0 or (val_set is not None and len(val_set) == 0):
        val_count = 0 if val_set is None else len(val_set)
        raise RuntimeError(
            f"Empty train/validation dataset: train={len(train_set)}, val={val_count}"
        )
    train_loader = make_loader(train_set, cfg, train=True)
    val_loader = (
        None if val_set is None else make_loader(val_set, cfg, train=False)
    )
    eval_pairs = build_eval_pairs(cfg, protocol)

    print(f"run_dir: {run_dir}")
    print(f"protocol: {protocol['name']} mode={protocol.get('training_mode', 'cv')}")
    print(f"train_ids: {protocol['train_ids']}")
    print(f"val_ids: {protocol['val_ids']}")
    print(f"general_test_ids: {protocol['general_test_ids']}")
    print(
        "fixed_special_test_ids: "
        f"{protocol['fixed_special_test_ids']}"
    )
    print(f"mixed_test_ids: {protocol['mixed_test_ids']}")
    print(f"train samples: {len(train_set)}")
    print(
        "coordinate adapter: "
        f"{data_cfg.get('coordinate_adapter', 'kinect_camera_xyz_mm_v1')}"
    )
    sampler_name = str(data_cfg.get("train_sampler", "random"))
    print(f"train sampler: {sampler_name}")
    print(
        "loader seeds: train={} eval={}".format(
            data_cfg["train_loader_seed"],
            data_cfg["eval_loader_seed"],
        )
    )
    print(f"validation samples: {0 if val_set is None else len(val_set)}")
    if fixed_budget:
        print("checkpoint policy: final epoch from a pre-specified fixed budget")
    for name, pair in eval_pairs.items():
        print(
            f"retrieval {name}: gallery={len(pair['gallery'])} "
            f"probe={len(pair['probe'])} "
            f"gallery_source={pair['gallery_id_source']} "
            f"probe_source={pair['probe_id_source']}"
        )

    if args.inspect_only:
        return

    device = resolve_device(train_cfg.get("device", "auto"))
    amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    model = model_builder(dict(cfg.get("model", {}))).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = make_scaler(amp_enabled)
    print(f"device: {device}")
    print(
        "model parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    if args.dry_run:
        batch = next(iter(train_loader))
        points = batch["points"].to(device)
        labels = batch["label"].to(device)
        with autocast_context(device, amp_enabled):
            outputs = model(points)
            ce_loss = classification_loss(outputs, labels, cfg)
            loss_cfg = cfg.get("loss", {})
            triplet_weight = float(loss_cfg.get("triplet_weight", 0.0))
            if triplet_weight > 0.0:
                triplet_feature_name = str(
                    loss_cfg.get("triplet_feature", "embedding")
                )
                triplet_loss = batch_hard_triplet_loss(
                    outputs[triplet_feature_name],
                    labels,
                    margin=float(loss_cfg.get("triplet_margin", 0.2)),
                )
            else:
                triplet_loss = ce_loss.new_zeros(())
            loss = (
                float(loss_cfg.get("ce_weight", 1.0)) * ce_loss
                + triplet_weight * triplet_loss
            )
        unique_labels, label_counts = torch.unique(
            labels.detach().cpu(),
            return_counts=True,
        )
        label_histogram = {
            int(label): int(count)
            for label, count in zip(unique_labels, label_counts)
        }
        print(f"dry_run points: {tuple(points.shape)}")
        print(f"dry_run label_counts: {label_histogram}")
        print("dry_run logits: " + str(tuple(outputs["logits"].shape)))
        print("dry_run embedding: " + str(tuple(outputs["embedding"].shape)))
        print(f"dry_run ce_loss: {float(ce_loss.item()):.4f}")
        print(f"dry_run triplet_loss: {float(triplet_loss.item()):.4f}")
        print(f"dry_run total_loss: {float(loss.item()):.4f}")
        if "residual_weight" in outputs:
            print(
                "dry_run residual_weight: "
                f"{float(outputs['residual_weight'].mean().item()):.6f}"
            )
        if "temporal_association_weight" in outputs:
            print(
                "dry_run temporal_association_weight: "
                f"{float(outputs['temporal_association_weight'].mean().item()):.6f}"
            )
        return

    wandb_run = init_wandb(cfg, run_dir, override=args.wandb)
    epochs = int(train_cfg.get("epochs", 50))
    save_every = int(train_cfg.get("save_every", 10))
    retrieval_every = int(
        train_cfg.get("retrieval_eval_every", 1)
    )
    best_metric = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
            amp_enabled,
            epoch,
        )
        val_metrics: dict[str, float] = {}
        retrieval_metrics: dict[str, Any] = {}
        if not fixed_budget:
            if val_loader is None:
                raise RuntimeError("Validation loader is missing in validation mode")
            val_metrics = evaluate_classification(
                model,
                val_loader,
                device,
                cfg,
                amp_enabled,
            )
        if (
            not fixed_budget
            and retrieval_every > 0
            and epoch % retrieval_every == 0
        ):
            retrieval_metrics = run_retrieval_eval(
                model,
                {"val": eval_pairs["val"]},
                cfg,
                device,
                amp_enabled,
            )
        selected = (
            None if fixed_budget else selection_value(retrieval_metrics)
        )

        if scheduler is not None:
            scheduler.step()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "lr": learning_rate,
            "selection_metric": selected,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_triplet_loss": train_metrics["triplet_loss"],
            "train_acc": train_metrics["acc"],
            "train_nonfinite_batches": train_metrics[
                "nonfinite_batches"
            ],
        }
        if val_metrics:
            record.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                    "val_nonfinite_batches": val_metrics[
                        "nonfinite_batches"
                    ],
                }
            )
        if "val" in retrieval_metrics:
            record.update(
                flatten_metrics(
                    "val_retrieval",
                    retrieval_metrics["val"],
                )
            )
        if "residual_weight" in train_metrics:
            record["train_residual_weight"] = train_metrics[
                "residual_weight"
            ]
        if "temporal_association_weight" in train_metrics:
            record["train_temporal_association_weight"] = train_metrics[
                "temporal_association_weight"
            ]
        append_jsonl(run_dir / "metrics.jsonl", record)

        if fixed_budget:
            print(
                f"epoch {epoch:03d}/{epochs:03d} "
                f"train_acc {train_metrics['acc']:.4f} "
                f"train_loss {train_metrics['loss']:.4f} "
                f"lr {learning_rate:.6f}"
            )
        else:
            print(
                f"epoch {epoch:03d}/{epochs:03d} "
                f"train_acc {train_metrics['acc']:.4f} "
                f"val_rank1 "
                f"{retrieval_metrics.get('val', {}).get('rank1', 0.0):.4f} "
                f"val_mAP "
                f"{retrieval_metrics.get('val', {}).get('mAP', 0.0):.4f} "
                f"lr {learning_rate:.6f}"
            )
        if wandb_run is not None:
            wandb_run.log(record, step=epoch)

        if not fixed_budget and selected is not None and selected > best_metric + 1e-12:
            best_metric = selected
            best_epoch = epoch
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                cfg,
            )
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                cfg,
            )
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric,
            cfg,
        )
        if fixed_budget and epoch == epochs:
            save_checkpoint(
                run_dir / "checkpoints" / "final.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                cfg,
            )

    selected_path = run_dir / "checkpoints" / (
        "final.pt" if fixed_budget else "best.pt"
    )
    if not selected_path.exists():
        raise RuntimeError(f"Selected checkpoint does not exist: {selected_path}")
    checkpoint = torch.load(selected_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if fixed_budget:
        best_metric = None
        best_epoch = epochs

    final_retrieval = run_retrieval_eval(
        model,
        eval_pairs,
        cfg,
        device,
        amp_enabled,
    )
    aggregates = compute_task_aggregates(final_retrieval, cfg)
    save_json(run_dir / "retrieval_metrics.json", final_retrieval)
    save_json(run_dir / "retrieval_aggregates.json", aggregates)
    summary: dict[str, Any] = {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "selection_pair": (
            "none_fixed_budget" if fixed_budget else "val"
        ),
        "checkpoint_policy": (
            "last_fixed_budget" if fixed_budget else "best_validation_mAP"
        ),
        "selected_checkpoint": str(selected_path),
        "selection_uses_fixed_special": False,
        "protocol": protocol,
        "retrieval": final_retrieval,
        "aggregates": aggregates,
        "run_dir": str(run_dir),
        "seed": seed,
    }
    current_weight = getattr(model, "current_residual_weight", None)
    if callable(current_weight):
        summary["residual_weight"] = float(
            current_weight().detach().item()
        )
    current_temporal_weight = getattr(
        model, "current_temporal_association_weight", None
    )
    if callable(current_temporal_weight):
        summary["temporal_association_weight"] = float(
            current_temporal_weight().detach().item()
        )
    save_json(run_dir / "summary.json", summary)

    print("final retrieval:")
    for name, metrics in final_retrieval.items():
        subject_macro = metrics.get("subject_macro", {})
        print(
            f"  {name}: rank1={metrics.get('rank1', 0.0):.4f} "
            f"mAP={metrics.get('mAP', 0.0):.4f} "
            f"subject_macro_mAP={subject_macro.get('mAP', 0.0):.4f}"
        )
    for name, metrics in aggregates.items():
        print(
            f"  aggregate/{name}: rank1={metrics['rank1']:.4f} "
            f"mAP={metrics['mAP']:.4f}"
        )

    if wandb_run is not None:
        log_data = {"best_epoch": best_epoch}
        if best_metric is not None:
            log_data["best_metric"] = best_metric
        for name, metrics in final_retrieval.items():
            log_data.update(flatten_metrics(f"final/{name}", metrics))
            log_data.update(
                flatten_metrics(
                    f"final_subject_macro/{name}",
                    metrics.get("subject_macro", {}),
                )
            )
        wandb_run.log(log_data)
        wandb_run.finish()

