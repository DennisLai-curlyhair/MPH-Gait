from __future__ import annotations

from typing import Any


DEFAULT_FIXED_SPECIAL_IDS = [3, 7, 18, 19, 26]
DEFAULT_DEVELOPMENT_GROUPS = [
    [1, 5, 9, 11, 20],
    [12, 17, 21, 22, 25],
    [2, 10, 14, 24, 27],
    [4, 8, 13, 16, 23],
    [6, 15, 28, 29],
]


def _as_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list of integer IDs")
    result = [int(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate IDs: {result}")
    return result


def _development_groups(protocol_cfg: dict[str, Any]) -> list[list[int]]:
    raw_groups = protocol_cfg.get("development_groups", DEFAULT_DEVELOPMENT_GROUPS)
    if not isinstance(raw_groups, (list, tuple)):
        raise TypeError("protocol.development_groups must be a list of ID lists")
    groups = [
        sorted(_as_int_list(group, f"development_groups[{index}]"))
        for index, group in enumerate(raw_groups)
    ]
    if len(groups) != 5:
        raise ValueError(f"Expected exactly five development groups, got {len(groups)}")
    return groups


def build_fixed_special5_protocol(
    cfg: dict[str, Any],
    available_ids: list[int] | tuple[int, ...] | set[int],
) -> dict[str, Any]:
    protocol_cfg = cfg.get("protocol", {})
    if not bool(protocol_cfg.get("enabled", True)):
        raise ValueError("Fixed-special5 experiments require protocol.enabled=true")

    groups = _development_groups(protocol_cfg)
    fixed_special_ids = sorted(
        _as_int_list(
            protocol_cfg.get("fixed_special_test_ids", DEFAULT_FIXED_SPECIAL_IDS),
            "fixed_special_test_ids",
        )
    )
    available = {int(person_id) for person_id in available_ids}

    flattened = [person_id for group in groups for person_id in group]
    development_ids = set(flattened)
    fixed_ids = set(fixed_special_ids)
    if len(flattened) != len(development_ids):
        raise ValueError("A development identity occurs in more than one group")
    if development_ids & fixed_ids:
        overlap = sorted(development_ids & fixed_ids)
        raise ValueError(f"Development and fixed-special identities overlap: {overlap}")

    configured_ids = development_ids | fixed_ids
    if configured_ids != available:
        missing = sorted(available - configured_ids)
        unknown = sorted(configured_ids - available)
        raise ValueError(
            "Protocol IDs do not match dataset IDs: "
            f"missing_from_protocol={missing}, absent_from_dataset={unknown}"
        )

    split_index = int(protocol_cfg.get("split_index", protocol_cfg.get("fold_index", 0)))
    if split_index < 0 or split_index >= len(groups):
        raise ValueError(f"split_index must be in [0, {len(groups) - 1}], got {split_index}")

    general_test_group_index = int(
        protocol_cfg.get("general_test_group_index", split_index)
    )
    val_group_index = protocol_cfg.get("val_group_index")
    if val_group_index is None:
        val_group_index = (
            general_test_group_index + int(protocol_cfg.get("val_group_offset", 1))
        ) % len(groups)
    val_group_index = int(val_group_index)

    if general_test_group_index < 0 or general_test_group_index >= len(groups):
        raise ValueError("general_test_group_index is outside development_groups")
    if val_group_index < 0 or val_group_index >= len(groups):
        raise ValueError("val_group_index is outside development_groups")
    if general_test_group_index == val_group_index:
        raise ValueError("Validation and general-test groups must differ")

    general_test_ids = sorted(groups[general_test_group_index])
    val_ids = sorted(groups[val_group_index])
    held_out = set(general_test_ids) | set(val_ids)
    train_ids = sorted(development_ids - held_out)
    mixed_test_ids = sorted(set(general_test_ids) | fixed_ids)

    overlaps = {
        "train_val": sorted(set(train_ids) & set(val_ids)),
        "train_general_test": sorted(set(train_ids) & set(general_test_ids)),
        "train_fixed_special": sorted(set(train_ids) & fixed_ids),
        "val_general_test": sorted(set(val_ids) & set(general_test_ids)),
        "val_fixed_special": sorted(set(val_ids) & fixed_ids),
    }
    if any(overlaps.values()):
        raise ValueError(f"Subject-disjoint protocol has identity overlap: {overlaps}")

    return {
        "enabled": True,
        "name": str(
            protocol_cfg.get(
                "name", "fixed_special5_subject_holdout_5split_v1"
            )
        ),
        "version": str(
            protocol_cfg.get(
                "version", "fixed_special5_subject_holdout_5split_v1_20260724"
            )
        ),
        "num_splits": len(groups),
        "split_index": split_index,
        "fold_index": split_index,
        "general_test_group_index": general_test_group_index,
        "val_group_index": val_group_index,
        "all_ids": sorted(available),
        "development_ids": sorted(development_ids),
        "fixed_special_test_ids": fixed_special_ids,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "general_test_ids": general_test_ids,
        "mixed_test_ids": mixed_test_ids,
        "development_groups": groups,
        "selection_uses_fixed_special": False,
        "overlaps": overlaps,
    }


def build_final24_protocol(
    cfg: dict[str, Any],
    available_ids: list[int] | tuple[int, ...] | set[int],
) -> dict[str, Any]:
    """Train on all 24 development identities and reserve Fixed-5 for evaluation."""
    protocol_cfg = cfg.get("protocol", {})
    if not bool(protocol_cfg.get("enabled", True)):
        raise ValueError("Final-24 requires protocol.enabled=true")

    groups = _development_groups(protocol_cfg)
    fixed_special_ids = sorted(
        _as_int_list(
            protocol_cfg.get("fixed_special_test_ids", DEFAULT_FIXED_SPECIAL_IDS),
            "fixed_special_test_ids",
        )
    )
    available = {int(person_id) for person_id in available_ids}
    flattened = [person_id for group in groups for person_id in group]
    development_ids = set(flattened)
    fixed_ids = set(fixed_special_ids)

    if len(flattened) != len(development_ids):
        raise ValueError("A development identity occurs in more than one group")
    if development_ids & fixed_ids:
        overlap = sorted(development_ids & fixed_ids)
        raise ValueError(f"Development and fixed-special identities overlap: {overlap}")

    configured_ids = development_ids | fixed_ids
    if configured_ids != available:
        missing = sorted(available - configured_ids)
        unknown = sorted(configured_ids - available)
        raise ValueError(
            "Final-24 IDs do not match dataset IDs: "
            f"missing_from_protocol={missing}, absent_from_dataset={unknown}"
        )
    if len(development_ids) != 24:
        raise ValueError(
            f"Final-24 requires exactly 24 development identities, got {len(development_ids)}"
        )

    overlaps = {
        "train_fixed_special": sorted(development_ids & fixed_ids),
    }
    if any(overlaps.values()):
        raise ValueError(f"Final-24 has identity overlap: {overlaps}")

    return {
        "enabled": True,
        "mode": "final24_fixed_budget",
        "training_mode": "fixed_budget",
        "name": str(protocol_cfg.get("name", "final24_fixed5_eval_v1")),
        "version": str(
            protocol_cfg.get("version", "final24_fixed5_eval_v1_20260817")
        ),
        "num_splits": 1,
        "split_index": 0,
        "fold_index": 0,
        "all_ids": sorted(available),
        "development_ids": sorted(development_ids),
        "fixed_special_test_ids": fixed_special_ids,
        "final_eval_ids": fixed_special_ids,
        "train_ids": sorted(development_ids),
        "val_ids": [],
        "general_test_ids": [],
        "mixed_test_ids": fixed_special_ids,
        "development_groups": groups,
        "selection_pair": None,
        "selection_metric": None,
        "selection_uses_fixed_special": False,
        "checkpoint_policy": "last_fixed_budget",
        "overlaps": overlaps,
    }


def build_protocol(
    cfg: dict[str, Any],
    available_ids: list[int] | tuple[int, ...] | set[int],
) -> dict[str, Any]:
    """Dispatch to the configured protocol without changing legacy defaults."""
    protocol_cfg = cfg.get("protocol", {})
    mode = str(protocol_cfg.get("mode", "fixed_special5_5split")).lower()
    if mode in {"final24", "final24_fixed_budget", "final24_fixed5_eval"}:
        return build_final24_protocol(cfg, available_ids)
    return build_fixed_special5_protocol(cfg, available_ids)
