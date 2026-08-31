#!/usr/bin/env python3
"""Validate that all Final-24 configs use the same identities and retrieval pairs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


FINAL24_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = FINAL24_ROOT.parent
sys.path.insert(0, str(RELEASE_ROOT / "shared"))

from gait_core.protocol import build_protocol  # noqa: E402


CONFIGS = (
    FINAL24_ROOT / "configs/pointnet_tmax_final24_len15.yaml",
    FINAL24_ROOT / "configs/mph_gait_final24_len15.yaml",
    FINAL24_ROOT / "configs/lidargaitpp_final24_protocol_len15.yaml",
)
EXPECTED_PAIRS = {
    "final24_fixed5_c1",
    "final24_fixed5_c457",
    "final24_fixed5_c4",
    "final24_fixed5_c5",
    "final24_fixed5_c7",
}


def main() -> None:
    protocols = []
    pair_sets = []
    for path in CONFIGS:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        protocol = build_protocol(config, list(range(1, 30)))
        pairs = set(config.get("retrieval", {}).get("pairs", {}))
        if pairs != EXPECTED_PAIRS:
            raise RuntimeError(f"Unexpected retrieval pairs in {path}: {sorted(pairs)}")
        protocols.append(protocol)
        pair_sets.append(pairs)

    signatures = {
        (
            tuple(protocol["train_ids"]),
            tuple(protocol["fixed_special_test_ids"]),
            protocol["checkpoint_policy"],
            protocol["name"],
        )
        for protocol in protocols
    }
    if len(signatures) != 1 or len(set(map(frozenset, pair_sets))) != 1:
        raise RuntimeError("Final-24 configs are not protocol-aligned")

    protocol = protocols[0]
    print(
        json.dumps(
            {
                "status": "ok",
                "configs": [str(path) for path in CONFIGS],
                "protocol": protocol["name"],
                "checkpoint_policy": protocol["checkpoint_policy"],
                "train_ids": protocol["train_ids"],
                "fixed_special_test_ids": protocol["fixed_special_test_ids"],
                "retrieval_pairs": sorted(EXPECTED_PAIRS),
                "c1_and_c457_separate": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
