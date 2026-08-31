#!/usr/bin/env python3
"""Summarize one Final-24 method while keeping C1 and C457 separate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PAIR_C1 = "final24_fixed5_c1"
PAIR_C457 = "final24_fixed5_c457"
CONDITION_PAIRS = (
    "final24_fixed5_c4",
    "final24_fixed5_c5",
    "final24_fixed5_c7",
)
EXPECTED_TRAIN_IDS = {
    1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 20, 21, 22, 23, 24, 25, 27, 28, 29,
}
EXPECTED_FIXED_IDS = {3, 7, 18, 19, 26}
METRICS = (
    "rank1",
    "rank5",
    "mAP",
    "subject_macro_rank1",
    "subject_macro_rank5",
    "subject_macro_mAP",
)
DISPLAY_NAMES = {
    "pointnet_tmax": "PointNet-TMax",
    "mph_gait": "MPH-Gait",
    "lidargaitpp": "LidarGait++",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.replace(",", " ").split()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"Seeds must be a non-empty unique list: {value!r}")
    return seeds


def metric_row(seed: int, pair_name: str, values: dict[str, Any]) -> dict[str, Any]:
    subject_macro = values.get("subject_macro", {})
    return {
        "seed": seed,
        "pair": pair_name,
        "num_gallery": int(values.get("num_gallery", 0)),
        "num_probe": int(values.get("num_probe", 0)),
        "num_subjects": int(subject_macro.get("num_subjects", 0)),
        "rank1": float(values.get("rank1", 0.0)),
        "rank5": float(values.get("rank5", 0.0)),
        "mAP": float(values.get("mAP", 0.0)),
        "subject_macro_rank1": float(subject_macro.get("rank1", 0.0)),
        "subject_macro_rank5": float(subject_macro.get("rank5", 0.0)),
        "subject_macro_mAP": float(subject_macro.get("mAP", 0.0)),
    }


def load_seed(
    method: str,
    seed_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if method == "lidargaitpp":
        metrics_path = seed_root / "final_eval" / "retrieval_metrics.json"
        summary_path = seed_root / "final_eval" / "summary.json"
        protocol_path = seed_root / "protocol.json"
    else:
        metrics_path = seed_root / "retrieval_metrics.json"
        summary_path = seed_root / "summary.json"
        protocol_path = seed_root / "protocol.json"

    required = (metrics_path, summary_path, protocol_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete Final-24 seed under {seed_root}: {missing}")

    metrics = load_json(metrics_path)
    summary = load_json(summary_path)
    protocol = load_json(protocol_path)
    if set(protocol.get("train_ids", [])) != EXPECTED_TRAIN_IDS:
        raise RuntimeError(f"Unexpected Final-24 train IDs in {protocol_path}")
    if set(protocol.get("fixed_special_test_ids", [])) != EXPECTED_FIXED_IDS:
        raise RuntimeError(f"Unexpected Fixed-5 IDs in {protocol_path}")
    if protocol.get("checkpoint_policy") != "last_fixed_budget":
        raise RuntimeError(f"Unexpected checkpoint policy in {protocol_path}")
    if protocol.get("selection_uses_fixed_special") is not False:
        raise RuntimeError(f"Fixed-5 selection leakage flag is invalid in {protocol_path}")
    if set(protocol.get("train_ids", [])) & set(
        protocol.get("fixed_special_test_ids", [])
    ):
        raise RuntimeError(f"Train/evaluation identity overlap in {protocol_path}")

    expected_pairs = {PAIR_C1, PAIR_C457, *CONDITION_PAIRS}
    missing_pairs = sorted(expected_pairs - set(metrics))
    if missing_pairs:
        raise RuntimeError(f"Missing Final-24 pairs in {metrics_path}: {missing_pairs}")
    checkpoint = str(
        summary.get("selected_checkpoint", summary.get("checkpoint", ""))
    )
    return metrics, protocol, checkpoint


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"num_seeds": len(rows)}
    for metric_name in METRICS:
        values = [float(row[metric_name]) for row in rows]
        result[metric_name] = {
            "mean": mean(values),
            "std": pstdev(values),
            "values": values,
        }
    result["num_gallery"] = sorted({int(row["num_gallery"]) for row in rows})
    result["num_probe"] = sorted({int(row["num_probe"]) for row in rows})
    result["num_subjects"] = sorted({int(row["num_subjects"]) for row in rows})
    return result


def format_mean_std(value: dict[str, Any]) -> str:
    return f"{value['mean']:.4f} +/- {value['std']:.4f}"


def pair_table(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| Seed | Query R1 | Query R5 | Query mAP | Subject-macro R1 | Subject-macro mAP |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {seed} | {rank1:.4f} | {rank5:.4f} | {mAP:.4f} | "
            "{subject_macro_rank1:.4f} | {subject_macro_mAP:.4f} |".format(**row)
        )
    values = aggregate(rows)
    lines.append(
        "| mean +/- std | {r1} | {r5} | {map_} | {sr1} | {smap} |".format(
            r1=format_mean_std(values["rank1"]),
            r5=format_mean_std(values["rank5"]),
            map_=format_mean_std(values["mAP"]),
            sr1=format_mean_std(values["subject_macro_rank1"]),
            smap=format_mean_std(values["subject_macro_mAP"]),
        )
    )
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(DISPLAY_NAMES),
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seeds", default="0 1 2")
    args = parser.parse_args()

    method = str(args.method)
    run_root = Path(args.run_root).resolve()
    seeds = parse_seeds(args.seeds)
    output_root = run_root / "summary"
    rows: list[dict[str, Any]] = []
    checkpoints: dict[str, str] = {}
    protocols: list[dict[str, Any]] = []

    for seed in seeds:
        seed_root = run_root / f"seed_{seed}"
        metrics, protocol, checkpoint = load_seed(method, seed_root)
        protocols.append(protocol)
        checkpoints[str(seed)] = checkpoint
        for pair_name in (PAIR_C1, PAIR_C457, *CONDITION_PAIRS):
            rows.append(metric_row(seed, pair_name, metrics[pair_name]))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair"])].append(row)
    aggregates = {
        pair_name: aggregate(pair_rows)
        for pair_name, pair_rows in sorted(grouped.items())
    }

    c1_rows = grouped[PAIR_C1]
    c457_rows = grouped[PAIR_C457]
    write_csv(output_root / "all_seed_metrics.csv", rows)
    write_csv(output_root / "c1_seed_metrics.csv", c1_rows)
    write_csv(output_root / "c457_seed_metrics.csv", c457_rows)
    write_csv(
        output_root / "c4_c5_c7_seed_metrics.csv",
        [
            row
            for pair_name in CONDITION_PAIRS
            for row in grouped[pair_name]
        ],
    )
    save_json(output_root / "c1_mean_std.json", aggregates[PAIR_C1])
    save_json(output_root / "c457_mean_std.json", aggregates[PAIR_C457])
    save_json(
        output_root / "summary.json",
        {
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "protocol_name": protocols[0]["name"],
            "checkpoint_policy": "last_fixed_budget",
            "train_ids": sorted(EXPECTED_TRAIN_IDS),
            "fixed_special_test_ids": sorted(EXPECTED_FIXED_IDS),
            "seeds": seeds,
            "checkpoints": checkpoints,
            "pair_aggregates": aggregates,
            "c1_and_c457_reported_separately": True,
        },
    )

    lines = [
        f"# {DISPLAY_NAMES[method]} Final-24 Len15 {len(seeds)}-Seed Summary",
        "",
        f"Run root: `{run_root}`",
        "",
        "Training: 24 development identities, C2/C3 V001-V016.",
        "",
        "Evaluation: Fixed-5 identities only; C1 and C4/C5/C7 are reported separately.",
        "",
        "Checkpoint: final checkpoint from the pre-specified fixed training budget. "
        "Fixed-5 is not used for checkpoint selection.",
        "",
    ]
    pair_table(lines, "C1 Personal Clothing", c1_rows)
    pair_table(lines, "C457 Special Clothing", c457_rows)
    lines.extend(
        [
            "## C4 / C5 / C7 Breakdown",
            "",
            "| Condition | Query R1 | Query mAP | Subject-macro mAP |",
            "|---|---:|---:|---:|",
        ]
    )
    for pair_name in CONDITION_PAIRS:
        values = aggregates[pair_name]
        lines.append(
            "| {name} | {r1} | {map_} | {smap} |".format(
                name=pair_name.rsplit("_", 1)[-1].upper(),
                r1=format_mean_std(values["rank1"]),
                map_=format_mean_std(values["mAP"]),
                smap=format_mean_std(values["subject_macro_mAP"]),
            )
        )
    lines.extend(
        [
            "",
            "Rank-5 is reported for completeness but should be interpreted cautiously: "
            "this gallery contains only five identities and multiple samples per identity.",
            "",
        ]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(output_root / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
