#!/usr/bin/env python3
"""Summarize a five-split Fixed-Special5 LidarGait++ main run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

MAIN_PAIRS = (
    "val",
    "test_general_to_personal",
    "test_fixed5_to_personal",
    "test_fixed5_to_special",
    "test_mixed_c2_to_c3",
    "test_mixed_c3_to_c2",
    "diagnostic_fixed5_only_to_special",
)

EXPECTED_METHOD = "official_code_lidargaitpp"
EXPECTED_COORDINATE_ADAPTER = (
    "kinect_xyz_mm_to_lidar_forward_lateral_height_m_v1"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-splits", type=int, default=5)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_root = run_root / "summary"
    split_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    commits: set[str] = set()
    methods: set[str] = set()
    coordinate_adapters: set[str] = set()

    for split_index in range(args.expected_splits):
        split_root = run_root / f"split_{split_index}"
        metrics_path = split_root / "final_eval" / "retrieval_metrics.json"
        selection_path = split_root / "checkpoint_selection.json"
        diagnostics_path = split_root / "final_eval" / "embedding_diagnostics.json"
        manifest_path = split_root / "manifest.json"
        provenance_path = split_root / "provenance.json"
        required = (
            metrics_path,
            selection_path,
            diagnostics_path,
            manifest_path,
            provenance_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Split {split_index} is incomplete: {missing}")

        metrics = load_json(metrics_path)
        selection = load_json(selection_path)
        diagnostics = load_json(diagnostics_path)
        manifest = load_json(manifest_path)
        provenance = load_json(provenance_path)
        commits.add(str(provenance["official_commit"]))
        methods.add(str(manifest.get("method", "")))
        coordinate = manifest.get("coordinate_adapter", {})
        coordinate_adapters.add(str(coordinate.get("adapter", "")))
        if manifest.get("selection_uses_fixed_special") is not False:
            raise RuntimeError(f"Split {split_index} selection leakage flag is invalid")
        if coordinate.get("output_axes") != ["forward", "lateral", "height_up"]:
            raise RuntimeError(
                f"Split {split_index} has unexpected output axes: "
                f"{coordinate.get('output_axes')}"
            )
        if coordinate.get("output_unit") != "meter":
            raise RuntimeError(
                f"Split {split_index} has unexpected coordinate unit: "
                f"{coordinate.get('output_unit')}"
            )

        selection_rows.append(
            {
                "split": split_index,
                "selected_iteration": selection["selected_iteration"],
                "val_rank1": selection["selected_val_rank1"],
                "val_rank5": selection["selected_val_rank5"],
                "val_mAP": selection["selected_val_mAP"],
                "feature_std_mean": diagnostics["feature_std_mean"],
                "off_diagonal_cosine_mean": diagnostics[
                    "off_diagonal_cosine_mean"
                ],
                "all_finite": diagnostics["all_finite"],
            }
        )
        for pair_name, pair_metrics in metrics.items():
            row = {
                "split": split_index,
                "pair": pair_name,
                "num_gallery": pair_metrics.get("num_gallery", 0),
                "num_probe": pair_metrics.get("num_probe", 0),
                "rank1": pair_metrics.get("rank1", 0.0),
                "rank5": pair_metrics.get("rank5", 0.0),
                "mAP": pair_metrics.get("mAP", 0.0),
                "subject_macro_rank1": pair_metrics.get("subject_macro", {}).get(
                    "rank1", 0.0
                ),
                "subject_macro_rank5": pair_metrics.get("subject_macro", {}).get(
                    "rank5", 0.0
                ),
                "subject_macro_mAP": pair_metrics.get("subject_macro", {}).get(
                    "mAP", 0.0
                ),
            }
            split_rows.append(row)
            for subject, subject_metrics in pair_metrics.get("subject", {}).items():
                subject_rows.append(
                    {
                        "split": split_index,
                        "pair": pair_name,
                        "subject": subject,
                        "rank1": subject_metrics.get("rank1", 0.0),
                        "rank5": subject_metrics.get("rank5", 0.0),
                        "mAP": subject_metrics.get("mAP", 0.0),
                        "num_probe": subject_metrics.get("num_probe", 0),
                    }
                )

    if len(commits) != 1:
        raise RuntimeError(f"Multiple OpenGait commits found: {sorted(commits)}")
    if methods != {EXPECTED_METHOD}:
        raise RuntimeError(f"Unexpected or mixed method IDs: {sorted(methods)}")
    if coordinate_adapters != {EXPECTED_COORDINATE_ADAPTER}:
        raise RuntimeError(
            "Unexpected or mixed coordinate adapters: "
            f"{sorted(coordinate_adapters)}"
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_rows:
        grouped[str(row["pair"])].append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for pair_name, rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {"pair": pair_name, "num_splits": len(rows)}
        for metric_name in (
            "rank1",
            "rank5",
            "mAP",
            "subject_macro_rank1",
            "subject_macro_rank5",
            "subject_macro_mAP",
        ):
            mean, std = mean_std([float(row[metric_name]) for row in rows])
            aggregate[f"{metric_name}_mean"] = mean
            aggregate[f"{metric_name}_std"] = std
        aggregate_rows.append(aggregate)

    write_csv(output_root / "retrieval_split_metrics.csv", split_rows)
    write_csv(output_root / "retrieval_mean_std.csv", aggregate_rows)
    write_csv(output_root / "checkpoint_selection.csv", selection_rows)
    write_csv(output_root / "subject_split_metrics.csv", subject_rows)

    aggregate_by_name = {row["pair"]: row for row in aggregate_rows}
    lines = [
        "# Fixed-Special5 LidarGait++ Axis-Aligned v2 Len15",
        "",
        f"Run root: `{run_root}`",
        "",
        f"OpenGait commit: `{next(iter(commits))}`",
        "",
        f"Method ID: `{next(iter(methods))}`",
        "",
        f"Coordinate adapter: `{next(iter(coordinate_adapters))}`",
        "",
        "Checkpoint selection: validation C1 query-micro mAP only.",
        "",
        "| Pair | Query R1 / R5 / mAP | Subject-macro R1 / R5 / mAP |",
        "|---|---:|---:|",
    ]
    for pair_name in MAIN_PAIRS:
        row = aggregate_by_name[pair_name]
        lines.append(
            "| {pair} | {r1:.4f} / {r5:.4f} / {map_:.4f} | "
            "{sr1:.4f} / {sr5:.4f} / {smap:.4f} |".format(
                pair=pair_name,
                r1=row["rank1_mean"],
                r5=row["rank5_mean"],
                map_=row["mAP_mean"],
                sr1=row["subject_macro_rank1_mean"],
                sr5=row["subject_macro_rank5_mean"],
                smap=row["subject_macro_mAP_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Selected Checkpoints",
            "",
            "| Split | Iteration | Val R1 / R5 / mAP | Feature std |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in selection_rows:
        lines.append(
            "| {split} | {iteration} | {r1:.4f} / {r5:.4f} / {map_:.4f} | {std:.6f} |".format(
                split=row["split"],
                iteration=row["selected_iteration"],
                r1=float(row["val_rank1"]),
                r5=float(row["val_rank5"]),
                map_=float(row["val_mAP"]),
                std=float(row["feature_std_mean"]),
            )
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(output_root / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
