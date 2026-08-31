from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


KEY_PAIRS = (
    "test_general_to_personal",
    "test_fixed5_to_personal",
    "test_fixed5_to_special",
    "test_fixed5_c4c5_normal",
    "test_fixed5_c7",
    "test_mixed_c2_to_c3",
    "test_mixed_c3_to_c2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize projection v2 fixed-special5 results."
    )
    parser.add_argument("run_groups", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tag", default="fixed_special5_summary")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fold_result_paths(run_group: Path) -> list[Path]:
    paths = []
    for split_dir in sorted(run_group.glob("split*")):
        candidates = sorted(split_dir.glob("*/retrieval_metrics.json"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one retrieval_metrics.json under {split_dir}, "
                f"found {len(candidates)}"
            )
        paths.append(candidates[0])
    if not paths:
        raise RuntimeError(f"No fold results found under {run_group}")
    return paths


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(
        {key for row in rows for key in row},
        key=lambda key: (
            key
            not in {
                "run_group",
                "method_id",
                "split_index",
                "pair",
                "subject",
            },
            key,
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std_rows(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in group_keys)].append(row)

    output = []
    ignored = set(group_keys) | {"split_index"}
    for group_values, group_rows in sorted(grouped.items()):
        result = dict(zip(group_keys, group_values))
        result["n_splits"] = len(group_rows)
        metric_keys = sorted(
            {
                key
                for row in group_rows
                for key in row
                if key not in ignored
            }
        )
        for key in metric_keys:
            values = [
                value
                for row in group_rows
                if (value := finite(row.get(key))) is not None
            ]
            if values:
                result[f"{key}_mean"] = float(np.mean(values))
                result[f"{key}_std"] = float(np.std(values))
        output.append(result)
    return output


def format_metric(row: dict[str, Any], prefix: str) -> str:
    mean = finite(row.get(f"{prefix}_mean"))
    std = finite(row.get(f"{prefix}_std"))
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def main() -> None:
    args = parse_args()
    fold_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for run_group in args.run_groups:
        for result_path in fold_result_paths(run_group):
            run_dir = result_path.parent
            protocol = load_json(run_dir / "protocol.json")
            config = load_json(run_dir / "config.json")
            metrics = load_json(result_path)
            method_id = str(
                config.get("method", {}).get("id", "unknown")
            )
            split_index = int(protocol["split_index"])
            base = {
                "run_group": run_group.name,
                "method_id": method_id,
                "split_index": split_index,
            }

            for pair_name, pair_metrics in sorted(metrics.items()):
                macro = pair_metrics.get("subject_macro", {})
                row = {
                    **base,
                    "pair": pair_name,
                    "num_gallery": pair_metrics.get("num_gallery", 0),
                    "num_probe": pair_metrics.get("num_probe", 0),
                    "query_rank1": pair_metrics.get("rank1", 0.0),
                    "query_rank5": pair_metrics.get("rank5", 0.0),
                    "query_mAP": pair_metrics.get("mAP", 0.0),
                    "subject_rank1": macro.get("rank1", 0.0),
                    "subject_rank5": macro.get("rank5", 0.0),
                    "subject_mAP": macro.get("mAP", 0.0),
                    "num_subjects": macro.get("num_subjects", 0),
                }
                fold_rows.append(row)
                for subject, values in sorted(
                    pair_metrics.get("subject", {}).items()
                ):
                    subject_rows.append(
                        {
                            **base,
                            "pair": pair_name,
                            "subject": subject,
                            "num_probe": values.get("num_probe", 0),
                            "rank1": values.get("rank1", 0.0),
                            "rank5": values.get("rank5", 0.0),
                            "mAP": values.get("mAP", 0.0),
                        }
                    )

            aggregate_path = run_dir / "retrieval_aggregates.json"
            if aggregate_path.exists():
                for aggregate_name, values in load_json(
                    aggregate_path
                ).items():
                    aggregate_rows.append(
                        {
                            **base,
                            "aggregate": aggregate_name,
                            "rank1": values.get("rank1", 0.0),
                            "rank5": values.get("rank5", 0.0),
                            "mAP": values.get("mAP", 0.0),
                        }
                    )

    fold_summary = mean_std_rows(
        fold_rows,
        ("run_group", "method_id", "pair"),
    )
    subject_summary = mean_std_rows(
        subject_rows,
        ("run_group", "method_id", "pair", "subject"),
    )
    aggregate_summary = mean_std_rows(
        aggregate_rows,
        ("run_group", "method_id", "aggregate"),
    )

    cohort_rows = []
    grouped_subjects: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in subject_summary:
        grouped_subjects[
            (
                str(row["run_group"]),
                str(row["method_id"]),
                str(row["pair"]),
            )
        ].append(row)
    for (run_group, method_id, pair), rows in sorted(
        grouped_subjects.items()
    ):
        cohort_row: dict[str, Any] = {
            "run_group": run_group,
            "method_id": method_id,
            "pair": pair,
            "num_subjects": len(rows),
        }
        for metric_name in ("rank1", "rank5", "mAP"):
            values = [
                value
                for row in rows
                if (
                    value := finite(row.get(f"{metric_name}_mean"))
                )
                is not None
            ]
            cohort_row[f"{metric_name}_subject_macro"] = (
                float(np.mean(values)) if values else 0.0
            )
        cohort_rows.append(cohort_row)

    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "retrieval_split_metrics.csv", fold_rows)
    write_csv(out_dir / "retrieval_split_mean_std.csv", fold_summary)
    write_csv(out_dir / "subject_split_metrics.csv", subject_rows)
    write_csv(out_dir / "subject_across_split_mean_std.csv", subject_summary)
    write_csv(out_dir / "subject_cohort_summary.csv", cohort_rows)
    write_csv(out_dir / "task_aggregate_split_metrics.csv", aggregate_rows)
    write_csv(
        out_dir / "task_aggregate_mean_std.csv",
        aggregate_summary,
    )

    lookup = {
        (str(row["run_group"]), str(row["pair"])): row
        for row in fold_summary
    }
    lines = [
        f"# {args.tag}",
        "",
        "數字為五組 development splits 的 mean +/- population std。"
        "同一 fixed-special5 cohort 重複測試，因此五組不是五個獨立"
        " special test sets。",
        "",
        "| Run group | Pair | Query R1 | Query mAP | Subject-macro mAP |",
        "|---|---|---:|---:|---:|",
    ]
    for run_group in args.run_groups:
        for pair in KEY_PAIRS:
            row = lookup.get((run_group.name, pair))
            if row is None:
                continue
            lines.append(
                "| "
                f"{run_group.name} | {pair} | "
                f"{format_metric(row, 'query_rank1')} | "
                f"{format_metric(row, 'query_mAP')} | "
                f"{format_metric(row, 'subject_mAP')} |"
            )
    lines.extend(
        [
            "",
            "Fixed-five 的主要統計應查看 "
            "`subject_across_split_mean_std.csv` 與 "
            "`subject_cohort_summary.csv`：先對同一 subject 跨 splits "
            "聚合，再對五位 subject 等權平均。",
            "",
        ]
    )
    (out_dir / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"summary: {out_dir}")


if __name__ == "__main__":
    main()

