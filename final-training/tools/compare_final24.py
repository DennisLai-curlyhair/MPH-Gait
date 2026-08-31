#!/usr/bin/env python3
"""Create separate cross-method C1 and C457 Final-24 comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PAIR_C1 = "final24_fixed5_c1"
PAIR_C457 = "final24_fixed5_c457"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def result_row(summary: dict[str, Any], pair_name: str) -> dict[str, Any]:
    values = summary["pair_aggregates"][pair_name]
    return {
        "method": summary["display_name"],
        "query_rank1_mean": values["rank1"]["mean"],
        "query_rank1_std": values["rank1"]["std"],
        "query_rank5_mean": values["rank5"]["mean"],
        "query_rank5_std": values["rank5"]["std"],
        "query_mAP_mean": values["mAP"]["mean"],
        "query_mAP_std": values["mAP"]["std"],
        "subject_macro_rank1_mean": values["subject_macro_rank1"]["mean"],
        "subject_macro_rank1_std": values["subject_macro_rank1"]["std"],
        "subject_macro_mAP_mean": values["subject_macro_mAP"]["mean"],
        "subject_macro_mAP_std": values["subject_macro_mAP"]["std"],
    }


def metric(value: float, std: float) -> str:
    return f"{value:.4f} +/- {std:.4f}"


def add_table(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| Method | Query R1 | Query R5 | Query mAP | Subject-macro R1 | Subject-macro mAP |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {method} | {r1} | {r5} | {map_} | {sr1} | {smap} |".format(
                method=row["method"],
                r1=metric(row["query_rank1_mean"], row["query_rank1_std"]),
                r5=metric(row["query_rank5_mean"], row["query_rank5_std"]),
                map_=metric(row["query_mAP_mean"], row["query_mAP_std"]),
                sr1=metric(
                    row["subject_macro_rank1_mean"],
                    row["subject_macro_rank1_std"],
                ),
                smap=metric(
                    row["subject_macro_mAP_mean"],
                    row["subject_macro_mAP_std"],
                ),
            )
        )
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointnet-root", required=True)
    parser.add_argument("--mph-root", required=True)
    parser.add_argument("--lidargaitpp-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    roots = (
        Path(args.pointnet_root).resolve(),
        Path(args.mph_root).resolve(),
        Path(args.lidargaitpp_root).resolve(),
    )
    summaries = [
        load_json(root / "summary" / "summary.json")
        for root in roots
    ]
    protocol_names = {summary["protocol_name"] for summary in summaries}
    train_ids = {tuple(summary["train_ids"]) for summary in summaries}
    fixed_ids = {
        tuple(summary["fixed_special_test_ids"]) for summary in summaries
    }
    if len(protocol_names) != 1 or len(train_ids) != 1 or len(fixed_ids) != 1:
        raise RuntimeError("Final-24 method summaries do not use the same protocol")

    c1_rows = [result_row(summary, PAIR_C1) for summary in summaries]
    c457_rows = [result_row(summary, PAIR_C457) for summary in summaries]
    output_dir = Path(args.output_dir).resolve()
    write_csv(output_dir / "comparison_c1.csv", c1_rows)
    write_csv(output_dir / "comparison_c457.csv", c457_rows)
    lines = [
        "# Final-24 Three-Method Comparison",
        "",
        "C1 and C457 are intentionally reported in separate tables.",
        "",
    ]
    add_table(lines, "C1 Personal Clothing", c1_rows)
    add_table(lines, "C457 Special Clothing", c457_rows)
    lines.extend(
        [
            "All three methods train on the same 24 development identities and "
            "evaluate the same Fixed-5 identities using their final fixed-budget checkpoint.",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps({"comparison": str(output_dir / "comparison.md")}, indent=2))


if __name__ == "__main__":
    main()
