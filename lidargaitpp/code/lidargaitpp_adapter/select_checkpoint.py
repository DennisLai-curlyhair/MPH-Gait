#!/usr/bin/env python3
"""Select one LidarGait++ checkpoint using validation C1 mAP only."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_iteration(path: str | Path) -> int:
    match = re.search(r"-(\d+)\.pt$", str(path))
    if not match:
        raise ValueError(f"Cannot parse checkpoint iteration from {path}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selection_root = Path(args.selection_root).resolve()
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(selection_root.glob("iter_*/summary.json")):
        summary = load_json(summary_path)
        if summary.get("pairs") != ["val"]:
            raise RuntimeError(
                f"Checkpoint selection output contains non-validation pairs: {summary_path}"
            )
        checkpoint = str(summary["checkpoint"])
        metrics = summary["retrieval_metrics"]["val"]
        rows.append(
            {
                "iteration": checkpoint_iteration(checkpoint),
                "checkpoint": checkpoint,
                "rank1": float(metrics.get("rank1", 0.0)),
                "rank5": float(metrics.get("rank5", 0.0)),
                "mAP": float(metrics.get("mAP", 0.0)),
                "subject_macro_mAP": float(
                    metrics.get("subject_macro", {}).get("mAP", 0.0)
                ),
                "summary": str(summary_path),
            }
        )
    if not rows:
        raise RuntimeError(f"No validation checkpoint summaries under {selection_root}")

    best = sorted(rows, key=lambda row: (-row["mAP"], row["iteration"]))[0]
    result = {
        "selection_pair": "val",
        "selection_metric": "query_micro_mAP",
        "selection_uses_fixed_special": False,
        "tie_break": "earliest_iteration",
        "selected_iteration": int(best["iteration"]),
        "selected_checkpoint": str(best["checkpoint"]),
        "selected_val_rank1": float(best["rank1"]),
        "selected_val_rank5": float(best["rank5"]),
        "selected_val_mAP": float(best["mAP"]),
        "selected_val_subject_macro_mAP": float(best["subject_macro_mAP"]),
        "candidates": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "iteration",
                "checkpoint",
                "rank1",
                "rank5",
                "mAP",
                "subject_macro_mAP",
                "summary",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
