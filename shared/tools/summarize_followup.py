#!/usr/bin/env python3
"""Summarize Fixed-Special5 follow-up runs across matched splits and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = ("rank1", "rank5", "mAP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_field(path: Path, prefix: str) -> str:
    for part in path.parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return "unknown"


def collect(result_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_path in sorted(result_root.rglob("retrieval_metrics.json")):
        config_path = metric_path.with_name("config.json")
        relative = metric_path.relative_to(result_root)
        parts = relative.parts
        if len(parts) < 6:
            continue
        suite, setting = parts[0], parts[1]
        seed = path_field(relative, "seed_")
        split = path_field(relative, "split_")

        if config_path.is_file():
            config = load_json(config_path)
            method = str(config.get("method", {}).get("id", "unknown"))
        elif metric_path.parent.name == "final_eval":
            manifest_path = metric_path.parent.parent / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = load_json(manifest_path)
            method = str(manifest.get("method", "unknown"))
            if int(
                manifest.get("randomness", {}).get("training_seed", -1)
            ) != int(seed):
                raise RuntimeError(f"Training seed mismatch for {metric_path}")
            if int(manifest.get("split_index", -1)) != int(split):
                raise RuntimeError(f"Split mismatch for {metric_path}")
        else:
            continue

        retrieval = load_json(metric_path)
        for pair, values in retrieval.items():
            if not isinstance(values, dict) or values.get("skipped") is True:
                continue
            for aggregation, metric_values in (
                ("query_micro", values),
                ("subject_macro", values.get("subject_macro", {})),
            ):
                if not isinstance(metric_values, dict):
                    continue
                for metric in METRICS:
                    value = metric_values.get(metric)
                    if isinstance(value, (int, float)):
                        rows.append(
                            {
                                "method": method,
                                "suite": suite,
                                "setting": setting,
                                "seed": int(seed),
                                "split": int(split),
                                "pair": pair,
                                "aggregation": aggregation,
                                "metric": metric,
                                "value": float(value),
                                "source": str(metric_path),
                            }
                        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(
            str(row[name])
            for name in (
                "method",
                "suite",
                "setting",
                "pair",
                "aggregation",
                "metric",
            )
        )
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        by_seed: dict[int, list[float]] = defaultdict(list)
        for row in group:
            by_seed[int(row["seed"])].append(float(row["value"]))
        seed_means = [statistics.fmean(values) for _, values in sorted(by_seed.items())]
        pooled = [float(row["value"]) for row in group]
        output.append(
            {
                "method": key[0],
                "suite": key[1],
                "setting": key[2],
                "pair": key[3],
                "aggregation": key[4],
                "metric": key[5],
                "mean_of_seed_split_means": statistics.fmean(seed_means),
                "seed_std": statistics.pstdev(seed_means) if len(seed_means) > 1 else 0.0,
                "pooled_split_std": statistics.pstdev(pooled) if len(pooled) > 1 else 0.0,
                "num_seeds": len(by_seed),
                "num_runs": len(group),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect(args.result_root.resolve())
    if not rows:
        raise SystemExit(f"No completed retrieval results under {args.result_root}")
    summary = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "fold_seed_metrics.csv", rows)
    write_csv(args.out_dir / "summary.csv", summary)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[summary] runs={len(rows)} groups={len(summary)} out={args.out_dir}")


if __name__ == "__main__":
    main()
