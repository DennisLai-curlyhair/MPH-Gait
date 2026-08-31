from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


KEY_PAIRS = (
    "val",
    "test_general_to_personal",
    "test_fixed5_to_personal",
    "test_fixed5_to_special",
    "test_fixed5_c4c5_normal",
    "test_fixed5_c7",
    "test_all_personal",
    "test_query_micro_overall",
    "test_mixed_c2_to_c3",
    "test_mixed_c3_to_c2",
    "diagnostic_fixed5_only_to_special",
)
METRICS = (
    "query_rank1",
    "query_rank5",
    "query_mAP",
    "subject_rank1",
    "subject_rank5",
    "subject_mAP",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize projection matched seed/split results."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metric: {value}")
    return result


def mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def pstdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def result_dir(run_group: Path, split: int) -> Path:
    candidates = sorted(
        (run_group / f"split{split}").glob("*/retrieval_metrics.json")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one completed result for {run_group} split={split}; "
            f"found {len(candidates)}"
        )
    return candidates[0].parent


def pair_values(pair: dict[str, Any]) -> dict[str, float]:
    subject = pair.get("subject_macro", {})
    return {
        "query_rank1": finite(pair["rank1"]),
        "query_rank5": finite(pair["rank5"]),
        "query_mAP": finite(pair["mAP"]),
        "subject_rank1": finite(subject["rank1"]),
        "subject_rank5": finite(subject["rank5"]),
        "subject_mAP": finite(subject["mAP"]),
    }


def load_cell(
    run_group: Path,
    *,
    expected_seed: int,
    expected_split: int,
    expected_protocol: str,
) -> dict[str, Any]:
    run_dir = result_dir(run_group, expected_split)
    config = load_json(run_dir / "config.json")
    protocol = load_json(run_dir / "protocol.json")
    retrieval = load_json(run_dir / "retrieval_metrics.json")
    summary = load_json(run_dir / "summary.json")
    metrics_rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    expected_train_seed = expected_seed + expected_split * 1009
    expected_eval_seed = expected_train_seed + 100_000
    checks = {
        "config_seed": int(
            config.get("experiment", {}).get("seed", -1)
        )
        == expected_seed,
        "summary_seed": int(summary.get("seed", -1)) == expected_seed,
        "loader_seed": int(config.get("data", {}).get("loader_seed", -1))
        == expected_seed,
        "train_loader_seed": int(
            config.get("data", {}).get("train_loader_seed", -1)
        )
        == expected_train_seed,
        "eval_loader_seed": int(
            config.get("data", {}).get("eval_loader_seed", -1)
        )
        == expected_eval_seed,
        "split_index": int(protocol.get("split_index", -1))
        == expected_split,
        "protocol_name": protocol.get("name") == expected_protocol,
        "selection_pair": summary.get("selection_pair") == "val",
        "fixed_special_not_selected": not bool(
            summary.get("selection_uses_fixed_special", True)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Seed/protocol contract failed for {run_dir}: {failed}"
        )

    nonfinite_batches = sum(
        finite(row.get("train_nonfinite_batches", 0.0))
        + finite(row.get("val_nonfinite_batches", 0.0))
        for row in metrics_rows
    )
    dropped_embeddings = sum(
        finite(values.get("gallery_dropped_nonfinite", 0.0))
        + finite(values.get("probe_dropped_nonfinite", 0.0))
        for values in retrieval.values()
    )
    if nonfinite_batches or dropped_embeddings:
        raise RuntimeError(
            f"Non-finite values found for {run_dir}: batches="
            f"{nonfinite_batches}, embeddings={dropped_embeddings}"
        )

    return {
        "run_dir": run_dir,
        "config": config,
        "protocol": protocol,
        "retrieval": retrieval,
        "summary": summary,
    }


def summarize_group(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    output: list[dict[str, Any]] = []
    for values, group_rows in sorted(grouped.items()):
        result = dict(zip(group_keys, values))
        result["num_cells"] = len(group_rows)
        for metric in METRICS:
            metric_values = [finite(row[metric]) for row in group_rows]
            result[f"{metric}_mean"] = mean(metric_values)
            result[f"{metric}_std"] = pstdev(metric_values)
        output.append(result)
    return output


def paired_differences(
    rows: list[dict[str, Any]], methods: list[str]
) -> list[dict[str, Any]]:
    lookup = {
        (row["method"], row["seed"], row["split"], row["pair"]): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    pairs = sorted({str(row["pair"]) for row in rows})
    cells = sorted({(int(row["seed"]), int(row["split"])) for row in rows})
    for method_a, method_b in combinations(methods, 2):
        for pair in pairs:
            result: dict[str, Any] = {
                "method_a": method_a,
                "method_b": method_b,
                "pair": pair,
                "num_cells": len(cells),
            }
            for metric in METRICS:
                deltas = [
                    finite(lookup[(method_b, seed, split, pair)][metric])
                    - finite(lookup[(method_a, seed, split, pair)][metric])
                    for seed, split in cells
                ]
                result[f"{metric}_delta_mean"] = mean(deltas)
                result[f"{metric}_delta_std"] = pstdev(deltas)
                result[f"{metric}_b_wins"] = sum(delta > 0 for delta in deltas)
                result[f"{metric}_ties"] = sum(delta == 0 for delta in deltas)
            output.append(result)
    return output


def metric_text(row: dict[str, Any], metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}"


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    methods = list(manifest["methods_order"])
    seeds = [int(seed) for seed in manifest["seeds"]]
    splits = [int(split) for split in manifest["splits"]]
    expected_protocol = str(manifest["protocol"])
    cell_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    protocol_by_cell: dict[tuple[int, int], dict[str, Any]] = {}

    for method in methods:
        method_info = manifest["methods"][method]
        for seed in seeds:
            run_group = Path(method_info["run_groups"][str(seed)])
            for split in splits:
                cell = load_cell(
                    run_group,
                    expected_seed=seed,
                    expected_split=split,
                    expected_protocol=expected_protocol,
                )
                protocol_key = (seed, split)
                protocol = cell["protocol"]
                if protocol_key in protocol_by_cell:
                    if protocol_by_cell[protocol_key] != protocol:
                        raise RuntimeError(
                            "Projection methods use different protocol data for "
                            f"seed={seed}, split={split}"
                        )
                else:
                    protocol_by_cell[protocol_key] = protocol

                audit_rows.append(
                    {
                        "method": method,
                        "method_id": method_info["method_id"],
                        "seed": seed,
                        "split": split,
                        "run_dir": str(cell["run_dir"]),
                        "best_epoch": cell["summary"].get("best_epoch"),
                        "selection_pair": cell["summary"].get(
                            "selection_pair"
                        ),
                    }
                )
                for pair, values in sorted(cell["retrieval"].items()):
                    cell_rows.append(
                        {
                            "method": method,
                            "method_id": method_info["method_id"],
                            "seed": seed,
                            "split": split,
                            "pair": pair,
                            **pair_values(values),
                        }
                    )

    expected_cells = len(methods) * len(seeds) * len(splits)
    if len(audit_rows) != expected_cells:
        raise RuntimeError(
            f"Expected {expected_cells} completed cells, got {len(audit_rows)}"
        )

    method_summary = summarize_group(cell_rows, ("method", "method_id", "pair"))
    seed_summary = summarize_group(
        cell_rows, ("method", "method_id", "seed", "pair")
    )
    split_summary = summarize_group(
        cell_rows, ("method", "method_id", "split", "pair")
    )
    paired_rows = paired_differences(cell_rows, methods)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "completed_cells.csv", audit_rows)
    write_csv(args.out_dir / "retrieval_cell_metrics.csv", cell_rows)
    write_csv(args.out_dir / "method_15cell_mean_std.csv", method_summary)
    write_csv(args.out_dir / "seed_5split_mean_std.csv", seed_summary)
    write_csv(args.out_dir / "split_3seed_mean_std.csv", split_summary)
    write_csv(args.out_dir / "paired_method_differences.csv", paired_rows)

    lookup = {
        (str(row["method"]), str(row["pair"])): row
        for row in method_summary
    }
    lines = [
        "# Projection Len15 Matched-Seed Summary",
        "",
        f"Seeds: `{seeds}`; splits: `{splits}`; "
        f"completed model fits: `{expected_cells}`.",
        "",
        "All checkpoints use validation C1 retrieval only; fixed-five special "
        "queries are excluded from model selection.",
        "",
        "| Method | Pair | Query R1 | Query R5 | Query mAP | Subject-macro mAP |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        for pair in KEY_PAIRS:
            row = lookup.get((method, pair))
            if row is None:
                continue
            lines.append(
                f"| {method} | {pair} | {metric_text(row, 'query_rank1')} | "
                f"{metric_text(row, 'query_rank5')} | "
                f"{metric_text(row, 'query_mAP')} | "
                f"{metric_text(row, 'subject_mAP')} |"
            )
    lines.extend(
        [
            "",
            "The table reports variation across all 15 matched seed/split cells. "
            "Use `seed_5split_mean_std.csv` to inspect seed sensitivity and "
            "`split_3seed_mean_std.csv` to inspect subject-split sensitivity.",
            "",
        ]
    )
    (args.out_dir / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    audit = {
        "manifest": str(args.manifest),
        "methods": methods,
        "seeds": seeds,
        "splits": splits,
        "expected_cells": expected_cells,
        "completed_cells": len(audit_rows),
        "protocol": expected_protocol,
        "seed_contract_valid": True,
        "protocol_equal_across_methods": True,
        "nonfinite_batches_or_embeddings": 0,
    }
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"matched summary: {args.out_dir}")


if __name__ == "__main__":
    main()
