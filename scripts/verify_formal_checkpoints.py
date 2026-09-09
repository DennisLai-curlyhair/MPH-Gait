#!/usr/bin/env python3
"""Verify downloaded formal checkpoints against the repository's pinned manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from package_formal_checkpoints import METHODS, ROOT, digest_file, read_csv


def verify(root: Path, methods: list[str]) -> dict:
    checked = 0
    failures = []
    for method in methods:
        directory = METHODS[method][0]
        manifest = ROOT / directory / "checkpoints/checkpoint_manifest.csv"
        rows = [row for row in read_csv(manifest) if row["method"] == method]
        if len(rows) != 15:
            failures.append({"method": method, "error": "expected 15 manifest rows"})
            continue
        for row in rows:
            name = row["release_filename"]
            expected = f"{method}/seed_{row['seed']}/split_{row['split']}/best.pt"
            if name != expected:
                failures.append({"method": method, "error": "invalid manifest path"})
                continue
            path = root / name
            if not path.is_file():
                failures.append({"file": name, "error": "missing"})
            elif path.stat().st_size != int(row["size_bytes"]):
                failures.append({"file": name, "error": "size mismatch"})
            elif digest_file(path) != row["sha256"]:
                failures.append({"file": name, "error": "SHA256 mismatch"})
            else:
                checked += 1
    return {"status": "ok" if not failures else "failed",
            "verified_checkpoints": checked, "methods": methods, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                        help="Directory containing pointnet_tmax/, mph_gait/, etc.")
    parser.add_argument("--method", action="append", choices=sorted(METHODS),
                        help="Verify selected method(s); default verifies all 90 checkpoints")
    args = parser.parse_args()
    result = verify(args.checkpoint_root.resolve(), list(dict.fromkeys(args.method or METHODS)))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
