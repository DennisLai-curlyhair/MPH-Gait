#!/usr/bin/env python3
"""Package the exact T=15 checkpoints identified by the public release manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "pointnet_tmax": ("pointnet-tmax", None, "pointnet-tmax"),
    "mph_gait": ("mph-gait", None, "mph-gait"),
    "lidargaitpp": ("lidargaitpp", None, "lidargaitpp"),
    "projection_rgb_depth": ("projection-baselines", "rgb_depth", "projection-rgb-depth"),
    "projection_gray_depth": ("projection-baselines", "gray_depth", "projection-gray-depth"),
    "projection_silhouette": ("projection-baselines", "silhouette", "projection-silhouette"),
}
FIELDS = ["method", "seed", "split", "release_filename", "size_bytes", "sha256", "download_url"]


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return digest_stream(stream)


def digest_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in FIELDS} for row in rows)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def collect_sources(release_root: Path, source_root: Path, source_manifest: Path) -> dict:
    source_index = {}
    for row in read_csv(source_manifest):
        source_index.setdefault(row["sha256"], []).append(row)
    plans = {}
    for method, (directory, representation, _) in METHODS.items():
        manifest = release_root / directory / "checkpoints/checkpoint_manifest.csv"
        rows = [row for row in read_csv(manifest) if row["method"] == method]
        cells = {(int(row["seed"]), int(row["split"])) for row in rows}
        if len(rows) != 15 or cells != {(s, f) for s in range(3) for f in range(5)}:
            raise ValueError(f"Expected exactly 3 seeds x 5 splits for {method}")
        entries = []
        for row in sorted(rows, key=lambda r: (int(r["seed"]), int(r["split"]))):
            expected_path = f"{method}/seed_{row['seed']}/split_{row['split']}/best.pt"
            if row["release_filename"] != expected_path:
                raise ValueError(f"Unexpected checkpoint destination: {row['release_filename']}")
            candidates = [item for item in source_index.get(row["sha256"], [])
                          if item["seed"] == row["seed"] and item["split"] == row["split"]]
            if len(candidates) != 1:
                raise ValueError(f"Expected one source record matching {expected_path}")
            source = (source_root / candidates[0]["source_experiment_path"]).resolve()
            if not source.is_relative_to(source_root):
                raise ValueError("Source checkpoint must resolve within --source-root")
            if source.stat().st_size != int(row["size_bytes"]):
                raise ValueError(f"Source checkpoint size mismatch: {expected_path}")
            cell = release_root / directory / "results/formal_len15_5split_3seed"
            if representation:
                cell /= representation
            cell = cell / f"seed_{row['seed']}" / f"split_{row['split']}"
            metadata = [cell / name for name in ("config.json", "protocol.json", "summary.json")]
            if method == "lidargaitpp":
                metadata.append(cell / "checkpoint_selection.json")
            if not all(path.is_file() for path in metadata):
                raise FileNotFoundError(f"Missing public run metadata for {expected_path}")
            config = json.loads(metadata[0].read_text(encoding="utf-8"))
            if config["data"]["clip_len"] != 15 or config["experiment"]["seed"] != int(row["seed"]):
                raise ValueError(f"Run metadata mismatch for {expected_path}")
            entries.append((row, source, metadata))
        plans[method] = {"entries": entries, "manifest": manifest}
    return plans


def verify_archive(path: Path, expected: dict[str, str]) -> None:
    seen = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unexpected archive member: {member.name}")
            if member.name in seen or member.name not in expected:
                raise ValueError(f"Unexpected or duplicate archive file: {member.name}")
            with archive.extractfile(member) as stream:
                if digest_stream(stream) != expected[member.name]:
                    raise ValueError(f"Archive content mismatch: {member.name}")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Archive file inventory does not match the copied package")


def package(release_root: Path, source_root: Path, source_manifest: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if output.is_relative_to(release_root):
        raise ValueError("Use an output directory outside the Git repository")
    plans = collect_sources(release_root, source_root, source_manifest)
    commit = subprocess.check_output(
        ["git", "-C", str(release_root), "rev-parse", "HEAD"], text=True).strip()
    weights = output / "weights"
    weights.mkdir(parents=True)
    records, archive_rows = [], []
    for method, plan in plans.items():
        folder = weights / method
        rows = []
        for row, source, metadata in plan["entries"]:
            target = weights / row["release_filename"]
            if digest_file(source) != row["sha256"]:
                raise ValueError(f"Source checkpoint hash mismatch: {row['release_filename']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest_file(target) != row["sha256"]:
                raise ValueError(f"Copied checkpoint hash mismatch: {row['release_filename']}")
            for path in metadata:
                shutil.copy2(path, target.parent / path.name)
            rows.append(row)
        write_csv(folder / "checkpoint_manifest.csv", rows)
        write_json(folder / "package_info.json", {
            "method": method, "protocol": "fixed_special5_subject_holdout_5split_v1",
            "clip_len": 15, "seeds": [0, 1, 2], "splits": [0, 1, 2, 3, 4],
            "checkpoint_count": len(rows), "checkpoint_format": "original_training_checkpoint",
            "checkpoint_bytes_unchanged": True, "selection_pair": "val",
            "selection_metric": "query_micro_mAP", "selection_uses_fixed_special": False,
            "release_repository": "https://github.com/DennisLai-curlyhair/MPH-Gait",
            "release_base_commit": commit,
            "public_checkpoint_manifest_sha256": digest_file(plan["manifest"]),
            "source_manifest_sha256": digest_file(source_manifest),
            "packager_sha256": digest_file(Path(__file__)),
            "raw_checkpoint_bytes": sum(int(row["size_bytes"]) for row in rows),
        })
        (folder / "README.md").write_text(
            f"# {method}: Fixed-Special5 T=15 Checkpoints\n\n"
            "This package contains 5 splits x 3 seeds selected using validation C1 query-micro mAP.\n"
            "Original checkpoint bytes and model tensors are unchanged. Training optimizer/scheduler\n"
            "states are retained when present. This is not an inference-only or Final-24 bundle.\n\n"
            "Extract into a new external checkpoint directory. From that directory, run:\n\n"
            f"```bash\nsha256sum --check {method}/SHA256SUMS\n```\n\n"
            "Each seed/split includes best.pt, config.json, protocol.json and summary.json.\n"
            "LidarGait++ also includes checkpoint_selection.json. Metadata are copied from the\n"
            "public experiment records; path placeholders must be configured for the local machine.\n"
            "Historical architecture names in the binary config may differ from public names;\n"
            "use the current repository's model factory and accompanying public config.\n\n"
            "Load only trusted checkpoints. Original PyTorch checkpoints may require pickle-aware\n"
            "loading. Resume behavior depends on the original trainer; the presence of optimizer\n"
            "states alone does not guarantee an identical continuation of training.\n\n"
            "Keep each checkpoint with its matching seed, split, input representation and axes.\n"
            "Do not mix gallery embeddings across checkpoints. No participant data are included.\n"
            "See https://github.com/DennisLai-curlyhair/MPH-Gait/blob/main/docs/CHECKPOINTS.md\n"
            "for release instructions and third-party attribution.\n", encoding="utf-8")
        hashes = {path.relative_to(weights).as_posix(): digest_file(path)
                  for path in sorted(folder.rglob("*")) if path.is_file()}
        checksums = folder / "SHA256SUMS"
        checksums.write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8")
        hashes[checksums.relative_to(weights).as_posix()] = digest_file(checksums)
        stem = METHODS[method][2]
        filename = f"{stem}_fixed-special5_len15_5split_3seed.tar.gz"
        temporary = output / (filename + ".partial")
        print(f"[archive] {method}: {len(rows)} verified original checkpoints", flush=True)
        with tarfile.open(temporary, "w:gz", compresslevel=6) as archive:
            archive.add(folder, arcname=method)
        verify_archive(temporary, hashes)
        destination = output / filename
        temporary.rename(destination)
        archive_rows.append({"method": method, "filename": filename,
                             "checkpoint_count": len(rows), "format": "tar.gz",
                             "size_bytes": destination.stat().st_size,
                             "sha256": digest_file(destination), "url": "TBD",
                             "archive_root": method})
        records.extend(rows)
        print(f"[ready] {filename}: {destination.stat().st_size} bytes", flush=True)
    write_csv(weights / "checkpoint_manifest.csv", records)
    write_json(output / "download_links.json", {
        "schema_version": 1, "distribution_url": "TBD",
        "scope": "formal_fixed_special5_len15_5split_3seed",
        "checkpoint_count": len(records), "archives": archive_rows})
    (output / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['filename']}\n" for row in archive_rows), encoding="utf-8")
    print(f"[complete] {len(records)} checkpoints, {len(archive_rows)} archives: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True,
                        help="Local CSV with source_experiment_path, seed, split and sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    package(ROOT, args.source_root.resolve(), args.source_manifest.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
