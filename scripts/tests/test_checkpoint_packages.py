from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package_formal_checkpoints import digest_file, package, verify_archive, write_csv
import verify_formal_checkpoints as verifier


class CheckpointPackageTest(unittest.TestCase):
    def test_existing_and_in_repo_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                package(root, root, root / "missing.csv", root)
            with self.assertRaises(ValueError):
                package(root, root, root / "missing.csv", root / "new-output")

    def test_archive_hash_and_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "best.pt"
            source.write_bytes(b"checkpoint-fixture")
            archive_path = root / "package.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(source, arcname="mph_gait/seed_0/split_0/best.pt")
            expected = {"mph_gait/seed_0/split_0/best.pt": digest_file(source)}
            verify_archive(archive_path, expected)
            with self.assertRaises(ValueError):
                verify_archive(archive_path, {next(iter(expected)): "0" * 64})
            with self.assertRaises(ValueError):
                verify_archive(archive_path, {})
            with self.assertRaises(ValueError):
                verify_archive(archive_path, {**expected, "missing.pt": "0" * 64})

    def test_archive_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "unsafe.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                member = tarfile.TarInfo("../escape.pt")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
            with self.assertRaises(ValueError):
                verify_archive(archive_path, {})

    def test_download_verifier_detects_corruption_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "repo/mph-gait/checkpoints/checkpoint_manifest.csv"
            manifest.parent.mkdir(parents=True)
            weights = root / "weights"
            rows = []
            for seed in range(3):
                for split in range(5):
                    relative = f"mph_gait/seed_{seed}/split_{split}/best.pt"
                    checkpoint = weights / relative
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(b"test")
                    rows.append({"method": "mph_gait", "seed": seed, "split": split,
                                 "release_filename": relative, "size_bytes": 4,
                                 "sha256": digest_file(checkpoint), "download_url": "TBD"})
            write_csv(manifest, rows)
            with patch.object(verifier, "ROOT", root / "repo"):
                self.assertEqual(verifier.verify(weights, ["mph_gait"])["verified_checkpoints"], 15)
                damaged = weights / rows[0]["release_filename"]
                damaged.write_bytes(b"fail")
                result = verifier.verify(weights, ["mph_gait"])
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["failures"][0]["error"], "SHA256 mismatch")
                damaged.unlink()
                self.assertEqual(verifier.verify(weights, ["mph_gait"])["failures"][0]["error"], "missing")


if __name__ == "__main__":
    unittest.main()
