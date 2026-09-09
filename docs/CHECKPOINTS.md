# Checkpoint Distribution

## Scope and Provenance

The formal release contains 90 selected checkpoints for
`fixed_special5_subject_holdout_5split_v1`: T=15, seeds 0/1/2, and splits 0-4.
Each model was selected using validation C1 query-micro mAP, not special-test
scores. LidarGait++ uses the recorded earliest-iteration tie rule.

| Method / representation | Checkpoints | Original checkpoint bytes | Compressed archive |
|---|---:|---:|---:|
| PointNet-TMax | 15 | 57.09 MB | 52.85 MB |
| MPH-Gait | 15 | 74.16 MB | 68.36 MB |
| Adapted LidarGait++ | 15 | 295.97 MB | 275.36 MB |
| Projection RGB-depth | 15 | 856.16 MB | 794.50 MB |
| Projection gray-depth | 15 | 856.02 MB | 794.15 MB |
| Projection silhouette | 15 | 856.02 MB | 794.92 MB |

Sizes use decimal MB. Original checkpoints total 3.00 GB; six archives total
approximately 2.78 GB including experiment metadata.

These are **byte-identical original training checkpoints**, not reserialized
inference exports. Existing optimizer/scheduler states and training configuration
are retained. Their SHA256 values match the pinned CSVs:

```text
pointnet-tmax/checkpoints/checkpoint_manifest.csv
mph-gait/checkpoints/checkpoint_manifest.csv
lidargaitpp/checkpoints/checkpoint_manifest.csv
projection-baselines/checkpoints/checkpoint_manifest.csv
```

Hashes identify the weights used for the stored
`results/formal_len15_5split_3seed/` metrics. Each archive includes the matching
public `config.json`, `protocol.json`, and `summary.json`; LidarGait++ also
includes `checkpoint_selection.json`. The public configurations normalize
historical model names and replace machine-specific paths with placeholders.

This package excludes T=30, clothing/video-count sweeps, spatial/temporal
ablation checkpoints, and Final-24 training. It does not claim to include every
research run. Source experiments and checkpoint hashes remain unchanged;
only download locations are updated in the public checkpoint CSVs.

## Files and Download Location

Download from the
[public checkpoint folder](https://drive.google.com/drive/folders/1LXsMvUE-496sVhIrpKOs6q_Fyo0gRbsW?usp=sharing)
and open its `checkpoints/` subfolder. Archive filenames, byte sizes, SHA256,
and individual archive pages are listed in
[`checkpoint_downloads.json`](checkpoint_downloads.json).

The JSON archive `url` fields and checkpoint CSV `download_url` fields point to
Google Drive file pages, not direct binary URLs. Open the page in a browser and
select **Download**. A method's 15 checkpoint rows share its archive link.
Download the required `.tar.gz` archives and `SHA256SUMS`; the `weights/`
subfolder is an uncompressed alternative and is not needed for archive installation.
The matching archive checksum file is
[`checkpoint_archives_SHA256SUMS`](checkpoint_archives_SHA256SUMS).

```text
pointnet-tmax_fixed-special5_len15_5split_3seed.tar.gz
mph-gait_fixed-special5_len15_5split_3seed.tar.gz
lidargaitpp_fixed-special5_len15_5split_3seed.tar.gz
projection-rgb-depth_fixed-special5_len15_5split_3seed.tar.gz
projection-gray-depth_fixed-special5_len15_5split_3seed.tar.gz
projection-silhouette_fixed-special5_len15_5split_3seed.tar.gz
SHA256SUMS
```

Each archive can be downloaded independently. Extracting all six into the same
new external directory creates:

```text
pointnet_tmax/seed_0/split_0/best.pt
mph_gait/seed_0/split_0/best.pt
lidargaitpp/seed_0/split_0/best.pt
projection_rgb_depth/seed_0/split_0/best.pt
projection_gray_depth/seed_0/split_0/best.pt
projection_silhouette/seed_0/split_0/best.pt
```

The parent of these method directories is `CHECKPOINT_ROOT`. Each method
directory also contains a checkpoint manifest, metadata/file checksums,
`package_info.json`, and a README. The existing `release_filename` column
is interpreted relative to `CHECKPOINT_ROOT`, not to the Git repository.

## Download Verification

After downloading all six archives and `SHA256SUMS`, run from their directory:

```bash
sha256sum --check SHA256SUMS
```

When downloading only some methods, verify their entries against the trusted
repository's checksum file. On GNU coreutils,
`sha256sum --ignore-missing --check SHA256SUMS` checks the files present.

Example extraction and metadata verification from an external directory:

```bash
tar -xzf mph-gait_fixed-special5_len15_5split_3seed.tar.gz
sha256sum --check mph_gait/SHA256SUMS
```

Then run from the Git repository root to verify the weights against its pinned
CSV, independently of the checksum files supplied in the archive:

```bash
python scripts/verify_formal_checkpoints.py \
  --checkpoint-root /path/to/extracted-checkpoints --method mph_gait
```

Omit `--method` to require all 90 checkpoints. Repeat `--method` to verify
several selected methods. The verifier checks sizes and SHA256 without loading
PyTorch pickle data; missing or mismatched files cause a nonzero exit status.

## Model Loading and Evaluation

Load only trusted, verified checkpoints. Original PyTorch training checkpoints
may require `weights_only=False`; this enables pickle loading and must not be
used for untrusted files.

For MPH-Gait, run from the repository root after installing its dependencies:

```python
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "shared")
sys.path.insert(0, "mph-gait/code")
from mph_gait.model import build_model

cell = Path("/path/to/extracted-checkpoints/mph_gait/seed_0/split_0")
config = json.loads((cell / "config.json").read_text())
checkpoint = torch.load(cell / "best.pt", map_location="cpu", weights_only=False)
model = build_model(config["model"]).eval()
model.load_state_dict(checkpoint["model"], strict=True)
```

Use the corresponding public factory for PointNet-TMax and projection models.
The names embedded in historical binary configs may differ from public names;
the accompanying public config and current factory define the intended model.
For LidarGait++, retain the pinned OpenGait implementation and its existing
adapter, training-class count, preprocessing, and evaluator. This package does
not replace those components with an approximate architecture.

Match every checkpoint to its seed, split, representation and coordinate
convention. Reproducing retrieval also requires the original gallery/probe
endpoints and sampling rules. Loading a model is not itself an accuracy
reproduction. Stored training state does not guarantee bitwise-identical resumed
training; continuation depends on the original trainer and random state.

## Deployment Weights

Final-24 seed 2 deployment bundles for PointNet-TMax, MPH-Gait, and LidarGait++
are already provided separately in
[MPH-Gait ID](https://github.com/DennisLai-curlyhair/MPH-Gait-ID/blob/main/docs/FINAL24_WEIGHTS.md).
They are inference-only exports trained on 24 identities and must not replace
the rotating-split weights for reproducing the formal results. Changing a
checkpoint requires compatible gallery enrollment and threshold calibration.

## Hosting and Maintenance

Do not add the 90 binaries or their archives to ordinary Git history. GitHub
Releases provides downloadable assets attached to a tagged code version; the
six archives are individually below its documented 2 GiB asset limit.
See [GitHub Releases documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
A separate cloud folder is also supported, with links recorded in the manifest.

Upload the six archives and `SHA256SUMS`, optionally adding the generated
`download_links.json` as an index. Do not also upload the uncompressed
`weights/` directory unless a second distribution format is intended.
The repository's `docs/checkpoint_downloads.json` and checkpoint CSVs provide
the current download locations. Archive metadata and the original cloud
`download_links.json` are packaging-time snapshots and may still contain
`TBD` URLs; use the repository index for current links. This does not affect
checkpoint bytes or checksum verification. Keep published archives immutable;
update repository links if the hosting location changes. Keep dataset and
checkpoint packages separately labeled.

Original code and weights from third-party methods retain their respective
ownership and applicable terms. Redistribution does not transfer ownership;
see [Third-party components](THIRD_PARTY.md).

Maintainers with the trusted source outputs and the local source CSV can
rebuild into a new directory:

```bash
python scripts/package_formal_checkpoints.py \
  --source-root /path/to/source-workspace \
  --source-manifest /path/to/source_checkpoint_manifest.csv \
  --output-dir /path/to/new-checkpoint-package

python -m unittest discover -s scripts/tests -v
```

The source CSV must include `source_experiment_path`, `seed`, `split`, and
`sha256`. Checkpoints are joined to the public manifests by hash and split/seed.
The script refuses existing output directories and verifies the sources,
copied weights and every file inside each compressed archive. Rebuilt archives
can have different archive hashes due to timestamps/compression metadata even
though their checkpoint bytes are identical; publish the new archive hashes.
