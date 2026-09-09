# Dataset Interface

The participant data are distributed separately from Git. Download
`dataset.tar.gz`, `dataset_proj.tar.gz`, and `SHA256SUMS` from the
[Google Drive dataset folder](https://drive.google.com/drive/folders/1Ec63IgVLfJkezbmzXADklmkVndpL73V2?usp=drive_link).
Archive metadata are recorded in [download_links.yaml](download_links.yaml).
Access/consent terms: **TBD**.

For access to original acquisition files (raw data, RGB images, and depth
images), please contact [dennis.y.lai@gmail.com](mailto:dennis.y.lai@gmail.com).

## Archive Layout

| Archive | Content | Extracted root | Repository interface |
|---|---|---|---|
| `dataset.tar.gz` | Foreground XYZ point arrays | `dataset/` | `dataset/local/pointcloud/` |
| `dataset_proj.tar.gz` | Three precomputed projection representations | `dataset_proj/` | `dataset/local/projection/` |

Both archives preserve their top-level directory:

```text
dataset/PersonRecognitionWalking_29/person_001/P1_C1_V001/clear_data_001.npy
dataset_proj/PersonRecognitionWalking_29/lidargait_rgb/person_001/P1_C1_V001/
dataset_proj/PersonRecognitionWalking_29/lidargait_gray/person_001/P1_C1_V001/
dataset_proj/PersonRecognitionWalking_29/lidargait_silhouette/person_001/P1_C1_V001/
```

The point-cloud package contains foreground arrays, not camera RGB images or
full-scene raw point clouds. `lidargait_rgb` means depth-derived pseudo-color,
not camera RGB. Projection metadata and the original frame numbering are
retained. Do not remove the initial frames or change axes in these archives:
the experiment loaders apply their configured preprocessing.

## Download, Verify, and Connect

Download both archives and `SHA256SUMS` to the same external data directory.
Compare checksums with the committed metadata, then run from that directory:

```bash
sha256sum --check SHA256SUMS
```

SHA256 is a content checksum, not an encryption key. An `OK` result confirms
that the downloaded archive matches its recorded checksum. If verification
fails, do not extract or use the file; download it again and check against the
checksum published in this repository. A checksum does not establish data
quality or authenticity unless the reference checksum is trusted.

After both archives pass verification:

```bash
tar -xzf dataset.tar.gz
tar -xzf dataset_proj.tar.gz
```

Use a new or empty extraction directory. Do not extract into the repository's
`dataset/` documentation directory. From the repository root, connect the
extracted top-level folders:

```bash
bash dataset/scripts/link_local_data.sh \
  /path/to/downloads/dataset \
  /path/to/downloads/dataset_proj
```

The existing experiment scripts resolve data through `dataset/local/` and
need no clothing, seed, video-count, or clip-length changes. All methods retain
the committed protocol in `metadata/fixed_special5_protocol.json`. Training
and evaluation splits are constructed by the loaders, not by the archives.

## Distribution Metadata

Both archive URLs in `download_links.yaml` point to the shared Google Drive
folder. Download the named files manually; these are not direct archive URLs,
and the manifest does not automatically download data. The recorded filenames,
SHA256, sizes, and archive roots identify this release. Replacing an archive
requires updating its metadata and both copies of `SHA256SUMS` in the
repository and cloud folder. Access terms and consent information remain
separate metadata fields. Do not commit data or archives to Git.

## Rebuilding Archives

GNU tar, Bash, gzip, and GNU coreutils are required. `pigz` is optional and
enables four-thread compression by default:

```bash
THREADS=4 bash dataset/scripts/package_local_data.sh \
  /path/to/source-parent /path/to/new-archives
```

The source parent must contain `dataset/` and `dataset_proj/`. The script
refuses existing archive outputs and symlinked inputs, preserves source data,
decompresses and compares every archived entry against its source, and emits
`SHA256SUMS`. New exports require fresh checksums; their bytes may differ if
the source metadata or compression tool changes.
