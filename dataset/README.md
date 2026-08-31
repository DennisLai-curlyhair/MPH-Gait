# Dataset Interface

The participant dataset is not committed to Git. This directory contains the
complete schema and split contract required to prepare it reproducibly.

Expected local paths:

```text
dataset/local/pointcloud/
dataset/local/projection/
```

Create links with:

```bash
bash dataset/scripts/link_local_data.sh /path/to/pointcloud /path/to/projection
```

Before publication, fill `download_links.yaml`, document consent/access terms,
and provide archive hashes. Do not publish participant data until the relevant
ethics, consent, and de-identification requirements are satisfied.
