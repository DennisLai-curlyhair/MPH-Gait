# Security and Data Handling

## Trusted Research Inputs

Use datasets, model checkpoints, and third-party source only from trusted
origins. Verify downloaded archives and checkpoints against the checksums
committed to this repository before extracting or loading them; see
[Checkpoint distribution](docs/CHECKPOINTS.md) and [Dataset interface](dataset/README.md).
A checksum establishes equality with a reference file, not that the file is
safe. It is useful only when the reference itself is trusted.

PyTorch checkpoints can contain pickle data. Loading an untrusted checkpoint
can execute code, especially through legacy loaders or `weights_only=False`.
Do not use the research training/evaluation tools as a service that accepts
arbitrary checkpoint uploads. Inspect metadata without unpickling when possible.

## Dependency Security

Reproduction environments are not security-hardened deployment environments.
Check the security advisories for the actual resolved dependencies before
installation. PyTorch versions through 2.5.1 are affected by
[CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6),
including a bypass of restricted checkpoint loading; 2.6.0 fixes that specific
issue. This does not establish that 2.6.0 is free of other vulnerabilities or
that unrestricted pickle loading becomes safe after an upgrade.

Validate model loading, CUDA extensions, coordinate conventions, and retrieval
results when changing dependencies. Do not silently substitute a different
training environment into published experimental evidence.

## Participant Data and Generated Files

Keep datasets outside Git and connect them through the documented local data
interface. Raw captures, foreground point clouds, registered embeddings, and
identity mappings may contain biometric or identifying information even when
RGB faces are absent. Pseudonymous subject IDs do not by themselves anonymize
these data. Use only data for which the required collection and distribution
permissions have been obtained.

The ignore rules cover local data, credentials, database sidecars, caches, and
large generated archives. Intended source code, published experiment metadata,
checkpoint manifests, and paper figures remain trackable. Review figures and
reports for private information before publishing them.

`.gitignore` is an accident-prevention aid, not an access-control mechanism. It
does not affect already tracked files, erase Git history, or override `git add
-f`. If a credential is exposed, revoke or rotate it promptly; deleting the
latest copy is not sufficient. Never add secrets to issues or pull requests.

## Reporting a Security Concern

Send security reports privately to
[dennis.y.lai@gmail.com](mailto:dennis.y.lai@gmail.com). Include the affected
commit, component, dependency versions, and a minimal description without real
credentials or participant data. Avoid publicly posting exploitable details
before the issue has been assessed.
