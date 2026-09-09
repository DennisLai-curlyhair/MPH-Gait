#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  printf 'Usage: bash %s SOURCE_ROOT OUTPUT_DIRECTORY\n' "$0" >&2
  printf 'SOURCE_ROOT must contain dataset/ and dataset_proj/.\n' >&2
  exit 2
fi

SOURCE_ROOT=$(realpath "$1")
OUTPUT_ROOT=$(realpath -m "$2")
THREADS=${THREADS:-4}
if [[ ! "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'THREADS must be a positive integer.\n' >&2
  exit 2
fi

for name in dataset dataset_proj; do
  if [[ ! -d "${SOURCE_ROOT}/${name}/PersonRecognitionWalking_29" ]]; then
    printf 'Missing dataset: %s/%s/PersonRecognitionWalking_29\n' "${SOURCE_ROOT}" "${name}" >&2
    exit 2
  fi
  case "${OUTPUT_ROOT}/" in
    "${SOURCE_ROOT}/${name}/"*)
      printf 'Archive output must be outside the source datasets.\n' >&2
      exit 2
      ;;
  esac
  if [[ -e "${OUTPUT_ROOT}/${name}.tar.gz" || -e "${OUTPUT_ROOT}/${name}.tar.gz.partial" ]]; then
    printf 'Refusing to overwrite archive for %s. Choose a new output directory.\n' "${name}" >&2
    exit 2
  fi
  link=$(find "${SOURCE_ROOT}/${name}" -type l -print -quit)
  if [[ -L "${SOURCE_ROOT}/${name}" || -n "${link}" ]]; then
    printf 'Source datasets must be real directories without symbolic links.\n' >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_ROOT}/SHA256SUMS" ]]; then
  printf 'Refusing to overwrite SHA256SUMS. Choose a new output directory.\n' >&2
  exit 2
fi

compressor=gzip
if command -v pigz >/dev/null 2>&1; then
  compressor="pigz -p ${THREADS}"
fi
mkdir -p "${OUTPUT_ROOT}"

for name in dataset dataset_proj; do
  temporary="${OUTPUT_ROOT}/${name}.tar.gz.partial"
  archive="${OUTPUT_ROOT}/${name}.tar.gz"
  printf '[compress] %s -> %s\n' "${name}" "${archive}"
  tar --sort=name --numeric-owner --use-compress-program="${compressor}" \
    --create --file="${temporary}" --directory="${SOURCE_ROOT}" -- "${name}"
  printf '[verify] Decompressing and comparing %s against the source files\n' "${name}"
  tar --use-compress-program="${compressor}" --compare \
    --file="${temporary}" --directory="${SOURCE_ROOT}"
  mv -- "${temporary}" "${archive}"
  printf '[ready] %s (%s bytes)\n' "${archive}" "$(stat -c %s "${archive}")"
done

(
  cd "${OUTPUT_ROOT}"
  sha256sum dataset.tar.gz dataset_proj.tar.gz > SHA256SUMS
  sha256sum --check SHA256SUMS
)
printf '[complete] Upload both archives and SHA256SUMS from %s\n' "${OUTPUT_ROOT}"
