#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATASET_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

if [[ "$#" -ne 2 ]]; then
  printf 'Usage: %s /path/to/pointcloud_dataset /path/to/projection_dataset\n' "$0" >&2
  exit 2
fi

POINT_SOURCE=$(realpath "$1")
PROJECTION_SOURCE=$(realpath "$2")
for source in "${POINT_SOURCE}" "${PROJECTION_SOURCE}"; do
  if [[ ! -d "${source}" ]]; then
    printf 'Missing dataset directory: %s\n' "${source}" >&2
    exit 2
  fi
done

link_once() {
  local target=$1
  local source=$2

  if [[ -L "${target}" && "$(realpath "${target}")" == "${source}" ]]; then
    printf '[ok] %s -> %s\n' "${target}" "${source}"
    return
  fi

  if [[ -d "${target}" && ! -L "${target}" ]]; then
    local entries
    entries=$(find "${target}" -mindepth 1 -maxdepth 1 ! -name .gitkeep -print -quit)
    if [[ -z "${entries}" ]]; then
      rm -f "${target}/.gitkeep"
      rmdir "${target}"
    fi
  fi

  if [[ -e "${target}" || -L "${target}" ]]; then
    printf 'Refusing to replace non-empty path: %s\n' "${target}" >&2
    exit 2
  fi

  ln -s "${source}" "${target}"
  printf '[linked] %s -> %s\n' "${target}" "${source}"
}

mkdir -p "${DATASET_ROOT}/local"
link_once "${DATASET_ROOT}/local/pointcloud" "${POINT_SOURCE}"
link_once "${DATASET_ROOT}/local/projection" "${PROJECTION_SOURCE}"
