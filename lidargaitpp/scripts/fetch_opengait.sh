#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHOD_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
TARGET="${METHOD_ROOT}/third_party/OpenGait"
PATCH="${METHOD_ROOT}/patches/opengait_runtime_compat.patch"
UPSTREAM=${OPENGAIT_UPSTREAM:-https://github.com/ShiqiYu/OpenGait.git}
COMMIT=f754f6f3831e9f83bb28f4e2f63dd43d8bcf9dc4

if [[ ! -d "${TARGET}/.git" ]]; then
  git clone "${UPSTREAM}" "${TARGET}"
fi

CURRENT=$(git -C "${TARGET}" rev-parse HEAD)
if [[ "${CURRENT}" != "${COMMIT}" ]]; then
  if [[ -n "$(git -C "${TARGET}" status --porcelain)" ]]; then
    printf 'OpenGait contains local changes; refusing to switch commits.\n' >&2
    exit 2
  fi
  git -C "${TARGET}" fetch origin "${COMMIT}"
  git -C "${TARGET}" checkout --detach "${COMMIT}"
fi

if git -C "${TARGET}" apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
  printf '[ok] compatibility patch already applied\n'
elif git -C "${TARGET}" apply --check "${PATCH}"; then
  git -C "${TARGET}" apply "${PATCH}"
else
  printf 'Compatibility patch cannot be applied cleanly.\n' >&2
  exit 2
fi

MODEL_HASH=$(sha256sum "${TARGET}/opengait/modeling/models/lidargaitv2.py" | awk '{print $1}')
UTILS_HASH=$(sha256sum "${TARGET}/opengait/modeling/models/lidargaitv2_utils.py" | awk '{print $1}')
[[ "${MODEL_HASH}" == "180cdb41da76cdd8950159469e72c525b4e555d1060edc89f3ee887f96aae748" ]]
[[ "${UTILS_HASH}" == "15ff5ff8f904f5b13d35d5d86f35d36814202769aed2bb0c12434acf43b881f8" ]]
printf '[verified] OpenGait %s and official LidarGait++ source hashes\n' "${COMMIT}"
