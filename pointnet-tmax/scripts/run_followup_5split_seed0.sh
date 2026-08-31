#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHOD_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RELEASE_ROOT=$(cd -- "${METHOD_ROOT}/.." && pwd)
CODE_ROOT="${METHOD_ROOT}/code"

export METHOD_NAME="PointNet-TMax"
export METHOD_MODULE="pc_v1"
export METHOD_ROOT
export RELEASE_ROOT
export CODE_ROOT
export BASE_CONFIG=${BASE_CONFIG:-${CODE_ROOT}/pc_v1/configs/pc_v1_len15.yaml}
export SEEDS=${SEEDS:-"0"}

exec bash "${RELEASE_ROOT}/shared/scripts/run_pointcloud_followup.sh"
