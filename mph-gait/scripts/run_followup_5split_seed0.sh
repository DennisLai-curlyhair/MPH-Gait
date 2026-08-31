#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHOD_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RELEASE_ROOT=$(cd -- "${METHOD_ROOT}/.." && pwd)
CODE_ROOT="${METHOD_ROOT}/code"

export METHOD_NAME="MPH-Gait"
export METHOD_MODULE="mph_gait"
export METHOD_ROOT
export RELEASE_ROOT
export CODE_ROOT
export BASE_CONFIG=${BASE_CONFIG:-${CODE_ROOT}/mph_gait/configs/mph_gait_len15.yaml}
export SEEDS=${SEEDS:-"0"}

exec bash "${RELEASE_ROOT}/shared/scripts/run_pointcloud_followup.sh"
