#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHOD_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RELEASE_ROOT=$(cd -- "${METHOD_ROOT}/.." && pwd)
CODE_ROOT="${METHOD_ROOT}/code"

export PYTHONPATH="${RELEASE_ROOT}/shared:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export EXPERIMENT_BASE=${EXPERIMENT_BASE:-${METHOD_ROOT}/outputs/lidargaitpp}
export SEEDS=${SEEDS:-"0"}
exec bash "${CODE_ROOT}/lidargaitpp_adapter/scripts/run_followup.sh"
