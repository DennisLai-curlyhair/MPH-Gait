#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHOD_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RELEASE_ROOT=$(cd -- "${METHOD_ROOT}/.." && pwd)
CODE_ROOT="${METHOD_ROOT}/code"
export PYTHONPATH="${RELEASE_ROOT}/shared:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXPERIMENT_BASE=${EXPERIMENT_BASE:-${METHOD_ROOT}/outputs/lidargaitpp}
export EXPERIMENT_BASE
exec bash "${CODE_ROOT}/lidargaitpp_adapter/scripts/run_len15_5split_3seed.sh"
