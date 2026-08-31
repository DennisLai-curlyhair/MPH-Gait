#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHOD_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RELEASE_ROOT=$(cd -- "${METHOD_ROOT}/.." && pwd)
CODE_ROOT="${METHOD_ROOT}/code"

export PYTHONPATH="${RELEASE_ROOT}/shared:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SEEDS=${SEEDS:-"0"}
cd "${CODE_ROOT}"
exec bash projection_baseline/scripts/run_followup.sh
