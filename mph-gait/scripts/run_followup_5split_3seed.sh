#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SEEDS=${SEEDS:-"0 1 2"}
exec bash "${SCRIPT_DIR}/run_followup_5split_seed0.sh"
