#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export SEEDS=${SEEDS:-"0 1 2"}
export SPLITS=${SPLITS:-"0 1 2 3 4"}
export SUITES=${SUITES:-"clip_length"}
export LENGTHS=${LENGTHS:-"15 30"}
export RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)-len15_len30_3seed}
export SKIP_COMPLETED=${SKIP_COMPLETED:-1}

exec bash "${SCRIPT_DIR}/run_followup_5split_seed0.sh"
