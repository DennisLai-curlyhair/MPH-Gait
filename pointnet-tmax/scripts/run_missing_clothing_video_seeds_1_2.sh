#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export SEEDS=${SEEDS:-"1 2"}
export SUITES=${SUITES:-"train_clothing train_video_count"}
export TRAIN_VARIANTS=${TRAIN_VARIANTS:-"c2 c3"}
export MAX_VIDEO_IDS=${MAX_VIDEO_IDS:-"1 2 4 8"}
export RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
export SKIP_COMPLETED=${SKIP_COMPLETED:-1}

exec bash "${SCRIPT_DIR}/run_followup_5split_3seed.sh"
