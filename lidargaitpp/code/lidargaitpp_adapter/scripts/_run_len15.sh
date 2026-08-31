#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CLIP_LEN=${CLIP_LEN:-15}
export MAX_TRAIN_VIDEO_ID=${MAX_TRAIN_VIDEO_ID:-16}
export EXPERIMENT_TAG=${EXPERIMENT_TAG:-len15_c2c3_full}
exec bash "${SCRIPT_DIR}/_run_setting.sh"
