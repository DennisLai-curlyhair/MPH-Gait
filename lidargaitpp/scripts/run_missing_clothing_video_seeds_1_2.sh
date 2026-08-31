#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHOD_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RELEASE_ROOT=$(cd -- "${METHOD_ROOT}/.." && pwd)

export SEEDS=${SEEDS:-"1 2"}
export SUITES=${SUITES:-"train_clothing train_video_count"}
export TRAIN_VARIANTS=${TRAIN_VARIANTS:-"c2 c3"}
export MAX_VIDEO_IDS=${MAX_VIDEO_IDS:-"1 2 4 8"}
export RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
export SKIP_COMPLETED=${SKIP_COMPLETED:-1}

bash "${SCRIPT_DIR}/run_followup_5split_3seed.sh"

if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" ]]; then
  exit 0
fi
if [[ " ${EXTRA_ARGS:-} " == *" --dry-run "* ]] || \
   [[ " ${EXTRA_ARGS:-} " == *" --inspect-only "* ]]; then
  exit 0
fi

RESULT_ROOT="${METHOD_ROOT}/outputs/lidargaitpp/followup/${RUN_STAMP}"
SUMMARY_ROOT="${METHOD_ROOT}/outputs/followup_summaries/${RUN_STAMP}"
conda run --no-capture-output -n "${CONDA_ENV:-pytorch}" python \
  "${RELEASE_ROOT}/shared/tools/summarize_followup.py" \
  --result-root "${RESULT_ROOT}" \
  --out-dir "${SUMMARY_ROOT}"
