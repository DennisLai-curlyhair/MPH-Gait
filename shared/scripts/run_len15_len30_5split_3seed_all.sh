#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

export SEEDS=${SEEDS:-"0 1 2"}
export SPLITS=${SPLITS:-"0 1 2 3 4"}
export SUITES=${SUITES:-"clip_length"}
export LENGTHS=${LENGTHS:-"15 30"}
export RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)-len15_len30_3seed}
export SKIP_COMPLETED=${SKIP_COMPLETED:-1}
export WANDB=${WANDB:-off}

bash "${RELEASE_ROOT}/pointnet-tmax/scripts/run_len15_len30_5split_3seed.sh"
bash "${RELEASE_ROOT}/mph-gait/scripts/run_len15_len30_5split_3seed.sh"
bash "${RELEASE_ROOT}/lidargaitpp/scripts/run_len15_len30_5split_3seed.sh"

printf '[done] len15/len30 matched-seed runs: %s\n' "${RUN_STAMP}"
