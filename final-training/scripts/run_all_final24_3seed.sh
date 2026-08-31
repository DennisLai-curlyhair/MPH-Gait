#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FINAL24_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RELEASE_ROOT=$(cd -- "${FINAL24_ROOT}/.." && pwd)
WORKSPACE_ROOT=$(cd -- "${RELEASE_ROOT}/.." && pwd)
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
SEEDS=${SEEDS:-"0 1 2"}
DRY_RUN=${DRY_RUN:-0}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${RELEASE_ROOT}/outputs/final-training}
POINTNET_OUTPUT_BASE="${EXPERIMENT_ROOT}/pointnet_tmax"
MPH_OUTPUT_BASE="${EXPERIMENT_ROOT}/mph-gait"
LIDAR_OUTPUT_BASE="${EXPERIMENT_ROOT}/lidargaitpp"
POINTNET_ROOT="${POINTNET_OUTPUT_BASE}/${RUN_STAMP}_final24_len15_3seed"
MPH_ROOT="${MPH_OUTPUT_BASE}/${RUN_STAMP}_final24_len15_3seed"
LIDAR_ROOT="${LIDAR_OUTPUT_BASE}/${RUN_STAMP}_final24_len15_3seed"
COMPARISON_ROOT="${EXPERIMENT_ROOT}/comparisons/${RUN_STAMP}_final24_len15_3seed"

COMMON_ENV=(
  "RUN_STAMP=${RUN_STAMP}"
  "SEEDS=${SEEDS}"
  "DRY_RUN=${DRY_RUN}"
)
env "${COMMON_ENV[@]}" "OUTPUT_BASE=${POINTNET_OUTPUT_BASE}" "RUN_ROOT=${POINTNET_ROOT}" "WANDB=${WANDB:-off}" bash "${SCRIPT_DIR}/run_pointnet_tmax_final24_3seed.sh"

env "${COMMON_ENV[@]}" "OUTPUT_BASE=${MPH_OUTPUT_BASE}" "RUN_ROOT=${MPH_ROOT}" "WANDB=${WANDB:-off}" bash "${SCRIPT_DIR}/run_mph_gait_final24_3seed.sh"

env "${COMMON_ENV[@]}" "OUTPUT_BASE=${LIDAR_OUTPUT_BASE}" "RUN_ROOT=${LIDAR_ROOT}" bash "${SCRIPT_DIR}/run_lidargaitpp_final24_3seed.sh"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

COMPARE_ARGS=(
  "${FINAL24_ROOT}/tools/compare_final24.py"
  --pointnet-root "${POINTNET_ROOT}"
  --mph-root "${MPH_ROOT}"
  --lidargaitpp-root "${LIDAR_ROOT}"
  --output-dir "${COMPARISON_ROOT}"
)
conda run --no-capture-output -n pytorch python "${COMPARE_ARGS[@]}"

printf '\n[all done] comparison=%s/comparison.md\n' "${COMPARISON_ROOT}"
