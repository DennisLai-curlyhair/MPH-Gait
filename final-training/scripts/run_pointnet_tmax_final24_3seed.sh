#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FINAL24_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RELEASE_ROOT=$(cd -- "${FINAL24_ROOT}/.." && pwd)
WORKSPACE_ROOT=$(cd -- "${RELEASE_ROOT}/.." && pwd)
METHOD_ROOT="${RELEASE_ROOT}/pointnet-tmax"
CODE_ROOT="${METHOD_ROOT}/code"
CONFIG=${CONFIG:-${FINAL24_ROOT}/configs/pointnet_tmax_final24_len15.yaml}
PYTHON_CMD=${PYTHON_CMD:-"conda run --no-capture-output -n pytorch python"}
SEEDS=${SEEDS:-"0 1 2"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
OUTPUT_BASE=${OUTPUT_BASE:-${RELEASE_ROOT}/outputs/final-training/pointnet-tmax}
RUN_ROOT=${RUN_ROOT:-${OUTPUT_BASE}/${RUN_STAMP}_final24_len15_3seed}
WANDB=${WANDB:-off}
EPOCHS=${EPOCHS:-50}
DRY_RUN=${DRY_RUN:-0}
SMOKE_TEST=${SMOKE_TEST:-0}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"
read -r -a EXTRA_ARR <<< "${EXTRA_ARGS}"
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARR+=(--inspect-only)
elif [[ "${SMOKE_TEST}" == "1" ]]; then
  EXTRA_ARR+=(--dry-run)
fi

mkdir -p "${OUTPUT_BASE}" "${RUN_ROOT}"
printf '%s\n' "${RUN_ROOT}" > "${OUTPUT_BASE}/latest_run.txt"
printf '%s\n' "${SEEDS}" > "${RUN_ROOT}/seeds.txt"
printf '%s\n' "${EPOCHS}" > "${RUN_ROOT}/fixed_epoch_budget.txt"

export PYTHONPATH="${RELEASE_ROOT}/shared:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_ROOT}"
for SEED in ${SEEDS}; do
  SEED_ROOT="${RUN_ROOT}/seed_${SEED}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${SEED_ROOT}/retrieval_metrics.json" ]]; then
    printf '[skip completed] PointNet-TMax seed=%s\n' "${SEED}"
    continue
  fi
  printf '\n[PointNet-TMax Final-24] seed=%s output=%s\n' "${SEED}" "${SEED_ROOT}"
  TRAIN_ARGS=(
    -m pc_v1.train
    --config "${CONFIG}"
    --seed "${SEED}"
    --epochs "${EPOCHS}"
    --run-dir "${SEED_ROOT}"
    --wandb "${WANDB}"
  )
  "${PYTHON_ARR[@]}" "${TRAIN_ARGS[@]}" "${EXTRA_ARR[@]}"
done

if [[ "${DRY_RUN}" != "1" && "${SMOKE_TEST}" != "1" ]]; then
  SUMMARY_ARGS=(
    "${FINAL24_ROOT}/tools/summarize_final24.py"
    --method pointnet_tmax
    --run-root "${RUN_ROOT}"
    --seeds "${SEEDS}"
  )
  "${PYTHON_ARR[@]}" "${SUMMARY_ARGS[@]}"
fi
printf '\n[done] %s\n' "${RUN_ROOT}"
