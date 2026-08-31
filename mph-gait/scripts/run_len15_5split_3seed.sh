#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
METHOD_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RELEASE_ROOT=$(cd -- "${METHOD_ROOT}/.." && pwd)
CODE_ROOT="${METHOD_ROOT}/code"
export PYTHONPATH="${RELEASE_ROOT}/shared:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_ROOT}"

PYTHON_CMD=${PYTHON_CMD:-"conda run --no-capture-output -n pytorch python"}
SEEDS=${SEEDS:-"0 1 2"}
SPLITS=${SPLITS:-"0 1 2 3 4"}
WANDB=${WANDB:-off}
DRY_RUN=${DRY_RUN:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}
CONFIG=${CONFIG:-mph_gait/configs/mph_gait_len15.yaml}

read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"
read -r -a EXTRA_ARR <<< "${EXTRA_ARGS}"
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  EXTRA_ARR+=(--inspect-only)
fi

for SEED in ${SEEDS}; do
  for SPLIT in ${SPLITS}; do
    printf '\n[MPH-Gait] seed=%s split=%s\n' "${SEED}" "${SPLIT}"
    "${PYTHON_ARR[@]}" -m mph_gait.train \
      --config "${CONFIG}" \
      --seed "${SEED}" \
      --split-index "${SPLIT}" \
      --run-name "formal_len15/seed_${SEED}/split_${SPLIT}" \
      --wandb "${WANDB}" \
      "${EXTRA_ARR[@]}"
  done
done

printf '\n[done] %s/outputs/mph_gait\n' "${METHOD_ROOT}"
