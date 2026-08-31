#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
METHOD_ROOT=$(cd "${CODE_ROOT}/.." && pwd)
RELEASE_ROOT=$(cd "${METHOD_ROOT}/.." && pwd)
export PYTHONPATH="${RELEASE_ROOT}/shared:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_ROOT}"

SEEDS=${SEEDS:-"0 1 2"}
SPLITS=${SPLITS:-"0 1 2 3 4"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
EXPERIMENT_BASE=${EXPERIMENT_BASE:-${METHOD_ROOT}/outputs/lidargaitpp_matched3seed}
MATCHED_RUN_ROOT=${MATCHED_RUN_ROOT:-${EXPERIMENT_BASE}/${RUN_STAMP}_lidargaitpp_len15_5split_3seed}
EVAL_SEED=${EVAL_SEED:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ "${EXPERIMENT_BASE}" != /* ]]; then
  EXPERIMENT_BASE="${CODE_ROOT}/${EXPERIMENT_BASE}"
fi
if [[ "${MATCHED_RUN_ROOT}" != /* ]]; then
  MATCHED_RUN_ROOT="${CODE_ROOT}/${MATCHED_RUN_ROOT}"
fi

mkdir -p "${MATCHED_RUN_ROOT}"
printf '%s\n' "${MATCHED_RUN_ROOT}" > "${EXPERIMENT_BASE}/latest_matched_run.txt"
printf '%s\n' "${SEEDS}" > "${MATCHED_RUN_ROOT}/seeds.txt"
printf '%s\n' "${SPLITS}" > "${MATCHED_RUN_ROOT}/splits.txt"

printf 'Official-code LidarGait++ matched-seed experiment\n'
printf 'run_root=%s\nseeds=%s\nsplits=%s\nevaluation_seed=%s\n' \
  "${MATCHED_RUN_ROOT}" "${SEEDS}" "${SPLITS}" "${EVAL_SEED}"

for SEED in ${SEEDS}; do
  if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    printf 'Invalid non-negative integer seed: %s\n' "${SEED}" >&2
    exit 2
  fi

  SEED_ROOT="${MATCHED_RUN_ROOT}/seed_${SEED}"
  printf '\n[matched seed %s] output=%s\n' "${SEED}" "${SEED_ROOT}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    for SPLIT in ${SPLITS}; do
      printf '[dry-run] training_seed=%s evaluation_seed=%s split=%s\n' \
        "${SEED}" "${EVAL_SEED}" "${SPLIT}"
    done
    continue
  fi

  TRAIN_SEED="${SEED}" \
  EVAL_SEED="${EVAL_SEED}" \
  SPLITS="${SPLITS}" \
  RUN_STAMP="${RUN_STAMP}" \
  RUN_LABEL="lidargaitpp_len15_seed${SEED}" \
  EXPERIMENT_BASE="${EXPERIMENT_BASE}" \
  RUN_ROOT="${SEED_ROOT}" \
    bash "${SCRIPT_DIR}/_run_len15.sh"
done

printf '\n[done] matched run: %s\n' "${MATCHED_RUN_ROOT}"
printf 'Each complete seed has its five-split summary under seed_<N>/summary/.\n'
