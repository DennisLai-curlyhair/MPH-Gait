#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FINAL24_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RELEASE_ROOT=$(cd -- "${FINAL24_ROOT}/.." && pwd)
WORKSPACE_ROOT=$(cd -- "${RELEASE_ROOT}/.." && pwd)
METHOD_ROOT="${RELEASE_ROOT}/lidargaitpp"
CODE_ROOT="${METHOD_ROOT}/code"
CONFIG=${CONFIG:-${FINAL24_ROOT}/configs/lidargaitpp_final24_protocol_len15.yaml}
CONDA_ENV=${CONDA_ENV:-pytorch}
SEEDS=${SEEDS:-"0 1 2"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
OUTPUT_BASE=${OUTPUT_BASE:-${RELEASE_ROOT}/outputs/final-training/lidargaitpp}
RUN_ROOT=${RUN_ROOT:-${OUTPUT_BASE}/${RUN_STAMP}_final24_len15_3seed}
CLIP_LEN=${CLIP_LEN:-15}
TOTAL_ITER=${TOTAL_ITER:-10000}
SAVE_ITER=${SAVE_ITER:-1000}
BATCH_ID=${BATCH_ID:-4}
BATCH_SEQ=${BATCH_SEQ:-2}
POINTS_NUM=${POINTS_NUM:-1024}
NUM_WORKERS=${NUM_WORKERS:-4}
EVAL_NUM_WORKERS=${EVAL_NUM_WORKERS:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-0.1}
EVAL_SEED=${EVAL_SEED:-0}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FORCE_PREPARE=${FORCE_PREPARE:-0}
PREPARE_ONLY=${PREPARE_ONLY:-0}
INSPECT_ONLY=${INSPECT_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
OPENGAIT_REPO=${OPENGAIT_REPO:-${METHOD_ROOT}/third_party/OpenGait}
DATA_ROOT=${DATA_ROOT:-${RELEASE_ROOT}/dataset/local/pointcloud}

if (( TOTAL_ITER <= 0 || SAVE_ITER <= 0 || TOTAL_ITER % SAVE_ITER != 0 )); then
  printf 'TOTAL_ITER must be positive and divisible by SAVE_ITER.\n' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="${RELEASE_ROOT}/shared:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_ROOT}"

run_python() {
  conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

mkdir -p "${OUTPUT_BASE}" "${RUN_ROOT}"
printf '%s\n' "${RUN_ROOT}" > "${OUTPUT_BASE}/latest_run.txt"
printf '%s\n' "${SEEDS}" > "${RUN_ROOT}/seeds.txt"
printf '%s\n' "${TOTAL_ITER}" > "${RUN_ROOT}/fixed_iteration_budget.txt"
printf '[LidarGait++ Final-24] run_root=%s seeds=%s total_iter=%s\n' "${RUN_ROOT}" "${SEEDS}" "${TOTAL_ITER}"

if [[ "${DRY_RUN}" == "1" ]]; then
  for SEED in ${SEEDS}; do
    printf '[dry-run] seed=%s train_ids=24 fixed_eval_ids=5 output=%s/seed_%s\n' "${SEED}" "${RUN_ROOT}" "${SEED}"
  done
  exit 0
fi

for SEED in ${SEEDS}; do
  SEED_ROOT="${RUN_ROOT}/seed_${SEED}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${SEED_ROOT}/final_eval/retrieval_metrics.json" ]]; then
    printf '[skip completed] LidarGait++ seed=%s\n' "${SEED}"
    continue
  fi

  BUILD_ARGS=(
    lidargaitpp_adapter/build_dataset.py
    --split-root "${SEED_ROOT}"
    --split-index 0
    --base-config "${CONFIG}"
    --data-root "${DATA_ROOT}"
    --opengait-repo "${OPENGAIT_REPO}"
    --clip-len "${CLIP_LEN}"
    --experiment-tag "final24_len15_seed${SEED}"
    --drop-first-frames 30
    --max-train-video-id 16
    --points-num "${POINTS_NUM}"
    --unit-scale 0.001
    --num-workers "${NUM_WORKERS}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --batch-id "${BATCH_ID}"
    --batch-seq "${BATCH_SEQ}"
    --learning-rate "${LEARNING_RATE}"
    --total-iter "${TOTAL_ITER}"
    --save-iter "${SAVE_ITER}"
    --training-seed "${SEED}"
    --evaluation-seed "${EVAL_SEED}"
  )
  if [[ "${FORCE_PREPARE}" == "1" ]]; then
    BUILD_ARGS+=(--force)
  fi
  if [[ "${INSPECT_ONLY}" == "1" ]]; then
    BUILD_ARGS+=(--inspect-only)
  fi
  run_python "${BUILD_ARGS[@]}"

  if [[ "${INSPECT_ONLY}" == "1" ]]; then
    printf '[inspected only] seed=%s\n' "${SEED}"
    continue
  fi
  if [[ "${PREPARE_ONLY}" == "1" ]]; then
    printf '[prepared only] seed=%s manifest=%s\n' "${SEED}" "${SEED_ROOT}/manifest.json"
    continue
  fi

  IFS=$'\t' read -r DATASET_NAME SAVE_NAME < <(
    run_python -c '
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg["data_cfg"]["dataset_name"], cfg["trainer_cfg"]["save_name"], sep="\t")
' "${SEED_ROOT}/configs/train.yaml"
  )
  TRAIN_WORK="${SEED_ROOT}/train_work"
  CHECKPOINT_DIR="${TRAIN_WORK}/output/${DATASET_NAME}/LidarGaitPlusPlus/${SAVE_NAME}/checkpoints"
  mkdir -p "${TRAIN_WORK}"
  ln -sfn "${OPENGAIT_REPO}/configs" "${TRAIN_WORK}/configs"

  FINAL_CHECKPOINT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "*-${TOTAL_ITER}.pt" -print -quit 2>/dev/null || true)
  if [[ -z "${FINAL_CHECKPOINT}" ]]; then
    printf '[train] LidarGait++ Final-24 seed=%s\n' "${SEED}"
    TRAIN_CMD=(
      conda run --no-capture-output -n "${CONDA_ENV}"
      python -m torch.distributed.launch
      --nproc_per_node=1
      "${CODE_ROOT}/lidargaitpp_adapter/opengait_entry.py"
      "${OPENGAIT_REPO}/opengait/main.py"
      --cfgs "${SEED_ROOT}/configs/train.yaml"
      --phase train
      --log_to_file
    )
    (
      cd "${TRAIN_WORK}"
      export OPENGAIT_SEED_OFFSET="${SEED}"
      export PYTHONHASHSEED="${SEED}"
      export PYTHONPATH="${OPENGAIT_REPO}:${PYTHONPATH}"
      "${TRAIN_CMD[@]}"
    )
    FINAL_CHECKPOINT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "*-${TOTAL_ITER}.pt" -print -quit)
  fi
  if [[ -z "${FINAL_CHECKPOINT}" || ! -f "${FINAL_CHECKPOINT}" ]]; then
    printf 'Final iteration checkpoint was not produced for seed %s.\n' "${SEED}" >&2
    exit 1
  fi

  EVALUATE_ARGS=(
    lidargaitpp_adapter/evaluate.py
    --config "${SEED_ROOT}/configs/final_eval.yaml"
    --manifest "${SEED_ROOT}/manifest.json"
    --checkpoint "${FINAL_CHECKPOINT}"
    --output-dir "${SEED_ROOT}/final_eval"
    --batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${EVAL_NUM_WORKERS}"
    --seed "${EVAL_SEED}"
    --opengait-repo "${OPENGAIT_REPO}"
  )
  run_python "${EVALUATE_ARGS[@]}"
done

if [[ "${PREPARE_ONLY}" != "1" && "${INSPECT_ONLY}" != "1" ]]; then
  SUMMARY_ARGS=(
    "${FINAL24_ROOT}/tools/summarize_final24.py"
    --method lidargaitpp
    --run-root "${RUN_ROOT}"
    --seeds "${SEEDS}"
  )
  run_python "${SUMMARY_ARGS[@]}"
fi
printf '\n[done] %s\n' "${RUN_ROOT}"
