#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
METHOD_ROOT=$(cd "${CODE_ROOT}/.." && pwd)
RELEASE_ROOT=$(cd "${METHOD_ROOT}/.." && pwd)
export PYTHONPATH="${RELEASE_ROOT}/shared:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_ROOT}"

CONDA_ENV=${CONDA_ENV:-pytorch}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
CLIP_LEN=${CLIP_LEN:-15}
MAX_TRAIN_VIDEO_ID=${MAX_TRAIN_VIDEO_ID:-16}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-len${CLIP_LEN}_c2c3_full}
RUN_LABEL=${RUN_LABEL:-lidargaitpp_${EXPERIMENT_TAG}}
EXPERIMENT_BASE=${EXPERIMENT_BASE:-${METHOD_ROOT}/outputs/lidargaitpp}
RUN_ROOT=${RUN_ROOT:-${EXPERIMENT_BASE}/${RUN_STAMP}_${RUN_LABEL}}
SPLITS=${SPLITS:-"0 1 2 3 4"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TOTAL_ITER=${TOTAL_ITER:-10000}
SAVE_ITER=${SAVE_ITER:-1000}
BATCH_ID=${BATCH_ID:-4}
BATCH_SEQ=${BATCH_SEQ:-2}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
EVAL_NUM_WORKERS=${EVAL_NUM_WORKERS:-0}
POINTS_NUM=${POINTS_NUM:-1024}
UNIT_SCALE=${UNIT_SCALE:-0.001}
LEARNING_RATE=${LEARNING_RATE:-0.1}
TRAIN_SEED=${TRAIN_SEED:-0}
EVAL_SEED=${EVAL_SEED:-0}
FORCE_PREPARE=${FORCE_PREPARE:-0}
PREPARE_ONLY=${PREPARE_ONLY:-0}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
SKIP_TRAIN_IF_COMPLETE=${SKIP_TRAIN_IF_COMPLETE:-1}
DRY_RUN=${DRY_RUN:-0}
OPENGAIT_REPO=${OPENGAIT_REPO:-${METHOD_ROOT}/third_party/OpenGait}
BASE_CONFIG=${BASE_CONFIG:-${RELEASE_ROOT}/shared/configs/fixed_special5_pointcloud_len15.yaml}
DATA_ROOT=${DATA_ROOT:-${RELEASE_ROOT}/dataset/local/pointcloud}

if [[ "${EXPERIMENT_BASE}" != /* ]]; then
  EXPERIMENT_BASE="${CODE_ROOT}/${EXPERIMENT_BASE}"
fi
if [[ "${RUN_ROOT}" != /* ]]; then
  RUN_ROOT="${CODE_ROOT}/${RUN_ROOT}"
fi
if ! [[ "${CLIP_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Invalid CLIP_LEN: %s\n' "${CLIP_LEN}" >&2
  exit 2
fi
if ! [[ "${MAX_TRAIN_VIDEO_ID}" =~ ^[1-9][0-9]*$ ]] || (( MAX_TRAIN_VIDEO_ID > 16 )); then
  printf 'MAX_TRAIN_VIDEO_ID must be in [1,16]: %s\n' "${MAX_TRAIN_VIDEO_ID}" >&2
  exit 2
fi
if (( TOTAL_ITER <= 0 || SAVE_ITER <= 0 || TOTAL_ITER % SAVE_ITER != 0 )); then
  printf 'TOTAL_ITER must be positive and divisible by SAVE_ITER.\n' >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

run_python() {
  conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

printf '[LidarGait++] tag=%s clip_len=%s max_train_video_id=%s seed=%s\n' "${EXPERIMENT_TAG}" "${CLIP_LEN}" "${MAX_TRAIN_VIDEO_ID}" "${TRAIN_SEED}"
printf '[LidarGait++] splits=%s run_root=%s\n' "${SPLITS}" "${RUN_ROOT}"
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

mkdir -p "${RUN_ROOT}"
printf '%s\n' "${RUN_ROOT}" > "${EXPERIMENT_BASE}/latest_run.txt"

for SPLIT in ${SPLITS}; do
  SPLIT_ROOT="${RUN_ROOT}/split_${SPLIT}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${SPLIT_ROOT}/final_eval/retrieval_metrics.json" && -f "${SPLIT_ROOT}/checkpoint_selection.json" ]]; then
    printf '[skip completed] split=%s\n' "${SPLIT}"
    continue
  fi

  BUILD_ARGS=(
    lidargaitpp_adapter/build_dataset.py
    --split-root "${SPLIT_ROOT}"
    --split-index "${SPLIT}"
    --base-config "${BASE_CONFIG}"
    --data-root "${DATA_ROOT}"
    --opengait-repo "${OPENGAIT_REPO}"
    --clip-len "${CLIP_LEN}"
    --experiment-tag "${EXPERIMENT_TAG}"
    --drop-first-frames 30
    --max-train-video-id "${MAX_TRAIN_VIDEO_ID}"
    --points-num "${POINTS_NUM}"
    --unit-scale "${UNIT_SCALE}"
    --num-workers "${NUM_WORKERS}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --batch-id "${BATCH_ID}"
    --batch-seq "${BATCH_SEQ}"
    --learning-rate "${LEARNING_RATE}"
    --total-iter "${TOTAL_ITER}"
    --save-iter "${SAVE_ITER}"
    --training-seed "${TRAIN_SEED}"
    --evaluation-seed "${EVAL_SEED}"
  )
  if [[ "${FORCE_PREPARE}" == "1" ]]; then
    BUILD_ARGS+=(--force)
  fi
  run_python "${BUILD_ARGS[@]}"

  if [[ "${PREPARE_ONLY}" == "1" ]]; then
    printf '[prepared only] split=%s audit=%s\n' "${SPLIT}" "${SPLIT_ROOT}/prepared/coordinate_audit.json"
    continue
  fi

  IFS=$'\t' read -r DATASET_NAME SAVE_NAME < <(
    run_python -c '
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg["data_cfg"]["dataset_name"], cfg["trainer_cfg"]["save_name"], sep="\t")
' "${SPLIT_ROOT}/configs/train.yaml"
  )
  TRAIN_WORK="${SPLIT_ROOT}/train_work"
  CHECKPOINT_DIR="${TRAIN_WORK}/output/${DATASET_NAME}/LidarGaitPlusPlus/${SAVE_NAME}/checkpoints"
  EXPECTED_CHECKPOINTS=$((TOTAL_ITER / SAVE_ITER))
  mkdir -p "${TRAIN_WORK}"
  ln -sfn "${OPENGAIT_REPO}/configs" "${TRAIN_WORK}/configs"

  EXISTING_CHECKPOINTS=0
  if [[ -d "${CHECKPOINT_DIR}" ]]; then
    EXISTING_CHECKPOINTS=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name '*.pt' | wc -l)
  fi
  if [[ "${SKIP_TRAIN_IF_COMPLETE}" == "1" && "${EXISTING_CHECKPOINTS}" -ge "${EXPECTED_CHECKPOINTS}" ]]; then
    printf '[skip train] split=%s checkpoints=%s\n' "${SPLIT}" "${EXISTING_CHECKPOINTS}"
  else
    printf '[train] split=%s tag=%s\n' "${SPLIT}" "${EXPERIMENT_TAG}"
    TRAIN_CMD=(
      conda run --no-capture-output -n "${CONDA_ENV}"
      python -m torch.distributed.launch
      --nproc_per_node=1
      "${CODE_ROOT}/lidargaitpp_adapter/opengait_entry.py"
      "${OPENGAIT_REPO}/opengait/main.py"
      --cfgs "${SPLIT_ROOT}/configs/train.yaml"
      --phase train
      --log_to_file
    )
    (
      cd "${TRAIN_WORK}"
      export OPENGAIT_SEED_OFFSET="${TRAIN_SEED}"
      export PYTHONHASHSEED="${TRAIN_SEED}"
      export PYTHONPATH="${OPENGAIT_REPO}:${PYTHONPATH}"
      "${TRAIN_CMD[@]}"
    )
  fi

  mapfile -t CHECKPOINTS < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name '*.pt' | sort)
  if [[ "${#CHECKPOINTS[@]}" -ne "${EXPECTED_CHECKPOINTS}" ]]; then
    printf 'Expected %s checkpoints for split %s, found %s under %s\n' "${EXPECTED_CHECKPOINTS}" "${SPLIT}" "${#CHECKPOINTS[@]}" "${CHECKPOINT_DIR}" >&2
    exit 1
  fi

  for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    BASENAME=$(basename "${CHECKPOINT}")
    if [[ ! "${BASENAME}" =~ -([0-9]+)\.pt$ ]]; then
      printf 'Cannot parse checkpoint iteration: %s\n' "${CHECKPOINT}" >&2
      exit 1
    fi
    ITERATION=$((10#${BASH_REMATCH[1]}))
    SELECTION_DIR=$(printf '%s/checkpoint_selection/iter_%05d' "${SPLIT_ROOT}" "${ITERATION}")
    if [[ ! -f "${SELECTION_DIR}/summary.json" ]]; then
      EVAL_ARGS=(
        lidargaitpp_adapter/evaluate.py
        --config "${SPLIT_ROOT}/configs/validation_eval.yaml"
        --manifest "${SPLIT_ROOT}/manifest.json"
        --checkpoint "${CHECKPOINT}"
        --output-dir "${SELECTION_DIR}"
        --pairs val
        --batch-size "${EVAL_BATCH_SIZE}"
        --num-workers "${EVAL_NUM_WORKERS}"
        --seed "${EVAL_SEED}"
        --opengait-repo "${OPENGAIT_REPO}"
      )
      run_python "${EVAL_ARGS[@]}"
    fi
  done

  SELECT_ARGS=(
    lidargaitpp_adapter/select_checkpoint.py
    --selection-root "${SPLIT_ROOT}/checkpoint_selection"
    --output "${SPLIT_ROOT}/checkpoint_selection.json"
  )
  run_python "${SELECT_ARGS[@]}"

  SELECTED_CHECKPOINT=$(conda run -n "${CONDA_ENV}" python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "${SPLIT_ROOT}/checkpoint_selection.json")
  FINAL_ARGS=(
    lidargaitpp_adapter/evaluate.py
    --config "${SPLIT_ROOT}/configs/final_eval.yaml"
    --manifest "${SPLIT_ROOT}/manifest.json"
    --checkpoint "${SELECTED_CHECKPOINT}"
    --output-dir "${SPLIT_ROOT}/final_eval"
    --batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${EVAL_NUM_WORKERS}"
    --seed "${EVAL_SEED}"
    --opengait-repo "${OPENGAIT_REPO}"
  )
  run_python "${FINAL_ARGS[@]}"
done

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  printf '[done: prepared only] %s\n' "${RUN_ROOT}"
  exit 0
fi

FINAL_COUNT=$(find "${RUN_ROOT}" -path '*/final_eval/retrieval_metrics.json' -type f | wc -l)
if [[ "${FINAL_COUNT}" -eq 5 ]]; then
  SUMMARY_ARGS=(lidargaitpp_adapter/summarize.py --run-root "${RUN_ROOT}" --expected-splits 5)
  run_python "${SUMMARY_ARGS[@]}"
fi

printf '[done] %s\n' "${RUN_ROOT}"
