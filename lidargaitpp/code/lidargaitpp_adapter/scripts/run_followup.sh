#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
METHOD_ROOT=$(cd -- "${CODE_ROOT}/.." && pwd)
RELEASE_ROOT=$(cd -- "${METHOD_ROOT}/.." && pwd)
export PYTHONPATH="${RELEASE_ROOT}/shared:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_ROOT}"

CONDA_ENV=${CONDA_ENV:-pytorch}
SEEDS=${SEEDS:-"0"}
SPLITS=${SPLITS:-"0 1 2 3 4"}
SUITES=${SUITES:-"clip_length train_clothing train_video_count"}
LENGTHS=${LENGTHS:-"1 4 8 30"}
# c2c3_balanced is identical to the 16-videos-per-ID setting below.
TRAIN_VARIANTS=${TRAIN_VARIANTS:-"c2 c3"}
MAX_VIDEO_IDS=${MAX_VIDEO_IDS:-"1 2 4 8"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
DRY_RUN=${DRY_RUN:-0}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
EVAL_SEED=${EVAL_SEED:-0}
OPENGAIT_REPO=${OPENGAIT_REPO:-${METHOD_ROOT}/third_party/OpenGait}
BASE_CONFIG=${BASE_CONFIG:-${RELEASE_ROOT}/shared/configs/fixed_special5_pointcloud_len15.yaml}
SOURCE_CONFIG=${BASE_CONFIG}
EXPERIMENT_BASE=${EXPERIMENT_BASE:-${METHOD_ROOT}/outputs/lidargaitpp}
CONFIG_ROOT="${METHOD_ROOT}/outputs/generated_configs/followup/${RUN_STAMP}"
RESULT_ROOT="${EXPERIMENT_BASE}/followup/${RUN_STAMP}"

mkdir -p "${CONFIG_ROOT}"

run_python() {
  conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

make_config() {
  local output=$1
  shift
  CONFIG_ARGS=("${RELEASE_ROOT}/shared/tools/make_followup_config.py" --base "${SOURCE_CONFIG}" --out "${output}")
  CONFIG_ARGS+=("$@")
  run_python "${CONFIG_ARGS[@]}"
}

run_setting() {
  local suite=$1
  local setting=$2
  local config=$3
  local clip_len=$4
  local max_train_video_id=$5

  for seed in ${SEEDS}; do
    run_root="${RESULT_ROOT}/${suite}/${setting}/seed_${seed}"
    printf '[LidarGait++ follow-up] suite=%s setting=%s seed=%s\n' "${suite}" "${setting}" "${seed}"
    export TRAIN_SEED="${seed}"
    export EVAL_SEED SPLITS RUN_STAMP EXPERIMENT_BASE OPENGAIT_REPO SKIP_COMPLETED DRY_RUN
    export RUN_ROOT="${run_root}"
    export CLIP_LEN="${clip_len}"
    export MAX_TRAIN_VIDEO_ID="${max_train_video_id}"
    export EXPERIMENT_TAG="${setting}"
    BASE_CONFIG="${config}" bash "${SCRIPT_DIR}/_run_setting.sh"
  done
}

if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" && ! -f "${OPENGAIT_REPO}/opengait/modeling/models/lidargaitv2.py" ]]; then
  printf 'OpenGait is missing: %s\n' "${OPENGAIT_REPO}" >&2
  printf 'Run: bash %s/lidargaitpp/scripts/fetch_opengait.sh\n' "${RELEASE_ROOT}" >&2
  exit 2
fi

for suite in ${SUITES}; do
  case "${suite}" in
    clip_length)
      for length in ${LENGTHS}; do
        setting="len${length}_c2c3_full"
        config="${CONFIG_ROOT}/${suite}_${setting}.yaml"
        make_config "${config}" --clip-len "${length}" --train-variant c2c3_full
        run_setting "${suite}" "${setting}" "${config}" "${length}" 16
      done
      ;;
    train_clothing)
      for variant in ${TRAIN_VARIANTS}; do
        setting="len15_${variant}"
        config="${CONFIG_ROOT}/${suite}_${setting}.yaml"
        make_config "${config}" --clip-len 15 --train-variant "${variant}"
        max_train_video_id=16
        if [[ "${variant}" == "c2c3_balanced" ]]; then
          max_train_video_id=8
        fi
        run_setting "${suite}" "${setting}" "${config}" 15 "${max_train_video_id}"
      done
      ;;
    train_video_count)
      for max_video_id in ${MAX_VIDEO_IDS}; do
        videos_per_id=$((max_video_id * 2))
        setting="len15_${videos_per_id}videos_per_id"
        config="${CONFIG_ROOT}/${suite}_${setting}.yaml"
        make_config "${config}" --clip-len 15 --max-video-id "${max_video_id}"
        run_setting "${suite}" "${setting}" "${config}" 15 "${max_video_id}"
      done
      ;;
    *)
      printf 'Unknown suite: %s\n' "${suite}" >&2
      exit 2
      ;;
  esac
done

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  printf '[dry run] generated configs: %s\n' "${CONFIG_ROOT}"
  exit 0
fi

printf '[done] LidarGait++ follow-up root: %s\n' "${RESULT_ROOT}"
