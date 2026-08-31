#!/usr/bin/env bash
set -euo pipefail

METHOD_NAME=${METHOD_NAME:?METHOD_NAME is required}
METHOD_MODULE=${METHOD_MODULE:?METHOD_MODULE is required}
METHOD_ROOT=${METHOD_ROOT:?METHOD_ROOT is required}
CODE_ROOT=${CODE_ROOT:?CODE_ROOT is required}
BASE_CONFIG=${BASE_CONFIG:?BASE_CONFIG is required}
RELEASE_ROOT=${RELEASE_ROOT:?RELEASE_ROOT is required}

PYTHON_CMD=${PYTHON_CMD:-"conda run --no-capture-output -n pytorch python"}
SEEDS=${SEEDS:-"0"}
SPLITS=${SPLITS:-"0 1 2 3 4"}
SUITES=${SUITES:-"clip_length train_clothing train_video_count"}
LENGTHS=${LENGTHS:-"1 4 8 30"}
# c2c3_balanced is identical to the 16-videos-per-ID setting below.
TRAIN_VARIANTS=${TRAIN_VARIANTS:-"c2 c3"}
MAX_VIDEO_IDS=${MAX_VIDEO_IDS:-"1 2 4 8"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
WANDB=${WANDB:-off}
DRY_RUN=${DRY_RUN:-0}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
AUTO_SUMMARY=${AUTO_SUMMARY:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

export PYTHONPATH="${RELEASE_ROOT}/shared:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_ROOT}"
read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"
read -r -a EXTRA_ARR <<< "${EXTRA_ARGS}"
shopt -s nullglob

CONFIG_ROOT="${METHOD_ROOT}/outputs/generated_configs/followup/${RUN_STAMP}"
SUMMARY_ROOT="${METHOD_ROOT}/outputs/followup_summaries/${RUN_STAMP}"
mkdir -p "${CONFIG_ROOT}"

OUTPUT_ROOT=$(
  "${PYTHON_ARR[@]}" -c '
from pathlib import Path
import sys
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(Path(config.get("experiment", {}).get("root", "../outputs")).resolve())
' "${BASE_CONFIG}"
)
RESULT_ROOT="${OUTPUT_ROOT}/followup/${RUN_STAMP}"

run_setting() {
  local suite=$1
  local setting=$2
  local config=$3

  for seed in ${SEEDS}; do
    for split in ${SPLITS}; do
      local run_name="followup/${RUN_STAMP}/${suite}/${setting}/seed_${seed}/split_${split}"
      local completed=(
        "${OUTPUT_ROOT}/${run_name}"/*_seed_"${seed}"/retrieval_metrics.json
      )
      if [[ "${SKIP_COMPLETED}" == "1" && ${#completed[@]} -gt 0 ]]; then
        printf '[skip completed] %s seed=%s split=%s\n' "${setting}" "${seed}" "${split}"
        continue
      fi
      printf '[%s] suite=%s setting=%s seed=%s split=%s\n' \
        "${METHOD_NAME}" "${suite}" "${setting}" "${seed}" "${split}"
      if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
        continue
      fi
      "${PYTHON_ARR[@]}" -m "${METHOD_MODULE}.train" \
        --config "${config}" \
        --seed "${seed}" \
        --split-index "${split}" \
        --run-name "${run_name}" \
        --wandb "${WANDB}" \
        "${EXTRA_ARR[@]}"
    done
  done
}

make_config() {
  local output=$1
  shift
  "${PYTHON_ARR[@]}" "${RELEASE_ROOT}/shared/tools/make_followup_config.py" \
    --base "${BASE_CONFIG}" \
    --out "${output}" \
    "$@"
}

for suite in ${SUITES}; do
  case "${suite}" in
    clip_length)
      for length in ${LENGTHS}; do
        setting="len${length}_c2c3_full"
        config="${CONFIG_ROOT}/${suite}_${setting}.yaml"
        make_config "${config}" --clip-len "${length}" --train-variant c2c3_full
        run_setting "${suite}" "${setting}" "${config}"
      done
      ;;
    train_clothing)
      for variant in ${TRAIN_VARIANTS}; do
        setting="len15_${variant}"
        config="${CONFIG_ROOT}/${suite}_${setting}.yaml"
        make_config "${config}" --clip-len 15 --train-variant "${variant}"
        run_setting "${suite}" "${setting}" "${config}"
      done
      ;;
    train_video_count)
      for max_video_id in ${MAX_VIDEO_IDS}; do
        videos_per_id=$((max_video_id * 2))
        setting="len15_${videos_per_id}videos_per_id"
        config="${CONFIG_ROOT}/${suite}_${setting}.yaml"
        make_config "${config}" --clip-len 15 --max-video-id "${max_video_id}"
        run_setting "${suite}" "${setting}" "${config}"
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

if [[ "${AUTO_SUMMARY}" != "0" && "${AUTO_SUMMARY}" != "false" ]] && \
   [[ " ${EXTRA_ARGS} " != *" --dry-run "* ]] && \
   [[ " ${EXTRA_ARGS} " != *" --inspect-only "* ]]; then
  "${PYTHON_ARR[@]}" "${RELEASE_ROOT}/shared/tools/summarize_followup.py" \
    --result-root "${RESULT_ROOT}" \
    --out-dir "${SUMMARY_ROOT}"
fi

printf '[done] results: %s\n' "${RESULT_ROOT}"
printf '[done] summary: %s\n' "${SUMMARY_ROOT}"
