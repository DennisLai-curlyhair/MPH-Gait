#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD=${PYTHON_CMD:-"conda run --no-capture-output -n pytorch python"}
METHODS=${METHODS:-"rgb_depth gray_depth silhouette"}
SEEDS=${SEEDS:-"0"}
SPLITS=${SPLITS:-"0 1 2 3 4"}
SUITES=${SUITES:-"clip_length train_clothing train_video_count"}
LENGTHS=${LENGTHS:-"1 4 8 30"}
TRAIN_VARIANTS=${TRAIN_VARIANTS:-"c2 c3"}
MAX_VIDEO_IDS=${MAX_VIDEO_IDS:-"1 2 4 8"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
WANDB=${WANDB:-off}
DRY_RUN=${DRY_RUN:-0}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
AUTO_SUMMARY=${AUTO_SUMMARY:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"
read -r -a EXTRA_ARR <<< "${EXTRA_ARGS}"
shopt -s nullglob

CONFIG_ROOT="../outputs/generated_configs/followup/${RUN_STAMP}"
SUMMARY_ROOT="../outputs/followup_summaries/${RUN_STAMP}"
mkdir -p "${CONFIG_ROOT}"
RUN_GROUPS=()

base_config() {
  case "$1" in
    rgb_depth|gray_depth|silhouette)
      printf 'projection_baseline/configs/%s_len15.yaml\n' "$1"
      ;;
    *)
      printf 'Unknown projection method: %s\n' "$1" >&2
      exit 2
      ;;
  esac
}

output_root() {
  "${PYTHON_ARR[@]}" -c '
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(Path(cfg["experiment"]["root"]).resolve())
' "$1"
}

make_config() {
  local base=$1
  local output=$2
  local seed=$3
  shift 3
  "${PYTHON_ARR[@]}" projection_baseline/make_config.py     --base "${base}"     --out "${output}"     --seed "${seed}"     "$@"
}

run_setting() {
  local method=$1
  local suite=$2
  local setting=$3
  shift 3

  local base root seed split config run_group run_name
  base=$(base_config "${method}")
  root=$(output_root "${base}")

  for seed in ${SEEDS}; do
    config="${CONFIG_ROOT}/${method}/${suite}/${setting}_seed${seed}.yaml"
    make_config "${base}" "${config}" "${seed}" "$@"
    run_group="${root}/followup/${RUN_STAMP}/${suite}/${setting}/seed_${seed}"
    RUN_GROUPS+=("${run_group}")

    for split in ${SPLITS}; do
      run_name="followup/${RUN_STAMP}/${suite}/${setting}/seed_${seed}/split${split}"
      if [[ "${SKIP_COMPLETED}" == "1" ]] &&          find "${root}/${run_name}" -mindepth 1 -name retrieval_metrics.json            -print -quit 2>/dev/null | grep -q .; then
        printf '[skip completed] method=%s setting=%s seed=%s split=%s\n'           "${method}" "${setting}" "${seed}" "${split}"
        continue
      fi

      printf '[projection] method=%s suite=%s setting=%s seed=%s split=%s\n'         "${method}" "${suite}" "${setting}" "${seed}" "${split}"
      if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
        continue
      fi
      "${PYTHON_ARR[@]}" -m projection_baseline.train         --config "${config}"         --seed "${seed}"         --split-index "${split}"         --run-name "${run_name}"         --wandb "${WANDB}"         "${EXTRA_ARR[@]}"
    done
  done
}

for method in ${METHODS}; do
  for suite in ${SUITES}; do
    case "${suite}" in
      clip_length)
        for length in ${LENGTHS}; do
          run_setting "${method}" "${suite}" "len${length}_c2c3_full"             --clip-len "${length}" --train-variant c2c3_full
        done
        ;;
      train_clothing)
        for variant in ${TRAIN_VARIANTS}; do
          run_setting "${method}" "${suite}" "len15_${variant}"             --clip-len 15 --train-variant "${variant}"
        done
        ;;
      train_video_count)
        for max_video_id in ${MAX_VIDEO_IDS}; do
          videos_per_id=$((max_video_id * 2))
          run_setting "${method}" "${suite}"             "len15_${videos_per_id}videos_per_id"             --clip-len 15 --max-video-id "${max_video_id}"
        done
        ;;
      *)
        printf 'Unknown suite: %s\n' "${suite}" >&2
        exit 2
        ;;
    esac
  done
done

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  printf '[dry run] generated configs: %s\n' "${CONFIG_ROOT}"
  exit 0
fi

if [[ "${AUTO_SUMMARY}" != "0" && "${AUTO_SUMMARY}" != "false" ]] &&    [[ " ${EXTRA_ARGS} " != *" --dry-run "* ]] &&    [[ " ${EXTRA_ARGS} " != *" --inspect-only "* ]]; then
  "${PYTHON_ARR[@]}" projection_baseline/summarize_results.py     --out-dir "${SUMMARY_ROOT}"     --tag "${RUN_STAMP}_projection_followup"     "${RUN_GROUPS[@]}"
fi

printf '[done] summary: %s\n' "${SUMMARY_ROOT}"
