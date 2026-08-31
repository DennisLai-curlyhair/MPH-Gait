#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RELEASE_ROOT=$(cd -- "${REPO_ROOT}/../.." && pwd)
export PYTHONPATH="${RELEASE_ROOT}/shared:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$REPO_ROOT"

PYTHON_CMD=${PYTHON_CMD:-"conda run -n pytorch python"}
METHODS=${METHODS:-"rgb_depth gray_depth silhouette"}
SEEDS=${SEEDS:-"0 1 2"}
SPLITS=${SPLITS:-"0 1 2 3 4"}
WANDB=${WANDB:-off}
EXTRA_ARGS=${EXTRA_ARGS:-}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
DRY_RUN=${DRY_RUN:-0}
AUTO_SUMMARY=${AUTO_SUMMARY:-1}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-../outputs/formal_len15_5split_3seed}

read -r -a PYTHON_ARR <<< "$PYTHON_CMD"
read -r -a SEED_ARR <<< "$SEEDS"

MANIFEST="${RESULT_ROOT}/manifests/${RUN_STAMP}.json"
CONFIG_ROOT="${RESULT_ROOT}/generated_configs/${RUN_STAMP}"
SUMMARY_ROOT="${RESULT_ROOT}/summaries/${RUN_STAMP}"

run_tag() {
  case "$1" in
    rgb_depth) printf '%s\n' "projection_rgb_depth_len15" ;;
    gray_depth) printf '%s\n' "projection_gray_depth_len15" ;;
    silhouette) printf '%s\n' "projection_silhouette_len15" ;;
    *) printf 'Unknown projection method: %s\n' "$1" >&2; return 1 ;;
  esac
}

printf '============================================================\n'
printf 'Projection len15 matched-seed evaluation\n'
printf 'run stamp : %s\n' "$RUN_STAMP"
printf 'methods   : %s\n' "$METHODS"
printf 'seeds     : %s\n' "$SEEDS"
printf 'splits    : %s\n' "$SPLITS"
printf 'result root: %s\n' "$RESULT_ROOT"
printf '============================================================\n'

"${PYTHON_ARR[@]}" \
  projection_baseline/prepare_projection_matched_seeds.py \
  --result-root "$RESULT_ROOT" \
  --run-stamp "$RUN_STAMP" \
  --methods "$METHODS" \
  --seeds "$SEEDS" \
  --splits "$SPLITS"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  printf '\n[dry run] Generated configs and manifest only.\n'
  printf '[manifest] %s\n' "$MANIFEST"
  exit 0
fi

for METHOD in $METHODS; do
  TAG=$(run_tag "$METHOD")
  for SEED in "${SEED_ARR[@]}"; do
    CONFIG="${CONFIG_ROOT}/${METHOD}_seed${SEED}.yaml"
    RUN_NAME="${RUN_STAMP}_${TAG}_seed${SEED}"

    printf '\n############################################################\n'
    printf 'Projection method=%s, matched seed=%s\n' "$METHOD" "$SEED"
    printf '############################################################\n'
    CONFIG="$CONFIG" \
      RUN_NAME="$RUN_NAME" \
      SPLITS="$SPLITS" \
      WANDB="$WANDB" \
      PYTHON_CMD="$PYTHON_CMD" \
      EXTRA_ARGS="$EXTRA_ARGS" \
      SKIP_COMPLETED="$SKIP_COMPLETED" \
      bash projection_baseline/scripts/run_folds.sh
  done
done

if [[ " $EXTRA_ARGS " == *" --dry-run "* ]] || \
   [[ " $EXTRA_ARGS " == *" --inspect-only "* ]]; then
  printf '\n[inspection run] Matched summary skipped.\n'
  exit 0
fi

if [[ "$AUTO_SUMMARY" != "0" && "$AUTO_SUMMARY" != "false" ]]; then
  "${PYTHON_ARR[@]}" \
    projection_baseline/summarize_projection_matched_seeds.py \
    --manifest "$MANIFEST" \
    --out-dir "$SUMMARY_ROOT"
fi

printf '\n[done] Projection matched-seed root: %s\n' "$RESULT_ROOT"
printf '[done] Manifest: %s\n' "$MANIFEST"
printf '[done] Summary: %s\n' "$SUMMARY_ROOT"
