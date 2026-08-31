#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-projection_baseline/configs/rgb_depth_len15.yaml}
SPLITS=${SPLITS:-"0 1 2 3 4"}
PYTHON_CMD=${PYTHON_CMD:-"conda run -n pytorch python"}
EXTRA_ARGS=${EXTRA_ARGS:-}
WANDB=${WANDB:-off}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

read -r -a PYTHON_ARR <<< "$PYTHON_CMD"
read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"

EXPERIMENT_ROOT=$(
  "${PYTHON_ARR[@]}" -c '
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(cfg.get("experiment", {}).get("root", "../outputs/projection"))
' "$CONFIG"
)

CONFIG_TAG=$(
  "${PYTHON_ARR[@]}" -c '
from pathlib import Path
import re
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
method = cfg.get("method", {}).get("id", "projection")
clip_len = cfg.get("data", {}).get("clip_len", "na")
tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{method}_len{clip_len}")
print(tag.strip("_"))
' "$CONFIG"
)

RUN_NAME=${RUN_NAME:-$(date +%Y%m%d-%H%M%S)_${CONFIG_TAG}}
RUN_ROOT="${EXPERIMENT_ROOT}/${RUN_NAME}"
CONFIG_SNAPSHOT="${RUN_ROOT}/config_snapshot.yaml"
mkdir -p "$RUN_ROOT"
cp "$CONFIG" "$CONFIG_SNAPSHOT"

printf '[config] %s\n' "$CONFIG"
printf '[snapshot] %s\n' "$CONFIG_SNAPSHOT"
printf '[run root] %s\n' "$RUN_ROOT"
printf '[splits] %s\n' "$SPLITS"

for SPLIT in $SPLITS; do
  SPLIT_ROOT="${RUN_ROOT}/split${SPLIT}"
  if [[ "$SKIP_COMPLETED" == "1" || "$SKIP_COMPLETED" == "true" ]]; then
    if find "$SPLIT_ROOT" -mindepth 2 -maxdepth 2 \
      -name retrieval_metrics.json -print -quit 2>/dev/null | grep -q .; then
      printf '[skip completed] split=%s under %s\n' "$SPLIT" "$SPLIT_ROOT"
      continue
    fi
  fi

  printf '\n========== projection split=%s ==========\n' "$SPLIT"
  "${PYTHON_ARR[@]}" projection_baseline/train.py \
    --config "$CONFIG_SNAPSHOT" \
    --split-index "$SPLIT" \
    --run-name "${RUN_NAME}/split${SPLIT}" \
    --wandb "$WANDB" \
    "${EXTRA_ARGS_ARR[@]}"
done

printf '\n[done] outputs: %s/split*/\n' "$RUN_ROOT"
