#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

WANDB=${WANDB:-off} bash "${ROOT}/pointnet-tmax/scripts/run_len15_5split_3seed.sh"
WANDB=${WANDB:-off} bash "${ROOT}/mph-gait/scripts/run_len15_5split_3seed.sh"
WANDB=${WANDB:-off} bash "${ROOT}/lidargaitpp/scripts/run_len15_5split_3seed.sh"
WANDB=${WANDB:-off} bash "${ROOT}/projection-baselines/scripts/run_len15_5split_3seed.sh"
