#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RUN_METHODS=${RUN_METHODS:-"pointnet_tmax mph_gait lidargaitpp projection"}
export SEEDS=${SEEDS:-"1 2"}
export SUITES=${SUITES:-"train_clothing train_video_count"}

for method in ${RUN_METHODS}; do
  printf '\n[matched-seed completion] method=%s seeds=%s suites=%s\n'     "${method}" "${SEEDS}" "${SUITES}"
  case "${method}" in
    pointnet_tmax)
      bash "${RELEASE_ROOT}/pointnet-tmax/scripts/run_followup_5split_3seed.sh"
      ;;
    mph_gait)
      bash "${RELEASE_ROOT}/mph-gait/scripts/run_followup_5split_3seed.sh"
      ;;
    lidargaitpp)
      bash "${RELEASE_ROOT}/lidargaitpp/scripts/run_followup_5split_3seed.sh"
      ;;
    projection)
      bash "${RELEASE_ROOT}/projection-baselines/scripts/run_followup_5split_3seed.sh"
      ;;
    *)
      printf 'Unknown method: %s\n' "${method}" >&2
      exit 2
      ;;
  esac
done
