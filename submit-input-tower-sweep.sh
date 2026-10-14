#!/usr/bin/env bash
# Queue the three input-tower functions with the existing cluster credentials.
set -euo pipefail
set +x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WANDB_PROJECT="${WANDB_PROJECT:-input tower Fourier sharpness}"
export SHARPNESS_SUBMIT_FILE="$script_dir/sweep_input_tower.sub"
exec bash "$script_dir/submit-wandb.sh" \
    -append "project_dir=$script_dir" \
    "$@"
