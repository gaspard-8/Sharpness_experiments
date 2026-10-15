#!/usr/bin/env bash
# Queue the two sparse Fourier targets with 20 inputs and degrees 1..20.
set -euo pipefail
set +x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WANDB_PROJECT="${WANDB_PROJECT:-single task Fourier sharpness}"
export SHARPNESS_SUBMIT_FILE="$script_dir/sweep_sparse_long.sub"
exec bash "$script_dir/submit-wandb.sh" \
    -append "project_dir=$script_dir" \
    "$@"
