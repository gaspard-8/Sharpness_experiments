#!/usr/bin/env bash
# Queue the two existing sparse targets at input lengths 5, 7, and 10.
set -euo pipefail
set +x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WANDB_PROJECT="${WANDB_PROJECT:-single task Fourier sharpness}"
export SHARPNESS_SUBMIT_FILE="$script_dir/sweep_sparse_small.sub"
exec bash "$script_dir/submit-wandb.sh" \
    -append "project_dir=$script_dir" \
    "$@"
