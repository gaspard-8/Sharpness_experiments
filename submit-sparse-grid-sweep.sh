#!/usr/bin/env bash
# Queue sixteen sparse Fourier runs with varying length and maximum degree.
set -euo pipefail
set +x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WANDB_PROJECT="sparse functions fourier analysis"
export SHARPNESS_SUBMIT_FILE="$script_dir/sweep_sparse_grid.sub"
exec bash "$script_dir/submit-wandb.sh" \
    -append "project_dir=$script_dir" \
    "$@"
