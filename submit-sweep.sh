#!/usr/bin/env bash
# Queue the 36-run arithmetic grid, loading the existing cluster W&B credentials.
# Optional short grid check: bash submit-sweep.sh -append 'experiment_args=--smoke-test'
set -euo pipefail
set +x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WANDB_PROJECT="${WANDB_PROJECT:-arithmetic tasks sharpness}"
export SHARPNESS_SUBMIT_FILE="$script_dir/sweep_arithmetic.sub"
exec bash "$script_dir/submit-wandb.sh" \
    -append "project_dir=$script_dir" \
    "$@"
