#!/usr/bin/env bash
# Queue four Fourier targets with the same large model and existing credentials.
# Short check: bash submit-fourier-sweep.sh -append 'experiment_args=--smoke-test'
set -euo pipefail
set +x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WANDB_PROJECT="${WANDB_PROJECT:-single task Fourier sharpness}"
export SHARPNESS_SUBMIT_FILE="$script_dir/sweep_fourier.sub"
exec bash "$script_dir/submit-wandb.sh" \
    -append "project_dir=$script_dir" \
    "$@"
