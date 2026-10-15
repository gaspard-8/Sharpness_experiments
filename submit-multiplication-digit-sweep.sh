#!/usr/bin/env bash
# Nine models for the hundreds digit of a three-digit multiplication.
set -euo pipefail
set +x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ ! -f "$script_dir/multiplication_fourier_targets/manifest.json" ]; then
    printf 'Prepare the target first: python sweep_multiplication_digit.py --prepare\n' >&2
    exit 1
fi
export WANDB_PROJECT="multiplication digit fourier analysis"
export SHARPNESS_SUBMIT_FILE="$script_dir/sweep_multiplication_digit.sub"
exec bash "$script_dir/submit-wandb.sh" \
    -append "project_dir=$script_dir" \
    "$@"
