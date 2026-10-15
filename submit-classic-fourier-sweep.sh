#!/usr/bin/env bash
# Queue one model for each prepared natural-function Fourier target.
set -euo pipefail
set +x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ ! -f "$script_dir/classic_fourier_targets/manifest.json" ]; then
    printf 'Prepare the spectra first: python sweep_classic_fourier.py --prepare\n' >&2
    exit 1
fi
export WANDB_PROJECT="sparse functions fourier analysis"
export SHARPNESS_SUBMIT_FILE="$script_dir/sweep_classic_fourier.sub"
exec bash "$script_dir/submit-wandb.sh" \
    -append "project_dir=$script_dir" \
    "$@"
