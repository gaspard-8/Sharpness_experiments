#!/usr/bin/env bash
# Submit a short arithmetic GPU check with the existing HTCondor/W&B setup.
# Tangent and sharpness loss-space searches are disabled.
# Pass Condor overrides as in submit-wandb.sh, for example:
#   ./submit.sh 'experiment_args=--smoke-test --grouped-selectors --no-loss-spaces'
#   ./submit.sh 'experiment_args=--grouped-selectors --digits-per-argument 2 --no-loss-spaces'
set -euo pipefail
set +x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WANDB_PROJECT="${WANDB_PROJECT:-arithmetic tasks sharpness}"

exec bash "$script_dir/submit-wandb.sh" \
    "project_dir=$script_dir" \
    'experiment_script=test_arithmetic.py' \
    'experiment_args=--smoke-test --grouped-selectors --no-loss-spaces' \
    'output=$(log_dir)/arithmetic.$(ClusterId).$(ProcId).out' \
    'error=$(log_dir)/arithmetic.$(ClusterId).$(ProcId).err' \
    'log=$(log_dir)/arithmetic.$(ClusterId).log' \
    "$@"
