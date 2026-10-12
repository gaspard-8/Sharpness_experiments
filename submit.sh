#!/usr/bin/env bash
# Submit a short arithmetic GPU check with the existing HTCondor/W&B setup.
# Tangent and sharpness loss-space searches are disabled.
# Pass Condor overrides as in submit-wandb.sh, for example:
#   ./submit.sh -append 'experiment_args=--smoke-test --grouped-selectors --no-loss-spaces'
#   ./submit.sh -append 'experiment_args=--grouped-selectors --digits-per-argument 2 --no-loss-spaces'
set -euo pipefail
set +x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WANDB_PROJECT="${WANDB_PROJECT:-arithmetic tasks sharpness}"

# Plain name=value arguments are read BEFORE test.sub and overwritten by its
# defaults. -append applies these settings immediately before the queue command.
exec bash "$script_dir/submit-wandb.sh" \
    -append "project_dir=$script_dir" \
    -append 'experiment_script=test_arithmetic.py' \
    -append 'experiment_args=--smoke-test --grouped-selectors --no-loss-spaces' \
    -append 'output=$(log_dir)/arithmetic.$(ClusterId).$(ProcId).out' \
    -append 'error=$(log_dir)/arithmetic.$(ClusterId).$(ProcId).err' \
    -append 'log=$(log_dir)/arithmetic.$(ClusterId).log' \
    "$@"
