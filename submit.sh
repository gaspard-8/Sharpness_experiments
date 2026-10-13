#!/usr/bin/env bash
# Submit the full 30,000-step arithmetic experiment with HTCondor/W&B.
# Tangent and sharpness loss-space searches are disabled.
# Pass Condor overrides as in submit-wandb.sh, for example:
#   ./submit.sh -append 'experiment_args=--smoke-test --grouped-selectors'
#   ./submit.sh -append 'experiment_args=--grouped-selectors --digits-per-argument 2'
set -euo pipefail
set +x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WANDB_PROJECT="${WANDB_PROJECT:-arithmetic tasks sharpness}"

# Plain name=value arguments are read BEFORE test.sub and overwritten by its
# defaults. -append applies these settings immediately before the queue command.
exec bash "$script_dir/submit-wandb.sh" \
    -append "project_dir=$script_dir" \
    -append 'experiment_script=test_arithmetic.py' \
    -append 'experiment_args=--steps 30000 --payload-len 4 --grouped-selectors' \
    -append 'output=$(log_dir)/arithmetic.$(ClusterId).$(ProcId).out' \
    -append 'error=$(log_dir)/arithmetic.$(ClusterId).$(ProcId).err' \
    -append 'log=$(log_dir)/arithmetic.$(ClusterId).log' \
    "$@"
