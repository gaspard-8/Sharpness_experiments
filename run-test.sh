#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

work_dir="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$work_dir"
mkdir -p results/wandb results/code

archive_results() {
    job_status=$?
    trap - EXIT
    printf '%s\n' "$job_status" > results/exit-code.txt
    if ! tar -czf results.tar.gz results; then
        printf 'Could not archive local results.\n' >&2
        exit 1
    fi
    exit "$job_status"
}
trap archive_results EXIT

for variable in WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY; do
    if [ -z "${!variable:-}" ]; then
        printf 'Missing %s; submit through submit-wandb.sh.\n' "$variable" >&2
        exit 1
    fi
done
export WANDB_MODE=online
export SHARPNESS_DEVICE=cuda
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Configure W&B before importing it. The container's home may be unwritable.
runtime_dir=$(mktemp -d "$work_dir/.wandb-runtime-XXXXXX")
export WANDB_DIR="$work_dir/results/wandb"
export WANDB_CACHE_DIR="$runtime_dir/cache"
export WANDB_CONFIG_DIR="$runtime_dir/config"
export WANDB_DATA_DIR="$work_dir/results/wandb-data"
export WANDB_ARTIFACT_DIR="$runtime_dir/artifacts"
export XDG_CACHE_HOME="$runtime_dir/xdg-cache"
export TMPDIR="$runtime_dir/tmp"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" \
    "$WANDB_ARTIFACT_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

cp test.py results/code/
cp -R src results/code/
printf 'Host: %s\nStarted (UTC): %s\n' "$(hostname)" "$(date -u +%FT%TZ)" > results/job-context.txt
python -u test.py "$@"
