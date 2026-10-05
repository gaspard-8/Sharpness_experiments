#!/usr/bin/env bash
set -euo pipefail
set +x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
credentials_file="$HOME/.config/wandb/cluster.env"
if [ ! -r "$credentials_file" ]; then
    printf 'Missing credentials file: %s\nSee the cluster setup in README.md.\n' "$credentials_file" >&2
    exit 1
fi
requested_project="${WANDB_PROJECT:-}"
requested_entity="${WANDB_ENTITY:-}"
source "$credentials_file"
: "${WANDB_API_KEY:?W&B API key is missing}"
if [[ "$WANDB_API_KEY" =~ [[:space:]] ]]; then
    printf 'W&B API key contains whitespace; re-enter it in cluster.env.\n' >&2
    exit 1
fi
export WANDB_API_KEY
export WANDB_PROJECT="${requested_project:-${WANDB_PROJECT:-multiple tasks sharpness}}"
export WANDB_ENTITY="${requested_entity:-${WANDB_ENTITY:-gaspardtomas-universit-t-des-saarlandes-saarland-university}}"

mkdir -p /scratch/gtomas/logs/sharpness_experiments \
    /scratch/gtomas/sharpness_experiments/results
chmod +x "$script_dir/run-test.sh"
exec condor_submit "$script_dir/test.sub" "$@"
