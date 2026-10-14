"""Run one of 36 arithmetic configurations, or list the grid without training.

    python sweep_arithmetic.py --list
    python sweep_arithmetic.py --run-index 0 --smoke-test --no-wandb --device cpu

The Condor submit file queues one process per configuration. All runs use the
same seed and four-digit payload, all seven tasks, and no loss-space searches.
Embedding size is the size of EACH token/position embedding; concatenating
them gives d_model = 2 * embedding_size. Smoke checks preserve the selected
architecture and batch size but reduce training and diagnostic sample counts.
"""

import argparse
from itertools import product
import json
from pathlib import Path

import test_arithmetic as experiment


EMBEDDING_SIZES = ( 32, 64, 128)
NUM_LAYERS = (1, 2, 4)
ATTENTION_HEADS = (2, 4, 8)
BATCH_SIZES = (64, 320)


def build_grid():
    return [dict(embedding_size=embedding, num_layers=layers, num_attn_heads=heads, batch_size=batch)
            for embedding, layers, heads, batch in product(
                EMBEDDING_SIZES, NUM_LAYERS, ATTENTION_HEADS, BATCH_SIZES)]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="Print the indexed grid and exit without training.")
    mode.add_argument("--run-index", type=int, help="Zero-based configuration index (0..35).")
    parser.add_argument("--steps", type=experiment.positive_int,
                        help="Training updates (default 30000, or 2 with --smoke-test).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wandb-group", default="arithmetic-hyperparameters")
    return parser


def experiment_arguments(args):
    grid = build_grid()
    if args.run_index is None or not 0 <= args.run_index < len(grid):
        raise ValueError(f"run-index must be between 0 and {len(grid) - 1}.")
    config = grid[args.run_index]
    name = (f"arithmetic-e{config['embedding_size']}-l{config['num_layers']}-"
            f"h{config['num_attn_heads']}-b{config['batch_size']}-seed{args.seed}")
    argv = ["--payload-len", "4", "--grouped-selectors",
            "--seed", str(args.seed), "--run-name", name, "--wandb-group", args.wandb_group,
            "--wandb" if args.wandb else "--no-wandb"]
    for option, value in config.items():
        argv.extend(["--" + option.replace("_", "-"), str(value)])
    if args.smoke_test:
        argv.append("--smoke-test")
    argv.extend(["--steps", str(args.steps if args.steps is not None else (2 if args.smoke_test else 50_000))])
    for option in ("device", "output_dir"):
        value = getattr(args, option)
        if value is not None:
            argv.extend(["--" + option.replace("_", "-"), str(value)])
    return argv


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps([dict(run_index=index, **config) for index, config in enumerate(build_grid())], indent=2))
        return
    try:
        run_args = experiment_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    print(f"Sweep configuration {args.run_index + 1}/{len(build_grid())}: {build_grid()[args.run_index]}", flush=True)
    return experiment.main(run_args)


if __name__ == "__main__":
    main()
