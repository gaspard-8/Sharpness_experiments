"""Sixteen runs: two sparse Fourier targets on eight length/degree pairs.

    env/bin/python sweep_sparse_grid.py --list
    env/bin/python sweep_sparse_grid.py --run-index 0 --smoke-test --no-wandb --device cpu

Lengths 10..20 and maximum degrees 3..6, excluding the completed (20, 6).
Each target gets eight runs of 50,000 updates. Model settings are unchanged.
"""

import argparse
import json
from pathlib import Path

import test_fourier as experiment


FUNCTION_NAMES = ("sparse_powerlaw", "sparse_geometric")
LENGTH_DEGREE_PAIRS = ((10, 3), (10, 6), (12, 5), (12, 6),
                      (15, 4), (15, 5), (20, 3), (20, 4))
MODEL_SETTINGS = dict(embedding_size=128, num_attn_heads=8, batch_size=64, num_layers=1)
WANDB_PROJECT = "sparse functions fourier analysis"


def build_grid():
    return [dict(function=name, input_len=length, max_degree=degree, steps=50_000,
                 eval_inputs=min(1024, 2 ** length // 4), **MODEL_SETTINGS)
            for length, degree in LENGTH_DEGREE_PAIRS for name in FUNCTION_NAMES]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="Inspect the sixteen configurations without training.")
    mode.add_argument("--run-index", type=int, help="Configuration index (0..15).")
    parser.add_argument("--steps", type=experiment.positive_int, help="Default 50000; smoke test 2.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--function-seed", type=int, default=17)
    parser.add_argument("--eval-seed", type=int, default=23)
    parser.add_argument("--sparse-terms", type=experiment.positive_int, default=16)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--geometric-decay", type=float, default=0.9)
    parser.add_argument("--eval-inputs", type=experiment.positive_int, help="Override the per-length held-out count.")
    parser.add_argument("--eval-batch-size", type=experiment.positive_int, default=320)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--sharpness-interval", type=experiment.nonnegative_int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-group", default="sparse-length-degree-functions")
    return parser


def resolved_grid(args):
    grid = build_grid()
    for chosen in grid:
        if args.steps is not None:
            chosen["steps"] = args.steps
        elif args.smoke_test:
            chosen["steps"] = 2
        if args.eval_inputs is not None:
            chosen["eval_inputs"] = args.eval_inputs
    return grid


def experiment_arguments(args):
    grid = resolved_grid(args)
    if args.run_index is None or not 0 <= args.run_index < len(grid):
        raise ValueError(f"run-index must be between 0 and {len(grid) - 1}.")
    chosen = grid[args.run_index]
    name = (f"fourier-{chosen['function']}-n{chosen['input_len']}-k{chosen['max_degree']}-"
            f"e128-h8-b64-steps{chosen['steps']}-seed{args.seed}-fseed{args.function_seed}")
    argv = ["--run-name", name, "--wandb-project", args.wandb_project, "--wandb-group", args.wandb_group,
            "--wandb" if args.wandb else "--no-wandb"]
    for key, value in chosen.items():
        if key != "num_layers":
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    if args.smoke_test:
        argv.append("--smoke-test")
    for key in ("seed", "function_seed", "eval_seed", "sparse_terms", "power", "geometric_decay",
                "eval_batch_size", "lr", "weight_decay", "sharpness_interval", "device", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    return argv


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    grid = resolved_grid(args)
    if args.list:
        print(json.dumps([dict(run_index=index, **chosen)
                          for index, chosen in enumerate(grid)], indent=2))
        return
    try:
        run_args = experiment_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    print(f"Sparse length/degree configuration {args.run_index + 1}/{len(grid)}: {grid[args.run_index]}", flush=True)
    return experiment.main(run_args)


if __name__ == "__main__":
    main()
