"""Two runs: the original sparse Fourier construction on 20 inputs, degrees 1..6.

    env/bin/python sweep_sparse_long.py --list
    env/bin/python sweep_sparse_long.py --run-index 0 --smoke-test --no-wandb --device cpu

Defaults: 20 inputs, degrees 1..6, one layer, width 256, eight heads,
batch size 64, and 30,000 updates. One target per model; other settings match
the original Fourier experiment.
"""

import argparse
import json
from pathlib import Path

import test_fourier as experiment


FUNCTION_NAMES = ("sparse_powerlaw", "sparse_geometric")
MODEL_SETTINGS = dict(embedding_size=128, num_attn_heads=8, batch_size=64, num_layers=1)


def build_grid():
    return [dict(function=name, **MODEL_SETTINGS) for name in FUNCTION_NAMES]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="Inspect the two configurations without training.")
    mode.add_argument("--run-index", type=int, help="Function index (0..1).")
    parser.add_argument("--steps", type=experiment.positive_int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--function-seed", type=int, default=17)
    parser.add_argument("--eval-seed", type=int, default=23)
    parser.add_argument("--input-len", type=experiment.positive_int, default=20)
    parser.add_argument("--max-degree", type=experiment.positive_int, default=6)
    parser.add_argument("--sparse-terms", type=experiment.positive_int, default=16)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--geometric-decay", type=float, default=0.9)
    parser.add_argument("--eval-inputs", type=experiment.positive_int)
    parser.add_argument("--eval-batch-size", type=experiment.positive_int, default=320)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--sharpness-interval", type=experiment.nonnegative_int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wandb-group", default="sparse-long-functions")
    return parser


def experiment_arguments(args):
    grid = build_grid()
    if args.run_index is None or not 0 <= args.run_index < len(grid):
        raise ValueError(f"run-index must be between 0 and {len(grid) - 1}.")
    chosen = grid[args.run_index]
    name = (f"fourier-{chosen['function']}-n{args.input_len}-k{args.max_degree or args.input_len}-"
            f"e{chosen['embedding_size']}-h{chosen['num_attn_heads']}-b{chosen['batch_size']}-seed{args.seed}")
    argv = ["--run-name", name, "--wandb-group", args.wandb_group,
            "--wandb" if args.wandb else "--no-wandb"]
    for key, value in chosen.items():
        if key != "num_layers":
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    if args.smoke_test:
        argv.append("--smoke-test")
    for key in ("steps", "seed", "function_seed", "eval_seed", "input_len", "max_degree",
                "sparse_terms", "power", "geometric_decay", "eval_inputs", "eval_batch_size",
                "lr", "weight_decay", "sharpness_interval", "device", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    return argv


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps([dict(run_index=index, **chosen, input_len=args.input_len,
                               max_degree=args.max_degree if args.max_degree is not None else args.input_len)
                          for index, chosen in enumerate(build_grid())], indent=2))
        return
    try:
        run_args = experiment_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    print(f"Original sparse configuration {args.run_index + 1}/{len(build_grid())}: {build_grid()[args.run_index]}", flush=True)
    return experiment.main(run_args)


if __name__ == "__main__":
    main()
