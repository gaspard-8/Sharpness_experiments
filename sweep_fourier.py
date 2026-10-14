"""Four runs: one per Fourier target, with identical large model settings.

    env/bin/python sweep_fourier.py --list
    env/bin/python sweep_fourier.py --run-index 0 --smoke-test --no-wandb --device cpu

Use the largest embeddings and head count from the arithmetic sweep, with
batch size 64 and the requested one transformer layer. Evaluation inputs and model
initialization are shared across targets; --seed changes initialization/training.
"""

import argparse
import json
from pathlib import Path

import test_fourier as experiment


FIXED_MODEL_SETTINGS = dict(embedding_size=128, num_attn_heads=8, batch_size=64, num_layers=1)


def build_grid():
    return [dict(function=function, **FIXED_MODEL_SETTINGS) for function in experiment.FUNCTION_NAMES]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="Print the indexed grid without training.")
    mode.add_argument("--run-index", type=int, help="Function index (0..3).")
    parser.add_argument("--steps", type=experiment.positive_int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--function-seed", type=int, default=17)
    parser.add_argument("--eval-seed", type=int, default=23)
    parser.add_argument("--input-len", type=experiment.positive_int, default=32)
    parser.add_argument("--max-degree", type=experiment.positive_int)
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
    parser.add_argument("--wandb-group", default="fourier-functions")
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
