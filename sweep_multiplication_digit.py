"""Nine multiplication-digit models: layers 1/2/4 crossed with widths 512/1024/2048.

    env/bin/python sweep_multiplication_digit.py --prepare
    env/bin/python sweep_multiplication_digit.py --list
    env/bin/python sweep_multiplication_digit.py --run-index 0 --smoke-test --no-wandb --device cpu

Full runs use 50,000 updates, batch 64, eight heads, and one digit target.
Smoke tests retain the architecture with two updates and smaller batches.
"""

import argparse
import json
from pathlib import Path

import test_multiplication_digit as experiment


def build_grid():
    return [dict(num_layers=layers, d_model=width, num_attn_heads=8)
            for layers in (1, 2, 4) for width in (512, 1024, 2048)]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--run-index", type=int, help="Architecture index (0..8).")
    parser.add_argument("--steps", type=experiment.positive_int)
    parser.add_argument("--batch-size", type=experiment.positive_int)
    parser.add_argument("--eval-inputs", type=experiment.positive_int)
    parser.add_argument("--eval-batch-size", type=experiment.positive_int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-seed", type=int, default=23)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--sharpness-interval", type=experiment.nonnegative_int)
    parser.add_argument("--target-dir", type=Path, default=experiment.TARGET_DIR)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default=experiment.WANDB_PROJECT)
    parser.add_argument("--wandb-group", default="multiplication-hundreds-model-sweep")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    return parser


def prepare_targets(args):
    from src.runtime import configure_cpu_environment
    configure_cpu_environment()
    from src.multiplication_fourier import prepare_spectrum, prepare_evaluation
    manifest = prepare_spectrum(args.target_dir)
    prepare_evaluation(args.target_dir, args.eval_inputs or 1024, args.eval_seed)
    return manifest


def experiment_arguments(args):
    grid = build_grid()
    if args.run_index is None or not 0 <= args.run_index < len(grid):
        raise ValueError(f"run-index must be between 0 and {len(grid) - 1}.")
    argv = ["--wandb" if args.wandb else "--no-wandb"]
    for key, value in grid[args.run_index].items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    if args.smoke_test:
        argv.append("--smoke-test")
    for key in ("steps", "batch_size", "eval_inputs", "eval_batch_size", "seed", "eval_seed", "lr",
                "weight_decay", "sharpness_interval", "target_dir", "wandb_project", "wandb_group", "output_dir", "device"):
        value = getattr(args, key)
        if value is not None:
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    return argv


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps([dict(run_index=i, **row, steps=args.steps or (2 if args.smoke_test else 50_000),
                               batch_size=args.batch_size or (4 if args.smoke_test else 64), input_len=6, vocab_size=10)
                          for i, row in enumerate(build_grid())], indent=2))
        return
    if args.prepare:
        manifest = prepare_targets(args)
        print(json.dumps({key: manifest[key] for key in ("definition", "indicator_degree_mass", "scalar_degree_mass",
                                                         "indicator_fraction_of_variance_by_degree",
                                                         "scalar_fraction_of_variance_by_degree")}, indent=2))
        return manifest
    try:
        run_args = experiment_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    print(f"Multiplication digit configuration {args.run_index + 1}/9: {build_grid()[args.run_index]}", flush=True)
    return experiment.main(run_args)


if __name__ == "__main__":
    main()
