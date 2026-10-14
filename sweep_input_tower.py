"""Three input-tower targets, one independent model per degree profile.

    env/bin/python sweep_input_tower.py --list
    env/bin/python sweep_input_tower.py --run-index 0 --smoke-test --no-wandb --device cpu

Default: 32 input bits, an 8-input core, degrees 1..8, and decreasing mass as
each additional input is introduced. Only the degree profile changes between
runs. Model settings: one layer, width 256, eight heads, batch size 64.
"""

import argparse
import json
from pathlib import Path

import test_fourier as experiment


MODEL_SETTINGS = dict(embedding_size=128, num_attn_heads=8, batch_size=64, num_layers=1)


def build_grid():
    return [dict(function=name, **MODEL_SETTINGS) for name in experiment.INPUT_TOWER_FUNCTION_NAMES]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="Inspect the three configurations without training.")
    mode.add_argument("--run-index", type=int, help="Function index (0..2).")
    parser.add_argument("--steps", type=experiment.positive_int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--function-seed", type=int, default=17)
    parser.add_argument("--eval-seed", type=int, default=23)
    parser.add_argument("--input-len", type=experiment.positive_int, default=32)
    parser.add_argument("--tower-core-size", type=experiment.positive_int, default=8)
    parser.add_argument("--max-degree", type=experiment.positive_int, help="Default: tower-core-size.")
    parser.add_argument("--terms-per-stage-degree", type=experiment.positive_int, default=16)
    parser.add_argument("--degree-power", type=float, default=2.0)
    parser.add_argument("--input-spectrum", choices=("powerlaw", "geometric"), default="powerlaw")
    parser.add_argument("--input-power", type=float, default=2.0)
    parser.add_argument("--input-decay", type=float, default=0.9)
    parser.add_argument("--input-order", choices=("prefix", "permuted"), default="prefix")
    parser.add_argument("--top-mass", type=float, default=0.9)
    parser.add_argument("--top-band", type=experiment.positive_int)
    parser.add_argument("--eval-inputs", type=experiment.positive_int)
    parser.add_argument("--eval-batch-size", type=experiment.positive_int, default=320)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--sharpness-interval", type=experiment.nonnegative_int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wandb-group", default="input-tower-functions")
    return parser


def experiment_arguments(args):
    grid = build_grid()
    if args.run_index is None or not 0 <= args.run_index < len(grid):
        raise ValueError(f"run-index must be between 0 and {len(grid) - 1}.")
    chosen = grid[args.run_index]
    degree = args.max_degree if args.max_degree is not None else args.tower_core_size
    name = (f"{chosen['function']}-n{args.input_len}-k{degree}-core{args.tower_core_size}-"
            f"{args.input_order}-e128-h8-b64-seed{args.seed}-fseed{args.function_seed}")
    argv = ["--run-name", name, "--wandb-group", args.wandb_group,
            "--wandb" if args.wandb else "--no-wandb"]
    for key, value in chosen.items():
        if key != "num_layers":
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    if args.smoke_test:
        argv.append("--smoke-test")
    translations = {"terms_per_stage_degree": "sparse_terms", "degree_power": "power"}
    for key in ("steps", "seed", "function_seed", "eval_seed", "input_len", "tower_core_size",
                "max_degree", "terms_per_stage_degree", "degree_power", "input_spectrum", "input_power",
                "input_decay", "input_order", "top_mass", "top_band", "eval_inputs", "eval_batch_size",
                "lr", "weight_decay", "sharpness_interval", "device", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            argv.extend(["--" + translations.get(key, key).replace("_", "-"), str(value)])
    return argv


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps([dict(run_index=index, **chosen, input_len=args.input_len,
                               tower_core_size=args.tower_core_size,
                               max_degree=args.max_degree if args.max_degree is not None else args.tower_core_size,
                               input_order=args.input_order, input_spectrum=args.input_spectrum)
                          for index, chosen in enumerate(build_grid())], indent=2))
        return
    try:
        run_args = experiment_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    print(f"Input-tower configuration {args.run_index + 1}/{len(build_grid())}: {build_grid()[args.run_index]}", flush=True)
    return experiment.main(run_args)


if __name__ == "__main__":
    main()
