"""Four natural-function runs with exact Fourier evaluation through degree 8.

    env/bin/python sweep_classic_fourier.py --prepare
    env/bin/python sweep_classic_fourier.py --list
    env/bin/python sweep_classic_fourier.py --run-index 0 --smoke-test --no-wandb --device cpu

Full natural outputs supply training labels. The Fourier cap limits evaluation
only. Model settings and 50,000 updates match the sparse length/degree sweep.
"""

import argparse
import json
from pathlib import Path

import test_fourier as experiment


TARGET_LENGTHS = (("classic_majority", 11), ("classic_weighted_majority", 13),
                  ("classic_tribes", 15), ("classic_mean", 32))
MODEL_SETTINGS = dict(embedding_size=128, num_attn_heads=8, batch_size=64, num_layers=1)
WANDB_PROJECT = "sparse functions fourier analysis"
TARGET_DIR = Path(__file__).resolve().parent / "classic_fourier_targets"


def build_grid():
    return [dict(function=name, input_len=length, max_degree=8, steps=50_000,
                 eval_inputs=min(1024, 2 ** length // 4), **MODEL_SETTINGS)
            for name, length in TARGET_LENGTHS]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Compute and verify all four exact spectra without training.")
    mode.add_argument("--list", action="store_true", help="Inspect the four runs without training or creating files.")
    mode.add_argument("--run-index", type=int, help="Function index (0..3).")
    parser.add_argument("--steps", type=experiment.positive_int, help="Default 50000; smoke test 2.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--function-seed", type=int, default=17, help="Permutation seed for weights 1..13.")
    parser.add_argument("--eval-seed", type=int, default=23)
    parser.add_argument("--classic-target-dir", type=Path, default=TARGET_DIR)
    parser.add_argument("--eval-inputs", type=experiment.positive_int)
    parser.add_argument("--eval-batch-size", type=experiment.positive_int, default=320)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--sharpness-interval", type=experiment.nonnegative_int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-group", default="classic-functions-fourier")
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
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


def prepare_targets(target_dir=TARGET_DIR, seed=17):
    from src.runtime import configure_cpu_environment
    cpus = configure_cpu_environment()
    import torch
    from src.classic_fourier import prepare_classic_function, spectrum_path

    torch.set_num_threads(cpus or 1)
    summaries = []
    for row in build_grid():
        function = prepare_classic_function(row["function"], row["input_len"], target_dir,
                                            max_degree=8, seed=seed)
        summaries.append(dict(function.metadata(), prepared_file=spectrum_path(target_dir, function.spec).name))
    manifest = dict(evaluation_max_degree=8, training_outputs="natural_0_to_1", targets=summaries)
    manifest_path = Path(target_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def experiment_arguments(args):
    grid = resolved_grid(args)
    if args.run_index is None or not 0 <= args.run_index < len(grid):
        raise ValueError(f"run-index must be between 0 and {len(grid) - 1}.")
    chosen = grid[args.run_index]
    name = (f"fourier-{chosen['function']}-n{chosen['input_len']}-evalk8-"
            f"e128-h8-b64-steps{chosen['steps']}-seed{args.seed}-fseed{args.function_seed}")
    argv = ["--run-name", name, "--wandb-project", args.wandb_project, "--wandb-group", args.wandb_group,
            "--wandb" if args.wandb else "--no-wandb"]
    for key, value in chosen.items():
        if key != "num_layers":
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    if args.smoke_test:
        argv.append("--smoke-test")
    for key in ("seed", "function_seed", "eval_seed", "classic_target_dir", "eval_batch_size", "lr",
                "weight_decay", "sharpness_interval", "device", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    return argv


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps([dict(run_index=i, **row) for i, row in enumerate(resolved_grid(args))], indent=2))
        return
    if args.prepare:
        manifest = prepare_targets(args.classic_target_dir, args.function_seed)
        for target in manifest["targets"]:
            print(f"{target['name']}: n={target['input_len']}, squared flip sensitivity="
                  f"{target['squared_flip_sensitivity']:.6f}, tail above 8="
                  f"{target['tail_mass_above_degree'][8]:.8f}, spectrum={target['prepared_file']}")
        return manifest
    try:
        run_args = experiment_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    print(f"Classic Fourier configuration {args.run_index + 1}/4: {resolved_grid(args)[args.run_index]}", flush=True)
    return experiment.main(run_args)


if __name__ == "__main__":
    main()
