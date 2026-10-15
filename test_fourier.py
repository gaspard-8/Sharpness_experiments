"""Train ONE Fourier target with a ONE-layer transformer on binary inputs.

    env/bin/python test_fourier.py --function sparse_powerlaw --smoke-test --no-wandb --device cpu

Supervision is always the full real-valued function. After every update,
one held-out model pass scores every cumulative degree approximation and,
for input-tower targets, each conditional input approximation.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import faulthandler
import json
import math
import os
from pathlib import Path
from uuid import uuid4

from src.runtime import DiagnosticProgress, configure_cpu_environment


FUNCTION_NAMES = ("concentrated_powerlaw", "concentrated_geometric",
                  "sparse_powerlaw", "sparse_geometric")
INPUT_TOWER_FUNCTION_NAMES = ("input_tower_increasing", "input_tower_random", "input_tower_top_heavy")
CLASSIC_FUNCTION_NAMES = ("classic_majority", "classic_weighted_majority", "classic_tribes", "classic_mean")


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer.")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--function", choices=(*FUNCTION_NAMES, *INPUT_TOWER_FUNCTION_NAMES, *CLASSIC_FUNCTION_NAMES), default=FUNCTION_NAMES[0])
    parser.add_argument("--input-len", type=positive_int, default=32, help="Binary payload length.")
    parser.add_argument("--max-degree", type=positive_int,
                        help="Fourier cutoff; classic evaluation defaults to/caps at 8, other targets default to input/core size.")
    parser.add_argument("--sparse-terms", type=positive_int, default=16, help="Max subsets per degree, or per stage/degree for input towers.")
    parser.add_argument("--power", type=float, default=2.0, help="Power-law mass exponent: mass_d proportional to d^(-power).")
    parser.add_argument("--geometric-decay", type=float, default=0.9, help="Geometric mass ratio between consecutive degrees.")
    parser.add_argument("--tower-core-size", type=positive_int, default=8,
                        help="Initial input core for input-tower functions; default max-degree equals this size.")
    parser.add_argument("--input-spectrum", choices=("powerlaw", "geometric"), default="powerlaw")
    parser.add_argument("--input-power", type=float, default=2.0)
    parser.add_argument("--input-decay", type=float, default=0.9)
    parser.add_argument("--input-order", choices=("prefix", "permuted"), default="prefix")
    parser.add_argument("--top-mass", type=float, default=0.9, help="Mass in top degree band for top-heavy input-tower target.")
    parser.add_argument("--top-band", type=positive_int, help="Number of top degrees; default ceil(max-degree / 4).")
    parser.add_argument("--function-seed", type=int, default=17, help="Fixed sparse support/sign seed, independent of model seed.")
    parser.add_argument("--classic-target-dir", type=Path, default=Path(__file__).resolve().parent / "classic_fourier_targets",
                        help="Prepared exact classic spectra; create with sweep_classic_fourier.py --prepare.")
    parser.add_argument("--seed", type=int, default=7, help="Model initialization and training sampling seed.")
    parser.add_argument("--eval-seed", type=int, default=23, help="Same held-out inputs across targets and architectures.")
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--embedding-size", type=positive_int, default=128,
                        help="Size of EACH token/position embedding; d_model is twice this value.")
    parser.add_argument("--num-attn-heads", type=positive_int, default=8)
    parser.add_argument("--eval-inputs", type=positive_int, help="Held-out examples; default 1024, smoke test 8.")
    parser.add_argument("--eval-batch-size", type=positive_int, default=320)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--sharpness-interval", type=nonnegative_int,
                        help="Output sharpness every N steps (default 100; smoke test 1); 0 disables it.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--wandb-project", help="W&B project; overrides WANDB_PROJECT.")
    parser.add_argument("--wandb-group")
    return parser


def build_model_config(args):
    if args.input_len < 1:
        raise ValueError("input-len must be positive.")
    d_model = 2 * args.embedding_size
    if d_model % args.num_attn_heads:
        raise ValueError("Twice embedding-size must be divisible by num-attn-heads.")
    return dict(d_model=d_model, num_attn_heads=args.num_attn_heads, num_layers=1,
                vocab_size=2, dropout=0.0, max_len=args.input_len)


def build_function(args):
    if args.function in CLASSIC_FUNCTION_NAMES:
        from src.classic_fourier import make_classic_function
        return make_classic_function(args.function, args.input_len,
                                     args.max_degree if args.max_degree is not None else 8,
                                     seed=args.function_seed, target_dir=args.classic_target_dir)
    if args.function in INPUT_TOWER_FUNCTION_NAMES:
        from src.input_tower import make_input_tower
        return make_input_tower(
            args.function, input_len=args.input_len, max_degree=args.max_degree,
            core_size=args.tower_core_size, terms_per_stage_degree=args.sparse_terms,
            input_spectrum=args.input_spectrum, input_power=args.input_power, input_decay=args.input_decay,
            degree_power=args.power, top_mass=args.top_mass, top_band=args.top_band,
            input_order=args.input_order, seed=args.function_seed,
        )
    from src.fourier import make_function
    return make_function(args.function, input_len=args.input_len, max_degree=args.max_degree,
                         sparse_terms=args.sparse_terms, power=args.power,
                         geometric_decay=args.geometric_decay, seed=args.function_seed)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cluster_cpus = configure_cpu_environment()
    import torch
    from src.fourier_training import TrainingConfig, train
    from src.model import TransformerModel

    if cluster_cpus is not None:
        torch.set_num_threads(cluster_cpus)
        torch.set_num_interop_threads(1)
    elif args.smoke_test:
        torch.set_num_threads(1)
    faulthandler.enable()
    default_device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(args.device or os.getenv("SHARPNESS_DEVICE", default_device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    try:
        model_config = build_model_config(args)
        if not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError("lr must be finite and positive.")
        if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
            raise ValueError("weight-decay must be finite and nonnegative.")
        function = build_function(args)
        config = TrainingConfig(eval_seed=args.eval_seed, sharpness_seed=args.eval_seed)
        if args.smoke_test:
            config.num_steps = 2
            config.batch_size = 16
            config.eval_num_inputs = 8
            config.sharpness_interval = 1
            config.sharpness_num_inputs = 4
            config.sharpness_num_perturbations = 1
        for value, field in ((args.steps, "num_steps"), (args.batch_size, "batch_size"),
                             (args.eval_inputs, "eval_num_inputs"), (args.eval_batch_size, "eval_batch_size")):
            if value is not None:
                setattr(config, field, value)
        if args.sharpness_interval is not None:
            config.sharpness_interval = args.sharpness_interval or None
        torch.manual_seed(args.seed)
        model = TransformerModel(**model_config).to(device)
        config.validate(function, model)
    except ValueError as error:
        parser.error(str(error))
    optimizer_config = {"lr": args.lr, "weight_decay": args.weight_decay}
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    results_root = Path(os.getenv("SHARPNESS_RESULTS_DIR", "results"))
    is_input_tower = args.function in INPUT_TOWER_FUNCTION_NAMES
    is_classic = args.function in CLASSIC_FUNCTION_NAMES
    output_dir = args.output_dir or results_root / ("classic_fourier" if is_classic else "input_tower" if is_input_tower else "fourier") / (
        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Choose a new or empty output directory to preserve previous results.")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or (
        f"fourier-{function.metric_name}-n{function.input_len}-k{function.max_degree}-"
        f"e{args.embedding_size}-h{args.num_attn_heads}-b{config.batch_size}-seed{args.seed}")
    wandb_project = args.wandb_project or os.getenv(
        "WANDB_PROJECT", "input tower Fourier sharpness" if is_input_tower else "single task Fourier sharpness")
    metadata = {
        "experiment": "single_task_classic_fourier" if is_classic else "single_task_input_tower" if is_input_tower else "single_task_fourier", "seed": args.seed, "device": str(device),
        "smoke_test": args.smoke_test, "run_name": run_name, "wandb_group": args.wandb_group,
        "wandb_project": wandb_project,
        "num_tasks": 1, "selector_size": 0, "function": function.metadata(),
        "evaluation": ("MSE(model(x), cumulative_degree_target_k) and MSE(model(x), conditional_input_target_k), post_update_every_step"
                       if is_input_tower else "MSE(model(x), full_natural_function) and MSE(model(x), constant_plus_degrees_through_k), post_update_every_step"
                       if is_classic else "MSE(model(x), cumulative_target_through_degree_k), post_update_every_step"),
        "epsilon_definition": ("squared_uniform_L2_omitted_mass_above_degree_cutoff_or_outside_input_order_prefix"
                               if is_input_tower else "squared_uniform_L2_tail_mass_above_degree_k"),
        "loss_name": "MSELoss", "loss_reduction": "mean", "optimizer_name": "AdamW",
        "model_embedding_size": args.embedding_size,
        **{f"model_{key}": value for key, value in model_config.items()},
        **{f"optimizer_{key}": value for key, value in optimizer_config.items()},
        **{f"train_{key}": value for key, value in asdict(config).items()},
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "target.json").write_text(json.dumps(function.metadata(), indent=2) + "\n")
    if is_classic:
        function.save_spectrum(output_dir / "target_spectrum.npz")
    print(f"Device: {device}; results: {output_dir.resolve()}", flush=True)
    degree_label = "evaluated degrees" if is_classic else "degrees"
    print(f"One target: {function.metric_name}; {function.input_len} input bits; {degree_label} 1..{function.max_degree}", flush=True)
    if is_input_tower:
        scope = function.metadata()["degree_profile_scope"]
        print(f"Input tower: core {function.core_size}, then one additional input per stage; degree profile scope: {scope}", flush=True)
    print(f"Model: {model_config}; batch size: {config.batch_size}; steps: {config.num_steps}", flush=True)
    progress = DiagnosticProgress(output_dir, device)
    run = None
    try:
        if args.wandb:
            import wandb
            run = wandb.init(
                entity=os.getenv("WANDB_ENTITY", "gaspardtomas-universit-t-des-saarlandes-saarland-university"),
                project=wandb_project,
                config=metadata, name=run_name, group=args.wandb_group,
                settings=wandb.Settings(_disable_stats=True),
            )
        result = train(model, function, optimizer, config, wandb_run=run, output_dir=output_dir)
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model_config, "config": metadata, "completed_steps": len(result.losses)},
                   output_dir / "checkpoint.pt")
        print(f"Completed {len(result.losses)} updates; held-out full-target MSE: {result.last_metrics['eval/loss']:.6f}", flush=True)
        progress({"stage": "run_complete"})
    except BaseException as error:
        progress({"stage": "failed", "error_type": type(error).__name__, "error": str(error)})
        if run is not None:
            run.finish(exit_code=1)
        raise
    else:
        if run is not None:
            run.finish()
    return output_dir


if __name__ == "__main__":
    main()
