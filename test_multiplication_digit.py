"""Predict floor(a*b/100)%10 from six decimal tokens, including leading zeros.

    env/bin/python test_multiplication_digit.py --smoke-test --no-wandb --device cpu

Training uses 10-class CE; evaluation also scores categorical and numeric
Fourier approximations through coordinate-support degree 6.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import faulthandler
import json
import math
import os
from pathlib import Path
import shutil
from uuid import uuid4

from src.runtime import DiagnosticProgress, configure_cpu_environment


TARGET_DIR = Path(__file__).resolve().parent / "multiplication_fourier_targets"
WANDB_PROJECT = "multiplication digit fourier analysis"


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
    parser.add_argument("--d-model", type=positive_int, default=512)
    parser.add_argument("--num-layers", type=positive_int, default=1)
    parser.add_argument("--num-attn-heads", type=positive_int, default=8)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--eval-inputs", type=positive_int)
    parser.add_argument("--eval-batch-size", type=positive_int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-seed", type=int, default=23)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--sharpness-interval", type=nonnegative_int)
    parser.add_argument("--target-dir", type=Path, default=TARGET_DIR)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-group", default="multiplication-hundreds-fourier")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    return parser


def model_config(args):
    if args.d_model % 2 or args.d_model % args.num_attn_heads:
        raise ValueError("Model width must be even and divisible by the number of heads.")
    return dict(d_model=args.d_model, num_layers=args.num_layers, num_attn_heads=args.num_attn_heads)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cpus = configure_cpu_environment()
    import torch
    from src.multiplication_digit import MultiplicationDigitTransformer, TrainingConfig, train
    from src.multiplication_fourier import load_manifest

    if cpus is not None:
        torch.set_num_threads(cpus)
        torch.set_num_interop_threads(1)
    elif args.smoke_test:
        torch.set_num_threads(1)
    faulthandler.enable()
    device = torch.device(args.device or os.getenv("SHARPNESS_DEVICE", "cuda" if torch.cuda.is_available()
                                                 else "mps" if torch.backends.mps.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    try:
        architecture = model_config(args)
        if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
            raise ValueError("Learning rate must be positive and weight decay nonnegative, both finite.")
        config = TrainingConfig(eval_seed=args.eval_seed, sharpness_seed=args.eval_seed)
        if args.smoke_test:
            config.num_steps, config.batch_size, config.eval_num_inputs = 2, 4, 8
            config.sharpness_interval, config.sharpness_num_inputs, config.sharpness_num_perturbations = 1, 4, 1
        for value, field in ((args.steps, "num_steps"), (args.batch_size, "batch_size"),
                             (args.eval_inputs, "eval_num_inputs"), (args.eval_batch_size, "eval_batch_size")):
            if value is not None:
                setattr(config, field, value)
        if args.sharpness_interval is not None:
            config.sharpness_interval = args.sharpness_interval or None
        config.validate()
        spectrum = load_manifest(args.target_dir, verify=True)
    except ValueError as error:
        parser.error(str(error))
    torch.manual_seed(args.seed)
    model = MultiplicationDigitTransformer(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    results_root = Path(os.getenv("SHARPNESS_RESULTS_DIR", "results"))
    output_dir = args.output_dir or results_root / "multiplication_digit" / (
        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Choose a new or empty output directory to preserve previous results.")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or (f"multiplication-hundreds-l{args.num_layers}-d{args.d_model}-"
                                 f"h{args.num_attn_heads}-b{config.batch_size}-steps{config.num_steps}-seed{args.seed}")
    metadata = {
        "experiment": "multiplication_hundreds_digit_fourier", "definition": spectrum["definition"],
        "run_name": run_name, "wandb_project": args.wandb_project, "wandb_group": args.wandb_group,
        "seed": args.seed, "device": str(device), "smoke_test": args.smoke_test,
        "num_tasks": 1, "selector_size": 0, "input_len": 6, "input_vocab_size": 10, "output_classes": 10,
        "loss_name": "CrossEntropyLoss", "optimizer_name": "AdamW", "optimizer_lr": args.lr,
        "optimizer_weight_decay": args.weight_decay, "model_embedding_size": args.d_model // 2,
        "model_feedforward_width": args.d_model, "model_dropout": 0., "model_max_len": 6,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "eval_interval": 1, "evaluation_degree_cap": 6, "excludes_reversed_operand_pairs": True,
        "degree_loss_definition": "mean_examples_sum_classes_squared_error(probabilities, cumulative_onehot_target)",
        "scalar_degree_loss_definition": "MSE(expected_digit_from_probabilities, cumulative_numeric_digit_target)",
        "spectrum_sha256": spectrum["spectrum_sha256"],
        **{f"model_{key}": value for key, value in architecture.items()},
        **{f"train_{key}": value for key, value in asdict(config).items()},
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "fourier_manifest.json").write_text(json.dumps(spectrum, indent=2) + "\n")
    shutil.copy2(args.target_dir / spectrum["spectrum_file"], output_dir / "target_spectrum.npz")
    print(f"Device: {device}; results: {output_dir.resolve()}", flush=True)
    print(f"Target: floor(a*b/100)%10, operands 000..999, six tokens; model: {architecture}", flush=True)
    print(f"Parameters: {metadata['model_parameter_count']:,}; batch: {config.batch_size}; updates: {config.num_steps}", flush=True)
    progress = DiagnosticProgress(output_dir, device)
    run = None
    try:
        if args.wandb:
            import wandb
            run = wandb.init(entity=os.getenv("WANDB_ENTITY", "gaspardtomas-universit-t-des-saarlandes-saarland-university"),
                             project=args.wandb_project, name=run_name, group=args.wandb_group, config=metadata,
                             settings=wandb.Settings(_disable_stats=True))
        metrics = train(model, optimizer, config, args.target_dir, output_dir, wandb_run=run, progress=progress)
        torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                        model_config=architecture, config=metadata, completed_steps=config.num_steps), output_dir / "checkpoint.pt")
        print(f"Completed {config.num_steps} updates; held-out accuracy: {metrics['eval/accuracy']:.4f}", flush=True)
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
