import argparse
import faulthandler
import os
import traceback
from pathlib import Path

from src.runtime import DiagnosticProgress, configure_cpu_environment

# Apply environment limits before torch/numpy/W&B load their thread pools.
cluster_cpus = configure_cpu_environment()

import torch
import wandb

if cluster_cpus is not None:
    torch.set_num_threads(cluster_cpus)
    torch.set_num_interop_threads(1)

from src.functions import SelectorFunction, Tribe_ws
from src.model import TransformerModel
from src.training import sample_batch, train

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sharpness experiment and log to W&B.")
    parser.add_argument("--smoke-test", action="store_true", help="Run two small updates and the enabled diagnostics.")
    parser.add_argument("--loss-spaces", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable the final tangent/sharpness spaces and overlaps (disabled by default).")
    args = parser.parse_args()
    faulthandler.enable()
    seed = 7
    default_device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = torch.device(os.getenv("SHARPNESS_DEVICE", default_device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; check GPU allocation, image, and driver.")
    print(f"Device: {device}; PyTorch: {torch.__version__}; W&B: {wandb.__version__}", flush=True)
    print(f"CPU threads: {torch.get_num_threads()}; interop threads: {torch.get_num_interop_threads()}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}; CUDA runtime: {torch.version.cuda}", flush=True)
    training_config = {
        "max_len": 20,
        "min_len": 20,
        "batch_size": 320,
        "num_steps": 30_000,
        "selector_probabilities": [0.125] * 8,
        "eval_interval": 10,
        "eval_num_inputs_per_function": 320,
        "eval_batch_size": 320,
        "eval_seed": seed,
        "sharpness_interval": 100,
        "sharpness_num_inputs": 32,
        "sharpness_num_perturbations": 4,
        "sharpness_rho": 0.02,
        "gradient_interval": 20,
        "gradient_num_inputs": 256,
        "gradient_seed": seed,
        "curvature_interval": 100,
        "curvature_num_points": 3,
        "space_enabled": args.loss_spaces,
        "space_delta": 0.02,
        "space_epsilon": 0.002,
        "space_num_inputs": 64,
        "space_seed": seed,
        "space_num_directions": 100,
        "space_direction_seed": seed,
        "space_hvp_budget": 10_000,
        "space_intersection_cosine": 0.999,
    }
    if args.smoke_test:
        training_config.update(
            num_steps=2, batch_size=16, eval_interval=1,
            eval_num_inputs_per_function=4, eval_batch_size=16,
            sharpness_interval=1, sharpness_num_inputs=4, sharpness_num_perturbations=1,
            gradient_interval=1, gradient_num_inputs=4,
            curvature_interval=1, curvature_num_points=1,
            space_num_inputs=4, space_num_directions=1,
            space_hvp_budget=4,
        )
    print(f"Final loss-space diagnostics: {'enabled' if training_config['space_enabled'] else 'disabled'}", flush=True)
    model_config = {
        "d_model": 64,
        "num_attn_heads": 4,
        "num_layers": 4,
        "vocab_size": 2,
        "dropout": 0.0,
        "max_len": training_config["max_len"],
    }
    optimizer_config = {"lr": 1e-4, "weight_decay": 1e-2}
    loss_config = {"name": "MSELoss", "reduction": "mean"}

    torch.manual_seed(seed)
    # Split the 21-bit payload (24 minus 3 selector bits) into 1 through 8 tribes.
    # Each tribe has width floor(21 / s); leftover trailing bits are ignored.
    tasks = [
        Tribe_ws(w=21, s=1),
        Tribe_ws(w=10, s=2),
        Tribe_ws(w=7, s=3),
        Tribe_ws(w=5, s=4),
        Tribe_ws(w=4, s=5),
        Tribe_ws(w=3, s=6),
        Tribe_ws(w=3, s=7),
        Tribe_ws(w=2, s=8),
    ]
    mix = SelectorFunction(tasks)

    model = TransformerModel(**model_config).to(device)
    criterion = torch.nn.MSELoss(reduction=loss_config["reduction"])
    inputs, labels = sample_batch(
        mix,
        batch_size=len(mix.functions),
        seq_len=training_config["max_len"],
        min_seq_len=training_config["min_len"],
        device=device,
        selector_probabilities=training_config["selector_probabilities"],
    )
    with torch.no_grad():
        outputs = model(inputs)
    assert outputs.shape == labels.shape
    assert torch.isfinite(criterion(outputs, labels.float()))

    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    results_directory = os.getenv("SHARPNESS_RESULTS_DIR")
    progress = DiagnosticProgress(Path(results_directory) if results_directory else None, device)
    wandb_config = {
        "seed": seed,
        "smoke_test": args.smoke_test,
        "device": device.type,
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "function_type": type(mix).__name__,
        "selector_size": mix.selector_size,
        "selector_functions": [function.metric_name for function in mix.functions],
        "loss_name": loss_config["name"],
        "loss_reduction": loss_config["reduction"],
        "optimizer_name": type(optimizer).__name__,
        **{f"model_{key}": value for key, value in model_config.items()},
        **{f"optimizer_{key}": value for key, value in optimizer_config.items()},
        **{f"train_{key}": value for key, value in training_config.items()},
    }
    run = wandb.init(
        entity=os.getenv(
            "WANDB_ENTITY",
            "gaspardtomas-universit-t-des-saarlandes-saarland-university",
        ),
        project=os.getenv("WANDB_PROJECT", "multiple tasks sharpness"),
        config=wandb_config,
        settings=wandb.Settings(_disable_stats=True),
    )
    print(f"W&B run: {run.url or run.id}", flush=True)

    try:
        train(
            model=model,
            function=mix,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            wandb_run=run,
            space_progress=progress,
            **training_config,
        )
    except BaseException as error:
        # Print the original failure before W&B shutdown can block or fail.
        traceback.print_exc()
        progress({"stage": "failed", "error_type": type(error).__name__, "error": str(error)})
        run.finish(exit_code=1)
        raise
    else:
        progress({"stage": "wandb_finish_start"})
        run.finish()
        progress({"stage": "run_complete"})


if __name__ == "__main__":
    main()
