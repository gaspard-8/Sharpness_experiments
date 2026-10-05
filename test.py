import argparse
import os

import torch
import wandb

from src.functions import (
    Isordered,
    Isrepeating,
    Majority_nm,
    MeanOfFunctions,
    Parity_n,
    SelectorFunction,
    Tribe_ws,
)
from src.model import TransformerModel
from src.training import sample_batch, train

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sharpness experiment and log to W&B.")
    parser.add_argument("--smoke-test", action="store_true", help="Run two small updates and all diagnostics.")
    args = parser.parse_args()
    seed = 7
    default_device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = torch.device(os.getenv("SHARPNESS_DEVICE", default_device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; check GPU allocation, image, and driver.")
    print(f"Device: {device}; PyTorch: {torch.__version__}; W&B: {wandb.__version__}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}; CUDA runtime: {torch.version.cuda}", flush=True)
    training_config = {
        "max_len": 24,
        "min_len": 24,
        "batch_size": 320,
        "num_steps": 10_000,
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
        "space_delta": 0.02,
        "space_epsilon": 0.01,
        "space_num_inputs": 64,
        "space_seed": seed,
        "space_num_directions": 16,
        "space_direction_seed": seed,
        "space_hvp_budget": 500,
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
    # Zero-based [n, m) windows within the 21-bit payload (24 minus 3 selectors).
    ordered = Isordered(n=0, m=10)
    repeating = Isrepeating(n=0, m=10)
    majority_first_10 = Majority_nm(n=0, m=10)
    tribe_3_4 = Tribe_ws(w=3, s=4)

    # The first two means reuse tasks that also appear on their own.
    average_majority_parity = MeanOfFunctions(
        [majority_first_10, Parity_n(n=5)], name="mean_majority_parity_5"
    )
    average_tribe_majority = MeanOfFunctions(
        [tribe_3_4, Majority_nm(n=10, m=21)], name="mean_tribe_majority_tail"
    )
    average_middle_parity = MeanOfFunctions(
        [Majority_nm(n=4, m=14), Parity_n(n=10)],
        name="mean_middle_majority_parity_10",
    )
    average_tribe_tail = MeanOfFunctions(
        [Tribe_ws(w=2, s=5), Majority_nm(n=11, m=21)],
        name="mean_tribe_2_5_majority_tail",
    )

    mix = SelectorFunction([
        ordered,
        repeating,
        majority_first_10,
        tribe_3_4,
        average_majority_parity,
        average_tribe_majority,
        average_middle_parity,
        average_tribe_tail,
    ])

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
    wandb_config = {
        "seed": seed,
        "smoke_test": args.smoke_test,
        "device": device.type,
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
            **training_config,
        )
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()
