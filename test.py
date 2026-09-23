import os

import torch
import wandb

from src.functions import First, Majority, Majority_n, Mean, Parity, Parity_n, SelectorFunction
from src.model import TransformerModel
from src.training import sample_batch, train





def main() -> None:
    seed = 7
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    training_config = {
        "max_len": 16,
        "min_len": 16,
        "batch_size": 320,
        "num_steps": 5_000,
        "balanced_selectors": True,
    }
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
    parity = Parity()
    majority = Majority()
    parity_5 = Parity_n(n=5)
    first = First()
    parity_10 = Parity_n(n=10)
    mean = Mean()
    majority_5 = Majority_n(n=5)
    majority_10 = Majority_n(n=10)
    mix = SelectorFunction([first, Parity_n(n=5),parity_10, majority_10])

    model = TransformerModel(**model_config).to(device)
    criterion = torch.nn.MSELoss(reduction=loss_config["reduction"])
    inputs, labels = sample_batch(
        mix,
        batch_size=len(mix.functions),
        seq_len=training_config["max_len"],
        min_seq_len=training_config["min_len"],
        device=device,
        balanced_selectors=True,
    )
    with torch.no_grad():
        outputs = model(inputs)
    assert outputs.shape == labels.shape
    assert torch.isfinite(criterion(outputs, labels.float()))

    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    wandb_config = {
        "seed": seed,
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
    finally:
        run.finish()


if __name__ == "__main__":
    main()
