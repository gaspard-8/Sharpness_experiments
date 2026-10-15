"""Single-target regression with cumulative degree/input losses at every update."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch.func import functional_call

from src.evaluation import EvaluationSet, make_evaluation_set
from src.fourier import FourierFunction, positive_integer
from src.training import sample_batch


@contextmanager
def evaluation_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


@dataclass
class FourierEvaluation:
    held_out: EvaluationSet
    cumulative_targets: torch.Tensor
    input_targets: torch.Tensor | None = None
    full_targets: torch.Tensor | None = None
    target_variance: float | None = None


def prepare_evaluation(function, num_inputs=1024, seed=23, device=torch.device("cpu")):
    held_out = make_evaluation_set(function, function.input_len, num_inputs, seed, device)
    # One small N x K target bank, computed once in CPU float64. The model
    # still runs on its configured device; no Fourier transform of it is needed.
    with torch.no_grad():
        targets = function.cumulative(held_out.inputs.cpu(), dtype=torch.float64)
        input_targets = None
        if getattr(function, "evaluate_input_tower", False):
            input_targets = function.input_cumulative(held_out.inputs.cpu(), dtype=torch.float64)
            # Both final cutoffs are the same target, including rounding.
            input_targets[:, -1] = targets[:, -1]
        full_targets = held_out.labels.cpu().double() if getattr(function, "separate_full_target", False) else None
        target_variance = function.variance if full_targets is not None else None
    return FourierEvaluation(held_out, targets, input_targets, full_targets, target_variance)


def evaluate_cumulative(model, evaluation, batch_size=320):
    """One set of batched model forwards supplies every degree/input loss.

    These are MSE(g(x), f_{<=k}(x)), NOT projected model losses or coefficient
    errors. For bandlimited targets the degree-K loss equals full-target loss.
    Natural targets may retain mass above the evaluated cutoffs. A perfectly learned
    full function has lower-degree loss equal to its omitted tail mass in
    expectation, so those losses need not converge to zero or be monotone.
    """
    positive_integer("eval_batch_size", batch_size)
    with evaluation_mode(model), torch.no_grad():
        predictions = torch.cat([model(inputs).detach().cpu().double()
                                 for inputs in evaluation.held_out.inputs.split(batch_size)])
        losses = (predictions[:, None] - evaluation.cumulative_targets).square().mean(dim=0)
    metrics = {f"eval/loss_degree_{degree}": float(loss)
               for degree, loss in enumerate(losses, start=1)}
    metrics["eval/loss"] = float(losses[-1])
    if evaluation.full_targets is not None:
        metrics["eval/loss"] = float((predictions - evaluation.full_targets).square().mean())
        if evaluation.target_variance is not None and evaluation.target_variance > 0:
            metrics["eval/loss_normalized"] = metrics["eval/loss"] / evaluation.target_variance
            metrics.update({f"eval/loss_degree_{degree}_normalized": float(loss) / evaluation.target_variance
                            for degree, loss in enumerate(losses, start=1)})
    if evaluation.input_targets is not None:
        input_losses = (predictions[:, None] - evaluation.input_targets).square().mean(dim=0)
        metrics.update({f"eval/loss_input_{count}": float(loss)
                        for count, loss in enumerate(input_losses, start=1)})
    return metrics


def output_sharpness(model, function, *, num_inputs=32, num_perturbations=4, rho=0.02, seed=23):
    """Same Gaussian output diagnostic as the Boolean run, with a private RNG."""
    generator = torch.Generator().manual_seed(seed)
    parameters = dict(model.named_parameters())
    device = next(model.parameters()).device
    inputs = torch.randint(0, 2, (num_inputs, function.input_len), generator=generator).to(device)
    total = 0.0
    with evaluation_mode(model), torch.no_grad():
        baseline = model(inputs)
        for _ in range(num_perturbations):
            noisy = {name: parameter + rho * torch.randn(parameter.shape, dtype=parameter.dtype,
                                                        generator=generator).to(device)
                     for name, parameter in parameters.items()
                     if parameter.requires_grad and not name.startswith("positional_encoding.")}
            predictions = functional_call(model, noisy, (inputs,))
            total += float((predictions - baseline).square().mean().cpu())
    return {"sharpness": total / num_perturbations}


@dataclass
class TrainingConfig:
    num_steps: int = 30_000
    batch_size: int = 64
    eval_interval: int = 1
    eval_num_inputs: int = 1024
    eval_batch_size: int = 320
    eval_seed: int = 23
    sharpness_interval: int | None = 100
    sharpness_num_inputs: int = 32
    sharpness_num_perturbations: int = 4
    sharpness_rho: float = 0.02
    sharpness_seed: int = 23

    def validate(self, function, model):
        for name in ("num_steps", "batch_size", "eval_num_inputs", "eval_batch_size",
                     "sharpness_num_inputs", "sharpness_num_perturbations"):
            positive_integer(name, getattr(self, name))
        if self.eval_interval != 1 or isinstance(self.eval_interval, bool):
            raise ValueError("The Fourier experiment evaluates at every step (eval_interval=1).")
        if self.sharpness_interval is not None:
            positive_integer("sharpness_interval", self.sharpness_interval)
        if not math.isfinite(self.sharpness_rho) or self.sharpness_rho < 0:
            raise ValueError("sharpness_rho must be finite and nonnegative.")
        if self.eval_num_inputs >= 2 ** function.input_len:
            raise ValueError("Evaluation must leave at least one input for training.")
        if len(model.encoder.layers) != 1:
            raise ValueError("The Fourier experiment requires exactly one transformer layer.")
        if model.positional_encoding.num_embeddings < function.input_len:
            raise ValueError("Model positional capacity is smaller than input_len.")


@dataclass
class TrainingResult:
    losses: list[float]
    last_metrics: dict[str, float]


def train(model, function: FourierFunction, optimizer, config: TrainingConfig, *,
          wandb_run=None, output_dir: Path | None = None):
    config.validate(function, model)
    device = next(model.parameters()).device
    evaluation = prepare_evaluation(function, config.eval_num_inputs, config.eval_seed, device)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluation_data = {"inputs": evaluation.held_out.inputs.cpu(),
                           "cumulative_targets": evaluation.cumulative_targets}
        if evaluation.full_targets is not None:
            evaluation_data.update(full_targets=evaluation.full_targets, target_variance=evaluation.target_variance)
        if evaluation.input_targets is not None:
            evaluation_data.update(input_targets=evaluation.input_targets,
                                   input_order=torch.tensor(function.input_order))
        torch.save(evaluation_data, output_dir / "evaluation.pt")
    path = output_dir / "metrics.jsonl" if output_dir is not None else None
    stream = path.open("w") if path is not None else None

    def record(step, metrics):
        if stream is not None:
            stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
            stream.flush()
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

    losses = []
    model.train()
    try:
        # Record the untrained baseline as well as every completed update.
        last_metrics = evaluate_cumulative(model, evaluation, config.eval_batch_size)
        record(0, last_metrics)
        for step in range(1, config.num_steps + 1):
            inputs, targets = sample_batch(function, config.batch_size,
                                          seq_len=function.input_len, min_seq_len=function.input_len,
                                          device=device, held_out=evaluation.held_out)
            predictions = model(inputs)
            loss = torch.nn.functional.mse_loss(predictions, targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite training loss at step {step}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            last_metrics = {"loss": losses[-1],
                            **evaluate_cumulative(model, evaluation, config.eval_batch_size)}
            if config.sharpness_interval is not None and step % config.sharpness_interval == 0:
                last_metrics.update(output_sharpness(
                    model, function, num_inputs=config.sharpness_num_inputs,
                    num_perturbations=config.sharpness_num_perturbations,
                    rho=config.sharpness_rho, seed=config.sharpness_seed + step,
                ))
            record(step, last_metrics)
    finally:
        if stream is not None:
            stream.close()
    return TrainingResult(losses, last_metrics)
