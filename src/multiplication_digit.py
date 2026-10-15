"""Six-token multiplication digit classification and Fourier-aware evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call

from src.fourier import positive_integer
from src.fourier_training import evaluation_mode
from src.model import TransformerModel
from src.multiplication_fourier import prepare_evaluation as prepared_evaluation


def inputs_from_codes(codes):
    powers = codes.new_tensor([100000, 10000, 1000, 100, 10, 1])
    return (codes[:, None] // powers) % 10


def codes_from_inputs(inputs):
    return (inputs * inputs.new_tensor([100000, 10000, 1000, 100, 10, 1])).sum(dim=-1)


def hundreds_digit(inputs):
    if inputs.ndim != 2 or inputs.shape[1] != 6:
        raise ValueError("Multiplication requires exactly six input digits.")
    if inputs.is_floating_point() or bool(((inputs < 0) | (inputs > 9)).any()):
        raise ValueError("Multiplication inputs must be integer digits 0..9.")
    powers = inputs.new_tensor([100, 10, 1])
    left, right = (inputs[:, :3] * powers).sum(-1), (inputs[:, 3:] * powers).sum(-1)
    return ((left * right // 100) % 10).long()


class MultiplicationDigitTransformer(TransformerModel):
    """Existing concatenated embedding encoder, with a 10-class final-position head."""

    def __init__(self, d_model=512, num_attn_heads=8, num_layers=1):
        super().__init__(d_model, num_attn_heads, num_layers, vocab_size=10, dropout=0., max_len=6)
        self.linear_head = nn.Linear(d_model, 10)

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != 6:
            raise ValueError("The multiplication model takes exactly six tokens.")
        positions = self.positional_encoding(torch.arange(6, device=inputs.device))
        embeddings = torch.cat((self.embedding(inputs), positions.expand(inputs.shape[0], -1, -1)), dim=-1)
        return self.linear_head(self.encoder(embeddings)[:, -1])


@dataclass
class TrainingConfig:
    num_steps: int = 50_000
    batch_size: int = 64
    eval_num_inputs: int = 1024
    eval_batch_size: int = 128
    eval_seed: int = 23
    sharpness_interval: int | None = 100
    sharpness_num_inputs: int = 32
    sharpness_num_perturbations: int = 4
    sharpness_rho: float = .02
    sharpness_seed: int = 23

    def validate(self):
        for key in ("num_steps", "batch_size", "eval_num_inputs", "eval_batch_size",
                    "sharpness_num_inputs", "sharpness_num_perturbations"):
            positive_integer(key, getattr(self, key))
        if self.eval_num_inputs >= 500_000:
            raise ValueError("Evaluation must leave operand-pair orbits for training.")
        if self.sharpness_interval is not None:
            positive_integer("sharpness_interval", self.sharpness_interval)
        if not math.isfinite(self.sharpness_rho) or self.sharpness_rho < 0:
            raise ValueError("Sharpness rho must be finite and nonnegative.")


@dataclass
class Evaluation:
    inputs: torch.Tensor
    labels: torch.Tensor
    indicator_targets: torch.Tensor
    scalar_targets: torch.Tensor
    excluded_codes: torch.Tensor


def prepare_evaluation(target_dir, config, device):
    payload = prepared_evaluation(target_dir, config.eval_num_inputs, config.eval_seed)
    codes = torch.from_numpy(payload["codes"]).long().to(device)
    excluded = torch.zeros(1_000_000, dtype=torch.bool, device=device)
    excluded[codes] = True
    excluded[(codes % 1000) * 1000 + codes // 1000] = True
    return Evaluation(torch.from_numpy(payload["inputs"]).long().to(device),
                      torch.from_numpy(payload["labels"]).long(),
                      torch.from_numpy(payload["indicator_targets"]).double(),
                      torch.from_numpy(payload["scalar_targets"]).double(), excluded)


def sample_batch(batch_size, evaluation):
    device = evaluation.excluded_codes.device
    codes = torch.randint(0, 1_000_000, (batch_size,), device=device)
    for _ in range(1000):
        overlap = evaluation.excluded_codes[codes]
        count = int(overlap.sum())
        if not count:
            inputs = inputs_from_codes(codes)
            return inputs, hundreds_digit(inputs)
        codes[overlap] = torch.randint(0, 1_000_000, (count,), device=device)
    raise RuntimeError("Could not sample outside held-out multiplication pairs.")


def evaluate(model, evaluation, batch_size):
    with evaluation_mode(model), torch.no_grad():
        logits = torch.cat([model(inputs).detach().cpu().double() for inputs in evaluation.inputs.split(batch_size)])
        probabilities = logits.softmax(dim=-1)
        brier = (probabilities[:, None] - evaluation.indicator_targets).square().sum(-1).mean(0)
        expected_digit = probabilities @ torch.arange(10, dtype=torch.float64)
        scalar = (expected_digit[:, None] - evaluation.scalar_targets).square().mean(0)
        metrics = {"eval/loss": float(nn.functional.cross_entropy(logits, evaluation.labels)),
                   "eval/accuracy": float((logits.argmax(-1) == evaluation.labels).double().mean()),
                   "eval/brier": float(brier[-1]), "eval/scalar_mse": float(scalar[-1])}
        metrics.update({f"eval/loss_degree_{degree}": float(value) for degree, value in enumerate(brier, 1)})
        metrics.update({f"eval/scalar_loss_degree_{degree}": float(value) for degree, value in enumerate(scalar, 1)})
    return metrics


def output_sharpness(model, config, seed):
    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    inputs = torch.randint(0, 10, (config.sharpness_num_inputs, 6), generator=generator).to(device)
    parameters = dict(model.named_parameters())
    total = 0.
    with evaluation_mode(model), torch.no_grad():
        baseline = model(inputs).softmax(-1)
        for _ in range(config.sharpness_num_perturbations):
            perturbed = {name: value + config.sharpness_rho * torch.randn(value.shape, dtype=value.dtype,
                                                                         generator=generator).to(device)
                         for name, value in parameters.items()
                         if value.requires_grad and not name.startswith("positional_encoding.")}
            probabilities = functional_call(model, perturbed, (inputs,)).softmax(-1)
            total += float((probabilities - baseline).square().sum(-1).mean().cpu())
    return {"sharpness": total / config.sharpness_num_perturbations}


def train(model, optimizer, config, target_dir, output_dir, wandb_run=None, progress=None):
    config.validate()
    device = next(model.parameters()).device
    evaluation = prepare_evaluation(target_dir, config, device)
    torch.save({"inputs": evaluation.inputs.cpu(), "labels": evaluation.labels,
                "indicator_targets": evaluation.indicator_targets, "scalar_targets": evaluation.scalar_targets,
                "excludes_reversed_pairs": True}, Path(output_dir) / "evaluation.pt")
    model.train()
    with (Path(output_dir) / "metrics.jsonl").open("w") as stream:
        def record(step, metrics):
            stream.write(json.dumps(dict(step=step, **metrics), allow_nan=False) + "\n")
            stream.flush()
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)

        metrics = evaluate(model, evaluation, config.eval_batch_size)
        record(0, metrics)
        for step in range(1, config.num_steps + 1):
            inputs, labels = sample_batch(config.batch_size, evaluation)
            loss = nn.functional.cross_entropy(model(inputs), labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite multiplication training loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            metrics = {"loss": float(loss.detach().cpu()), **evaluate(model, evaluation, config.eval_batch_size)}
            if config.sharpness_interval is not None and step % config.sharpness_interval == 0:
                metrics.update(output_sharpness(model, config, config.sharpness_seed + step))
            record(step, metrics)
            if progress is not None and (step == 1 or step % 1000 == 0):
                progress({"stage": "training", "step": step, "eval_loss": metrics["eval/loss"],
                          "eval_accuracy": metrics["eval/accuracy"]})
    return metrics
