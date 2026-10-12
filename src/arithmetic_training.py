"""Arithmetic training and adapters for the unchanged Boolean diagnostics.

The adapter's binary codes are private routing indices, never model tokens.
The actual model always receives the configured decimal/grouped selector and
digit arguments, followed by teacher-forced answer tokens. Every legacy loss
diagnostic therefore measures the SAME sequence cross entropy as training.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch.func import functional_call

from src.arithmetic import (
    PAD, ArithmeticSelector, ArithmeticTransformer, SequenceCrossEntropy,
    _positive_int, evaluate_fixed_set, evaluation_mode, make_evaluation_set,
    sample_batch, sample_payloads, sequence_metrics, teacher_tokens,
)
from src.functions import SelectorFunction
from src.metrics import gradient_metrics, multi_task_curvature, task_affinity_metrics
from src.spaces import loss_spaces


class _ProbeLabels:
    def __init__(self, name: str, targets: torch.Tensor):
        self.metric_name = name
        self.targets = targets

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        return self.targets[indices[:, 0].long()]


class DiagnosticAdapter(torch.nn.Module):
    """Route legacy task/payload-index inputs onto fixed arithmetic probe banks.

    Common uniform digit payloads are shared across all functions. If a payload
    violates a task's domain (division by zero), that task resamples it with a
    private seed. Each marginal remains uniform over its own valid domain.
    Probes are fixed across checkpoints; they are not held-out evaluation data.
    """

    def __init__(self, model: ArithmeticTransformer, selector: ArithmeticSelector,
                 max_output_tokens: int, num_inputs: int = 64, seed: int = 0):
        super().__init__()
        _positive_int("num_inputs", num_inputs)
        self.model = model
        self.selector = selector
        device = next(model.parameters()).device
        common = torch.randint(0, 10, (num_inputs, selector.payload_len),
                               generator=torch.Generator().manual_seed(seed))
        inputs, labels = [], []
        for index, function in enumerate(selector.functions):
            payloads = sample_payloads(function, num_inputs, selector.payload_len, initial=common,
                                      generator=torch.Generator().manual_seed(seed + 1))
            batch = selector.inputs_for(index, payloads).to(device)
            inputs.append(batch)
            labels.append(selector.targets(batch, max_output_tokens))
        self.register_buffer("probe_inputs", torch.stack(inputs), persistent=False)
        self.register_buffer("probe_targets", torch.stack(labels), persistent=False)
        self.function = SelectorFunction([
            _ProbeLabels(name, targets) for name, targets in zip(selector.names, self.probe_targets)
        ])
        self.register_buffer("payloads", torch.arange(num_inputs, device=device)[:, None], persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ids = self.function.function_ids(inputs)
        rows = inputs[:, self.function.selector_size].long()
        prompts = self.probe_inputs[ids, rows]
        targets = self.probe_targets[ids, rows]
        return self.model.teacher_logits(prompts, targets)


def average_direction_sharpness(adapter: DiagnosticAdapter, num_perturbations: int = 4,
                                rho: float = 0.02, seed: int = 0) -> dict[str, float]:
    """Gaussian parameter noise; mean squared change in token probabilities.

    Excludes learned positional embeddings as in the Boolean diagnostic. For
    each answer, average across vocabulary and non-PAD positions (including EOS),
    then average examples and tasks equally. Teacher-forced contexts stay fixed.
    Unlike discontinuous decoded strings, this is a smooth sequence-output metric.
    Uses a private CPU RNG and functional replacements, preserving training state.
    """
    _positive_int("num_perturbations", num_perturbations)
    if not 0 <= rho < float("inf"):
        raise ValueError("rho must be finite and nonnegative.")
    model = adapter.model
    generator = torch.Generator().manual_seed(seed)
    parameters = dict(model.named_parameters())
    perturbable = {name: value for name, value in parameters.items()
                   if value.requires_grad and not name.startswith("positional_encoding.")}
    inputs, targets = adapter.probe_inputs, adapter.probe_targets
    masks = targets != PAD
    totals = [0.] * len(inputs)
    with evaluation_mode(model), torch.no_grad():
        originals = [model.teacher_logits(x, y).softmax(-1) for x, y in zip(inputs, targets)]
        for _ in range(num_perturbations):
            noisy = {
                name: parameter + rho * torch.randn(parameter.shape, dtype=parameter.dtype,
                                                    generator=generator).to(parameter.device)
                for name, parameter in perturbable.items()
            }
            for index, (x, y, mask, baseline) in enumerate(zip(inputs, targets, masks, originals)):
                output = functional_call(model, noisy, (teacher_tokens(x, y), x.size(1)))[:, x.size(1):]
                change = (output.softmax(-1) - baseline).square().mean(-1)
                totals[index] += float(((change * mask).sum(-1) / mask.sum(-1)).mean())
    values = [total / num_perturbations for total in totals]
    return {"sharpness": sum(values) / len(values), **{
        f"sharpness/{name}": value for name, value in zip(adapter.selector.names, values)
    }}


@dataclass
class TrainingConfig:
    num_steps: int = 30_000
    batch_size: int = 320
    max_output_tokens: int = 8
    seed: int = 7
    eval_interval: Optional[int] = 10
    eval_num_inputs_per_function: int = 320
    eval_batch_size: int = 320
    eval_seed: int = 7
    gradient_interval: Optional[int] = 20
    gradient_num_inputs: int = 256
    gradient_seed: int = 7
    affinity_interval: Optional[int] = 100
    affinity_num_inputs: int = 64
    affinity_eval_batch_size: int = 256
    affinity_seed: int = 7
    curvature_interval: Optional[int] = 100
    curvature_num_points: int = 3
    sharpness_interval: Optional[int] = 100
    sharpness_num_inputs: int = 32
    sharpness_num_perturbations: int = 4
    sharpness_rho: float = 0.02
    sharpness_seed: int = 7
    sensitivity_num_inputs: int = 1024
    space_enabled: bool = False
    space_delta: float = 0.02
    space_epsilon: float = 0.002
    space_num_inputs: int = 64
    space_seed: int = 7
    space_num_directions: int = 100
    space_direction_seed: int = 7
    space_hvp_budget: int = 10_000
    space_intersection_cosine: float = 0.999

    def validate(self, selector: ArithmeticSelector, model: ArithmeticTransformer) -> None:
        for name in ("num_steps", "batch_size", "max_output_tokens", "eval_num_inputs_per_function",
                     "eval_batch_size", "gradient_num_inputs", "affinity_num_inputs",
                     "affinity_eval_batch_size", "curvature_num_points", "sharpness_num_inputs",
                     "sharpness_num_perturbations", "sensitivity_num_inputs", "space_num_inputs",
                     "space_num_directions", "space_hvp_budget"):
            _positive_int(name, getattr(self, name))
        for name in ("eval_interval", "gradient_interval", "affinity_interval",
                     "curvature_interval", "sharpness_interval"):
            if getattr(self, name) is not None:
                _positive_int(name, getattr(self, name))
        if self.max_output_tokens < selector.required_output_tokens:
            raise ValueError(f"max_output_tokens must be >= {selector.required_output_tokens} "
                             "to fit every answer and EOS; targets cannot be truncated.")
        if selector.prompt_len + self.max_output_tokens > model.positional_encoding.num_embeddings:
            raise ValueError("Model positional capacity is too small for prompts and answers.")
        if not 0 <= self.sharpness_rho < float("inf"):
            raise ValueError("sharpness_rho must be finite and nonnegative.")
        if not isinstance(self.space_enabled, bool):
            raise ValueError("space_enabled must be boolean.")
        if self.space_enabled and (not 0 < self.space_delta < float("inf") or
                                   not 0 <= self.space_epsilon < float("inf") or
                                   not 0 < self.space_intersection_cosine <= 1):
            raise ValueError("Invalid loss-space delta, epsilon or intersection cosine.")


@dataclass
class TrainingResult:
    losses: list[float]
    last_metrics: dict[str, float]


def _due(interval: Optional[int], step: int) -> bool:
    return interval is not None and step % interval == 0


def train(model: ArithmeticTransformer, selector: ArithmeticSelector,
          optimizer: torch.optim.Optimizer, config: Optional[TrainingConfig] = None, *,
          selector_probabilities: Optional[Sequence[float]] = None,
          balanced_selectors: bool = False, wandb_run: Any = None,
          output_dir: Optional[Path] = None, space_progress=None) -> TrainingResult:
    """Train with fixed evaluation and the existing scientific diagnostics.

    Diagnostics also run without W&B. Local metrics/checkpoints are optional;
    only scalar metrics are returned, while W&B also receives an affinity table.
    Training sampling, evaluation and every diagnostic have independent RNGs.
    """
    config = config or TrainingConfig()
    config.validate(selector, model)
    device = next(model.parameters()).device
    criterion = SequenceCrossEntropy()
    generator = torch.Generator().manual_seed(config.seed)
    held_out = make_evaluation_set(selector, config.max_output_tokens,
                                  config.eval_num_inputs_per_function, config.eval_seed, device
                                  ) if config.eval_interval is not None else None

    def probe(count, seed):
        return DiagnosticAdapter(model, selector, config.max_output_tokens, count, seed)

    gradient_probe = probe(config.gradient_num_inputs, config.gradient_seed) if any(
        interval is not None and interval <= config.num_steps
        for interval in (config.gradient_interval, config.curvature_interval)
    ) else None
    affinity_probe = probe(config.affinity_num_inputs, config.affinity_seed) if (
        config.affinity_interval is not None and config.affinity_interval <= config.num_steps) else None
    sharpness_probe = probe(config.sharpness_num_inputs, config.sharpness_seed) if (
        config.sharpness_interval is not None and config.sharpness_interval <= config.num_steps) else None
    space_probe = probe(config.space_num_inputs, config.space_seed) if config.space_enabled else None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity = {f"sensitivity/{name}": function.avg_sensitivity(
        selector.payload_len, config.sensitivity_num_inputs, config.seed)
        for name, function in zip(selector.names, selector.functions)}
    losses = []
    model.train()
    for step in range(1, config.num_steps + 1):
        before = None
        if _due(config.curvature_interval, step):
            before = {name: parameter.detach().clone() for name, parameter in gradient_probe.named_parameters()
                      if parameter.requires_grad}
        inputs, targets = sample_batch(
            selector, config.batch_size, config.max_output_tokens, device=device, generator=generator,
            selector_probabilities=selector_probabilities, balanced_selectors=balanced_selectors,
            held_out=held_out,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model.teacher_logits(inputs, targets)
        loss = criterion(logits, targets)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite arithmetic training loss at step {step}.")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            metrics = sequence_metrics(selector, inputs, logits.detach(), targets)
        losses.append(float(loss.detach()))
        if step == 1:
            metrics.update(sensitivity)
        if _due(config.eval_interval, step):
            metrics.update({f"eval/{key}": value for key, value in evaluate_fixed_set(
                model, selector, held_out, config.max_output_tokens, config.eval_batch_size).items()})
        if _due(config.gradient_interval, step):
            metrics.update(gradient_metrics(gradient_probe, gradient_probe.function,
                                           criterion, gradient_probe.payloads))
        if _due(config.affinity_interval, step):
            rates_by_id = {id(parameter): float(group["lr"]) for group in optimizer.param_groups
                           for parameter in group["params"]}
            rates = {name: rates_by_id.get(id(parameter), 0.)
                     for name, parameter in affinity_probe.named_parameters() if parameter.requires_grad}
            metrics.update(task_affinity_metrics(affinity_probe, affinity_probe.function, criterion,
                                                affinity_probe.payloads, learning_rate=rates,
                                                eval_batch_size=config.affinity_eval_batch_size))
        if before is not None:
            metrics.update(multi_task_curvature(gradient_probe, gradient_probe.function, criterion,
                                               gradient_probe.payloads, before,
                                               num_points=config.curvature_num_points))
        if _due(config.sharpness_interval, step):
            metrics.update(average_direction_sharpness(sharpness_probe, config.sharpness_num_perturbations,
                                                       config.sharpness_rho, config.sharpness_seed))
        if space_probe is not None and step == config.num_steps:
            space_metrics, directions = loss_spaces(
                space_probe, space_probe.function, criterion, space_probe.payloads,
                delta=config.space_delta, epsilon=config.space_epsilon,
                num_directions=config.space_num_directions, direction_seed=config.space_direction_seed,
                hvp_budget=config.space_hvp_budget, intersection_cosine=config.space_intersection_cosine,
                progress=space_progress,
            )
            metrics.update(space_metrics)
            directions["arithmetic_selector_codes"] = selector.codes
            directions["loss_definition"] = "mean_per_example_mean_token_cross_entropy_including_eos"
            if output_dir is not None:
                artifact_path = output_dir / f"spaces_step_{step}.pt"
                torch.save(directions, artifact_path)
                if wandb_run is not None and hasattr(wandb_run, "log_artifact"):
                    import wandb
                    artifact = wandb.Artifact(f"arithmetic-loss-spaces-{wandb_run.id}", type="loss-spaces")
                    artifact.add_file(str(artifact_path))
                    wandb_run.log_artifact(artifact, aliases=[f"step-{step}"])
        if output_dir is not None:
            with (output_dir / "metrics.jsonl").open("a") as stream:
                # Undefined gradient cosines are JSON null, not nonstandard NaN.
                record = {key: value if math.isfinite(value) else None
                          for key, value in metrics.items()}
                stream.write(json.dumps({"step": step, **record}, allow_nan=False) + "\n")
        if wandb_run is not None:
            logged = dict(metrics)
            if _due(config.affinity_interval, step):
                import wandb
                logged["affinity/matrix"] = wandb.Table(
                    columns=["source / target", *selector.names],
                    data=[[source, *[metrics[f"affinity/{source}_to_{target}"] for target in selector.names]]
                          for source in selector.names],
                )
            wandb_run.log(logged, step=step)
    return TrainingResult(losses, metrics)
