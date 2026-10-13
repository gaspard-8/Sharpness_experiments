from typing import Any, List, Optional, Sequence, Tuple, Union

import torch as t

from src.evaluation import EvaluationSet, evaluate_fixed_set, make_evaluation_set
from src.functions import BoolFunction, SelectorFunction
from src.metrics import (
    _unique_metric_names, average_direction_sharpness, batch_metrics,
    task_affinity_metrics,
)
from src.model import TransformerModel


SelectorProbabilities = Union[Sequence[float], t.Tensor]


def sample_batch(
    function: BoolFunction,
    batch_size: int = 32,
    seq_len: int = 1024,
    min_seq_len: int = 16,
    device: t.device = t.device("cpu"),
    balanced_selectors: bool = False,
    selector_probabilities: Optional[SelectorProbabilities] = None,
    held_out: Optional[EvaluationSet] = None,
) -> Tuple[t.Tensor, t.Tensor]:
    """Sample binary sequences and labels for a boolean function.

    A batch has one sequence length, sampled uniformly between
    ``min_seq_len`` and ``seq_len``. When ``balanced_selectors`` is enabled for
    a ``SelectorFunction``, every selected function occurs at least once in
    the batch. Alternatively, ``selector_probabilities`` samples function
    indices in ``function.functions`` order. It must contain one nonnegative
    probability per function, sum to 1, and cannot be combined with balanced
    sampling. Payload bits remain uniform random. When ``held_out`` is given,
    reject exact held-out inputs, resampling payloads while keeping selectors
    and the sampled sequence length unchanged.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if min_seq_len < 1 or min_seq_len > seq_len:
        raise ValueError("min_seq_len must be between 1 and seq_len.")
    if isinstance(function, SelectorFunction) and min_seq_len <= function.selector_size:
        raise ValueError("Selector inputs need at least one payload token.")
    if balanced_selectors and not isinstance(function, SelectorFunction):
        raise ValueError("balanced_selectors requires a SelectorFunction.")
    if balanced_selectors and batch_size < len(function.functions):
        raise ValueError("batch_size must cover every selected function.")
    if selector_probabilities is not None:
        if not isinstance(function, SelectorFunction):
            raise ValueError("selector_probabilities requires a SelectorFunction.")
        if balanced_selectors:
            raise ValueError("selector_probabilities and balanced_selectors cannot be combined.")
        probabilities = t.as_tensor(
            selector_probabilities, dtype=t.float32, device=device
        )
        if probabilities.ndim != 1 or probabilities.numel() != len(
            function.functions
        ):
            raise ValueError("selector_probabilities needs one value per selected function.")
        if not bool(t.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
            raise ValueError("selector_probabilities must be finite and nonnegative.")
        if not bool(t.isclose(
            probabilities.sum(), probabilities.new_tensor(1.0), atol=1e-6, rtol=0
        )):
            raise ValueError("selector_probabilities must sum to 1.")

    sampled_seq_len = int(
        t.randint(min_seq_len, seq_len + 1, (), device=device).item()
    )
    if selector_probabilities is not None or balanced_selectors:
        if selector_probabilities is not None:
            function_ids = t.multinomial(probabilities, batch_size, replacement=True)
        else:
            function_ids = t.arange(batch_size, device=device) % len(function.functions)
            function_ids = function_ids[t.randperm(batch_size, device=device)]

        payload = t.randint(
            0, 2, (batch_size, sampled_seq_len - function.selector_size),
            device=device, dtype=t.long,
        )
        if function.selector_size:
            shifts = t.arange(function.selector_size - 1, -1, -1, device=device)
            selectors = ((function_ids[:, None] >> shifts) & 1).long()
            x = t.cat((selectors, payload), dim=1)
        else:
            x = payload
    else:
        x = t.randint(
            0, 2, (batch_size, sampled_seq_len), device=device, dtype=t.long
        )
    if held_out is not None:
        selector_size = function.selector_size if isinstance(function, SelectorFunction) else 0
        for _ in range(1000):
            overlapping = held_out.overlapping_rows(x)
            if not overlapping:
                break
            x[overlapping, selector_size:] = t.randint(
                0, 2, (len(overlapping), sampled_seq_len - selector_size),
                device=device, dtype=t.long,
            )
        else:
            raise RuntimeError("Unable to sample training inputs outside the evaluation set; reduce its size.")
    return x, function(x)


def evaluate(
    model: TransformerModel,
    function: BoolFunction,
    criterion: t.nn.modules.loss._Loss,
    max_len: int = 1024,
    min_len: int = 16,
    batch_size: int = 32,
    num_steps: int = 100,
    device: t.device = t.device("cpu"),
    selector_probabilities: Optional[SelectorProbabilities] = None,
) -> float:
    """Return the average loss over sampled evaluation batches."""

    if num_steps < 1:
        raise ValueError("num_steps must be positive.")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    try:
        with t.no_grad():
            for _ in range(num_steps):
                x, y = sample_batch(
                    function,
                    batch_size=batch_size,
                    seq_len=max_len,
                    min_seq_len=min_len,
                    device=device,
                    selector_probabilities=selector_probabilities,
                )
                loss = criterion(model(x), y.float())
                total_loss += float(loss.detach().cpu())
    finally:
        model.train(was_training)

    return total_loss / num_steps


def train(
    model: TransformerModel,
    function: BoolFunction,
    criterion: t.nn.modules.loss._Loss,
    optimizer: t.optim.Optimizer,
    max_len: int = 1024,
    min_len: int = 16,
    batch_size: int = 32,
    num_steps: int = 1000,
    device: t.device = t.device("cpu"),
    wandb_run: Any = None,
    balanced_selectors: bool = False,
    selector_probabilities: Optional[SelectorProbabilities] = None,
    sharpness_interval: int = 100,
    sharpness_num_inputs: int = 32,
    sharpness_num_perturbations: int = 4,
    sharpness_rho: float = 0.02,
    eval_interval: Optional[int] = None,
    eval_num_inputs_per_function: int = 320,
    eval_batch_size: int = 320,
    eval_seed: int = 0,
    affinity_interval: Optional[int] = 100,
    affinity_num_inputs: int = 64,
    affinity_eval_batch_size: int = 256,
    affinity_seed: int = 0,
) -> List[float]:
    """Train the model and optionally log per-batch metrics to W&B.

    ``wandb_run`` is optional so the training helper remains usable without
    W&B. When supplied, W&B automatically receives overall loss and accuracy.
    For a ``SelectorFunction``, it also receives a per-function loss and
    accuracy using each selected function's ``metric_name``.
    ``selector_probabilities`` is passed to ``sample_batch`` for selector tasks.
    When logging, average direction sharpness is estimated every
    ``sharpness_interval`` optimizer updates using uniform input bits.
    Directed task affinities are measured every ``affinity_interval`` updates
    (default 100; None disables them), using TAG's relative loss reduction
    after an SGD lookahead step scaled by the optimizer's current learning
    rates. Source gradients and target losses share a fixed probe batch.
    Positive affinity means help, negative harm. AdamW moments and weight
    decay are not used in this diagnostic; training still uses its optimizer.
    W&B receives pairwise histories and a source-row/target-column matrix.
    ``affinity_num_inputs`` controls the probe size; target forwards are
    capped at ``affinity_eval_batch_size`` examples. These uniform probes are
    not reserved from training and do not certify held-out generalization.
    The same probe computes ``local_curvature``: the observed target loss
    change minus its first-order prediction, in loss units. Both matrices
    have source rows and target columns; only their pairwise entries log.
    Set ``eval_interval`` to reserve a fixed, balanced set at ``max_len`` and
    log post-update ``eval/loss`` and ``eval/accuracy`` (also per task).
    Reserved inputs are excluded from training even without a W&B logger.
    Evaluation uses ``no_grad()``, restores model modes, and runs only when
    logging. Its private seed does not consume training randomness.
    """

    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if sharpness_interval < 1:
        raise ValueError("sharpness_interval must be positive.")
    if sharpness_num_inputs < 1 or sharpness_num_perturbations < 1:
        raise ValueError("sharpness sample counts must be positive.")
    if not 0 <= sharpness_rho < float("inf"):
        raise ValueError("sharpness_rho must be finite and nonnegative.")
    if eval_interval is not None:
        if not isinstance(eval_interval, int) or isinstance(eval_interval, bool) or eval_interval < 1:
            raise ValueError("eval_interval must be a positive integer or None.")
        if not isinstance(eval_batch_size, int) or isinstance(eval_batch_size, bool) or eval_batch_size < 1:
            raise ValueError("eval_batch_size must be a positive integer.")
    if affinity_interval is not None:
        if not isinstance(affinity_interval, int) or isinstance(affinity_interval, bool) or affinity_interval < 1:
            raise ValueError("affinity_interval must be a positive integer or None.")
        for label, value in (("affinity_num_inputs", affinity_num_inputs),
                             ("affinity_eval_batch_size", affinity_eval_batch_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{label} must be a positive integer.")
    evaluation_set = None
    if eval_interval is not None:
        evaluation_set = make_evaluation_set(
            function, max_len, eval_num_inputs_per_function, eval_seed, device,
        )

    affinity_payloads = None
    if wandb_run is not None and affinity_interval is not None and num_steps >= affinity_interval:
        selector_size = function.selector_size if isinstance(function, SelectorFunction) else 0
        if max_len <= selector_size:
            raise ValueError("Affinity inputs need at least one payload token.")
        generator = t.Generator(device="cpu").manual_seed(affinity_seed)
        affinity_payloads = t.randint(
            0, 2, (affinity_num_inputs, max_len - selector_size), generator=generator,
        ).to(device)

    model.train()
    losses: List[float] = []
    for step in range(num_steps):
        x, y = sample_batch(
            function,
            batch_size=batch_size,
            seq_len=max_len,
            min_seq_len=min_len,
            device=device,
            balanced_selectors=balanced_selectors,
            selector_probabilities=selector_probabilities,
            held_out=evaluation_set,
        )
        outputs = model(x)
        loss = criterion(outputs, y.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if wandb_run is not None:
            metrics = batch_metrics(function, criterion, x, outputs.detach(), y)
            if evaluation_set is not None and (step + 1) % eval_interval == 0:
                metrics.update({
                    f"eval/{name}": value for name, value in evaluate_fixed_set(
                        model, function, criterion, evaluation_set, eval_batch_size,
                    ).items()
                })
            if affinity_payloads is not None and (step + 1) % affinity_interval == 0:
                optimizer_rates = {
                    id(parameter): float(group["lr"])
                    for group in optimizer.param_groups for parameter in group["params"]
                }
                affinity = task_affinity_metrics(
                    model, function, criterion, affinity_payloads,
                    learning_rate={name: optimizer_rates.get(id(parameter), 0.)
                                   for name, parameter in model.named_parameters() if parameter.requires_grad},
                    eval_batch_size=affinity_eval_batch_size,
                )
                metrics.update(affinity)
                import wandb

                functions = function.functions if isinstance(function, SelectorFunction) else [function]
                names = _unique_metric_names(functions)
                for kind in ("affinity", "local_curvature"):
                    metrics[f"{kind}/matrix"] = wandb.Table(
                        columns=["source / target", *names],
                        data=[[source, *[affinity[f"{kind}/{source}_to_{target}"]
                                         for target in names]] for source in names],
                    )
            if (step + 1) % sharpness_interval == 0:
                metrics.update(
                    average_direction_sharpness(
                        model,
                        function,
                        seq_len=max_len,
                        num_inputs=sharpness_num_inputs,
                        num_perturbations=sharpness_num_perturbations,
                        rho=sharpness_rho,
                        device=device,
                    )
                )
            wandb_run.log(metrics, step=step + 1)

    return losses
