from typing import Any, Callable, Dict, List, Optional, Tuple

import torch as t

from src.functions import BoolFunction, SelectorFunction
from src.metrics import batch_metrics
from src.model import TransformerModel


BatchMetricFn = Callable[[t.Tensor, t.Tensor, t.Tensor], Dict[str, float]]


def sample_batch(
    function: BoolFunction,
    batch_size: int = 32,
    seq_len: int = 1024,
    min_seq_len: int = 16,
    device: t.device = t.device("cpu"),
    balanced_selectors: bool = False,
) -> Tuple[t.Tensor, t.Tensor]:
    """Sample binary sequences and labels for a boolean function.

    A batch has one sequence length, sampled uniformly between
    ``min_seq_len`` and ``seq_len``. When ``balanced_selectors`` is enabled for
    a ``SelectorFunction``, every selected function occurs at least once in
    the batch while the payload bits remain random.
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

    sampled_seq_len = int(
        t.randint(min_seq_len, seq_len + 1, (), device=device).item()
    )
    x = t.randint(
        0,
        2,
        (batch_size, sampled_seq_len),
        device=device,
        dtype=t.long,
    )
    if balanced_selectors and function.selector_size:
        function_ids = t.arange(batch_size, device=device) % len(function.functions)
        function_ids = function_ids[t.randperm(batch_size, device=device)]
        for bit in range(function.selector_size):
            shift = function.selector_size - bit - 1
            x[:, bit] = (function_ids // (2 ** shift)) % 2
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
) -> float:
    """Return the average loss over randomly sampled evaluation batches."""

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
    batch_metric_fn: Optional[BatchMetricFn] = None,
    balanced_selectors: bool = False,
) -> List[float]:
    """Train the model and optionally log per-batch metrics to W&B.

    ``wandb_run`` is optional so the training helper remains usable without
    W&B. When supplied, W&B automatically receives overall loss and accuracy.
    For a ``SelectorFunction``, it also receives a per-function loss and
    accuracy using each selected function's ``metric_name``.
    ``batch_metric_fn`` can add experiment-specific metrics to those defaults.
    """

    if num_steps < 1:
        raise ValueError("num_steps must be positive.")

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
            if batch_metric_fn is not None:
                metrics.update(batch_metric_fn(x, outputs.detach(), y))
            wandb_run.log(metrics, step=step)

    return losses
