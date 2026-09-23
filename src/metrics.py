"""Reusable loss and accuracy metrics for boolean-function experiments."""

from collections import Counter, defaultdict
from typing import Dict, Iterable

import torch as t

from src.functions import BoolFunction, SelectorFunction


def _loss_value(
    criterion: t.nn.modules.loss._Loss,
    outputs: t.Tensor,
    labels: t.Tensor,
) -> float:
    """Evaluate the configured loss and turn any unreduced result into a scalar."""

    loss = criterion(outputs, labels.float())
    if loss.ndim:
        loss = loss.mean()
    return float(loss.detach().cpu())


def _unique_metric_names(functions: Iterable[BoolFunction]) -> list[str]:
    """Return readable, collision-free names for a selector's functions."""

    functions = list(functions)
    counts = Counter(function.metric_name for function in functions)
    occurrences: defaultdict[str, int] = defaultdict(int)
    names: list[str] = []
    for function in functions:
        base_name = function.metric_name
        if counts[base_name] == 1:
            names.append(base_name)
            continue
        occurrence = occurrences[base_name]
        names.append(f"{base_name}_{occurrence}")
        occurrences[base_name] += 1
    return names


def _function_metrics(
    function: BoolFunction,
    criterion: t.nn.modules.loss._Loss,
    inputs: t.Tensor,
    outputs: t.Tensor,
    labels: t.Tensor,
) -> Dict[str, float]:
    """Return the loss and, when defined, accuracy for one function."""

    metrics = {"loss": _loss_value(criterion, outputs, labels)}
    correct = function.accuracy_mask(outputs, labels, inputs)
    if correct is not None:
        metrics["accuracy"] = float(correct.float().mean().detach().cpu())
    return metrics


def batch_metrics(
    function: BoolFunction,
    criterion: t.nn.modules.loss._Loss,
    inputs: t.Tensor,
    outputs: t.Tensor,
    labels: t.Tensor,
) -> Dict[str, float]:
    """Compute named loss and accuracy metrics for a sampled batch.

    A regular function produces ``loss`` and, if it defines accuracy,
    ``accuracy``. A ``SelectorFunction`` additionally produces
    ``loss/<function-name>`` and ``accuracy/<function-name>`` for every
    selected function represented in the batch. Its aggregate accuracy is
    sample-weighted across the selected functions that define an accuracy.
    """

    if not isinstance(function, SelectorFunction):
        return _function_metrics(function, criterion, inputs, outputs, labels)

    metrics: Dict[str, float] = {"loss": _loss_value(criterion, outputs, labels)}
    function_ids = function.function_ids(inputs)
    payloads = inputs[:, function.selector_size :]
    total_correct = 0
    total_accuracy_examples = 0

    for function_id, (selected_function, metric_name) in enumerate(
        zip(function.functions, _unique_metric_names(function.functions))
    ):
        mask = function_ids == function_id
        if not mask.any():
            continue

        selected_metrics = _function_metrics(
            selected_function,
            criterion,
            payloads[mask],
            outputs[mask],
            labels[mask],
        )
        metrics[f"loss/{metric_name}"] = selected_metrics["loss"]

        correct = selected_function.accuracy_mask(
            outputs[mask], labels[mask], payloads[mask]
        )
        if correct is not None:
            correct_count = int(correct.sum().detach().cpu())
            example_count = correct.numel()
            metrics[f"accuracy/{metric_name}"] = correct_count / example_count
            total_correct += correct_count
            total_accuracy_examples += example_count

    invalid_mask = function_ids >= len(function.functions)
    if invalid_mask.any():
        metrics["selectors/invalid_rate"] = float(
            invalid_mask.float().mean().detach().cpu()
        )

    if total_accuracy_examples:
        metrics["accuracy"] = total_correct / total_accuracy_examples
    return metrics
