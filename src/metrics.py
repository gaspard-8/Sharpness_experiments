"""Reusable metrics for boolean-function experiments."""

from collections import Counter, defaultdict
from typing import Dict, Iterable, Optional

from numpy.polynomial.legendre import leggauss
import torch as t
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.functions import BoolFunction, SelectorFunction


def _diagnostic_device(parameters: Iterable[t.Tensor]) -> t.device:
    """Keep diagnostic algebra on CUDA; use CPU for float64 on other backends."""
    device = next(iter(parameters)).device
    return device if device.type == "cuda" else t.device("cpu")


def _mean_loss(
    criterion: t.nn.modules.loss._Loss,
    outputs: t.Tensor,
    labels: t.Tensor,
) -> t.Tensor:
    """Return a differentiable loss averaged over the probe outputs."""

    loss = criterion(outputs, labels.to(outputs))
    if loss.ndim:
        return loss.mean()
    if getattr(criterion, "reduction", None) == "sum":
        return loss / outputs.numel()
    return loss


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


def gradient_metrics(
    model: t.nn.Module,
    function: BoolFunction,
    criterion: t.nn.modules.loss._Loss,
    payloads: t.Tensor,
    norm_epsilon: float = 1e-12,
) -> Dict[str, float]:
    """Measure task loss-gradient norms and pairwise cosine similarities.

    ``payloads`` is a shared batch of binary inputs without selector tokens.
    Each valid task gets its own selector prefix and a mean loss, independent
    of training selector probabilities. Gradients include all trainable
    parameters (including positional embeddings), with unused parameters
    contributing zeros. Cosines are NaN when either norm is <= norm_epsilon.

    Evaluation mode disables dropout. Parameter values, existing ``.grad``
    buffers and module training modes are preserved. Reuse the same payloads
    across calls to compare checkpoints on a fixed probe set.
    """

    if payloads.ndim != 2 or min(payloads.shape) < 1:
        raise ValueError("payloads must be a nonempty 2D batch.")
    if not 0 <= norm_epsilon < float("inf"):
        raise ValueError("norm_epsilon must be finite and nonnegative.")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Gradient metrics require trainable model parameters.")
    compute_device = _diagnostic_device(parameters)

    functions = function.functions if isinstance(function, SelectorFunction) else [function]
    names = _unique_metric_names(functions)
    selector_size = function.selector_size if isinstance(function, SelectorFunction) else 0
    shifts = t.arange(selector_size - 1, -1, -1, device=payloads.device)
    training_modes = [(module, module.training) for module in model.modules()]
    gradients = []
    model.eval()
    try:
        with t.enable_grad():
            for code, selected_function in enumerate(functions):
                if selector_size:
                    prefix = ((code >> shifts) & 1).expand(payloads.size(0), -1)
                    inputs = t.cat((prefix, payloads), dim=1)
                else:
                    inputs = payloads
                outputs = model(inputs)
                loss = _mean_loss(criterion, outputs, selected_function(payloads))

                task_gradients = (
                    t.autograd.grad(loss, parameters, allow_unused=True)
                    if loss.requires_grad else [None] * len(parameters)
                )
                # Flatten on the model device, then transfer once if necessary.
                # Cast only after transfer: MPS cannot represent float64.
                gradients.append(t.cat([
                    gradient.detach().reshape(-1)
                    if gradient is not None else t.zeros_like(parameter).reshape(-1)
                    for parameter, gradient in zip(parameters, task_gradients)
                ]).to(compute_device).double())
    finally:
        for module, was_training in training_modes:
            module.training = was_training

    gradients = t.stack(gradients)
    norms = t.linalg.vector_norm(gradients, dim=1)
    valid = norms > norm_epsilon
    normalized = gradients / t.where(valid, norms, t.ones_like(norms))[:, None]
    alignments = (normalized @ normalized.T).clamp(-1, 1).cpu()
    norms, valid = norms.cpu(), valid.cpu()
    metrics = {f"gradient_norm/{name}": float(norm) for name, norm in zip(names, norms)}
    for i, name_i in enumerate(names):
        for j in range(i + 1, len(names)):
            alignment = float("nan")
            if valid[i] and valid[j]:
                alignment = float(alignments[i, j])
            metrics[f"gradient_alignment/{name_i}_vs_{names[j]}"] = alignment
    return metrics


def multi_task_curvature(
    model: t.nn.Module,
    function: BoolFunction,
    criterion: t.nn.modules.loss._Loss,
    payloads: t.Tensor,
    parameters_before: Dict[str, t.Tensor],
    num_points: int = 3,
) -> Dict[str, float]:
    """Estimate Definition 3 of PCGrad across one actual optimizer step.

    H = integral_0^1 g.T Hessian(L)(theta + a*(theta_next-theta)) g da,
    where L is the SUM of mean losses of valid tasks and g = grad L(theta)
    stays fixed at the pre-step parameters. ``model`` contains theta_next;
    ``parameters_before`` must snapshot every trainable parameter at theta.
    The direction is not normalized or replaced with the optimizer update.

    Use Gauss-Legendre quadrature and exact autograd Hessian-vector products,
    without constructing a full Hessian. Task losses share fixed ``payloads``
    without selectors, and are not weighted by training probabilities. All
    trainable parameters, including positional embeddings, participate.
    Dropout is disabled; model values, .grad buffers, and modes are preserved.
    """

    if payloads.ndim != 2 or min(payloads.shape) < 1:
        raise ValueError("payloads must be a nonempty 2D batch.")
    if not isinstance(num_points, int) or isinstance(num_points, bool) or num_points < 1:
        raise ValueError("num_points must be a positive integer.")
    parameters = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not parameters:
        raise ValueError("Curvature requires trainable model parameters.")
    compute_device = _diagnostic_device(parameters.values())
    if parameters_before.keys() != parameters.keys():
        raise ValueError("parameters_before must contain every trainable parameter exactly once.")
    for name, parameter in parameters.items():
        previous = parameters_before[name]
        if previous.shape != parameter.shape or previous.device != parameter.device or previous.dtype != parameter.dtype:
            raise ValueError(f"Snapshot shape, device, or dtype mismatch for {name}.")

    functions = function.functions if isinstance(function, SelectorFunction) else [function]
    selector_size = function.selector_size if isinstance(function, SelectorFunction) else 0
    shifts = t.arange(selector_size - 1, -1, -1, device=payloads.device)
    batches = []
    for code, selected_function in enumerate(functions):
        prefix = ((code >> shifts) & 1).expand(payloads.size(0), -1)
        inputs = t.cat((prefix, payloads), dim=1) if selector_size else payloads
        batches.append((inputs, selected_function(payloads)))

    def path_parameters(fraction: float) -> Dict[str, t.Tensor]:
        return {
            name: t.lerp(parameters_before[name].detach(), parameter.detach(), fraction).requires_grad_()
            for name, parameter in parameters.items()
        }

    def loss_at(point: Dict[str, t.Tensor], inputs: t.Tensor, labels: t.Tensor) -> t.Tensor:
        return _mean_loss(criterion, functional_call(model, point, (inputs,)), labels)

    nodes, weights = leggauss(num_points)
    training_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    curvature = t.zeros((), dtype=t.float64, device=compute_device)
    try:
        # Fused attention backwards may not implement second derivatives.
        # Restrict this diagnostic to math attention and restore the backend
        # settings on exit, so ordinary training keeps its existing kernels.
        with t.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            start = path_parameters(0.0)
            direction = [t.zeros_like(parameter) for parameter in start.values()]
            for inputs, labels in batches:
                loss = loss_at(start, inputs, labels)
                if loss.requires_grad:
                    gradients = t.autograd.grad(loss, tuple(start.values()), allow_unused=True)
                    for total, gradient in zip(direction, gradients):
                        if gradient is not None:
                            total.add_(gradient.detach())
            reduction_direction = [vector.to(compute_device).double() for vector in direction]

            for node, weight in zip(nodes, weights):
                point = path_parameters(float((node + 1) / 2))
                # Hessians add across tasks. Differentiate one task at a time
                # to bound memory, always using the same TOTAL gradient g.
                for inputs, labels in batches:
                    loss = loss_at(point, inputs, labels)
                    if not loss.requires_grad:
                        continue
                    gradients = t.autograd.grad(
                        loss, tuple(point.values()), create_graph=True, allow_unused=True,
                    )
                    products = [
                        (gradient * vector).sum()
                        for gradient, vector in zip(gradients, direction)
                        if gradient is not None and gradient.requires_grad
                    ]
                    if not products:
                        continue
                    directional_derivative = sum(products)
                    hvp = t.autograd.grad(directional_derivative, tuple(point.values()), allow_unused=True)
                    # CUDA reductions stay on device. The CPU fallback transfers
                    # before casting so float64 is never used on MPS.
                    value = sum(
                        t.sum(product.detach().to(compute_device).double() * vector)
                        for product, vector in zip(hvp, reduction_direction) if product is not None
                    )
                    curvature += float(weight / 2) * value
    finally:
        for module, was_training in training_modes:
            module.training = was_training

    return {"curvature/multi_task": float(curvature.cpu())}


def average_direction_sharpness(
    model: t.nn.Module,
    function: BoolFunction,
    seq_len: int,
    num_inputs: int = 32,
    num_perturbations: int = 4,
    rho: float = 0.02,
    device: Optional[t.device] = None,
) -> Dict[str, float]:
    """Estimate E_x E_delta[(T(theta + delta, x) - T(theta, x))**2].

    Input bits are uniform at the fixed ``seq_len``. For a selector, payload
    bits are sampled separately for each selector code, so function-specific
    estimates exclude selector tokens. Their equally weighted mean gives the
    full-input-space estimate. Unused selector codes contribute as well.

    Each parameter coordinate receives independent N(0, rho**2) noise. As in
    Hahn and Rofin, learned positional encodings are excluded. The same input
    samples and perturbation draws are used for every selector code.
    """

    if num_inputs < 1 or num_perturbations < 1:
        raise ValueError("num_inputs and num_perturbations must be positive.")
    if not 0 <= rho < float("inf"):
        raise ValueError("rho must be finite and nonnegative.")

    selector_size = (
        function.selector_size if isinstance(function, SelectorFunction) else 0
    )
    if seq_len <= selector_size:
        raise ValueError("seq_len must include at least one payload token.")

    parameters = dict(model.named_parameters())
    if device is None:
        device = next(iter(parameters.values())).device if parameters else t.device("cpu")
    perturbable = {
        name: parameter for name, parameter in parameters.items()
        if not name.startswith("positional_encoding.")
    }
    num_codes = 1 << selector_size
    shifts = t.arange(selector_size - 1, -1, -1, device=device)
    inputs_by_code = []
    for code in range(num_codes):
        payload = t.randint(0, 2, (num_inputs, seq_len - selector_size), device=device)
        if selector_size:
            prefix = ((code >> shifts) & 1).expand(num_inputs, -1)
            inputs_by_code.append(t.cat((prefix, payload), dim=1))
        else:
            inputs_by_code.append(payload)

    was_training = model.training
    model.eval()
    try:
        with t.no_grad():
            originals = [model(inputs) for inputs in inputs_by_code]
            squared_changes = [t.zeros((), device=device) for _ in range(num_codes)]
            for _ in range(num_perturbations):
                noisy_parameters = {
                    name: parameter + rho * t.randn_like(parameter)
                    for name, parameter in perturbable.items()
                }
                for code, inputs in enumerate(inputs_by_code):
                    perturbed_output = functional_call(model, noisy_parameters, (inputs,))
                    squared_changes[code] += (
                        (perturbed_output - originals[code]).square().mean()
                    )
            code_values = [
                float((value / num_perturbations).cpu()) for value in squared_changes
            ]
    finally:
        model.train(was_training)

    if isinstance(function, SelectorFunction):
        names = _unique_metric_names(function.functions)
        metrics = {
            f"sharpness/{name}": code_values[index]
            for index, name in enumerate(names)
        }
        if len(names) < num_codes:
            metrics["sharpness/invalid_selectors"] = sum(
                code_values[len(names):]
            ) / (num_codes - len(names))
    else:
        metrics = {f"sharpness/{function.metric_name}": code_values[0]}
    metrics["sharpness"] = sum(code_values) / num_codes
    return metrics
