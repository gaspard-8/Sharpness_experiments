"""Task-loss directions from matrix-free, thick-restart block Lanczos."""

from __future__ import annotations

from typing import Any, Callable

import torch as t
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.functions import BoolFunction, SelectorFunction
from src.metrics import _diagnostic_device, _mean_loss, _unique_metric_names
from src.subspaces import pairwise_overlaps


def _flatten(parts: tuple[t.Tensor | None, ...], parameters: list[t.Tensor]) -> t.Tensor:
    # Join on the model device first: CUDA needs no transfer, and the MPS
    # fallback needs one transfer rather than one per parameter tensor.
    return t.cat([
        part.detach().reshape(-1)
        if part is not None else t.zeros_like(parameter).reshape(-1)
        for part, parameter in zip(parts, parameters)
    ]).to(_diagnostic_device(parameters))


def _cpu_artifact(value: Any) -> Any:
    """Copy finished results for portable serialization, after all GPU algebra."""
    if isinstance(value, t.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_artifact(item) for key, item in value.items()}
    return value


def _project(vector: t.Tensor, basis: t.Tensor) -> t.Tensor:
    # Two passes prevent loss of orthogonality, particularly in flat spaces.
    for _ in range(2):
        vector = vector - basis @ (basis.T @ vector)
    return vector


def _orient_columns(vectors: t.Tensor) -> t.Tensor:
    if vectors.numel():
        indices = vectors.abs().argmax(dim=0)
        vectors = vectors * vectors[indices, t.arange(vectors.shape[1], device=vectors.device)].sign()
    return vectors


def _lanczos(
    hvp: Callable[[t.Tensor], t.Tensor],
    parameter_count: int,
    budget: int,
    seeds: list[t.Tensor],
    block_size: int,
    generator: t.Generator,
    dtype: t.dtype,
    max_basis_dim: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, dict[str, Any]]:
    """Spend a shared HVP budget with bounded, thick-restarted storage.

    Keep the lowest/highest block_size Ritz modes and task-gradient seeds at
    each restart. Transform their cached H*Q images with the same coordinates,
    so retained vectors cost no new HVPs. Expand from their residuals, with
    independent random vectors at breakdown. Stop at the budget, or once a
    complete parameter basis has been processed. The reduced eigensystem and
    both P-by-window arrays remain bounded even when the budget is 10,000.
    """
    requested_window = max(512, 4 * block_size + len(seeds)) if max_basis_dim is None else max_basis_dim
    if not isinstance(requested_window, int) or requested_window < 1:
        raise ValueError("max_basis_dim must be a positive integer.")
    limit = min(budget, parameter_count, requested_window)
    if budget > limit and parameter_count > limit and limit < 2 * block_size + len(seeds) + 2:
        raise ValueError("The Lanczos window needs room for retained modes, gradient seeds and expansion.")
    device = generator.device
    basis = t.empty((parameter_count, limit), dtype=dtype, device=device)
    images = t.empty_like(basis)
    filled = 0
    product_count = 0
    random_restarts = 0
    lanczos_restarts = 0
    tolerance = 32 * t.finfo(dtype).eps

    def append(vector: t.Tensor) -> bool:
        nonlocal filled
        vector = vector.detach().to(device=device, dtype=dtype)
        original_norm = t.linalg.vector_norm(vector, dtype=t.float64)
        if not bool(t.isfinite(vector).all()):
            raise ValueError("Non-finite vector during Lanczos measurement.")
        if original_norm == 0:
            return False
        vector = _project(vector, basis[:, :filled])
        norm = t.linalg.vector_norm(vector, dtype=t.float64)
        if norm <= tolerance * original_norm:
            return False
        basis[:, filled] = vector / norm
        filled += 1
        return True

    def append_random() -> None:
        for _ in range(8):
            if append(t.randn(parameter_count, generator=generator, dtype=dtype, device=device)):
                return
        raise RuntimeError("Could not restart Lanczos with an independent vector.")

    # Retain the task gradient's linear loss term, and improve coverage of
    # repeated eigenvalues with other task gradients and independent seeds.
    initial_size = min(limit, block_size)
    for seed in seeds:
        if filled == initial_size:
            break
        append(seed)
    while filled < initial_size:
        append_random()

    column = 0
    while True:
        while column < limit and product_count < budget:
            if column == filled:
                append_random()
                random_restarts += 1
            image = hvp(basis[:, column]).detach().to(device=device, dtype=dtype)
            if image.shape != (parameter_count,) or not bool(t.isfinite(image).all()):
                raise ValueError("Hessian-vector product must be a finite parameter vector.")
            images[:, column] = image
            column += 1
            product_count += 1
            if filled < limit:
                append(image)
            if progress is not None and (product_count == 1 or product_count % 500 == 0
                                         or product_count == budget or column == parameter_count):
                progress({"stage": "hvp", "hvp_count": product_count,
                          "active_dim": column, "lanczos_restarts": lanczos_restarts})

        # Only processed columns have cached images. A final partial cycle
        # may have an additional generated, but unprocessed, basis vector.
        active = basis[:, :column]
        active_images = images[:, :column]
        reduced = (active.T @ active_images).double()
        reduced = (reduced + reduced.T) / 2
        if product_count == budget or column == parameter_count:
            return active, active_images, reduced, {
                "hvp_count": product_count,
                "random_restarts": random_restarts,
                "lanczos_restarts": lanczos_restarts,
                "max_basis_dim": limit,
                "full_space_covered": column == parameter_count,
            }

        # Thick restart retains both spectral ends, rather than discarding
        # the accumulated eigenvector estimates and starting over randomly.
        if progress is not None:
            progress({"stage": "lanczos_restart", "hvp_count": product_count,
                      "active_dim": column, "lanczos_restarts": lanczos_restarts})
        _, modes = t.linalg.eigh(reduced)
        ends = sorted(set(range(block_size)) | set(range(column - block_size, column)))
        retained_coordinates = [modes[:, index] for index in ends]
        retained = t.stack(retained_coordinates, dim=1)
        for seed in seeds:
            coordinate = (active.T @ seed.to(device=device, dtype=dtype)).double()
            original_norm = coordinate.norm()
            coordinate = _project(coordinate, retained)
            norm = coordinate.norm()
            if original_norm > 0 and norm > tolerance * original_norm:
                retained = t.cat((retained, (coordinate / norm)[:, None]), dim=1)
        transformed = retained.to(dtype=dtype)
        retained_basis = active @ transformed
        retained_images = active_images @ transformed
        # Reorthogonalize carried full vectors and transform their images
        # identically. This limits accumulated float32 error across restarts.
        filled = 0
        for vector, image in zip(retained_basis.T, retained_images.T):
            for _ in range(2):
                coefficients = basis[:, :filled].T @ vector
                vector = vector - basis[:, :filled] @ coefficients
                image = image - images[:, :filled] @ coefficients
            norm = t.linalg.vector_norm(vector, dtype=t.float64)
            if norm <= tolerance:
                continue
            basis[:, filled], images[:, filled] = vector / norm, image / norm
            filled += 1
        column = filled  # All retained images have already been computed.
        lanczos_restarts += 1
        # Hessian residuals of retained modes seed the next expansion without
        # consuming products. Repeated/flat eigenvalues use random restarts.
        retained_count = filled
        for index in range(retained_count):
            if filled == limit:
                break
            append(images[:, index])


def _complement(vectors: t.Tensor, dimension: int) -> t.Tensor:
    """An orthogonal complement in the small Krylov coordinate space."""
    if vectors.shape[1] == 0:
        return t.eye(dimension, dtype=t.float64, device=vectors.device)
    orthogonal, _ = t.linalg.qr(vectors, mode="complete")
    return orthogonal[:, vectors.shape[1]:]


def _maximize_quadratic(hessian: t.Tensor, linear: t.Tensor) -> t.Tensor:
    """Maximize b.T*u + (u.T*A*u)/2 on the reduced unit sphere.

    Solve the symmetric trust-region secular equation, including its hard
    case. A largest Hessian eigenvector alone misses first-order sharpness.
    """
    eigenvalues, eigenvectors = t.linalg.eigh(hessian)
    coefficients = eigenvectors.T @ linear
    linear_norm = t.linalg.vector_norm(coefficients)
    if linear_norm == 0:
        return eigenvectors[:, -1]
    largest = eigenvalues[-1]
    gaps = largest - eigenvalues
    scale = t.maximum(eigenvalues.abs().max(), linear_norm).clamp_min(1e-30)
    at_top = gaps <= 1e-12 * scale
    regular = t.zeros_like(coefficients)
    regular[~at_top] = coefficients[~at_top] / gaps[~at_top]
    if t.linalg.vector_norm(coefficients[at_top]) <= 1e-12 * linear_norm and regular.norm() <= 1:
        # Complete the norm in the top eigenspace when the secular root lies
        # exactly at the largest eigenvalue.
        regular[t.where(at_top)[0][-1]] = t.sqrt((1 - regular.square().sum()).clamp_min(0))
        return eigenvectors @ regular

    lower = largest
    upper = lower + linear_norm
    floor = t.finfo(t.float64).eps * scale
    for _ in range(80):
        middle = (lower + upper) / 2
        norm = t.linalg.vector_norm(coefficients / (middle - eigenvalues).clamp_min(floor))
        # Tensor selection avoids 80 CUDA-to-host synchronizations per solve.
        lower = t.where(norm > 1, middle, lower)
        upper = t.where(norm > 1, upper, middle)
    direction = coefficients / (upper - eigenvalues).clamp_min(floor)
    return eigenvectors @ (direction / direction.norm())


def loss_spaces(
    model: t.nn.Module,
    function: BoolFunction,
    criterion: t.nn.modules.loss._Loss,
    payloads: t.Tensor,
    delta: float,
    epsilon: float,
    num_directions: int = 100,
    direction_seed: int = 0,
    hvp_budget: int = 10_000,
    intersection_cosine: float = 0.999,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Estimate loss spaces with one shared HVP budget per task.

    A thick-restart block Lanczos space approximates the task Hessian with
    bounded storage. Extreme modes and task gradients survive each restart.
    Sharpness greedily maximizes the second-order, both-sign loss increase in
    that space, including the gradient term. Tangent directions minimize
    curvature in its gradient-orthogonal part. At a stationary point these
    reduce to the largest/smallest Ritz eigenvectors, respectively.

    Every proposed unit vector is checked against the actual mean loss at both
    +/-delta. No extra HVPs or perturbed-loss gradient optimization are used.
    Counts concern discovered directions, not certified full dimensions.
    Neither Ritz approximations nor the Taylor model guarantee exact extrema
    of finite-radius loss or safety of arbitrary linear combinations.
    Parameters, gradients, module modes and training RNG are preserved.
    """
    if payloads.ndim != 2 or min(payloads.shape) < 1:
        raise ValueError("payloads must be a nonempty 2D batch.")
    if not 0 < delta < float("inf") or not 0 <= epsilon < float("inf"):
        raise ValueError("delta must be positive and epsilon nonnegative; both finite.")
    for name, value in (("num_directions", num_directions), ("hvp_budget", hvp_budget)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not 0 < intersection_cosine <= 1:
        raise ValueError("intersection_cosine must be in (0, 1].")

    named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named_parameters:
        raise ValueError("Loss spaces require trainable parameters.")
    names = [name for name, _ in named_parameters]
    parameters = [p for _, p in named_parameters]
    compute_device = _diagnostic_device(parameters)
    parameter_count = sum(p.numel() for p in parameters)
    sizes = [p.numel() for p in parameters]
    cap = min(num_directions, parameter_count)
    tasks = function.functions if isinstance(function, SelectorFunction) else [function]
    task_names = _unique_metric_names(tasks)
    selector_size = function.selector_size if isinstance(function, SelectorFunction) else 0
    shifts = t.arange(selector_size - 1, -1, -1, device=payloads.device)
    batches = []
    for code, task in enumerate(tasks):
        prefix = ((code >> shifts) & 1).expand(payloads.size(0), -1)
        inputs = t.cat((prefix, payloads), dim=1) if selector_size else payloads
        batches.append((inputs, task(payloads)))

    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with t.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            generator = t.Generator(device=compute_device).manual_seed(direction_seed)

            def report(stage: str, **fields: Any) -> None:
                if progress is not None:
                    progress({"stage": stage, "solver_device": str(compute_device), **fields})

            report("baseline_gradients", task_count=len(tasks), parameter_count=parameter_count,
                   hvp_budget=hvp_budget, direction_cap=cap)

            def task_loss(index: int, direction: t.Tensor | None = None, sign: float = 0.0) -> t.Tensor:
                inputs, labels = batches[index]
                if direction is None:
                    outputs = model(inputs)
                else:
                    point = {
                        name: parameter.detach() + sign * delta * piece.reshape_as(parameter).to(
                            dtype=parameter.dtype, device=parameter.device,
                        )
                        for (name, parameter), piece in zip(named_parameters, direction.split(sizes))
                    }
                    outputs = functional_call(model, point, (inputs,))
                return _mean_loss(criterion, outputs, labels)

            baselines = []
            task_gradients = []
            for index in range(len(tasks)):
                loss = task_loss(index)
                value = loss.detach()
                if not bool(t.isfinite(loss.detach())):
                    raise ValueError("Non-finite baseline task loss during space measurement.")
                baselines.append(value)
                grads = t.autograd.grad(loss, parameters, allow_unused=True) if loss.requires_grad else (
                    None,
                ) * len(parameters)
                task_gradients.append(_flatten(grads, parameters))

            def make_hvp(index: int) -> Callable[[t.Tensor], t.Tensor]:
                # Reuse one differentiable baseline gradient graph per task.
                # autograd.grad never writes training .grad buffers.
                loss = task_loss(index)
                grads = t.autograd.grad(
                    loss, parameters, create_graph=True, allow_unused=True,
                ) if loss.requires_grad else (None,) * len(parameters)

                def hvp(vector: t.Tensor) -> t.Tensor:
                    products = [
                        (gradient * piece.reshape_as(parameter).to(
                            dtype=parameter.dtype, device=parameter.device,
                        )).sum()
                        for gradient, parameter, piece in zip(grads, parameters, vector.split(sizes))
                        if gradient is not None and gradient.requires_grad
                    ]
                    values = t.autograd.grad(
                        sum(products), parameters, retain_graph=True, allow_unused=True,
                    ) if products else (None,) * len(parameters)
                    return _flatten(values, parameters)

                return hvp

            def loss_changes(index: int, direction: t.Tensor) -> t.Tensor:
                with t.no_grad():
                    # Subtract in float64, matching the previous Python-float
                    # subtraction. Only the MPS fallback needs a transfer here.
                    return t.stack([task_loss(index, direction, sign).detach()
                                    for sign in (+1, -1)]).to(compute_device).double() - baselines[index].to(compute_device).double()

            metrics: dict[str, float] = {
                "spaces/search_dim": float(parameter_count),
                "spaces/parameter_count": float(parameter_count),
                "spaces/direction_cap": float(cap),
                "spaces/hvp_budget": float(hvp_budget),
            }
            task_results = {}
            for index, name in enumerate(task_names):
                report("task_start", task=name, task_index=index + 1, task_count=len(tasks),
                       hvp_budget=hvp_budget, direction_cap=cap)
                hvp = make_hvp(index)
                seeds = [task_gradients[index]] + [gradient for j, gradient in enumerate(task_gradients)
                                                  if j != index]
                basis, images, reduced, solver = _lanczos(
                    hvp, parameter_count, hvp_budget, seeds, max(cap, len(tasks)),
                    generator, task_gradients[index].dtype,
                    **({"progress": lambda event: report(event["stage"], task=name,
                        **{key: value for key, value in event.items() if key != "stage"})}
                       if progress is not None else {}),
                )
                del hvp  # Release this task's differentiation graph.
                dimension = basis.shape[1]
                report("eigensystem", task=name, hvp_count=solver["hvp_count"], active_dim=dimension)
                gradient = task_gradients[index].double()
                reduced_gradient = (basis.T @ task_gradients[index]).double()
                eigenvalues, eigenvectors = t.linalg.eigh(reduced)
                # Check both spectral ends using cached H*Q, costing no HVPs.
                selected = sorted(set(range(min(cap, dimension))) |
                                  set(range(max(0, dimension - cap), dimension)))
                coordinates = eigenvectors[:, selected].to(dtype=basis.dtype)
                ritz_vectors = basis @ coordinates
                ritz_images = images @ coordinates
                residuals = (ritz_images - ritz_vectors * eigenvalues[selected].to(basis.dtype)).double().norm(dim=0)
                relative = residuals / t.maximum(
                    ritz_images.double().norm(dim=0), eigenvalues[selected].abs(),
                ).clamp_min(1e-30)
                result = {
                    "baseline_loss": float(baselines[index].cpu()), "loss_gradient": gradient.float(),
                    "hvp_count": solver["hvp_count"], "krylov_dim": dimension,
                    "hvp_budget_exhausted": solver["hvp_count"] == hvp_budget,
                    "krylov_restarts": solver["random_restarts"],
                    "lanczos_restarts": solver["lanczos_restarts"],
                    "krylov_max_dim": solver["max_basis_dim"],
                    "full_space_covered": solver["full_space_covered"],
                    "ritz_eigenvalues": eigenvalues,
                    "ritz_checked_indices": t.tensor(selected, device=compute_device),
                    "ritz_residual_norms": residuals, "ritz_relative_residuals": relative,
                }
                task_results[name] = result
                metrics[f"spaces/hvp_count/{name}"] = float(solver["hvp_count"])
                metrics[f"spaces/krylov_dim/{name}"] = float(dimension)
                metrics[f"spaces/hvp_budget_exhausted/{name}"] = float(solver["hvp_count"] == hvp_budget)
                metrics[f"spaces/lanczos_restarts/{name}"] = float(solver["lanczos_restarts"])
                metrics[f"spaces/full_space_covered/{name}"] = float(solver["full_space_covered"])
                metrics["spaces/krylov_max_dim"] = float(solver["max_basis_dim"])
                metrics[f"spaces/ritz_max_relative_residual/{name}"] = float(relative.max())

                # Tangency to a nonstationary level set means g.T*v = 0.
                normals = (reduced_gradient / reduced_gradient.norm())[:, None] if gradient.norm() > 0 else t.empty(
                    (dimension, 0), dtype=t.float64, device=compute_device,
                )
                tangent_coordinates = _complement(normals, dimension)
                tangent_hessian = tangent_coordinates.T @ reduced @ tangent_coordinates
                tangent_values, tangent_modes = t.linalg.eigh(tangent_hessian)
                tangent_coordinates = tangent_coordinates @ tangent_modes

                report("direction_checks", task=name)

                for kind, maximize in (("sharpness", True), ("tangent", False)):
                    accepted = []
                    accepted_coordinates = []
                    changes = []
                    curvatures = []
                    predicted = []
                    direction_residuals = []
                    stop_harm = None
                    available = dimension if maximize else tangent_coordinates.shape[1]
                    for search_index in range(min(cap, available)):
                        if maximize:
                            previous = t.stack(accepted_coordinates, dim=1) if accepted_coordinates else t.empty(
                                (dimension, 0), dtype=t.float64, device=compute_device,
                            )
                            complement = _complement(previous, dimension)
                            coordinate = complement @ _maximize_quadratic(
                                delta**2 * (complement.T @ reduced @ complement),
                                delta * (complement.T @ reduced_gradient),
                            )
                        else:
                            coordinate = tangent_coordinates[:, search_index]
                        direction = (basis @ coordinate.to(basis.dtype)).double()
                        image = (images @ coordinate.to(images.dtype)).double()
                        norm = direction.norm()
                        direction, image = direction / norm, image / norm
                        actual_changes = loss_changes(index, direction)
                        harm = actual_changes.max()
                        if not bool(t.isfinite(actual_changes).all()):
                            raise ValueError("Non-finite perturbed task loss during space measurement.")
                        if search_index == 0:
                            metrics[f"spaces/{'max' if maximize else 'min'}_harm/{name}"] = float(harm)
                        if not (harm > epsilon if maximize else harm <= epsilon):
                            stop_harm = float(harm)
                            break
                        curvature = t.dot(direction, image)
                        first_order = t.dot(gradient, direction)
                        if maximize:
                            stationarity = delta**2 * image + delta * gradient
                            prior_basis = t.stack(accepted, dim=1) if accepted else t.empty(
                                (parameter_count, 0), dtype=t.float64, device=compute_device,
                            )
                            stationarity = _project(stationarity, prior_basis)
                            residual = stationarity - t.dot(direction, stationarity) * direction
                        else:
                            normal = (gradient / gradient.norm())[:, None] if gradient.norm() > 0 else t.empty(
                                (parameter_count, 0), dtype=t.float64, device=compute_device,
                            )
                            residual = _project(image, normal) - tangent_values[search_index] * direction
                        accepted.append(direction)
                        accepted_coordinates.append(coordinate)
                        changes.append(actual_changes)
                        curvatures.append(curvature)
                        predicted.append(delta * first_order.abs() + delta**2 * curvature / 2)
                        direction_residuals.append(residual.norm())

                    directions = t.stack(accepted, dim=1) if accepted else t.empty(
                        (parameter_count, 0), dtype=t.float64, device=compute_device,
                    )
                    oriented = _orient_columns(directions)
                    # Flipping orientation swaps the +/- loss-change columns.
                    loss_columns = t.stack(changes).float() if changes else t.empty(
                        (0, 2), dtype=t.float32, device=compute_device,
                    )
                    if accepted:
                        flipped = (directions * oriented).sum(dim=0) < 0
                        loss_columns[flipped] = loss_columns[flipped].flip(dims=[1])
                    result[kind] = oriented.float()
                    result[f"{kind}_loss_changes"] = loss_columns
                    result[f"{kind}_curvatures"] = t.stack(curvatures) if curvatures else t.empty(
                        0, dtype=t.float64, device=compute_device,
                    )
                    result[f"{kind}_predicted_harms"] = t.stack(predicted) if predicted else t.empty(
                        0, dtype=t.float64, device=compute_device,
                    )
                    residual_name = "stationarity_residual_norms" if maximize else "eigen_residual_norms"
                    result[f"{kind}_{residual_name}"] = t.stack(direction_residuals) if direction_residuals else t.empty(
                        0, dtype=t.float64, device=compute_device,
                    )
                    result[f"{kind}_stop_harm"] = stop_harm
                    result[f"{kind}_cap_reached"] = len(accepted) == cap
                    result[f"{kind}_candidate_exhausted"] = stop_harm is None and len(accepted) < cap
                    metrics[f"spaces/{kind}_dim/{name}"] = float(len(accepted))
                    metrics[f"spaces/{kind}_cap_reached/{name}"] = float(len(accepted) == cap)
                    metrics[f"spaces/{kind}_candidate_exhausted/{name}"] = float(result[f"{kind}_candidate_exhausted"])
                    if direction_residuals:
                        metrics[f"spaces/{kind}_max_residual/{name}"] = float(t.stack(direction_residuals).max())
                    if stop_harm is not None:
                        metrics[f"spaces/{kind}_stop_harm/{name}"] = stop_harm
                report("task_complete", task=name, hvp_count=solver["hvp_count"],
                       tangent_dim=result["tangent"].shape[1], sharpness_dim=result["sharpness"].shape[1])
                del basis, images, ritz_vectors, ritz_images

            report("interference_start")
            interference = {}
            for source_index, source in enumerate(task_names):
                for target_index, target in enumerate(task_names):
                    if source_index == target_index:
                        continue
                    sharp = task_results[source]["sharpness"].double()
                    tangent = task_results[target]["tangent"].double()
                    if sharp.shape[1] and tangent.shape[1]:
                        left, cosines, _ = t.linalg.svd(sharp.T @ tangent, full_matrices=False)
                        selected = cosines >= intersection_cosine
                        proposed = _orient_columns(sharp @ left[:, selected])
                        proposed_cosines = cosines[selected]
                        source_changes = t.stack([loss_changes(source_index, vector) for vector in proposed.T]) if proposed.shape[1] else t.empty(
                            (0, 2), dtype=t.float64, device=compute_device,
                        )
                        target_changes = t.stack([loss_changes(target_index, vector) for vector in proposed.T]) if proposed.shape[1] else t.empty_like(source_changes)
                        verified = (source_changes.amax(dim=1) > epsilon) & (target_changes.amax(dim=1) <= epsilon)
                        directions = proposed[:, verified].float()
                        selected_cosines = proposed_cosines[verified].float()
                        source_changes = source_changes[verified].float()
                        target_changes = target_changes[verified].float()
                    else:
                        directions = t.empty((parameter_count, 0), dtype=t.float32, device=compute_device)
                        selected_cosines = t.empty(0, dtype=t.float32, device=compute_device)
                        source_changes = target_changes = t.empty((0, 2), dtype=t.float32, device=compute_device)
                    key = f"{source}_to_{target}"
                    interference[key] = {
                        "directions": directions, "cosines": selected_cosines,
                        "source_loss_changes": source_changes,
                        "target_loss_changes": target_changes,
                    }
                    metrics[f"spaces/interference_dim/{key}"] = float(directions.shape[1])
                report("interference_source_complete", task=source)
            report("same_kind_overlaps_start")
            overlaps = pairwise_overlaps(
                task_results, intersection_cosine, loss_changes=loss_changes,
                epsilon=epsilon, progress=progress,
            )
            for kind, pairs in overlaps.items():
                for key, result in pairs.items():
                    metrics[f"spaces/{kind}_overlap_dim/{key}"] = float(result["directions"].shape[1])
                    metrics[f"spaces/{kind}_overlap_verified_dim/{key}"] = float(result["loss_verified"].sum())
                    if result["principal_cosines"].numel():
                        metrics[f"spaces/{kind}_overlap_max_cosine/{key}"] = float(result["principal_cosines"].max())
            report("artifact_cpu_export")
    finally:
        for module, was_training in modes:
            module.training = was_training

    artifact = {
        "format_version": 5,
        "method": "matrix_free_thick_restart_block_lanczos_loss_spaces",
        "parameter_names": names,
        "parameter_shapes": [tuple(p.shape) for p in parameters],
        "tasks": task_results,
        "interference": interference,
        "overlaps": overlaps,
        "delta": delta,
        "epsilon": epsilon,
        "num_directions": num_directions,
        "intersection_cosine": intersection_cosine,
        "probe_payloads": payloads.detach().cpu(),
        "direction_seed": direction_seed,
        "hvp_budget": hvp_budget,
        "krylov_max_dim": int(metrics["spaces/krylov_max_dim"]),
        "tangent_constraint": "orthogonal_to_task_loss_gradient",
        "solver_device": str(compute_device),
    }
    artifact = _cpu_artifact(artifact)
    report("measurement_complete")
    return metrics, artifact
