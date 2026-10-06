import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch as t

from src.evaluation import EvaluationSet, evaluate_fixed_set, make_evaluation_set
from src.functions import BoolFunction, SelectorFunction
from src.metrics import average_direction_sharpness, batch_metrics, gradient_metrics, multi_task_curvature
from src.model import TransformerModel
from src.spaces import loss_spaces


BatchMetricFn = Callable[[t.Tensor, t.Tensor, t.Tensor], Dict[str, float]]
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
    batch_metric_fn: Optional[BatchMetricFn] = None,
    balanced_selectors: bool = False,
    selector_probabilities: Optional[SelectorProbabilities] = None,
    sharpness_interval: int = 100,
    sharpness_num_inputs: int = 32,
    sharpness_num_perturbations: int = 4,
    sharpness_rho: float = 0.02,
    gradient_interval: int = 20,
    gradient_num_inputs: int = 256,
    gradient_seed: int = 0,
    curvature_interval: int = 100,
    curvature_num_points: int = 3,
    eval_interval: Optional[int] = None,
    eval_num_inputs_per_function: int = 320,
    eval_batch_size: int = 320,
    eval_seed: int = 0,
    space_delta: Optional[float] = None,
    space_epsilon: float = 0.002,
    space_num_inputs: int = 16,
    space_seed: int = 0,
    space_num_directions: int = 100,
    space_direction_seed: int = 0,
    space_hvp_budget: int = 10_000,
    space_intersection_cosine: float = 0.999,
    space_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    space_enabled: bool = False,
) -> List[float]:
    """Train the model and optionally log per-batch metrics to W&B.

    ``wandb_run`` is optional so the training helper remains usable without
    W&B. When supplied, W&B automatically receives overall loss and accuracy.
    For a ``SelectorFunction``, it also receives a per-function loss and
    accuracy using each selected function's ``metric_name``.
    ``batch_metric_fn`` can add experiment-specific metrics to those defaults.
    ``selector_probabilities`` is passed to ``sample_batch`` for selector tasks.
    When logging, average direction sharpness is estimated every
    ``sharpness_interval`` optimizer updates using uniform input bits.
    Gradient norms and pairwise cosines are measured every ``gradient_interval``
    updates on ``gradient_num_inputs`` fixed uniform payloads at ``max_len``.
    A separate generator seeded with ``gradient_seed`` keeps probe sampling
    independent of training randomness. Measurements only run when logging.
    PCGrad multi-task curvature is measured every ``curvature_interval`` steps
    along that optimizer update using ``curvature_num_points`` quadrature
    points. It uses the same fixed payloads and the sum of task mean losses.
    Set ``eval_interval`` to reserve a fixed, balanced set at ``max_len`` and
    log post-update ``eval/loss`` and ``eval/accuracy`` (also per task).
    Reserved inputs are excluded from training even without a W&B logger.
    Evaluation uses ``no_grad()``, restores model modes, and runs only when
    logging. Its private seed does not consume training randomness.
    Task-loss spaces are disabled by default. Set ``space_enabled=True`` and
    a positive ``space_delta`` to measure them after the final update.
    Matrix-free thick-restart block Lanczos uses ``space_hvp_budget`` products
    per task, shared between sharpness and gradient-orthogonal tangent modes.
    Its active basis is bounded while eigenvector estimates survive restarts.
    Every returned direction passes both-sign finite-radius loss checks.
    ``space_num_directions`` caps the number returned for each task and each
    kind; it does not restrict the search to a parameter subspace. Full vectors
    and both-sign loss changes are saved as W&B artifacts. The reusable helper
    leaves this costly measure disabled by default.
    """

    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if sharpness_interval < 1:
        raise ValueError("sharpness_interval must be positive.")
    if sharpness_num_inputs < 1 or sharpness_num_perturbations < 1:
        raise ValueError("sharpness sample counts must be positive.")
    if not 0 <= sharpness_rho < float("inf"):
        raise ValueError("sharpness_rho must be finite and nonnegative.")
    if gradient_interval < 1 or gradient_num_inputs < 1:
        raise ValueError("gradient_interval and gradient_num_inputs must be positive.")
    if not isinstance(curvature_interval, int) or isinstance(curvature_interval, bool) or curvature_interval < 1:
        raise ValueError("curvature_interval must be a positive integer.")
    if not isinstance(curvature_num_points, int) or isinstance(curvature_num_points, bool) or curvature_num_points < 1:
        raise ValueError("curvature_num_points must be a positive integer.")
    if eval_interval is not None:
        if not isinstance(eval_interval, int) or isinstance(eval_interval, bool) or eval_interval < 1:
            raise ValueError("eval_interval must be a positive integer or None.")
        if not isinstance(eval_batch_size, int) or isinstance(eval_batch_size, bool) or eval_batch_size < 1:
            raise ValueError("eval_batch_size must be a positive integer.")
    if not isinstance(space_enabled, bool):
        raise ValueError("space_enabled must be a boolean.")
    if space_enabled:
        if space_delta is None or not 0 < space_delta < float("inf") or not 0 <= space_epsilon < float("inf"):
            raise ValueError("space_delta must be positive and space_epsilon nonnegative; both finite.")
        if not isinstance(space_num_inputs, int) or isinstance(space_num_inputs, bool) or space_num_inputs < 1:
            raise ValueError("space_num_inputs must be a positive integer.")
        if not isinstance(space_num_directions, int) or isinstance(space_num_directions, bool) or space_num_directions < 1:
            raise ValueError("space_num_directions must be a positive integer.")
        if not isinstance(space_hvp_budget, int) or isinstance(space_hvp_budget, bool) or space_hvp_budget < 1:
            raise ValueError("space_hvp_budget must be a positive integer.")
        if not 0 < space_intersection_cosine <= 1:
            raise ValueError("space_intersection_cosine must be in (0, 1].")
    evaluation_set = None
    if eval_interval is not None:
        evaluation_set = make_evaluation_set(
            function, max_len, eval_num_inputs_per_function, eval_seed, device,
        )

    gradient_payloads = None
    if wandb_run is not None and num_steps >= min(gradient_interval, curvature_interval):
        selector_size = function.selector_size if isinstance(function, SelectorFunction) else 0
        if max_len <= selector_size:
            raise ValueError("Gradient inputs need at least one payload token.")
        generator = t.Generator(device="cpu").manual_seed(gradient_seed)
        gradient_payloads = t.randint(
            0, 2, (gradient_num_inputs, max_len - selector_size), generator=generator,
        ).to(device)

    space_payloads = None
    if wandb_run is not None and space_enabled:
        selector_size = function.selector_size if isinstance(function, SelectorFunction) else 0
        if max_len <= selector_size:
            raise ValueError("Space inputs need at least one payload token.")
        generator = t.Generator(device="cpu").manual_seed(space_seed)
        space_payloads = t.randint(
            0, 2, (space_num_inputs, max_len - selector_size), generator=generator,
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
        measure_curvature = wandb_run is not None and (step + 1) % curvature_interval == 0
        if measure_curvature:
            parameters_before = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters() if parameter.requires_grad
            }
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if wandb_run is not None:
            metrics = batch_metrics(function, criterion, x, outputs.detach(), y)
            if batch_metric_fn is not None:
                metrics.update(batch_metric_fn(x, outputs.detach(), y))
            if evaluation_set is not None and (step + 1) % eval_interval == 0:
                metrics.update({
                    f"eval/{name}": value for name, value in evaluate_fixed_set(
                        model, function, criterion, evaluation_set, eval_batch_size,
                    ).items()
                })
            if (step + 1) % gradient_interval == 0:
                metrics.update(gradient_metrics(model, function, criterion, gradient_payloads))
            if measure_curvature:
                metrics.update(multi_task_curvature(
                    model, function, criterion, gradient_payloads,
                    parameters_before, num_points=curvature_num_points,
                ))
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
            if space_payloads is not None and step + 1 == num_steps:
                space_metrics, directions = loss_spaces(
                    model, function, criterion, space_payloads, delta=space_delta,
                    epsilon=space_epsilon,
                    num_directions=space_num_directions,
                    direction_seed=space_direction_seed,
                    hvp_budget=space_hvp_budget,
                    intersection_cosine=space_intersection_cosine,
                    progress=space_progress,
                )
                metrics.update(space_metrics)
                if hasattr(wandb_run, "log_artifact"):
                    import wandb

                    artifact = wandb.Artifact(
                        f"loss-spaces-{wandb_run.id}", type="loss-spaces",
                        metadata={"step": step + 1, "delta": space_delta,
                                  "epsilon": space_epsilon,
                                  "num_directions": space_num_directions,
                                  "num_inputs": space_num_inputs,
                                  "hvp_budget": space_hvp_budget,
                                  "method": "matrix_free_thick_restart_block_lanczos_loss_spaces",
                                  "krylov_max_dim": directions.get("krylov_max_dim"),
                                  "format_version": directions.get("format_version"),
                                  "overlap_kinds": ["sharpness", "tangent"],
                                  "search_scope": "all_trainable_parameters",
                                  "intersection_cosine": space_intersection_cosine},
                    )
                    with tempfile.TemporaryDirectory(prefix="loss-spaces-") as directory:
                        path = Path(directory) / f"spaces_step_{step + 1}.pt"
                        directions["run_id"] = wandb_run.id
                        if space_progress is not None:
                            space_progress({"stage": "artifact_save_start", "step": step + 1})
                        t.save(directions, path)
                        if space_progress is not None:
                            space_progress({"stage": "artifact_saved", "bytes": path.stat().st_size})
                        artifact.add_file(str(path))
                        wandb_run.log_artifact(artifact, aliases=[f"step-{step + 1}"])
                        if space_progress is not None:
                            space_progress({"stage": "artifact_upload_queued", "step": step + 1})
            wandb_run.log(metrics, step=step + 1)

    return losses
