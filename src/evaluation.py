"""Fixed, reproducible held-out inputs and forward-only evaluation."""

from dataclasses import dataclass, field
from typing import Dict

import torch as t

from src.functions import BoolFunction, SelectorFunction
from src.metrics import batch_metrics


def _input_keys(inputs: t.Tensor) -> list[bytes]:
    # Exact keys (no hash collisions), including all selector and payload bits.
    return [row.tobytes() for row in inputs.detach().to(device="cpu", dtype=t.uint8).numpy()]


@dataclass
class EvaluationSet:
    inputs: t.Tensor
    labels: t.Tensor
    _keys: set[bytes] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._keys = set(_input_keys(self.inputs))

    def overlapping_rows(self, inputs: t.Tensor) -> list[int]:
        if inputs.size(1) != self.inputs.size(1):
            return []
        return [index for index, key in enumerate(_input_keys(inputs)) if key in self._keys]


def make_evaluation_set(
    function: BoolFunction,
    seq_len: int,
    num_inputs_per_function: int = 320,
    seed: int = 0,
    device: t.device = t.device("cpu"),
) -> EvaluationSet:
    """Reserve unique uniform inputs at one length, equally across valid tasks.

    A private CPU generator leaves training randomness untouched. Each task
    must have at least one possible input left for training.
    """
    if not isinstance(num_inputs_per_function, int) or isinstance(num_inputs_per_function, bool) or num_inputs_per_function < 1:
        raise ValueError("num_inputs_per_function must be a positive integer.")
    selector_size = function.selector_size if isinstance(function, SelectorFunction) else 0
    payload_len = seq_len - selector_size
    if payload_len < 1:
        raise ValueError("Evaluation inputs need at least one payload token.")
    if num_inputs_per_function >= 2 ** payload_len:
        raise ValueError("Evaluation set must leave at least one input per function for training; reduce num_inputs_per_function.")
    generator = t.Generator(device="cpu").manual_seed(seed)
    num_functions = len(function.functions) if isinstance(function, SelectorFunction) else 1
    batches = []
    for function_id in range(num_functions):
        if payload_len <= 16:
            # Avoid slow rejection sampling when much of a small space is held out.
            codes = t.randperm(2 ** payload_len, generator=generator)[:num_inputs_per_function]
            shifts = t.arange(payload_len - 1, -1, -1)
            payloads = (codes[:, None] >> shifts) & 1
        else:
            unique: dict[bytes, t.Tensor] = {}
            while len(unique) < num_inputs_per_function:
                candidates = t.randint(
                    0, 2, (num_inputs_per_function - len(unique), payload_len), generator=generator,
                )
                unique.update(zip(_input_keys(candidates), candidates))
            payloads = t.stack(list(unique.values()))
        shifts = t.arange(selector_size - 1, -1, -1)
        selectors = ((function_id >> shifts) & 1).expand(num_inputs_per_function, -1)
        batches.append(t.cat((selectors, payloads), dim=1))
    inputs = t.cat(batches).to(device)
    return EvaluationSet(inputs, function(inputs))


def evaluate_fixed_set(
    model: t.nn.Module,
    function: BoolFunction,
    criterion: t.nn.modules.loss._Loss,
    evaluation_set: EvaluationSet,
    batch_size: int = 320,
) -> Dict[str, float]:
    """Measure the full set after batched forwards, with no_grad, not inference mode.

    Aggregating predictions before scoring preserves exact sample weighting,
    task names and criterion reductions, even for incomplete final batches.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("evaluation batch_size must be a positive integer.")
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with t.no_grad():
            outputs = t.cat([model(inputs) for inputs in evaluation_set.inputs.split(batch_size)])
            return batch_metrics(function, criterion, evaluation_set.inputs, outputs, evaluation_set.labels)
    finally:
        for module, training in modes:
            module.training = training
