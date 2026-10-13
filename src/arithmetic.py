"""Digit arithmetic tasks and an autoregressive version of the existing transformer.

Only new code lives here: the Boolean experiment and its public APIs are untouched.
Digits are big-endian (``0012`` means 12); answers use canonical decimal notation.
EOS terminates answers, PAD is ignored by the loss, and ANSWER starts decoding.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
import math
import re
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.evaluation import EvaluationSet
from src.metrics import _unique_metric_names
from src.model import TransformerModel


EOS = 10
MINUS = 11
PAD = 12
ANSWER = 13
INPUT_VOCAB_SIZE = 14
OUTPUT_VOCAB_SIZE = 12  # Digits, EOS and a minus sign.


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _digits_tensor(inputs: torch.Tensor) -> None:
    if inputs.ndim != 2 or min(inputs.shape) < 1:
        raise ValueError("Expected a nonempty 2D digit batch.")
    if inputs.dtype not in (torch.int32, torch.int64):
        raise ValueError("Digit tokens must use integer tensors.")
    if bool(((inputs < 0) | (inputs > 9)).any()):
        raise ValueError("Arithmetic inputs and selectors must contain digits 0..9.")


class ArithmeticFunction(ABC):
    """Extend by implementing ``evaluate`` and ``max_result``.

    ``digits_per_argument=None`` splits the entire payload equally by arity;
    otherwise read that many digits per argument and ignore extra payload digits.
    ``sensitivity`` is an optional supplied value. ``avg_sensitivity`` estimates
    the sum of digit influences: replace one digit uniformly by a DIFFERENT digit
    and test whether the integer answer changes. Domain-invalid mutations have
    zero contribution; original inputs are uniform on the function's valid domain.
    Selector digits never participate. This reduces to Boolean flip sensitivity
    when the alphabet has two symbols.
    """

    operation = "arithmetic"

    def __init__(self, arity: int = 2, *, digits_per_argument: Optional[int] = None,
                 sensitivity: Optional[float] = None, name: Optional[str] = None):
        self.arity = _positive_int("arity", arity)
        self.digits_per_argument = digits_per_argument
        if digits_per_argument is not None:
            _positive_int("digits_per_argument", digits_per_argument)
        if sensitivity is not None and not 0 <= sensitivity < float("inf"):
            raise ValueError("sensitivity must be finite and nonnegative.")
        self.sensitivity = sensitivity
        self._name = name

    @property
    def metric_name(self) -> str:
        return self._name or f"{self.operation}_{self.arity}"

    def argument_width(self, payload_len: int) -> int:
        _positive_int("payload_len", payload_len)
        if self.digits_per_argument is None:
            if payload_len % self.arity:
                raise ValueError("Payload length must be divisible by each function's arity.")
            width = payload_len // self.arity
        else:
            width = self.digits_per_argument
        if width < 1 or width * self.arity > payload_len:
            raise ValueError("Payload is too short for the function's arguments.")
        return width

    def arguments(self, payloads: torch.Tensor) -> list[tuple[int, ...]]:
        _digits_tensor(payloads)
        width = self.argument_width(payloads.size(1))
        # Python integers avoid silent overflow for long digit sequences/products.
        return [tuple(int("".join(map(str, row[i * width:(i + 1) * width])))
                      for i in range(self.arity))
                for row in payloads.detach().cpu().tolist()]

    def valid_arguments(self, arguments: tuple[int, ...]) -> bool:
        return True

    def valid_mask(self, payloads: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.valid_arguments(args) for args in self.arguments(payloads)],
                            device=payloads.device, dtype=torch.bool)

    def __call__(self, payloads: torch.Tensor) -> list[int]:
        arguments = self.arguments(payloads)
        if not all(self.valid_arguments(args) for args in arguments):
            raise ValueError(f"Input is outside the domain of {self.metric_name} (zero divisor).")
        values = [self.evaluate(args) for args in arguments]
        if any(not isinstance(value, int) for value in values):
            raise ValueError("Arithmetic functions must return integer results.")
        return values

    @abstractmethod
    def evaluate(self, arguments: tuple[int, ...]) -> int:
        """Evaluate already-decoded, valid, nonnegative integer arguments."""

    @abstractmethod
    def max_result(self, argument_width: int) -> int:
        """Upper bound on the absolute result, for output capacity validation."""

    def required_output_tokens(self, argument_width: int) -> int:
        return len(str(self.max_result(argument_width))) + 1

    def avg_sensitivity(self, seq_len: int, num_inputs: int = 1024, seed: int = 0) -> float:
        if self.sensitivity is not None:
            return float(self.sensitivity)
        _positive_int("num_inputs", num_inputs)
        width = self.argument_width(seq_len)
        generator = torch.Generator().manual_seed(seed)
        payloads = sample_payloads(self, num_inputs, seq_len, generator=generator)
        baseline = self(payloads)
        changes = 0
        for position in range(width * self.arity):
            mutated = payloads.clone()
            mutated[:, position] = (mutated[:, position] + torch.randint(
                1, 10, (num_inputs,), generator=generator)) % 10
            valid = self.valid_mask(mutated)
            if bool(valid.any()):
                before = [value for value, keep in zip(baseline, valid.tolist()) if keep]
                # Use the payload API so digit tasks retain leading zeros.
                changes += sum(a != b for a, b in zip(before, self(mutated[valid])))
        return changes / num_inputs


class Addition(ArithmeticFunction):
    operation = "addition"

    def evaluate(self, arguments):
        return sum(arguments)

    def max_result(self, argument_width):
        return self.arity * (10 ** argument_width - 1)

    def avg_sensitivity(self, seq_len, num_inputs=1024, seed=0):
        if self.sensitivity is not None:
            return float(self.sensitivity)
        return float(self.arity * self.argument_width(seq_len))


class _UnaryDigitArithmetic(ArithmeticFunction):
    """One fixed-width digit argument; leading zeros are part of the input."""

    def __init__(self, arity=1, **kwargs):
        if arity != 1:
            raise ValueError(f"{type(self).__name__} requires arity 1.")
        super().__init__(arity=arity, **kwargs)

    def __call__(self, payloads):
        _digits_tensor(payloads)
        width = self.argument_width(payloads.size(1))
        return [self.evaluate_digits(row) for row in payloads[:, :width].detach().cpu().tolist()]

    def evaluate(self, arguments):
        # Decoded integers have canonical width unless one is explicitly supplied.
        digits = str(arguments[0]).zfill(self.digits_per_argument or 1)
        return self.evaluate_digits(list(map(int, digits)))

    @abstractmethod
    def evaluate_digits(self, digits):
        """Evaluate a single argument without dropping its leading zero digits."""

    def max_result(self, argument_width):
        return 9


class AdditionMod10(_UnaryDigitArithmetic):
    """Sum all digits of one argument, modulo 10 (e.g. 987 -> 4)."""

    operation = "addition_mod_10"

    def evaluate_digits(self, digits):
        return sum(digits) % 10

    def avg_sensitivity(self, seq_len, num_inputs=1024, seed=0):
        if self.sensitivity is not None:
            return float(self.sensitivity)
        # Every replacement by a different digit changes the sum modulo 10.
        return float(self.argument_width(seq_len))


class Majority(_UnaryDigitArithmetic):
    """Most frequent digit of one argument; ties select the smallest digit."""

    operation = "majority"

    def evaluate_digits(self, digits):
        counts = [0] * 10
        for digit in digits:
            counts[digit] += 1
        return max(range(10), key=counts.__getitem__)


class Multiplication(ArithmeticFunction):
    operation = "multiplication"

    def evaluate(self, arguments):
        return math.prod(arguments)

    def max_result(self, argument_width):
        return (10 ** argument_width - 1) ** self.arity

    def avg_sensitivity(self, seq_len, num_inputs=1024, seed=0):
        if self.sensitivity is not None:
            return float(self.sensitivity)
        width = self.argument_width(seq_len)
        return self.arity * width * (1 - 10.0 ** (-width)) ** (self.arity - 1)


class _BinaryArithmetic(ArithmeticFunction):
    def __init__(self, arity=2, **kwargs):
        if arity != 2:
            raise ValueError(f"{type(self).__name__} requires arity 2.")
        super().__init__(arity=arity, **kwargs)

    def max_result(self, argument_width):
        return 10 ** argument_width - 1


class AbsoluteDifference(_BinaryArithmetic):
    """Unsigned subtraction: abs(a - b), explicitly not signed subtraction."""

    operation = "absolute_difference"

    def evaluate(self, arguments):
        return abs(arguments[0] - arguments[1])


class Subtraction(_BinaryArithmetic):
    operation = "subtraction"

    def evaluate(self, arguments):
        return arguments[0] - arguments[1]

    def required_output_tokens(self, argument_width):
        return super().required_output_tokens(argument_width) + 1


class IntegerDivision(_BinaryArithmetic):
    operation = "integer_division"

    def valid_arguments(self, arguments):
        return arguments[1] != 0

    def evaluate(self, arguments):
        return arguments[0] // arguments[1]


class Remainder(IntegerDivision):
    operation = "remainder"

    def evaluate(self, arguments):
        return arguments[0] % arguments[1]


@dataclass(frozen=True)
class FunctionGroup:
    name: str
    functions: Sequence[ArithmeticFunction]


def _code_width(count: int) -> int:
    return max(1, len(str(count - 1)))


class ArithmeticSelector:
    """Flat decimal selectors or group-prefix + within-group decimal selectors.

    Example: groups [[add, difference], [multiply, divide, remainder]] use
    00, 01, 10, 11, 12. Unassigned codes raise instead of silently producing 0.
    Repeated functions remain distinct selectable tasks and receive unique names.
    """

    def __init__(self, functions: Optional[Sequence[ArithmeticFunction]] = None, *,
                 groups: Optional[Sequence[FunctionGroup]] = None, payload_len: int = 4):
        self.payload_len = _positive_int("payload_len", payload_len)
        if (functions is None) == (groups is None):
            raise ValueError("Supply functions OR groups.")
        self.group_names = []
        self.group_ids = []
        self.codes = []
        if groups is not None:
            groups = list(groups)
            if not groups or any(not group.functions for group in groups):
                raise ValueError("Groups must be nonempty.")
            self.group_names = [group.name for group in groups]
            if len(set(self.group_names)) != len(groups) or any(
                not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in self.group_names
            ):
                raise ValueError("Group names must be unique nonempty metric-safe identifiers.")
            group_width = _code_width(len(groups))
            operation_width = _code_width(max(len(group.functions) for group in groups))
            self.functions = []
            for group_id, group in enumerate(groups):
                for operation_id, function in enumerate(group.functions):
                    self.functions.append(function)
                    self.group_ids.append(group_id)
                    code = f"{group_id:0{group_width}d}{operation_id:0{operation_width}d}"
                    self.codes.append(tuple(map(int, code)))
        else:
            self.functions = list(functions)
            if not self.functions:
                raise ValueError("At least one function is required.")
            width = _code_width(len(self.functions))
            self.codes = [tuple(map(int, f"{index:0{width}d}"))
                          for index in range(len(self.functions))]
        if not all(isinstance(function, ArithmeticFunction) for function in self.functions):
            raise TypeError("Every task must implement ArithmeticFunction.")
        for function in self.functions:
            function.argument_width(payload_len)
        self.names = _unique_metric_names(self.functions)
        # Catch collisions between an explicit name and a generated duplicate suffix.
        if len(set(self.names)) != len(self.names):
            raise ValueError("Task metric names collide; supply distinct explicit names.")
        self.selector_size = len(self.codes[0])
        self.prompt_len = self.selector_size + payload_len

    @property
    def required_output_tokens(self) -> int:
        return max(function.required_output_tokens(function.argument_width(self.payload_len))
                   for function in self.functions)

    def function_ids(self, inputs: torch.Tensor) -> torch.Tensor:
        _digits_tensor(inputs)
        if inputs.size(1) != self.prompt_len:
            raise ValueError(f"Expected {self.prompt_len} selector + argument tokens.")
        codes = inputs.new_tensor(self.codes)
        matches = (inputs[:, None, :self.selector_size] == codes[None]).all(-1)
        if not bool(matches.any(-1).all()):
            raise ValueError("Unknown arithmetic selector code.")
        return matches.long().argmax(-1)

    def inputs_for(self, function_id: int, payloads: torch.Tensor) -> torch.Tensor:
        _digits_tensor(payloads)
        if payloads.size(1) != self.payload_len:
            raise ValueError("Incorrect payload length.")
        if not 0 <= function_id < len(self.functions):
            raise ValueError("Unknown function index.")
        prefix = payloads.new_tensor(self.codes[function_id]).expand(payloads.size(0), -1)
        return torch.cat((prefix, payloads), dim=1)

    def __call__(self, inputs: torch.Tensor) -> list[int]:
        ids = self.function_ids(inputs)
        values = [0] * inputs.size(0)
        for index, function in enumerate(self.functions):
            rows = (ids == index).nonzero(as_tuple=True)[0]
            if rows.numel():
                answers = function(inputs[rows, self.selector_size:])
                for row, answer in zip(rows.cpu().tolist(), answers):
                    values[row] = answer
        return values

    def targets(self, inputs: torch.Tensor, max_output_tokens: int) -> torch.Tensor:
        return encode_answers(self(inputs), max_output_tokens, inputs.device)


def encode_answers(values: Sequence[int], max_output_tokens: int,
                   device: torch.device = torch.device("cpu")) -> torch.Tensor:
    _positive_int("max_output_tokens", max_output_tokens)
    result = torch.full((len(values), max_output_tokens), PAD, dtype=torch.long, device=device)
    for row, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Answers must be integers.")
        digits = ([MINUS] if value < 0 else []) + list(map(int, str(abs(value)))) + [EOS]
        if len(digits) > max_output_tokens:
            raise ValueError("Answer and EOS exceed max_output_tokens; targets are never truncated.")
        result[row, :len(digits)] = result.new_tensor(digits)
    return result


def decode_answer(tokens: Sequence[int]) -> Optional[int]:
    """Return None for missing EOS or a noncanonical decimal answer."""
    tokens = list(tokens)
    if EOS not in tokens:
        return None
    end = tokens.index(EOS)
    digits = tokens[:end]
    negative = bool(digits and digits[0] == MINUS)
    if negative:
        digits = digits[1:]
    if not digits or any(token < 0 or token > 9 for token in digits):
        return None
    if len(digits) > 1 and digits[0] == 0:
        return None
    if any(token != PAD for token in tokens[end + 1:]):
        return None
    value = int("".join(map(str, digits)))
    return None if negative and value == 0 else (-value if negative else value)


def sample_payloads(function: ArithmeticFunction, count: int, payload_len: int, *,
                    generator: Optional[torch.Generator] = None,
                    initial: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Uniform digits conditioned on the task domain, using a CPU RNG."""
    _positive_int("count", count)
    function.argument_width(payload_len)
    payloads = (torch.randint(0, 10, (count, payload_len), generator=generator)
                if initial is None else initial.clone().cpu())
    if payloads.shape != (count, payload_len):
        raise ValueError("Initial payloads have the wrong shape.")
    for _ in range(1000):
        invalid = ~function.valid_mask(payloads)
        if not bool(invalid.any()):
            return payloads
        payloads[invalid] = torch.randint(0, 10, (int(invalid.sum()), payload_len), generator=generator)
    raise RuntimeError("Unable to sample valid arithmetic inputs.")


def sample_batch(selector: ArithmeticSelector, batch_size: int, max_output_tokens: int, *,
                 device: torch.device = torch.device("cpu"),
                 generator: Optional[torch.Generator] = None,
                 selector_probabilities: Optional[Sequence[float]] = None,
                 balanced_selectors: bool = False,
                 held_out: Optional[EvaluationSet] = None) -> tuple[torch.Tensor, torch.Tensor]:
    _positive_int("batch_size", batch_size)
    count = len(selector.functions)
    if selector_probabilities is not None and balanced_selectors:
        raise ValueError("Choose balanced_selectors OR selector_probabilities.")
    if balanced_selectors:
        if batch_size < count:
            raise ValueError("Balanced batch must cover every task.")
        ids = (torch.arange(batch_size) % count)[torch.randperm(batch_size, generator=generator)]
    else:
        probabilities = torch.ones(count) / count if selector_probabilities is None else torch.as_tensor(
            selector_probabilities, dtype=torch.float32)
        if probabilities.shape != (count,) or not bool(torch.isfinite(probabilities).all()) or bool(
            (probabilities < 0).any()
        ) or not bool(torch.isclose(probabilities.sum(), torch.tensor(1.), atol=1e-6, rtol=0)):
            raise ValueError("Selector probabilities must be finite, nonnegative and sum to 1, one per task.")
        ids = torch.multinomial(probabilities, batch_size, replacement=True, generator=generator)
    inputs = torch.empty((batch_size, selector.prompt_len), dtype=torch.long)
    for index, function in enumerate(selector.functions):
        rows = (ids == index).nonzero(as_tuple=True)[0]
        if rows.numel():
            inputs[rows] = selector.inputs_for(index, sample_payloads(
                function, rows.numel(), selector.payload_len, generator=generator))
    if held_out is not None:
        for _ in range(1000):
            overlaps = held_out.overlapping_rows(inputs)
            if not overlaps:
                break
            for index, function in enumerate(selector.functions):
                rows = [row for row in overlaps if int(ids[row]) == index]
                if rows:
                    inputs[rows] = selector.inputs_for(index, sample_payloads(
                        function, len(rows), selector.payload_len, generator=generator))
        else:
            raise RuntimeError("Cannot avoid held-out inputs; reduce the evaluation set.")
    inputs = inputs.to(device)
    return inputs, selector.targets(inputs, max_output_tokens)


def make_evaluation_set(selector: ArithmeticSelector, max_output_tokens: int,
                        num_inputs_per_function: int = 320, seed: int = 0,
                        device: torch.device = torch.device("cpu")) -> EvaluationSet:
    _positive_int("num_inputs_per_function", num_inputs_per_function)
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for index, function in enumerate(selector.functions):
        # Exact enumeration makes small/nearly exhausted domains terminate reliably.
        if selector.payload_len <= 4:
            codes = torch.arange(10 ** selector.payload_len)
            powers = 10 ** torch.arange(selector.payload_len - 1, -1, -1)
            candidates = codes[:, None] // powers % 10
            candidates = candidates[function.valid_mask(candidates)]
            if num_inputs_per_function >= len(candidates):
                raise ValueError("Evaluation must leave at least one valid training input per task.")
            payloads = candidates[torch.randperm(len(candidates), generator=generator)[:num_inputs_per_function]]
        else:
            if num_inputs_per_function >= 10 ** selector.payload_len:
                raise ValueError("Evaluation set exhausts the input domain.")
            # Find one extra valid input to prove training is not exhausted.
            requested = num_inputs_per_function + 1
            unique = {}
            for _ in range(1000):
                candidates = sample_payloads(function, requested - len(unique),
                                             selector.payload_len, generator=generator)
                unique.update((tuple(row), row) for row in candidates.tolist())
                if len(unique) == requested:
                    break
            else:
                raise ValueError("Unable to reserve unique evaluation inputs; reduce its size.")
            payloads = torch.tensor(list(unique.values())[:num_inputs_per_function], dtype=torch.long)
        batches.append(selector.inputs_for(index, payloads))
    inputs = torch.cat(batches).to(device)
    return EvaluationSet(inputs, selector.targets(inputs, max_output_tokens))


@contextmanager
def evaluation_mode(model: nn.Module):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode


class ArithmeticTransformer(TransformerModel):
    """Same embeddings, encoder layers and widths; vocabulary head + prefix mask.

    The prompt is bidirectional as before. Answer positions see the full prompt
    and earlier answer tokens only. No target token can leak into its prediction.
    """

    def __init__(self, d_model=64, num_attn_heads=4, num_layers=4, dropout=0.0,
                 max_len=64):
        super().__init__(d_model, num_attn_heads, num_layers,
                         vocab_size=INPUT_VOCAB_SIZE, dropout=dropout, max_len=max_len)
        self.linear_head = nn.Linear(d_model, OUTPUT_VOCAB_SIZE)

    def forward(self, tokens: torch.Tensor, prompt_len: int = 0) -> torch.Tensor:
        if tokens.ndim != 2 or not 0 <= prompt_len < tokens.size(1):
            raise ValueError("Expected a token batch with at least one decoding position.")
        length = tokens.size(1)
        if length > self.positional_encoding.num_embeddings:
            raise ValueError("Sequence exceeds the model's positional capacity.")
        positions = self.positional_encoding(torch.arange(length, device=tokens.device))
        embeddings = torch.cat((self.embedding(tokens), positions.expand(tokens.size(0), -1, -1)), -1)
        mask = torch.ones((length, length), dtype=torch.bool, device=tokens.device).triu(1)
        mask[:prompt_len, :prompt_len] = False
        return self.linear_head(self.encoder(embeddings, mask=mask))

    def teacher_logits(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tokens = teacher_tokens(inputs, targets)
        return self(tokens, inputs.size(1))[:, inputs.size(1):]

    @torch.no_grad()
    def generate(self, inputs: torch.Tensor, max_output_tokens: int) -> torch.Tensor:
        _positive_int("max_output_tokens", max_output_tokens)
        if inputs.size(1) + max_output_tokens > self.positional_encoding.num_embeddings:
            raise ValueError("Generation limit exceeds the model's positional capacity.")
        result = inputs.new_full((inputs.size(0), max_output_tokens), PAD)
        context = torch.cat((inputs, inputs.new_full((inputs.size(0), 1), ANSWER)), 1)
        finished = torch.zeros(inputs.size(0), device=inputs.device, dtype=torch.bool)
        with evaluation_mode(self):
            for position in range(max_output_tokens):
                token = self(context, inputs.size(1))[:, -1].argmax(-1)
                token = torch.where(finished, token.new_full(token.shape, PAD), token)
                result[:, position] = token
                finished |= token == EOS
                if bool(finished.all()):
                    break
                context = torch.cat((context, token[:, None]), 1)
        return result


def teacher_tokens(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.cat((inputs, inputs.new_full((inputs.size(0), 1), ANSWER), targets[:, :-1]), 1)


class SequenceCrossEntropy(nn.Module):
    """Mean over examples of each answer's mean CE, including EOS, excluding PAD.

    Accept float labels because the reused diagnostics convert their targets to
    the logits' dtype. Token IDs are converted back to long before cross entropy.
    """

    reduction = "mean"

    def per_example(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        mask = targets != PAD
        if not bool(mask.any(-1).all()):
            raise ValueError("Each answer needs at least one non-padding token.")
        losses = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=PAD, reduction="none")
        return losses.sum(-1) / mask.sum(-1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.per_example(logits, targets).mean()


def sequence_metrics(selector: ArithmeticSelector, inputs: torch.Tensor,
                     logits: torch.Tensor, targets: torch.Tensor,
                     generated: Optional[torch.Tensor] = None) -> dict[str, float]:
    losses = SequenceCrossEntropy().per_example(logits, targets)
    valid = targets != PAD
    matches = logits.argmax(-1) == targets
    values = {
        "loss": losses,
        "token_accuracy": (matches & valid).sum(-1) / valid.sum(-1),
        "teacher_forced_accuracy": (matches | ~valid).all(-1).float(),
    }
    if generated is not None:
        values["accuracy"] = (generated == targets).all(-1).float()
    metrics = {key: float(value.mean()) for key, value in values.items()}
    ids = selector.function_ids(inputs)
    for index, name in enumerate(selector.names):
        mask = ids == index
        if bool(mask.any()):
            metrics.update({f"{key}/{name}": float(value[mask].mean()) for key, value in values.items()})
    for group_id, name in enumerate(selector.group_names):
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for index, group in enumerate(selector.group_ids):
            if group == group_id:
                mask |= ids == index
        if bool(mask.any()):
            metrics.update({f"{key}/group/{name}": float(value[mask].mean()) for key, value in values.items()})
    return metrics


def evaluate_fixed_set(model: ArithmeticTransformer, selector: ArithmeticSelector,
                       evaluation_set: EvaluationSet, max_output_tokens: int,
                       batch_size: int = 320) -> dict[str, float]:
    _positive_int("batch_size", batch_size)
    logits, generated = [], []
    with evaluation_mode(model), torch.no_grad():
        for inputs, targets in zip(evaluation_set.inputs.split(batch_size), evaluation_set.labels.split(batch_size)):
            logits.append(model.teacher_logits(inputs, targets))
            generated.append(model.generate(inputs, max_output_tokens))
        return sequence_metrics(selector, evaluation_set.inputs, torch.cat(logits),
                                evaluation_set.labels, torch.cat(generated))
