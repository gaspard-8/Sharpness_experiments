from abc import ABC, abstractmethod
import re
from typing import Optional

import torch as t



class BoolFunction(ABC):
    def __init__(self, negative_value: int = 0, name: Optional[str] = None):
        self.negative_value = negative_value
        self._name = name

    @property
    def metric_name(self) -> str:
        """Stable name used for metric keys."""

        if self._name is not None:
            return self._name
        return re.sub(r"(?<!^)(?=[A-Z])", "_", type(self).__name__).lower()

    @property
    @abstractmethod
    def bool_output(self) -> bool: ...


    @abstractmethod
    def avg_sensitivity(self, seq_len: int) -> float: ...

    @abstractmethod
    def sharpness(self, seq_len: int) -> float: ...

    @abstractmethod
    def __call__(self, x: t.Tensor) -> t.Tensor: ...

    def postprocess(self, x: t.Tensor) -> t.Tensor:
        if self.negative_value != 0:
            return 2*x - 1
        else:
            return x

    def accuracy_mask(
        self,
        outputs: t.Tensor,
        labels: t.Tensor,
        inputs: t.Tensor,
    ) -> Optional[t.Tensor]:
        """Return per-example correctness, or ``None`` when it is undefined.

        Binary functions use the midpoint between their negative and positive
        targets as the decision boundary. Continuous functions should override
        this method when they have a meaningful discrete accuracy definition.
        """

        del inputs
        if not self.bool_output:
            return None

        threshold = 0.5 if self.negative_value == 0 else 0.0
        return (outputs >= threshold).eq(labels >= threshold)

class Parity(BoolFunction):
    bool_output = True
    def avg_sensitivity(self, seq_len: int) -> float:
        return seq_len
    def sharpness(self, seq_len: int) -> float:
        raise NotImplementedError("Sharpness is not implemented for Parity function.")
    
    def __init__(self, negative_value: int = 0, name: Optional[str] = None):
        super().__init__(negative_value=negative_value, name=name)
    def __call__(self, x: t.Tensor) -> t.Tensor:
        return self.postprocess(t.sum(x, dim=-1) % 2)

class Majority(BoolFunction):
    bool_output = True
    def avg_sensitivity(self, seq_len: int) -> float:
            raise NotImplementedError("Average sensitivity is not implemented for Majority function.")
    
    def sharpness(self, seq_len: int) -> float:
            raise NotImplementedError("Sharpness is not implemented for Majority function.")
    
    def __init__(self, negative_value: int = 0, name: Optional[str] = None):
        super().__init__(negative_value=negative_value, name=name)

    def __call__(self, x: t.Tensor) -> t.Tensor:
        return self.postprocess(t.sum(x, dim=-1) > (x.size(-1) // 2)).long()

class Majority_n(BoolFunction):
    bool_output = True

    def __init__(self, n: int, negative_value: int = 0, name: Optional[str] = None):
            super().__init__(negative_value=negative_value, name=name)
            self.n = n

    @property
    def metric_name(self) -> str:
            return self._name or f"Majority_{self.n}"
    
    def avg_sensitivity(self, seq_len: int) -> float:
            raise NotImplementedError("Average sensitivity is not implemented for Majority function.")
    
    def sharpness(self, seq_len: int) -> float:
            raise NotImplementedError("Sharpness is not implemented for Majority function.")
    
    def __init__(self, n: int, negative_value: int = 0, name: Optional[str] = None):
        super().__init__(negative_value=negative_value, name=name)
        self.n = n

    def __call__(self, x: t.Tensor) -> t.Tensor:
        return self.postprocess(t.sum(x[:, :self.n], dim=-1) > (self.n // 2)).long()

class Majority_nm(Majority):
    """Strict majority of bits in the zero-based, half-open interval [n, m)."""

    def __init__(
        self,
        n: int,
        m: int,
        negative_value: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(negative_value=negative_value, name=name)
        if n < 0 or m <= n:
            raise ValueError("Majority_nm requires 0 <= n < m.")
        self.n = n
        self.m = m

    @property
    def metric_name(self) -> str:
        return self._name or f"Majority_{self.n}_{self.m}"

    def __call__(self, x: t.Tensor) -> t.Tensor:
        if x.size(-1) < self.m:
            raise ValueError("Majority_nm requires at least m input bits.")
        return super().__call__(x[..., self.n:self.m])


class Mean(BoolFunction):
    bool_output = False

    def avg_sensitivity(self, seq_len: int) -> float:
        raise NotImplementedError("Average sensitivity is not implemented for Mean function.")
    
    def sharpness(self, seq_len: int) -> float:
        raise NotImplementedError("Sharpness is not implemented for Mean function.")
    
    def __init__(self, negative_value: int = 0, name: Optional[str] = None):
        super().__init__(negative_value=negative_value, name=name)

    def __call__(self, x: t.Tensor) -> t.Tensor:
        return self.postprocess(x.float().mean(dim=-1))

    def accuracy_mask(
        self,
        outputs: t.Tensor,
        labels: t.Tensor,
        inputs: t.Tensor,
    ) -> t.Tensor:
        """Treat a mean prediction as correct when it predicts the right count."""

        payload_len = inputs.size(-1)
        if self.negative_value == 0:
            predicted_mean = outputs.clamp(0.0, 1.0)
            target_mean = labels
        else:
            predicted_mean = ((outputs + 1.0) / 2.0).clamp(0.0, 1.0)
            target_mean = (labels + 1.0) / 2.0

        predicted_ones = (predicted_mean * payload_len).round()
        target_ones = (target_mean * payload_len).round()
        return predicted_ones.eq(target_ones)

class _PairWindowFunction(BoolFunction):
    """Shared bounds and metric names for adjacent-pair functions."""

    bool_output = False

    def __init__(
        self,
        n: int = 0,
        m: Optional[int] = None,
        *,
        negative_value: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(negative_value=negative_value, name=name)
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("Window start n must be a nonnegative integer.")
        if m is not None:
            if isinstance(m, bool) or not isinstance(m, int):
                raise ValueError("Window end m must be an integer or None.")
            if m - n < 2:
                raise ValueError("The window [n, m) must contain at least two input bits.")
        self.n = n
        self.m = m

    @property
    def metric_name(self) -> str:
        base_name = super().metric_name
        if self._name is not None or (self.n == 0 and self.m is None):
            return base_name
        end = self.m if self.m is not None else "end"
        return f"{base_name}_{self.n}_{end}"

    def _window(self, x: t.Tensor) -> t.Tensor:
        if x.ndim == 0:
            raise ValueError(f"{type(self).__name__} requires at least two input bits in its window.")
        end = x.size(-1) if self.m is None else self.m
        if end > x.size(-1):
            raise ValueError(f"{type(self).__name__} requires at least m input bits.")
        if end - self.n < 2:
            raise ValueError(f"{type(self).__name__} requires at least two input bits in its window.")
        return x[..., self.n:end]


class Isordered(_PairWindowFunction):
    """One minus the mean pair penalty: 10 -> 2, all other pairs -> 0.

    Only pairs fully inside x[..., n:m] are scored; m defaults to the end.
    For window length L, return 1 - 2 * count(10) / (L - 1). Ordered
    windows score 1; alternating windows approach 0 as their length grows.
    Outputs are returned without additional postprocessing.
    """

    bool_output = False

    def __init__(self, n: int = 0, m: Optional[int] = None, name: Optional[str] = None):
        super().__init__(n=n, m=m, negative_value=-1, name=name)

    def avg_sensitivity(self, seq_len: int) -> float:
        raise NotImplementedError("Average sensitivity is not implemented for Isordered.")

    def sharpness(self, seq_len: int) -> float:
        raise NotImplementedError("Sharpness is not implemented for Isordered.")

    def __call__(self, x: t.Tensor) -> t.Tensor:
        x = self._window(x)
        unordered = x[..., :-1].eq(1) & x[..., 1:].eq(0)
        return 1.0 - 2.0 * unordered.float().mean(dim=-1)

    def accuracy_mask(
        self,
        outputs: t.Tensor,
        labels: t.Tensor,
        inputs: t.Tensor,
    ) -> t.Tensor:
        """Predict the number of 10 pairs by rounding to its nearest integer.

        Counts range from zero to floor(window_length / 2). Midpoint ties follow
        torch.round's ties-to-even rule on the count.
        """

        window_length = self._window(inputs).size(-1)
        pair_count = window_length - 1
        max_count = window_length // 2
        predicted_count = ((1.0 - outputs) * pair_count / 2.0).clamp(0, max_count)
        target_count = (1.0 - labels) * pair_count / 2.0
        return predicted_count.round().eq(target_count.round())


class Isrepeating(_PairWindowFunction):
    """Fraction of adjacent pairs in x[..., n:m] that repeat.

    Scores are 00/11 -> 1 and 01/10 -> 0; m defaults to the input end.
    Pairs crossing either window boundary are excluded.
    """

    bool_output = False

    def __init__(self, n: int = 0, m: Optional[int] = None, name: Optional[str] = None):
        super().__init__(n=n, m=m, name=name)

    def avg_sensitivity(self, seq_len: int) -> float:
        raise NotImplementedError("Average sensitivity is not implemented for Isrepeating.")

    def sharpness(self, seq_len: int) -> float:
        raise NotImplementedError("Sharpness is not implemented for Isrepeating.")

    def __call__(self, x: t.Tensor) -> t.Tensor:
        x = self._window(x)
        return x[..., :-1].eq(x[..., 1:]).float().mean(dim=-1)

    def accuracy_mask(
        self,
        outputs: t.Tensor,
        labels: t.Tensor,
        inputs: t.Tensor,
    ) -> t.Tensor:
        """Round to the nearest repeating-pair count, with ties to even."""

        pair_count = self._window(inputs).size(-1) - 1
        predicted_count = (outputs.clamp(0.0, 1.0) * pair_count).round()
        target_count = (labels * pair_count).round()
        return predicted_count.eq(target_count)


class First(BoolFunction):
    bool_output = True

    def avg_sensitivity(self, seq_len: int) -> float:
        return 1/seq_len

    def sharpness(self, seq_len: int) -> float:
        raise NotImplementedError("Sharpness is not implemented for First function.")
    
    def __init__(self, negative_value: int = 0, name: Optional[str] = None):
        super().__init__(negative_value=negative_value, name=name)

    def __call__(self, x: t.Tensor) -> t.Tensor:
        return self.postprocess(x[:, 0])

class Parity_n(BoolFunction):
    bool_output = True
    def __init__(
        self,
        n: int,
        bool_output: bool = False,
        negative_value: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(negative_value=negative_value, name=name)
        self.n = n

    @property
    def metric_name(self) -> str:
        return self._name or f"parity_{self.n}"
    def sharpness(self, seq_len: int) -> float:
        raise NotImplementedError("Sharpness is not implemented for Parity_n function.")

    def avg_sensitivity(self, seq_len: int) -> float:
        raise NotImplementedError("Average sensitivity is not implemented for Parity_n function.")
    def __call__(self, x: t.Tensor) -> t.Tensor:
        return self.postprocess(t.sum(x[:, :self.n], dim=-1) % 2)


class Tribe_ws(BoolFunction):
    """OR of s disjoint ANDs, each using w consecutive input bits.

    Only the first w * s bits are used. Additional input bits are ignored.
    Binary accuracy is inherited from BoolFunction for both output encodings.
    """

    bool_output = True

    def __init__(
        self,
        w: int,
        s: int,
        negative_value: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(negative_value=negative_value, name=name)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (w, s)):
            raise ValueError("Tribe_ws requires positive integers w and s.")
        self.w = w
        self.s = s

    @property
    def metric_name(self) -> str:
        return self._name or f"tribe_{self.w}_{self.s}"

    def avg_sensitivity(self, seq_len: int) -> float:
        raise NotImplementedError("Average sensitivity is not implemented for Tribe_ws.")

    def sharpness(self, seq_len: int) -> float:
        raise NotImplementedError("Sharpness is not implemented for Tribe_ws.")

    def __call__(self, x: t.Tensor) -> t.Tensor:
        if x.ndim == 0 or x.size(-1) < self.w * self.s:
            raise ValueError("Tribe_ws requires at least w * s input bits.")
        tribes = x[..., :self.w * self.s].reshape(*x.shape[:-1], self.s, self.w)
        return self.postprocess(tribes.eq(1).all(dim=-1).any(dim=-1).long())


class MeanOfFunctions(BoolFunction):
    """Average all supplied functions on the same input as a single task.

    Each child controls its own output encoding; the mean is returned without
    additional postprocessing. No input bits are consumed as selectors.
    """

    bool_output = False

    def __init__(self, functions: list[BoolFunction], name: Optional[str] = None):
        super().__init__(name=name)
        if not functions:
            raise ValueError("MeanOfFunctions needs at least one function.")
        self.functions = list(functions)

    def avg_sensitivity(self, seq_len: int) -> float:
        raise NotImplementedError("Average sensitivity is not implemented for MeanOfFunctions.")

    def sharpness(self, seq_len: int) -> float:
        raise NotImplementedError("Sharpness is not implemented for MeanOfFunctions.")

    def __call__(self, x: t.Tensor) -> t.Tensor:
        return t.stack([function(x).float() for function in self.functions]).mean(dim=0)

    def accuracy_mask(
        self,
        outputs: t.Tensor,
        labels: t.Tensor,
        inputs: t.Tensor,
    ) -> Optional[t.Tensor]:
        """Round the mean of binary outputs to its nearest discrete level.

        As with Mean, midpoint ties use torch.round's ties-to-even rule.
        Accuracy remains undefined when a child has nonbinary outputs.
        """

        del inputs
        if not all(function.bool_output for function in self.functions):
            return None

        count = len(self.functions)
        signed_count = sum(function.negative_value != 0 for function in self.functions)
        # Shift the sum to start at zero. Fully signed sums advance by two;
        # unsigned or mixed-encoding sums advance by one.
        step = 2 if signed_count == count else 1
        max_level = (count + signed_count) // step
        predicted_level = ((outputs * count + signed_count) / step).clamp(0, max_level)
        target_level = (labels * count + signed_count) / step
        return predicted_level.round().eq(target_level.round())


class SelectorFunction(BoolFunction):
    bool_output = False
    def __init__(
        self,
        functions: list[BoolFunction],
        negative_value: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(negative_value=negative_value, name=name)
        if not functions:
            raise ValueError("SelectorFunction needs at least one function.")
        self.functions = functions
        self.selector_size = (len(functions) - 1).bit_length()

    def sharpness(self, seq_len: int) -> float:
        raise NotImplementedError("Sharpness is not implemented for SelectorFunction.")
    def avg_sensitivity(self, seq_len: int) -> float:
        raise NotImplementedError("Average sensitivity is not implemented for SelectorFunction.")

    def function_ids(self, x: t.Tensor) -> t.Tensor:
        """Return the selected function index for every sequence in ``x``."""
        if self.selector_size == 0:
            return t.zeros(x.size(0), device=x.device, dtype=t.long)

        selector = x[:, :self.selector_size]
        weights = 2 ** t.arange(
            self.selector_size - 1, -1, -1, device=x.device
        )
        return t.sum(selector * weights, dim=-1).long()

    def __call__(self, x: t.Tensor) -> t.Tensor:
        function_id = self.function_ids(x)
        payload = x[:, self.selector_size:]

        result = t.zeros(x.size(0), device=x.device, dtype=t.float32)
        for i, function in enumerate(self.functions):
            mask = (function_id == i)
            if mask.any():
                result[mask] = function(payload[mask]).float()
        return result 
