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
