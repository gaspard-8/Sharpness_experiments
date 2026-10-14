"""Sparse, real-valued Walsh-Fourier targets on uniform binary inputs.

chi_S(x) = product_{i in S} (1 - 2*x_i). Fourier mass means the SUM of
squared coefficients at a degree, not the magnitude of each coefficient.
All targets have zero mean and unit squared L2 norm under the uniform measure.
"""

from itertools import combinations
import math
import random

import torch

from src.functions import BoolFunction


FUNCTION_NAMES = (
    "concentrated_powerlaw", "concentrated_geometric",
    "sparse_powerlaw", "sparse_geometric",
)


def positive_integer(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


class FourierFunction(BoolFunction):
    """One specified Fourier polynomial; no selector tokens or output threshold.

    Concentrated targets use one nested subset {0, ..., d-1} at each degree.
    Sparse targets use a reproducible random selection of distinct subsets,
    equal coefficient magnitudes within a degree, and independent random signs.
    The two layouts have identical per-degree mass for the same spectrum.
    """

    bool_output = False

    def __init__(self, input_len=32, max_degree=None, *,
                 layout="concentrated", spectrum="powerlaw", sparse_terms=16,
                 power=2.0, geometric_decay=0.9, seed=17):
        positive_integer("input_len", input_len)
        max_degree = input_len if max_degree is None else max_degree
        positive_integer("max_degree", max_degree)
        positive_integer("sparse_terms", sparse_terms)
        if max_degree > input_len:
            raise ValueError("max_degree cannot exceed input_len.")
        if layout not in ("concentrated", "sparse"):
            raise ValueError("layout must be concentrated or sparse.")
        if spectrum not in ("powerlaw", "geometric"):
            raise ValueError("spectrum must be powerlaw or geometric.")
        if not math.isfinite(power) or power <= 0:
            raise ValueError("power must be finite and positive.")
        if not 0 < geometric_decay < 1:
            raise ValueError("geometric_decay must be strictly between 0 and 1.")
        super().__init__(name=f"{layout}_{spectrum}")
        self.input_len = input_len
        self.max_degree = max_degree
        self.layout = layout
        self.spectrum = spectrum
        self.sparse_terms = sparse_terms
        self.power = float(power)
        self.geometric_decay = float(geometric_decay)
        self.seed = seed
        raw_mass = [degree ** (-power) if spectrum == "powerlaw"
                    else geometric_decay ** (degree - 1)
                    for degree in range(1, max_degree + 1)]
        total = math.fsum(raw_mass)
        self.degree_mass = tuple(mass / total for mass in raw_mass)
        if any(mass <= 0 for mass in self.degree_mass) or any(
                lower <= upper for lower, upper in zip(self.degree_mass, self.degree_mass[1:])):
            raise ValueError("Spectrum must retain positive, strictly decreasing mass at every degree; use a milder decay.")
        # Separate Python RNG: support generation does not alter training RNGs.
        generator = random.Random(seed)
        subsets, coefficients, degrees, offsets = [], [], [], [0]
        for degree, mass in enumerate(self.degree_mass, start=1):
            count = 1 if layout == "concentrated" else min(sparse_terms, math.comb(input_len, degree))
            if layout == "concentrated":
                support = [tuple(range(degree))]
            elif math.comb(input_len, degree) <= sparse_terms:
                support = list(combinations(range(input_len), degree))
            else:
                chosen = set()
                while len(chosen) < count:
                    chosen.add(tuple(sorted(generator.sample(range(input_len), degree))))
                support = sorted(chosen)
            for subset in support:
                sign = generator.choice((-1, 1)) if layout == "sparse" else 1
                subsets.append(subset)
                coefficients.append(sign * math.sqrt(mass / count))
                degrees.append(degree)
            offsets.append(len(subsets))
        self.subsets = tuple(subsets)
        self.coefficients = tuple(coefficients)
        self.degrees = tuple(degrees)
        self.offsets = tuple(offsets)
        # epsilon_k = ||f - f_{<=k}||_2^2, including epsilon_0 = 1.
        self.tail_mass = tuple(math.fsum(self.degree_mass[k:]) for k in range(max_degree + 1))
        self._device_tensors = {}

    def _tensors(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._device_tensors:
            incidence = torch.zeros(self.input_len, len(self.subsets), dtype=dtype)
            for column, subset in enumerate(self.subsets):
                incidence[list(subset), column] = 1
            self._device_tensors[key] = (
                incidence.to(device), torch.tensor(self.coefficients, device=device, dtype=dtype),
            )
        return self._device_tensors[key]

    def basis(self, inputs, dtype=torch.float32):
        if inputs.ndim != 2 or inputs.size(1) != self.input_len:
            raise ValueError(f"Fourier inputs must have shape (batch, {self.input_len}).")
        incidence, _ = self._tensors(inputs.device, dtype)
        # Binary dot products count selected ones exactly at these lengths.
        return 1 - 2 * torch.remainder(inputs.to(dtype) @ incidence, 2)

    def components(self, inputs, dtype=torch.float32):
        """Columns are exact-degree target components, from degree 1 to K."""
        basis = self.basis(inputs, dtype)
        _, coefficients = self._tensors(inputs.device, dtype)
        terms = basis * coefficients
        return torch.stack([terms[:, start:end].sum(dim=1)
                            for start, end in zip(self.offsets, self.offsets[1:])], dim=1)

    def cumulative(self, inputs, dtype=torch.float32):
        """Column k-1 is f_{<=k}(x), without renormalizing the truncation."""
        return self.components(inputs, dtype).cumsum(dim=1)

    def input_cumulative(self, inputs, dtype=torch.float32):
        """E[f(X) | the first k inputs in input_order], for k=1..input_len.

        Under uniform independent bits, retain exactly the characters whose
        complete support is contained in the available input positions.
        """
        order = getattr(self, "input_order", tuple(range(self.input_len)))
        ranks = {position: rank for rank, position in enumerate(order, start=1)}
        last_ranks = torch.tensor([max(ranks[position] for position in subset)
                                  for subset in self.subsets], device=inputs.device)
        mask = (last_ranks[:, None] <= torch.arange(1, self.input_len + 1, device=inputs.device)).to(dtype)
        _, coefficients = self._tensors(inputs.device, dtype)
        return (self.basis(inputs, dtype) * coefficients) @ mask

    def approximation(self, inputs, degree):
        if not isinstance(degree, int) or isinstance(degree, bool) or not 0 <= degree <= self.max_degree:
            raise ValueError("approximation degree must be between 0 and max_degree.")
        if degree == 0:
            # Still validate the input shape.
            self.basis(inputs)
            return torch.zeros(inputs.size(0), device=inputs.device)
        return self.cumulative(inputs)[:, degree - 1]

    def approximation_degree(self, epsilon):
        """Minimum k whose squared L2 tail mass is at most epsilon."""
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative.")
        return next(k for k, tail in enumerate(self.tail_mass) if tail <= epsilon)

    def __call__(self, inputs):
        return self.cumulative(inputs)[:, -1]

    def avg_sensitivity(self, seq_len):
        raise NotImplementedError("Boolean sensitivity is not defined for this real-valued target.")

    def sharpness(self, seq_len):
        raise NotImplementedError("Use the model output-sharpness diagnostic for this target.")

    def metadata(self):
        return {
            "name": self.metric_name, "basis": "chi_S(x)=product(1-2*x_i)",
            "measure": "uniform_independent_binary_bits", "input_len": self.input_len,
            "max_degree": self.max_degree, "layout": self.layout, "spectrum": self.spectrum,
            "sparse_terms": self.sparse_terms, "power": self.power,
            "geometric_decay": self.geometric_decay, "seed": self.seed,
            "mean": 0.0, "squared_l2_norm": 1.0,
            "degree_mass": list(self.degree_mass), "tail_mass_above_degree": list(self.tail_mass),
            "terms_per_degree": [end - start for start, end in zip(self.offsets, self.offsets[1:])],
            "terms": [{"subset": list(subset), "coefficient": coefficient}
                      for subset, coefficient in zip(self.subsets, self.coefficients)],
        }


def make_function(name, **kwargs):
    if name not in FUNCTION_NAMES:
        raise ValueError(f"Unknown Fourier function {name!r}; choose from {FUNCTION_NAMES}.")
    layout, spectrum = name.split("_", maxsplit=1)
    return FourierFunction(layout=layout, spectrum=spectrum, **kwargs)
