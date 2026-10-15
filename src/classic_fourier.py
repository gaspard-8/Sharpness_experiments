"""Exact Walsh spectra for natural functions; training always uses the full source.

Subset mask bit i denotes input position i. Dense truth tables use the same
least-significant-bit convention. Mean has an analytic sparse degree-1 spectrum.
"""

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from src.fourier import positive_integer
from src.functions import BoolFunction, Majority, Mean, Tribe_ws, WeightedVoting


CLASSIC_FUNCTION_NAMES = ("classic_majority", "classic_weighted_majority", "classic_tribes", "classic_mean")


def fwht(values):
    """Unnormalized Walsh-Hadamard transform; returns a new float64 array."""
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 1 or result.size < 1 or result.size & (result.size - 1):
        raise ValueError("FWHT requires a nonempty vector of power-of-two length.")
    width = 1
    while width < result.size:
        blocks = result.reshape(-1, 2 * width)
        left, right = blocks[:, :width].copy(), blocks[:, width:].copy()
        blocks[:, :width], blocks[:, width:] = left + right, left - right
        width *= 2
    return result


def definition(name, input_len, seed=17):
    positive_integer("input_len", input_len)
    if name not in CLASSIC_FUNCTION_NAMES:
        raise ValueError(f"Unknown classic function: {name}")
    if input_len > (32 if name == "classic_mean" else 20):
        raise ValueError("Exact classic spectra support at most 20 bits, or 32 for analytic mean.")
    parameters = {}
    if name == "classic_weighted_majority":
        weights = list(range(1, input_len + 1))
        random.Random(seed).shuffle(weights)
        parameters = {"weights": weights, "threshold": 0, "ties": "negative"}
    elif name == "classic_tribes":
        if input_len % 3:
            raise ValueError("Classic tribes requires an input length divisible by 3.")
        parameters = {"width": 3, "groups": input_len // 3}
    elif name == "classic_majority":
        parameters = {"threshold": input_len // 2, "ties": "negative"}
    return dict(version=1, name=name, input_len=input_len, seed=seed,
                output_encoding="natural_0_to_1", parameters=parameters)


def source_function(spec):
    name, length, params = spec["name"], spec["input_len"], spec["parameters"]
    if name == "classic_majority":
        return Majority()
    if name == "classic_weighted_majority":
        return WeightedVoting(length, params["weights"])
    if name == "classic_tribes":
        return Tribe_ws(params["width"], params["groups"])
    return Mean()


def popcounts(masks, input_len):
    counts = np.zeros(len(masks), dtype=np.int64)
    for bit in range(input_len):
        counts += ((masks >> bit) & 1).astype(np.int64)
    return counts


def spectrum_digest(spec, masks, coefficients):
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    digest.update(np.asarray(masks, dtype="<u8").tobytes())
    digest.update(np.asarray(coefficients, dtype="<f8").tobytes())
    return digest.hexdigest()


def spectrum_path(target_dir, spec):
    return Path(target_dir) / f"{spec['name']}-n{spec['input_len']}-seed{spec['seed']}.npz"


class ClassicFourierFunction(BoolFunction):
    """A full natural target with exact cumulative projections only through degree 8."""

    separate_full_target = True

    def __init__(self, spec, masks, coefficients, max_degree=8):
        positive_integer("max_degree", max_degree)
        if max_degree > min(8, spec["input_len"]):
            raise ValueError("Classic evaluation max-degree must not exceed 8 or input length.")
        super().__init__(name=spec["name"])
        self.spec = spec
        self.input_len = spec["input_len"]
        self.max_degree = max_degree
        self.source = source_function(spec)
        self.masks = np.asarray(masks, dtype=np.uint64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        expected_masks = (np.array([0] + [1 << bit for bit in range(self.input_len)], dtype=np.uint64)
                          if spec["name"] == "classic_mean" else np.arange(2 ** self.input_len, dtype=np.uint64))
        if not np.array_equal(self.masks, expected_masks) or self.coefficients.shape != self.masks.shape:
            raise ValueError("Spectrum masks or coefficient shape do not match the classic definition.")
        if not np.isfinite(self.coefficients).all():
            raise ValueError("Classic spectrum contains nonfinite coefficients.")
        self.degrees = popcounts(self.masks, self.input_len)
        self.degree_mass = np.bincount(self.degrees, weights=self.coefficients ** 2,
                                       minlength=self.input_len + 1)
        self.variance = float(self.degree_mass[1:].sum())
        self.squared_flip_sensitivity = float(4 * np.arange(self.input_len + 1) @ self.degree_mass)
        self._cumulative_table = None

    @property
    def bool_output(self):
        return self.source.bool_output

    def __call__(self, inputs):
        self._validate_inputs(inputs)
        return self.source(inputs).float()

    def _validate_inputs(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != self.input_len:
            raise ValueError(f"Classic inputs must have shape (batch, {self.input_len}).")

    def cumulative(self, inputs, dtype=torch.float32):
        self._validate_inputs(inputs)
        if self.spec["name"] == "classic_mean":
            # This is the exact mathematical mean, including its constant 1/2.
            return inputs.to(dtype=dtype).mean(dim=1, keepdim=True).expand(-1, self.max_degree).clone()
        if self._cumulative_table is None:
            table = np.empty((len(self.coefficients), self.max_degree), dtype=np.float64)
            running = np.full(len(self.coefficients), self.coefficients[0])
            for degree in range(1, self.max_degree + 1):
                running += fwht(np.where(self.degrees == degree, self.coefficients, 0.))
                table[:, degree - 1] = running
            self._cumulative_table = table
        values = inputs.detach().cpu().numpy().astype(np.uint64)
        codes = (values * (np.uint64(1) << np.arange(self.input_len, dtype=np.uint64))).sum(axis=1).astype(np.intp)
        return torch.from_numpy(self._cumulative_table[codes]).to(device=inputs.device, dtype=dtype)

    def avg_sensitivity(self, seq_len):
        if seq_len != self.input_len or not self.bool_output:
            raise ValueError("Boolean average sensitivity requires a binary target at its defined length.")
        return self.squared_flip_sensitivity

    def sharpness(self, seq_len):
        raise NotImplementedError("Use the model output-sharpness diagnostic.")

    def metadata(self):
        active = self.coefficients != 0
        return {
            "name": self.metric_name, "definition": self.spec,
            "input_len": self.input_len, "max_degree": self.max_degree,
            "full_target_degree": int(self.degrees[active].max()),
            "training_target": "full_natural_function_without_truncation_or_rescaling",
            "basis": "chi_S(x)=product(1-2*x_i)", "subset_mask_convention": "bit_i_is_input_position_i",
            "measure": "uniform_independent_binary_bits", "mean": float(self.coefficients[0]),
            "squared_l2_norm": float(self.degree_mass.sum()), "variance": self.variance,
            "degree_mass_includes_constant": True, "degree_mass": self.degree_mass.tolist(),
            "tail_mass_above_degree": [float(self.degree_mass[k + 1:].sum()) for k in range(self.input_len + 1)],
            "squared_flip_sensitivity": self.squared_flip_sensitivity,
            "average_sensitivity": self.squared_flip_sensitivity if self.bool_output else None,
            "variance_normalized_average_degree": self.squared_flip_sensitivity / (4 * self.variance),
            "per_input_squared_flip_sensitivity": [
                float(4 * (self.coefficients[((self.masks >> bit) & 1).astype(bool)] ** 2).sum())
                for bit in range(self.input_len)],
            "computation": "analytic_degree_1" if self.spec["name"] == "classic_mean" else "exact_full_cube_float64_FWHT",
            "nonzero_coefficients": int(active.sum()), "coefficient_file": "target_spectrum.npz",
            "spectrum_sha256": spectrum_digest(self.spec, self.masks, self.coefficients),
        }

    def save_spectrum(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f"Refusing to replace an existing spectrum: {path}")
        np.savez_compressed(path, definition=json.dumps(self.spec, sort_keys=True), masks=self.masks,
                            coefficients=self.coefficients,
                            sha256=spectrum_digest(self.spec, self.masks, self.coefficients))


def compute_spectrum(spec, max_degree=8):
    length = spec["input_len"]
    if spec["name"] == "classic_mean":
        masks = np.array([0] + [1 << bit for bit in range(length)], dtype=np.uint64)
        coefficients = np.array([.5] + [-.5 / length] * length, dtype=np.float64)
    else:
        masks = np.arange(2 ** length, dtype=np.uint64)
        values = np.empty(len(masks), dtype=np.float64)
        source = source_function(spec)
        shifts = np.arange(length, dtype=np.uint64)
        for start in range(0, len(masks), 4096):
            codes = masks[start:start + 4096]
            inputs = torch.from_numpy(((codes[:, None] >> shifts) & 1).astype(np.int64))
            values[start:start + len(codes)] = source(inputs).numpy()
        coefficients = fwht(values) / len(values)
        if not np.allclose(fwht(coefficients), values, atol=1e-12, rtol=0):
            raise ValueError("Classic Fourier reconstruction failed.")
        if not math.isclose(float((coefficients ** 2).sum()), float((values ** 2).mean()), abs_tol=1e-12):
            raise ValueError("Classic Fourier Parseval verification failed.")
    return ClassicFourierFunction(spec, masks, coefficients, max_degree)


def make_classic_function(name, input_len, max_degree=8, *, seed=17, target_dir=None):
    spec = definition(name, input_len, seed)
    if target_dir is None:
        return compute_spectrum(spec, max_degree)
    path = spectrum_path(target_dir, spec)
    if not path.is_file():
        raise ValueError(f"Missing prepared classic spectrum: {path}. Run sweep_classic_fourier.py --prepare first.")
    with np.load(path, allow_pickle=False) as data:
        stored_spec = json.loads(str(data["definition"]))
        masks, coefficients = data["masks"], data["coefficients"]
        if stored_spec != spec or str(data["sha256"]) != spectrum_digest(spec, masks, coefficients):
            raise ValueError(f"Classic spectrum definition or checksum mismatch: {path}")
    return ClassicFourierFunction(spec, masks, coefficients, max_degree)


def prepare_classic_function(name, input_len, target_dir, *, max_degree=8, seed=17):
    spec = definition(name, input_len, seed)
    path = spectrum_path(target_dir, spec)
    if path.exists():
        return make_classic_function(name, input_len, max_degree, seed=seed, target_dir=target_dir)
    function = compute_spectrum(spec, max_degree)
    function.save_spectrum(path)
    return function
