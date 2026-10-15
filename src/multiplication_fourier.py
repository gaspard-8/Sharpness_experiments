"""Exact product-space Fourier analysis of the hundreds digit of multiplication.

Inputs are six independent uniform decimal symbols, big-endian a then b.
Degree is coordinate support size, as in O'Donnell Chapter 8. Real Helmert
contrasts fix one orthonormal basis; grouped degree masses are basis invariant.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


SPEC = dict(version=1, alphabet_size=10, input_len=6, operands="000_to_999",
            input_order="a_hundreds,a_tens,a_units,b_hundreds,b_tens,b_units",
            output="floor(a*b/100)%10", measure="uniform_product", digit_from_right=3)
SPECTRUM_FILE = "multiplication_hundreds_spectrum.npz"


def alphabet_basis(size=10):
    if not isinstance(size, int) or isinstance(size, bool) or size < 2:
        raise ValueError("Alphabet size must be an integer at least two.")
    basis = np.zeros((size, size), dtype=np.float64)
    basis[0] = 1.
    for row in range(1, size):
        factor = math.sqrt(size / (row * (row + 1)))
        basis[row, :row] = factor
        basis[row, row] = -row * factor
    return basis


def product_transform(values, basis, coordinates, inverse=False):
    result = np.asarray(values, dtype=np.float64)
    size = len(basis)
    if result.shape[:coordinates] != (size,) * coordinates:
        raise ValueError("Product transform dimensions do not match the alphabet.")
    matrix = basis.T if inverse else basis / size
    for axis in range(coordinates):
        result = np.moveaxis(np.tensordot(matrix, result, axes=(1, axis)), 0, axis)
    return result


def support_degrees(size, coordinates):
    degrees = np.zeros((size,) * coordinates, dtype=np.uint8)
    nonconstant = (np.arange(size) != 0).astype(np.uint8)
    for axis in range(coordinates):
        shape = [1] * coordinates
        shape[axis] = size
        degrees += nonconstant.reshape(shape)
    return degrees


def grouped_mass(coefficients, degrees, coordinates):
    squared = np.square(coefficients)
    if squared.ndim == degrees.ndim + 1:
        return np.stack([squared[degrees == degree].sum(axis=0) for degree in range(coordinates + 1)])
    return np.array([squared[degrees == degree].sum() for degree in range(coordinates + 1)])


def digit_table():
    codes = np.arange(1_000_000, dtype=np.int64)
    return (((codes // 1000) * (codes % 1000) // 100) % 10).astype(np.uint8)


def cumulative_projections(values, inputs, coordinates):
    """Exact ANOVA projections at given points, including the constant term.

    Conditional means average omitted coordinates uniformly; a subset Mobius
    transform isolates each orthogonal component. This avoids synthesizing a
    million-point inverse transform separately for every evaluation cutoff.
    """
    values = np.asarray(values, dtype=np.float64)
    inputs = np.asarray(inputs, dtype=np.int64)
    size = values.shape[0]
    if values.shape[:coordinates] != (size,) * coordinates or inputs.ndim != 2 or inputs.shape[1] != coordinates:
        raise ValueError("Invalid product-space values or input shape.")
    if np.any(inputs < 0) or np.any(inputs >= size):
        raise ValueError("Input outside product alphabet.")
    components = []
    for mask in range(1 << coordinates):
        included = [axis for axis in range(coordinates) if mask & (1 << axis)]
        omitted = tuple(axis for axis in range(coordinates) if not mask & (1 << axis))
        conditional = values.mean(axis=omitted) if omitted else values
        if included:
            selected = conditional[tuple(inputs[:, axis] for axis in included)]
        else:
            selected = np.broadcast_to(conditional, (len(inputs),) + conditional.shape)
        components.append(np.array(selected, copy=True))
    components = np.stack(components)
    for axis in range(coordinates):
        for mask in range(1 << coordinates):
            if mask & (1 << axis):
                components[mask] -= components[mask ^ (1 << axis)]
    levels = np.stack([components[[bin(mask).count("1") == degree for mask in range(1 << coordinates)]].sum(axis=0)
                       for degree in range(coordinates + 1)])
    return np.moveaxis(levels.cumsum(axis=0), 0, 1)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_spectrum(target_dir):
    target_dir = Path(target_dir)
    path = target_dir / SPECTRUM_FILE
    manifest_path = target_dir / "manifest.json"
    if path.exists() and manifest_path.exists():
        return load_manifest(target_dir, verify=True)
    if path.exists() or manifest_path.exists():
        raise ValueError("Incomplete existing spectrum; choose a new target directory.")
    target_dir.mkdir(parents=True, exist_ok=True)
    labels = digit_table()
    values = np.eye(10, dtype=np.float64)[labels].reshape((10,) * 6 + (10,))
    basis = alphabet_basis()
    coefficients = product_transform(values, basis, 6)
    reconstructed = product_transform(coefficients, basis, 6, inverse=True)
    reconstruction_error = float(np.abs(reconstructed - values).max())
    del reconstructed
    degrees = support_degrees(10, 6)
    class_mass = grouped_mass(coefficients, degrees, 6)
    indicator_mass = class_mass.sum(axis=1)
    scalar_coefficients = coefficients @ np.arange(10, dtype=np.float64)
    scalar_mass = grouped_mass(scalar_coefficients, degrees, 6)
    if reconstruction_error > 1e-10 or not math.isclose(float(indicator_mass.sum()), 1., abs_tol=1e-10):
        raise ValueError("Categorical Fourier reconstruction or Parseval verification failed.")
    scalar_energy = float(np.square(labels.astype(np.float64)).mean())
    if not math.isclose(float(scalar_mass.sum()), scalar_energy, abs_tol=1e-10):
        raise ValueError("Scalar digit Fourier Parseval verification failed.")
    probabilities = np.bincount(labels, minlength=10) / len(labels)
    manifest = {
        "definition": SPEC, "basis": "uniform_Helmert_real_orthonormal_constant_first",
        "basis_formula": "B_0(x)=1; B_j(x)=sqrt(10/(j*(j+1))) for x<j, -j*sqrt(10/(j*(j+1))) for x=j, 0 otherwise",
        "degree_definition": "number_of_nonconstant_input_coordinates_in_multiindex",
        "reference": "https://www.cs.cmu.edu/~odonnell/papers/Analysis-of-Boolean-Functions-by-Ryan-ODonnell.pdf#page=208",
        "domain_size": len(labels), "class_probabilities": probabilities.tolist(),
        "indicator_degree_mass": indicator_mass.tolist(), "per_class_degree_mass": class_mass.tolist(),
        "indicator_variance": float(indicator_mass[1:].sum()),
        "indicator_fraction_of_variance_by_degree": (indicator_mass[1:] / indicator_mass[1:].sum()).tolist(),
        "indicator_tail_mass_above_degree": [float(indicator_mass[k + 1:].sum()) for k in range(7)],
        "scalar_mean": float(labels.mean()), "scalar_variance": float(labels.astype(np.float64).var()),
        "scalar_degree_mass": scalar_mass.tolist(),
        "scalar_fraction_of_variance_by_degree": (scalar_mass[1:] / scalar_mass[1:].sum()).tolist(),
        "scalar_tail_mass_above_degree": [float(scalar_mass[k + 1:].sum()) for k in range(7)],
        "indicator_variance_normalized_average_degree": float(np.arange(1, 7) @ indicator_mass[1:] / indicator_mass[1:].sum()),
        "scalar_variance_normalized_average_degree": float(np.arange(1, 7) @ scalar_mass[1:] / scalar_mass[1:].sum()),
        "reconstruction_max_absolute_error": reconstruction_error, "spectrum_file": SPECTRUM_FILE,
    }
    np.savez_compressed(path, definition=json.dumps(SPEC, sort_keys=True), basis=basis,
                        labels=labels, indicator_coefficients=coefficients, scalar_coefficients=scalar_coefficients)
    manifest["spectrum_sha256"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(target_dir, verify=False):
    manifest_path = Path(target_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Prepare the multiplication spectrum with sweep_multiplication_digit.py --prepare first.")
    manifest = json.loads(manifest_path.read_text())
    if manifest["definition"] != SPEC:
        raise ValueError("Prepared multiplication target definition mismatch.")
    path = Path(target_dir) / manifest["spectrum_file"]
    if not path.is_file() or (verify and sha256_file(path) != manifest["spectrum_sha256"]):
        raise ValueError("Multiplication spectrum is missing or its checksum does not match.")
    return manifest


def prepare_evaluation(target_dir, count=1024, seed=23):
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count < 500_000:
        raise ValueError("Evaluation count must be positive and below 500000.")
    manifest = load_manifest(target_dir)
    path = Path(target_dir) / f"evaluation-n{count}-seed{seed}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data["spectrum_sha256"]) != manifest["spectrum_sha256"]:
                raise ValueError("Prepared evaluation belongs to a different multiplication spectrum.")
            return {key: data[key] for key in ("codes", "inputs", "labels", "indicator_targets", "scalar_targets")}
    codes = np.random.default_rng(seed).choice(1_000_000, count, replace=False)
    powers = 10 ** np.arange(5, -1, -1)
    inputs = (codes[:, None] // powers) % 10
    with np.load(Path(target_dir) / manifest["spectrum_file"], allow_pickle=False) as data:
        labels = data["labels"]
    values = np.eye(10, dtype=np.float64)[labels].reshape((10,) * 6 + (10,))
    cumulative = cumulative_projections(values, inputs, 6)[:, 1:]
    expected = np.eye(10)[labels[codes]]
    if not np.allclose(cumulative[:, -1], expected, atol=1e-12, rtol=0):
        raise ValueError("Full-degree evaluation projection does not reproduce the target.")
    cumulative[:, -1] = expected
    scalar = cumulative @ np.arange(10, dtype=np.float64)
    payload = dict(codes=codes, inputs=inputs, labels=labels[codes].astype(np.int64),
                   indicator_targets=cumulative, scalar_targets=scalar)
    np.savez_compressed(path, **payload, spectrum_sha256=manifest["spectrum_sha256"])
    return payload
