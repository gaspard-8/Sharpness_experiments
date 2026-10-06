"""Principal-angle overlaps of discovered parameter direction spans."""

from __future__ import annotations

from typing import Any, Callable

import torch


def _orient(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.shape[1]:
        pivots = vectors.abs().argmax(dim=0)
        signs = vectors[pivots, torch.arange(vectors.shape[1], device=vectors.device)].sign()
        vectors = vectors * signs
    return vectors


def _principal_overlap(first: torch.Tensor, second: torch.Tensor, cosine: float) -> dict[str, Any]:
    """Inputs are orthonormal bases; return symmetric unit midpoint directions.

    For principal vectors a and b with a.T b = c, (a+b)/sqrt(2+2c)
    is equally close to both spans. Distinct retained midpoints are orthogonal.
    c=1 is an exact common direction; c>=cosine is an approximate overlap.
    """
    if not first.shape[1] or not second.shape[1]:
        return {"directions": first.new_empty((first.shape[0], 0)).float(),
                "cosines": first.new_empty(0), "principal_cosines": first.new_empty(0)}
    left, cosines, right = torch.linalg.svd(first.T @ second, full_matrices=False)
    cosines = cosines.clamp(0, 1)
    selected = cosines >= cosine
    shared = first @ left[:, selected] + second @ right.T[:, selected]
    shared = shared / torch.sqrt(2 + 2 * cosines[selected])
    return {"directions": _orient(shared).float(), "cosines": cosines[selected],
            "principal_cosines": cosines}


def pairwise_overlaps(
    tasks: dict[str, dict[str, Any]],
    intersection_cosine: float,
    *,
    loss_changes: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
    epsilon: float | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute both same-kind overlaps once per unordered task pair.

    Geometric counts describe spans. Optional finite-loss checks separately
    identify shared basis directions that still satisfy the kind for BOTH tasks.
    This uses no Hessian-vector products. Without a model, checks stay unknown.
    """
    if not 0 < intersection_cosine <= 1:
        raise ValueError("intersection_cosine must be in (0, 1].")
    if loss_changes is not None and epsilon is None:
        raise ValueError("Loss verification requires epsilon.")
    names = list(tasks)
    overlaps = {}
    for kind in ("sharpness", "tangent"):
        # Normalize each span once; float32 export/restart drift should not
        # affect the principal angles or change them under basis rotations.
        bases = {
            name: torch.linalg.qr(tasks[name][kind].double(), mode="reduced").Q
            for name in names
        }
        pairs = {}
        for index, first in enumerate(names):
            for other_index in range(index + 1, len(names)):
                second = names[other_index]
                result = _principal_overlap(bases[first], bases[second], intersection_cosine)
                result.update({"task_a": first, "task_b": second,
                               "loss_changes_a": None, "loss_changes_b": None,
                               "loss_verified": None})
                if loss_changes is not None:
                    directions = result["directions"]
                    changes = [
                        torch.stack([loss_changes(task_index, vector.double()) for vector in directions.T])
                        if directions.shape[1] else bases[first].new_empty((0, 2))
                        for task_index in (index, other_index)
                    ]
                    if not all(bool(torch.isfinite(value).all()) for value in changes):
                        raise ValueError("Non-finite perturbed loss in overlap verification.")
                    harms = [value.amax(dim=1) for value in changes]
                    verified = ((harms[0] > epsilon) & (harms[1] > epsilon) if kind == "sharpness"
                                else (harms[0] <= epsilon) & (harms[1] <= epsilon))
                    result.update({"loss_changes_a": changes[0].float(),
                                   "loss_changes_b": changes[1].float(), "loss_verified": verified})
                pairs[f"{first}_and_{second}"] = result
                if progress is not None:
                    progress({"stage": "overlap_pair_complete", "kind": kind,
                              "task_a": first, "task_b": second,
                              "overlap_dim": result["directions"].shape[1]})
        overlaps[kind] = pairs
    return overlaps
