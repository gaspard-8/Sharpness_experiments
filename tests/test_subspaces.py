import math
import unittest

import torch

from src.functions import Parity, SelectorFunction
from src.spaces import loss_spaces
from src.subspaces import pairwise_overlaps
from tests.test_spaces import AxisModel, MeanOutputLoss, DiagonalLossModel


class PairwiseOverlapsTest(unittest.TestCase):
    def test_distinct_sharp_axes_and_common_tangent_axis(self):
        metrics, artifact = loss_spaces(
            AxisModel(), SelectorFunction([Parity(), Parity()]), MeanOutputLoss(),
            torch.tensor([[0], [1]]), delta=1., epsilon=.5, num_directions=3,
        )
        key = "parity_0_and_parity_1"
        self.assertEqual(metrics[f"spaces/sharpness_overlap_dim/{key}"], 0)
        self.assertEqual(metrics[f"spaces/tangent_overlap_dim/{key}"], 1)
        self.assertEqual(metrics[f"spaces/tangent_overlap_verified_dim/{key}"], 1)
        pair = artifact["overlaps"]["tangent"][key]
        torch.testing.assert_close(pair["directions"].abs(), torch.tensor([[0.], [0.], [1.]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(pair["loss_changes_a"], torch.zeros(1, 2))
        torch.testing.assert_close(pair["loss_changes_b"], torch.zeros(1, 2))
        self.assertEqual(len(artifact["overlaps"]["tangent"]), 1)
        self.assertEqual(metrics["spaces/interference_dim/parity_0_to_parity_1"], 1)
        self.assertEqual(metrics["spaces/hvp_count/parity_0"], 3)

    def test_identical_task_spans_and_joint_loss_checks(self):
        metrics, artifact = loss_spaces(
            DiagonalLossModel(torch.tensor([0., 0., 2., 6.])),
            SelectorFunction([Parity(), Parity()]), MeanOutputLoss(),
            torch.tensor([[0], [1]]), delta=1., epsilon=.1, num_directions=4,
        )
        key = "parity_0_and_parity_1"
        for kind in ("sharpness", "tangent"):
            self.assertEqual(metrics[f"spaces/{kind}_overlap_dim/{key}"], 2)
            self.assertEqual(metrics[f"spaces/{kind}_overlap_verified_dim/{key}"], 2)
            pair = artifact["overlaps"][kind][key]
            torch.testing.assert_close(pair["directions"].T @ pair["directions"], torch.eye(2), atol=1e-6, rtol=0)
            self.assertTrue(bool(pair["loss_verified"].all()))

    def test_principal_overlap_is_symmetric_and_rotation_invariant(self):
        eye = torch.eye(4, dtype=torch.float64)
        first = eye[:, :2]
        angle = .02
        second = torch.stack((math.cos(angle) * eye[:, 0] + math.sin(angle) * eye[:, 2], eye[:, 1]), dim=1)
        rotation = torch.tensor([[.6, -.8], [.8, .6]], dtype=torch.float64)

        def calculate(a, b):
            tasks = {"a": {"sharpness": a, "tangent": a}, "b": {"sharpness": b, "tangent": b}}
            return pairwise_overlaps(tasks, .999)["sharpness"]["a_and_b"]

        base = calculate(first, second)
        self.assertEqual(base["directions"].shape, (4, 2))
        for value in (calculate(second, first), calculate(first @ rotation, second @ rotation.T)):
            torch.testing.assert_close(value["cosines"], base["cosines"])
            torch.testing.assert_close(value["directions"] @ value["directions"].T,
                                       base["directions"] @ base["directions"].T, atol=1e-6, rtol=0)
        v = base["directions"].double()
        torch.testing.assert_close(v.T @ v, torch.eye(2, dtype=torch.float64), atol=1e-6, rtol=0)
        torch.testing.assert_close((first.T @ v).square().sum(0), (second.T @ v).square().sum(0), atol=1e-6, rtol=0)
        self.assertIsNone(base["loss_verified"])

    def test_geometric_overlap_can_fail_joint_finite_loss_check(self):
        axis = torch.tensor([[1.], [0.]])
        tasks = {name: {"sharpness": axis, "tangent": axis} for name in ("a", "b")}

        def changes(index, vector):
            return torch.tensor([.2, .2]) if index == 0 else torch.tensor([0., 0.])

        shared = pairwise_overlaps(tasks, .999, loss_changes=changes, epsilon=.1)
        for kind in ("sharpness", "tangent"):
            pair = shared[kind]["a_and_b"]
            self.assertEqual(pair["directions"].shape[1], 1)
            self.assertFalse(bool(pair["loss_verified"].any()))


if __name__ == "__main__":
    unittest.main()
