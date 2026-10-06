import copy
import math
import unittest
from unittest.mock import patch

import torch

from src.functions import First, SelectorFunction
from src.metrics import task_affinity_metrics
from src.model import TransformerModel
from src.training import train


class AffinityProbeModel(torch.nn.Module):
    def __init__(self, features, selector_size=2, offsets=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.positional_encoding = torch.nn.Embedding(1, 1, dtype=torch.float64)
        torch.nn.init.zeros_(self.positional_encoding.weight)
        self.unused = torch.nn.Parameter(torch.tensor(7., dtype=torch.float64))
        self.frozen = torch.nn.Parameter(torch.tensor(3., dtype=torch.float64), requires_grad=False)
        self.register_buffer("features", torch.tensor(features, dtype=torch.float64))
        self.register_buffer("offsets", torch.zeros(len(features), dtype=torch.float64)
                             if offsets is None else torch.tensor(offsets, dtype=torch.float64))
        self.selector_size = selector_size
        self.points, self.seen_codes = [], []

    def forward(self, inputs):
        ids = torch.zeros(inputs.size(0), dtype=torch.long, device=inputs.device)
        for index in range(self.selector_size):
            ids = ids * 2 + inputs[:, index]
        self.seen_codes.extend(ids.tolist())
        self.points.append(torch.stack([self.weight, self.positional_encoding.weight[0, 0]]).detach().clone())
        return (self.weight * self.features[ids, 0]
                + self.positional_encoding.weight[0, 0] * self.features[ids, 1]
                + self.offsets[ids])


class CapturingRun:
    def __init__(self):
        self.logged = []

    def log(self, metrics, step):
        self.logged.append((step, metrics))


class TaskAffinityTest(unittest.TestCase):
    def test_exact_directed_matrix_and_equal_movement_include_position_parameters(self):
        model = AffinityProbeModel([[1, 0], [-2, 0], [0, 3], [2, 0]])
        function = SelectorFunction([First()] * 4)
        probes = torch.ones((3, 1), dtype=torch.long)
        metrics = task_affinity_metrics(model, function, torch.nn.MSELoss(), probes, probes,
                                        radius=.1, eval_batch_size=5)
        expected = [[.19, -.44, 0, .36], [-.21, .36, 0, -.44],
                    [0, 0, .51, 0], [.19, -.44, 0, .36]]
        for i, row in enumerate(expected):
            for j, value in enumerate(row):
                self.assertAlmostEqual(metrics[f"affinity/first_{i}_to_first_{j}"], value, places=12)
        self.assertEqual([metrics[f"affinity/source_gradient_norm/first_{i}"] for i in range(4)],
                         [2., 4., 6., 4.])
        moved_points = [point for point in model.points if point.norm() > 0]
        self.assertTrue(moved_points)
        for point in moved_points:
            self.assertAlmostEqual(float(point.norm()), .1, places=12)

    def test_learned_target_can_be_harmed_and_zero_source_is_undefined(self):
        model = AffinityProbeModel([[1, 0], [1, 0]], selector_size=1, offsets=[0, 1])
        probes = torch.ones((3, 1), dtype=torch.long)
        metrics = task_affinity_metrics(model, SelectorFunction([First()] * 2),
                                        torch.nn.MSELoss(), probes, probes, radius=.1)
        self.assertEqual(metrics["affinity/baseline_loss/first_1"], 0.)
        self.assertAlmostEqual(metrics["affinity/first_0_to_first_1"], -.01, places=12)
        self.assertEqual(metrics["affinity/source_valid/first_1"], 0.)
        self.assertTrue(math.isnan(metrics["affinity/first_1_to_first_0"]))
        self.assertTrue(math.isnan(metrics["affinity/first_1_to_first_1"]))

    def test_target_evaluation_is_separate_and_reductions_and_chunks_preserve_mean_loss(self):
        # Gradient probes ask for output 1, but target probes ask for 0.
        # Moving towards the source labels must therefore harm target loss.
        for reduction in ("mean", "sum", "none"):
            for batch_size in (1, 5, 100):
                with self.subTest(reduction=reduction, batch_size=batch_size):
                    model = AffinityProbeModel([[1, 0]] * 3)
                    metrics = task_affinity_metrics(
                        model, SelectorFunction([First()] * 3), torch.nn.MSELoss(reduction=reduction),
                        torch.ones((2, 1), dtype=torch.long), torch.zeros((7, 1), dtype=torch.long),
                        radius=.1, eval_batch_size=batch_size,
                    )
                    for i in range(3):
                        for j in range(3):
                            self.assertAlmostEqual(metrics[f"affinity/first_{i}_to_first_{j}"], -.01, places=12)
                    self.assertEqual(set(model.seen_codes), {0, 1, 2})

    def test_tiny_and_nonfinite_source_gradients_are_not_normalized(self):
        for scale in (0., 1e-15, float("nan"), float("inf")):
            with self.subTest(scale=scale):
                model = AffinityProbeModel([[scale, 0], [1, 0]], selector_size=1)
                probes = torch.ones((2, 1), dtype=torch.long)
                metrics = task_affinity_metrics(model, SelectorFunction([First()] * 2),
                                                torch.nn.MSELoss(), probes, probes)
                self.assertEqual(metrics["affinity/source_valid/first_0"], 0.)
                self.assertTrue(math.isnan(metrics["affinity/first_0_to_first_1"]))
                self.assertEqual(metrics["affinity/source_valid/first_1"], 1.)

    def test_single_task_without_selector_and_frozen_or_unused_parameters(self):
        for function in (First(name="custom"), SelectorFunction([First(name="custom")])):
            model = AffinityProbeModel([[1, 0]], selector_size=0)
            probes = torch.ones((3, 1), dtype=torch.long)
            metrics = task_affinity_metrics(model, function, torch.nn.MSELoss(), probes, probes, radius=.1)
            self.assertAlmostEqual(metrics["affinity/custom_to_custom"], .19, places=12)
            self.assertEqual(float(model.frozen), 3.)
            self.assertEqual(float(model.unused.detach()), 7.)

    def test_state_gradients_and_modes_survive_success_and_perturbed_forward_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                model = AffinityProbeModel([[1, 0]] * 2, selector_size=1)
                model.train()
                model.positional_encoding.eval()
                modes = [module.training for module in model.modules()]
                state = copy.deepcopy(model.state_dict())
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.grad = torch.full_like(parameter, 9.)
                previous_gradients = [parameter.grad for parameter in model.parameters()]
                probes = torch.ones((2, 1), dtype=torch.long)
                original_forward = model.forward

                def forward(inputs):
                    if fail and model.weight.detach().abs() > 0:
                        raise RuntimeError("perturbed forward failure")
                    return original_forward(inputs)

                with patch.object(model, "forward", side_effect=forward), torch.no_grad():
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "perturbed forward failure"):
                            task_affinity_metrics(model, SelectorFunction([First()] * 2),
                                                  torch.nn.MSELoss(), probes, probes)
                    else:
                        task_affinity_metrics(model, SelectorFunction([First()] * 2),
                                              torch.nn.MSELoss(), probes, probes)
                self.assertEqual([module.training for module in model.modules()], modes)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, state[name], rtol=0, atol=0)
                for parameter, previous in zip(model.parameters(), previous_gradients):
                    self.assertIs(parameter.grad, previous)
                    if previous is not None:
                        torch.testing.assert_close(previous, torch.full_like(previous, 9.))

    def test_real_transformer_repeatability_and_rng_preservation(self):
        torch.manual_seed(5)
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4, dropout=.3)
        payloads = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 1, 1]])
        rng_before = torch.get_rng_state().clone()
        state = copy.deepcopy(model.state_dict())
        results = [task_affinity_metrics(model, SelectorFunction([First()] * 2),
                                        torch.nn.MSELoss(), payloads, payloads, eval_batch_size=4)
                   for _ in range(2)]
        for name, value in results[0].items():
            if name != "affinity/compute_seconds":
                self.assertEqual(value, results[1][name])
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_invalid_configuration(self):
        model = AffinityProbeModel([[1, 0]], selector_size=0)
        payloads = torch.ones((2, 1), dtype=torch.long)
        for kwargs in ({"radius": 0}, {"radius": -1}, {"radius": float("nan")},
                       {"eval_batch_size": 0}, {"eval_batch_size": True}, {"norm_epsilon": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                task_affinity_metrics(model, First(), torch.nn.MSELoss(), payloads, payloads, **kwargs)


class TaskAffinityLoggingTest(unittest.TestCase):
    def test_default_interval_matrix_orientation_and_fixed_independent_probes(self):
        model = AffinityProbeModel([[1, 0]] * 2, selector_size=1).float()
        run = CapturingRun()
        weights = []

        def capture(*args, **kwargs):
            weights.append(float(args[0].weight.detach()))
            return {"affinity/first_0_to_first_0": .1, "affinity/first_0_to_first_1": -.2,
                    "affinity/first_1_to_first_0": .3, "affinity/first_1_to_first_1": .4}

        with patch("src.training.task_affinity_metrics", side_effect=capture) as measured:
            train(model, SelectorFunction([First()] * 2), torch.nn.MSELoss(),
                  torch.optim.SGD(model.parameters(), lr=.01), max_len=4, min_len=4,
                  batch_size=4, num_steps=201, wandb_run=run, affinity_num_inputs=7,
                  affinity_seed=13, gradient_interval=1000, curvature_interval=1000,
                  sharpness_interval=1000)
        self.assertEqual([step for step, metrics in run.logged if "affinity/matrix" in metrics], [100, 200])
        self.assertEqual(measured.call_count, 2)
        first, second = measured.call_args_list
        expected = torch.randint(0, 2, (2, 7, 3), generator=torch.Generator().manual_seed(13))
        torch.testing.assert_close(first.args[3], expected[0])
        torch.testing.assert_close(first.args[4], expected[1])
        # Views share the same fixed underlying probes across checkpoints.
        self.assertEqual(first.args[3].data_ptr(), second.args[3].data_ptr())
        self.assertEqual(first.args[4].data_ptr(), second.args[4].data_ptr())
        self.assertNotEqual(first.args[3].data_ptr(), first.args[4].data_ptr())
        self.assertGreater(weights[0], 0.)
        for step, metrics in run.logged:
            if "affinity/matrix" in metrics:
                table = metrics["affinity/matrix"]
                self.assertEqual(table.columns, ["source / target", "first_0", "first_1"])
                self.assertEqual(table.data, [["first_0", .1, -.2], ["first_1", .3, .4]])

    def test_adamw_training_and_optimizer_state_are_unchanged(self):
        torch.manual_seed(6)
        initial = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4, dropout=.2)
        outcomes = []
        for interval in (None, 1):
            model = copy.deepcopy(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            torch.manual_seed(7)
            losses = train(model, SelectorFunction([First()] * 2), torch.nn.MSELoss(), optimizer,
                           max_len=4, min_len=4, batch_size=4, num_steps=3,
                           wandb_run=CapturingRun(), affinity_interval=interval,
                           affinity_num_inputs=3, affinity_seed=11)
            outcomes.append((losses, model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        for name, value in outcomes[0][1].items():
            torch.testing.assert_close(value, outcomes[1][1][name], rtol=0, atol=0)
        for parameter_id, state in outcomes[0][2]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, outcomes[1][2]["state"][parameter_id][name], rtol=0, atol=0)
        self.assertEqual(outcomes[0][2]["param_groups"], outcomes[1][2]["param_groups"])
        torch.testing.assert_close(outcomes[0][3], outcomes[1][3], rtol=0, atol=0)

    def test_no_logger_or_disabled_interval_skips_measurement(self):
        for logger, interval in ((None, 1), (CapturingRun(), None)):
            model = AffinityProbeModel([[1, 0]], selector_size=0).float()
            with patch("src.training.task_affinity_metrics") as measured:
                train(model, First(), torch.nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=.01),
                      max_len=2, min_len=2, batch_size=2, num_steps=2,
                      wandb_run=logger, affinity_interval=interval)
            measured.assert_not_called()


if __name__ == "__main__":
    unittest.main()
