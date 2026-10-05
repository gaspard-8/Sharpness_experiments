import copy
import math
import unittest
from unittest.mock import patch

import torch

from src.functions import First, Parity, SelectorFunction
from src.metrics import gradient_metrics
from src.model import TransformerModel
from src.training import train


class LinearProbeModel(torch.nn.Module):
    """Two gradient coordinates, one in the positional embedding."""

    def __init__(self, features: torch.Tensor, selector_size: int = 2):
        super().__init__()
        self.register_buffer("features", features)
        self.selector_size = selector_size
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.positional_encoding = torch.nn.Embedding(1, 1)
        torch.nn.init.zeros_(self.positional_encoding.weight)
        self.unused = torch.nn.Parameter(torch.tensor(5.0))
        self.frozen = torch.nn.Parameter(torch.tensor(3.0), requires_grad=False)
        self.seen_inputs = []

    def forward(self, inputs):
        self.seen_inputs.append(inputs.detach().clone())
        ids = torch.zeros(inputs.size(0), dtype=torch.long, device=inputs.device)
        for index in range(self.selector_size):
            ids = 2 * ids + inputs[:, index]
        features = self.features[ids]
        if features.ndim == 3:
            features = features[torch.arange(inputs.size(0)), inputs[:, -1]]
        return self.weight * features[:, 0] + self.positional_encoding.weight[0, 0] * features[:, 1]


class CapturingRun:
    def __init__(self):
        self.logged = []

    def log(self, metrics, step):
        self.logged.append((step, metrics))


class GradientMetricsTest(unittest.TestCase):
    def test_analytic_alignment_and_norms_include_positions_and_duplicate_tasks(self):
        model = LinearProbeModel(torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [1., 0.]]))
        function = SelectorFunction([First()] * 4)
        payloads = torch.ones((3, 1), dtype=torch.long)
        model.train()
        model.positional_encoding.eval()
        original_modes = [module.training for module in model.modules()]
        before = copy.deepcopy(model.state_dict())
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 7.0)
        previous_grads = [parameter.grad for parameter in model.parameters()]

        with torch.no_grad():
            metrics = gradient_metrics(model, function, torch.nn.MSELoss(), payloads)

        self.assertEqual(metrics, {
            "gradient_norm/first_0": 2.0,
            "gradient_norm/first_1": 2.0,
            "gradient_norm/first_2": 2.0,
            "gradient_norm/first_3": 2.0,
            "gradient_alignment/first_0_vs_first_1": -1.0,
            "gradient_alignment/first_0_vs_first_2": 0.0,
            "gradient_alignment/first_0_vs_first_3": 1.0,
            "gradient_alignment/first_1_vs_first_2": 0.0,
            "gradient_alignment/first_1_vs_first_3": -1.0,
            "gradient_alignment/first_2_vs_first_3": 0.0,
        })
        for code, inputs in enumerate(model.seen_inputs):
            torch.testing.assert_close(inputs[:, 2:], payloads)
            torch.testing.assert_close(function.function_ids(inputs), torch.full((3,), code))
        self.assertEqual([module.training for module in model.modules()], original_modes)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        for parameter, previous in zip(model.parameters(), previous_grads):
            self.assertIs(parameter.grad, previous)
            if previous is not None:
                torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7.0))

    def test_cosine_of_mean_gradients_and_loss_reduction_are_batch_size_independent(self):
        # Per-example gradients: task 0 = (2, 0), (0, 4);
        # task 1 = (4, 0), (0, 2). Mean gradients have cosine 4/5,
        # whereas averaging the two per-example cosines would give 1.
        model = LinearProbeModel(
            torch.tensor([[[1., 0.], [0., -2.]], [[2., 0.], [0., -1.]]]),
            selector_size=1,
        )
        function = SelectorFunction([First(negative_value=-1)] * 2)
        for reduction in ("mean", "sum", "none"):
            for repeats in (1, 3):
                with self.subTest(reduction=reduction, repeats=repeats):
                    metrics = gradient_metrics(
                        model, function, torch.nn.MSELoss(reduction=reduction),
                        torch.tensor([[0], [1]]).repeat(repeats, 1),
                    )
                    self.assertAlmostEqual(metrics["gradient_norm/first_0"], math.sqrt(5), places=6)
                    self.assertAlmostEqual(metrics["gradient_norm/first_1"], math.sqrt(5), places=6)
                    self.assertAlmostEqual(metrics["gradient_alignment/first_0_vs_first_1"], 0.8, places=6)

    def test_zero_and_tiny_gradients_have_undefined_cosine(self):
        function = SelectorFunction([First(), Parity()])
        for scale in (0.0, 1e-15):
            with self.subTest(scale=scale):
                model = LinearProbeModel(torch.tensor([[scale, 0.], [1., 0.]]), selector_size=1)
                metrics = gradient_metrics(model, function, torch.nn.MSELoss(), torch.ones((2, 1), dtype=torch.long))
                self.assertLessEqual(metrics["gradient_norm/first"], 1e-12)
                self.assertEqual(metrics["gradient_norm/parity"], 2.0)
                self.assertTrue(math.isnan(metrics["gradient_alignment/first_vs_parity"]))

    def test_single_task_and_unused_selector_codes(self):
        for function, selector_size, expected_count in (
            (First(name="custom"), 0, 1),
            (SelectorFunction([First(name="custom")]), 0, 1),
            (SelectorFunction([First()] * 3), 2, 6),
        ):
            with self.subTest(function=function):
                model = LinearProbeModel(torch.ones((4, 2)), selector_size=selector_size)
                metrics = gradient_metrics(model, function, torch.nn.MSELoss(), torch.ones((2, 1), dtype=torch.long))
                self.assertEqual(len(metrics), expected_count)
                self.assertEqual(len(model.seen_inputs), 3 if selector_size else 1)
                self.assertFalse(any("invalid" in key for key in metrics))
                if not selector_size:
                    self.assertEqual(set(metrics), {"gradient_norm/custom"})

    def test_real_transformer_is_repeatable_and_leaves_gradients_unset(self):
        torch.manual_seed(5)
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4, dropout=0.3)
        model.train()
        function = SelectorFunction([First(), Parity()])
        payloads = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0]])
        rng_before = torch.get_rng_state().clone()

        first = gradient_metrics(model, function, torch.nn.MSELoss(), payloads)
        second = gradient_metrics(model, function, torch.nn.MSELoss(), payloads)

        self.assertEqual(first, second)
        self.assertTrue(all(math.isfinite(value) for value in first.values()))
        self.assertGreater(first["gradient_norm/first"], 0)
        self.assertLessEqual(abs(first["gradient_alignment/first_vs_parity"]), 1)
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        torch.testing.assert_close(torch.get_rng_state(), rng_before)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_transformer_metrics_match_cpu(self):
        torch.manual_seed(5)
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4)
        mps_model = copy.deepcopy(model).to("mps")
        function = SelectorFunction([First(), Parity()])
        payloads = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0]])

        expected = gradient_metrics(model, function, torch.nn.MSELoss(), payloads)
        actual = gradient_metrics(mps_model, function, torch.nn.MSELoss(), payloads.to("mps"))

        self.assertEqual(set(actual), set(expected))
        for key in expected:
            with self.subTest(metric=key):
                self.assertTrue(math.isfinite(actual[key]))
                torch.testing.assert_close(actual[key], expected[key], rtol=1e-4, atol=1e-5)
        self.assertTrue(mps_model.training)
        self.assertTrue(all(parameter.grad is None for parameter in mps_model.parameters()))

    def test_model_modes_are_restored_on_failure(self):
        model = LinearProbeModel(torch.ones((4, 2)))
        model.train()
        model.positional_encoding.eval()
        modes = [module.training for module in model.modules()]
        with patch.object(model, "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                gradient_metrics(model, SelectorFunction([First()] * 4), torch.nn.MSELoss(), torch.ones((2, 1), dtype=torch.long))
        self.assertEqual([module.training for module in model.modules()], modes)


class GradientLoggingTest(unittest.TestCase):
    def test_logging_uses_fixed_shared_probes_after_updates(self):
        model = LinearProbeModel(torch.ones((4, 2)))
        run = CapturingRun()
        probe_weights = []

        def capture(*args):
            probe_weights.append(float(args[0].weight.detach()))
            return {"gradient_norm/first_0": 1.0}

        with patch("src.training.gradient_metrics", side_effect=capture) as measured:
            train(
                model, SelectorFunction([First()] * 4), torch.nn.MSELoss(),
                torch.optim.SGD(model.parameters(), lr=0.01),
                max_len=4, min_len=4, batch_size=8, num_steps=5,
                wandb_run=run, gradient_interval=2, gradient_num_inputs=7,
                gradient_seed=13, selector_probabilities=[0., 0., 0., 1.],
            )
        self.assertEqual([step for step, metrics in run.logged if "gradient_norm/first_0" in metrics], [2, 4])
        self.assertEqual(measured.call_count, 2)
        probes = [call.args[3] for call in measured.call_args_list]
        self.assertIs(probes[0], probes[1])
        torch.testing.assert_close(probes[0], torch.randint(0, 2, (7, 2), generator=torch.Generator().manual_seed(13)))
        self.assertTrue(all(weight > 0 for weight in probe_weights))
        self.assertNotEqual(probe_weights[0], probe_weights[1])

    def test_metrics_do_not_change_sgd_training_or_randomness(self):
        torch.manual_seed(8)
        original = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4, dropout=0.2)
        final_models = []
        final_rngs = []
        losses = []
        run = CapturingRun()
        for logger in (None, run):
            model = copy.deepcopy(original)
            torch.manual_seed(9)
            losses.append(train(
                model, SelectorFunction([First(), Parity()]), torch.nn.MSELoss(),
                torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.7),
                max_len=4, min_len=4, batch_size=4, num_steps=3,
                wandb_run=logger, gradient_interval=1, gradient_num_inputs=4,
                gradient_seed=11,
            ))
            final_models.append(model)
            final_rngs.append(torch.get_rng_state())
        self.assertEqual(losses[0], losses[1])
        torch.testing.assert_close(final_rngs[0], final_rngs[1])
        for a, b in zip(final_models[0].parameters(), final_models[1].parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for _, metrics in run.logged:
            self.assertTrue(math.isfinite(metrics["gradient_alignment/first_vs_parity"]))


if __name__ == "__main__":
    unittest.main()
