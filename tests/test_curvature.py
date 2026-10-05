import copy
import math
import unittest
from unittest.mock import patch

import torch

from src.functions import First, Parity, SelectorFunction
from src.metrics import multi_task_curvature
from src.model import TransformerModel
from src.training import train


def snapshot(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }


class TwoTaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.positional_encoding = torch.nn.Embedding(1, 1, dtype=torch.float64)
        torch.nn.init.constant_(self.positional_encoding.weight, 2.0)
        self.unused = torch.nn.Parameter(torch.tensor(7.0, dtype=torch.float64))
        self.frozen = torch.nn.Parameter(torch.tensor(9.0, dtype=torch.float64), requires_grad=False)

    def forward(self, inputs):
        return self.weight + inputs[:, 0] * self.positional_encoding.weight[0, 0]


class PolynomialModel(torch.nn.Module):
    def __init__(self, value=1.0, power=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
        self.power = power

    def forward(self, inputs):
        return self.weight.pow(self.power).expand(inputs.size(0))


class CapturingRun:
    def __init__(self):
        self.logged = []

    def log(self, metrics, step):
        self.logged.append((step, metrics))


class CurvatureTest(unittest.TestCase):
    def test_task_sum_and_fixed_initial_direction_include_learned_task_curvature(self):
        # L_A=(w-1)^2 and L_B=(w+p-1)^2 at (w,p)=(1,2).
        # A is learned, but its Hessian still contributes to the total:
        # g=(4,4), H=[[4,2],[2,2]], and g^T H g=160.
        model = TwoTaskModel()
        before = snapshot(model)
        with torch.no_grad():
            model.weight.fill_(1.4)
            model.positional_encoding.weight.fill_(1.9)
        function = SelectorFunction([First(), First()])

        for reduction in ("mean", "sum", "none"):
            with self.subTest(reduction=reduction):
                result = multi_task_curvature(
                    model, function, torch.nn.MSELoss(reduction=reduction),
                    torch.ones((7, 1), dtype=torch.long), before,
                )
                self.assertAlmostEqual(result["curvature/multi_task"], 160.0, places=10)

    def test_quadrature_integrates_the_step_path_instead_of_only_one_endpoint(self):
        # L=w^4, w:1->3, g(1)=4. Integral = 16*12*integral(1+2a)^2 da = 832.
        model = PolynomialModel()
        before = snapshot(model)
        with torch.no_grad():
            model.weight.fill_(3.0)
        payloads = torch.zeros((4, 1), dtype=torch.long)
        for count, expected in ((1, 768.0), (3, 832.0), (5, 832.0)):
            with self.subTest(num_points=count):
                result = multi_task_curvature(
                    model, First(), torch.nn.MSELoss(), payloads, before, num_points=count,
                )
                self.assertAlmostEqual(result["curvature/multi_task"], expected, places=9)

    def test_negative_curvature_is_not_clamped(self):
        # L=(w^2-1)^2, w:1/4->1/2, g(1/4)=-15/16;
        # integral Hessian = integral(12w^2-4) = -9/4.
        model = PolynomialModel(value=0.25)
        before = snapshot(model)
        with torch.no_grad():
            model.weight.fill_(0.5)
        result = multi_task_curvature(
            model, First(), torch.nn.MSELoss(), torch.ones((3, 1), dtype=torch.long), before,
        )
        self.assertAlmostEqual(result["curvature/multi_task"], -2025 / 1024, places=10)

    def test_zero_total_gradient_gives_zero_even_along_a_curved_path(self):
        model = PolynomialModel(value=0.0)
        before = snapshot(model)
        with torch.no_grad():
            model.weight.fill_(2.0)
        result = multi_task_curvature(
            model, First(), torch.nn.MSELoss(), torch.zeros((3, 1), dtype=torch.long), before,
        )
        self.assertEqual(result, {"curvature/multi_task": 0.0})

    def test_linear_loss_has_zero_hessian(self):
        model = PolynomialModel(power=1)
        result = multi_task_curvature(
            model, First(), lambda outputs, labels: outputs.mean(),
            torch.zeros((3, 1), dtype=torch.long), snapshot(model),
        )
        self.assertEqual(result, {"curvature/multi_task": 0.0})

    def test_model_state_gradients_rng_and_attention_settings_are_preserved(self):
        model = TwoTaskModel()
        model.train()
        model.positional_encoding.eval()
        modes = [module.training for module in model.modules()]
        before = snapshot(model)
        state = copy.deepcopy(model.state_dict())
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 7)
        grad_refs = [parameter.grad for parameter in model.parameters()]
        rng = torch.get_rng_state().clone()
        attention_settings = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
        with torch.no_grad():
            multi_task_curvature(
                model, SelectorFunction([First(), First()]), torch.nn.MSELoss(),
                torch.ones((2, 1), dtype=torch.long), before,
            )
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, state[name])
        for parameter, previous in zip(model.parameters(), grad_refs):
            self.assertIs(parameter.grad, previous)
            if previous is not None:
                torch.testing.assert_close(previous, torch.full_like(previous, 7))
        self.assertEqual([module.training for module in model.modules()], modes)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertEqual(attention_settings, (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        ))

    def test_modes_and_parameters_are_restored_after_failure(self):
        model = TwoTaskModel()
        before = snapshot(model)
        model.train()
        model.positional_encoding.eval()
        modes = [module.training for module in model.modules()]
        parameter_refs = list(model.parameters())
        with patch.object(model, "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                multi_task_curvature(
                    model, First(), torch.nn.MSELoss(), torch.ones((2, 1), dtype=torch.long), before,
                )
        self.assertEqual([module.training for module in model.modules()], modes)
        for parameter, original in zip(model.parameters(), parameter_refs):
            self.assertIs(parameter, original)

    def test_invalid_snapshot_and_quadrature_count_are_rejected(self):
        model = TwoTaskModel()
        args = (model, First(), torch.nn.MSELoss(), torch.ones((2, 1), dtype=torch.long))
        for count in (0, -1, 1.5, True):
            with self.subTest(num_points=count), self.assertRaises(ValueError):
                multi_task_curvature(*args, snapshot(model), num_points=count)
        with self.assertRaisesRegex(ValueError, "every trainable parameter"):
            multi_task_curvature(*args, {})
        wrong = snapshot(model)
        wrong["weight"] = wrong["weight"].float()
        with self.assertRaisesRegex(ValueError, "dtype mismatch"):
            multi_task_curvature(*args, wrong)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_transformer_curvature_matches_cpu(self):
        torch.manual_seed(9)
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4)
        before = snapshot(model)
        inputs = torch.tensor([[0, 0, 1, 0], [1, 1, 1, 0], [1, 0, 0, 1]])
        function = SelectorFunction([First(), Parity()])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        torch.nn.MSELoss()(model(inputs), function(inputs)).backward()
        optimizer.step()
        payloads = inputs[:, 1:]
        expected = multi_task_curvature(model, function, torch.nn.MSELoss(), payloads, before)
        mps_model = copy.deepcopy(model).to("mps")
        actual = multi_task_curvature(
            mps_model, function, torch.nn.MSELoss(), payloads.to("mps"),
            {name: value.to("mps") for name, value in before.items()},
        )
        self.assertTrue(math.isfinite(actual["curvature/multi_task"]))
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


class CurvatureLoggingTest(unittest.TestCase):
    def test_independent_interval_captures_the_actual_step_and_reuses_fixed_probes(self):
        model = PolynomialModel(power=1).float()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        run = CapturingRun()
        endpoints = []
        original_step = optimizer.step

        def step(*args, **kwargs):
            previous = snapshot(model)
            result = original_step(*args, **kwargs)
            endpoints.append((previous, snapshot(model)))
            return result

        measured_steps = []

        def measure(*args, **kwargs):
            measured_steps.append((args[4], snapshot(args[0])))
            return {"curvature/multi_task": 123.0}

        with patch.object(optimizer, "step", side_effect=step), patch(
            "src.training.multi_task_curvature", side_effect=measure,
        ) as curvature, patch("src.training.gradient_metrics", return_value={}) as gradients:
            train(
                model, First(), torch.nn.MSELoss(), optimizer,
                max_len=4, min_len=4, batch_size=4, num_steps=5, wandb_run=run,
                gradient_interval=3, gradient_num_inputs=7, gradient_seed=13,
                curvature_interval=2, curvature_num_points=5,
            )
        self.assertEqual([step for step, metrics in run.logged if "curvature/multi_task" in metrics], [2, 4])
        torch.testing.assert_close(measured_steps, [endpoints[1], endpoints[3]], rtol=0, atol=0)
        probes = curvature.call_args_list[0].args[3]
        self.assertIs(probes, curvature.call_args_list[1].args[3])
        self.assertIs(probes, gradients.call_args.args[3])
        torch.testing.assert_close(probes, torch.randint(0, 2, (7, 4), generator=torch.Generator().manual_seed(13)))
        self.assertEqual(curvature.call_args.kwargs["num_points"], 5)

    def test_curvature_does_not_change_adamw_training_or_optimizer_state(self):
        torch.manual_seed(8)
        original = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4, dropout=0.2)
        results = []
        run = CapturingRun()
        for logger in (None, run):
            model = copy.deepcopy(original)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            torch.manual_seed(9)
            losses = train(
                model, SelectorFunction([First(), Parity()]), torch.nn.MSELoss(), optimizer,
                max_len=4, min_len=4, batch_size=4, num_steps=3, wandb_run=logger,
                gradient_interval=1, gradient_num_inputs=4, gradient_seed=11,
                curvature_interval=1, curvature_num_points=2,
            )
            results.append((losses, model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
        self.assertTrue(all(math.isfinite(metrics["curvature/multi_task"]) for _, metrics in run.logged))


if __name__ == "__main__":
    unittest.main()
