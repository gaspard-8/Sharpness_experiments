import unittest
from unittest.mock import patch

import torch

from src.functions import Majority, Mean, Parity, Parity_n, SelectorFunction
from src.metrics import average_direction_sharpness, batch_metrics
from src.model import TransformerModel
from src.training import evaluate, sample_batch, train


class CapturingRun:
    def __init__(self) -> None:
        self.logged_metrics: list[dict[str, float]] = []
        self.steps: list[int] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.steps.append(step)
        self.logged_metrics.append(metrics)


class SumOfBitsModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.positional_encoding = torch.nn.Embedding(4, 1)
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        positions = self.positional_encoding.weight[: inputs.size(1)].sum()
        return self.weight * inputs.float().sum(dim=1) + positions


def distinguishing_payloads() -> torch.Tensor:
    """Payloads whose expected labels distinguish all current functions."""

    payloads = torch.zeros((4, 22), dtype=torch.long)
    payloads[0, [0, 5]] = 1  # Parity-5 is 1, full parity is 0.
    payloads[1, [0, 1, 5]] = 1  # Parity-5 is 0, full parity is 1.
    payloads[2, :12] = 1  # Strict majority is 1.
    payloads[3, ::2] = 1  # Mean is 11/22 = 0.5.
    return payloads


class FunctionLabelsTest(unittest.TestCase):
    def test_parity_5_uses_exactly_the_first_five_payload_bits(self) -> None:
        function = Parity_n(5)
        inputs = torch.zeros((22, 22), dtype=torch.long)
        inputs[torch.arange(22), torch.arange(22)] = 1

        labels = function(inputs)
        expected = torch.tensor([1] * 5 + [0] * 17)

        torch.testing.assert_close(labels, expected)

    def test_balanced_selector_labels_match_independent_formulas(self) -> None:
        torch.manual_seed(1234)
        function = SelectorFunction([Parity_n(5), Parity(), Majority(), Mean()])
        inputs, labels = sample_batch(
            function,
            batch_size=400,
            seq_len=24,
            min_seq_len=24,
            balanced_selectors=True,
        )
        function_ids = function.function_ids(inputs)
        payloads = inputs[:, function.selector_size :]

        torch.testing.assert_close(
            torch.bincount(function_ids, minlength=4),
            torch.full((4,), 100, dtype=torch.long),
        )

        expected = torch.empty_like(labels)
        parity_5_mask = function_ids == 0
        parity_mask = function_ids == 1
        majority_mask = function_ids == 2
        mean_mask = function_ids == 3
        expected[parity_5_mask] = (
            payloads[parity_5_mask, :5].sum(dim=-1) % 2
        ).float()
        expected[parity_mask] = (
            payloads[parity_mask].sum(dim=-1) % 2
        ).float()
        expected[majority_mask] = (
            payloads[majority_mask].sum(dim=-1) > payloads.size(1) // 2
        ).float()
        expected[mean_mask] = payloads[mean_mask].float().mean(dim=-1)

        torch.testing.assert_close(labels, expected)

    def test_selector_probabilities_control_function_frequencies(self) -> None:
        torch.manual_seed(1234)
        function = SelectorFunction([Parity(), Majority(), Mean()])
        inputs, _ = sample_batch(
            function,
            batch_size=4000,
            seq_len=6,
            min_seq_len=6,
            selector_probabilities=[0.1, 0.3, 0.6],
        )

        ids = function.function_ids(inputs)
        frequencies = torch.bincount(ids, minlength=3).float() / len(ids)
        torch.testing.assert_close(
            frequencies, torch.tensor([0.1, 0.3, 0.6]), atol=0.03, rtol=0
        )
        self.assertTrue((ids < len(function.functions)).all())

    def test_one_hot_selector_probability_encodes_the_chosen_function(self) -> None:
        function = SelectorFunction([Parity(), Majority(), Mean()])
        inputs, labels = sample_batch(
            function,
            batch_size=128,
            seq_len=6,
            min_seq_len=6,
            selector_probabilities=torch.tensor([0.0, 0.0, 1.0]),
        )

        torch.testing.assert_close(
            inputs[:, : function.selector_size],
            torch.tensor([1, 0]).expand(128, -1),
        )
        torch.testing.assert_close(labels, Mean()(inputs[:, function.selector_size :]))
        self.assertEqual(inputs[:, function.selector_size :].unique().numel(), 2)

    def test_invalid_selector_probabilities_are_rejected(self) -> None:
        function = SelectorFunction([Parity(), Majority(), Mean()])
        for probabilities in ([0.5, 0.5], [0.2, 0.3, 0.4], [0.0, -0.1, 1.1], [float("nan"), 0.0, 1.0]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                sample_batch(function, selector_probabilities=probabilities)

        with self.assertRaises(ValueError):
            sample_batch(function, balanced_selectors=True, selector_probabilities=[0.2, 0.3, 0.5])
        with self.assertRaises(ValueError):
            sample_batch(Parity(), selector_probabilities=[1.0])


class BatchMetricsTest(unittest.TestCase):
    def test_selector_metrics_use_the_selected_functions_names(self) -> None:
        function = SelectorFunction(
            [Parity_n(5), Parity(), Majority(), Mean()]
        )
        selectors = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        payloads = distinguishing_payloads()
        inputs = torch.cat((selectors, payloads), dim=1)
        labels = function(inputs)
        outputs = torch.tensor([0.8, 0.4, 0.2, 0.49])

        metrics = batch_metrics(function, torch.nn.MSELoss(), inputs, outputs, labels)

        self.assertEqual(
            set(metrics),
            {
                "loss",
                "accuracy",
                "loss/parity_5",
                "accuracy/parity_5",
                "loss/parity",
                "accuracy/parity",
                "loss/majority",
                "accuracy/majority",
                "loss/mean",
                "accuracy/mean",
            },
        )
        self.assertEqual(metrics["accuracy/parity_5"], 1.0)
        self.assertEqual(metrics["accuracy/parity"], 0.0)
        self.assertEqual(metrics["accuracy/majority"], 0.0)
        self.assertEqual(metrics["accuracy/mean"], 1.0)
        self.assertEqual(metrics["accuracy"], 0.5)

    def test_custom_metric_names_are_preserved(self) -> None:
        function = SelectorFunction([Parity(name="payload_parity"), Mean()])
        inputs = torch.tensor(
            [
                [0, 1, 0, 1],
                [1, 1, 1, 0],
            ]
        )
        labels = function(inputs)
        outputs = torch.tensor([0.9, 0.6])

        metrics = batch_metrics(function, torch.nn.MSELoss(), inputs, outputs, labels)

        self.assertIn("accuracy/payload_parity", metrics)
        self.assertIn("loss/payload_parity", metrics)

    def test_train_logs_selector_metrics_without_a_custom_callback(self) -> None:
        function = SelectorFunction([Parity(), Mean()])
        model = TransformerModel(
            d_model=8,
            num_attn_heads=2,
            num_layers=1,
            max_len=4,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        run = CapturingRun()

        losses = train(
            model=model,
            function=function,
            criterion=torch.nn.MSELoss(),
            optimizer=optimizer,
            max_len=4,
            min_len=4,
            batch_size=2,
            num_steps=1,
            wandb_run=run,
            balanced_selectors=True,
        )

        self.assertEqual(len(losses), 1)
        self.assertEqual(len(run.logged_metrics), 1)
        self.assertEqual(
            set(run.logged_metrics[0]),
            {
                "loss",
                "accuracy",
                "loss/parity",
                "accuracy/parity",
                "loss/mean",
                "accuracy/mean",
            },
        )

    def test_train_and_evaluate_forward_selector_probabilities(self) -> None:
        function = SelectorFunction([Parity(), Mean()])
        model = TransformerModel(
            d_model=8, num_attn_heads=2, num_layers=1, max_len=4
        )
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        probabilities = [0.0, 1.0]

        with patch("src.training.sample_batch", wraps=sample_batch) as sampled:
            train(
                model, function, criterion, optimizer, max_len=4, min_len=4,
                batch_size=2, num_steps=1, selector_probabilities=probabilities,
            )
            evaluate(
                model, function, criterion, max_len=4, min_len=4,
                batch_size=2, num_steps=1, selector_probabilities=probabilities,
            )

        self.assertEqual(sampled.call_count, 2)
        for call in sampled.call_args_list:
            self.assertIs(call.kwargs["selector_probabilities"], probabilities)


class SharpnessTest(unittest.TestCase):
    def test_selector_sharpness_reuses_function_estimates_for_uniform_global_mean(self) -> None:
        function = SelectorFunction([Parity(), Mean(), Majority()])
        model = SumOfBitsModel()
        model.train()
        original_weight = model.weight.detach().clone()
        original_positions = model.positional_encoding.weight.detach().clone()
        payloads = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])

        with patch("src.metrics.t.randint", return_value=payloads) as sampled, patch(
            "src.metrics.t.randn_like", side_effect=torch.ones_like
        ) as perturbed:
            metrics = average_direction_sharpness(
                model, function, seq_len=4, num_inputs=4,
                num_perturbations=2, rho=2.0,
            )

        self.assertEqual(sampled.call_count, 4)  # one payload sample per code
        self.assertEqual(perturbed.call_count, 2)  # positional encoding is fixed
        self.assertEqual(model.forward_calls, 12)  # baselines + perturbed forwards
        self.assertEqual(metrics, {
            "sharpness/parity": 6.0,
            "sharpness/mean": 18.0,
            "sharpness/majority": 18.0,
            "sharpness/invalid_selectors": 38.0,
            "sharpness": 20.0,
        })
        torch.testing.assert_close(model.weight, original_weight)
        torch.testing.assert_close(model.positional_encoding.weight, original_positions)
        self.assertTrue(model.training)

    def test_real_transformer_sharpness_is_finite_and_restores_eval_mode(self) -> None:
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4)
        model.eval()
        parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

        metrics = average_direction_sharpness(
            model, Parity(), seq_len=4, num_inputs=2,
            num_perturbations=1, rho=0.02,
        )

        self.assertEqual(metrics["sharpness"], metrics["sharpness/parity"])
        self.assertGreaterEqual(metrics["sharpness"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["sharpness"])))
        self.assertFalse(model.training)
        for parameter, before in zip(model.parameters(), parameters_before):
            torch.testing.assert_close(parameter, before)

    def test_train_estimates_sharpness_after_each_interval_only(self) -> None:
        model = SumOfBitsModel()
        run = CapturingRun()
        with patch(
            "src.training.average_direction_sharpness",
            return_value={"sharpness": 1.5, "sharpness/parity": 1.5},
        ) as sharpness:
            train(
                model, Parity(), torch.nn.MSELoss(),
                torch.optim.SGD(model.parameters(), lr=0.001),
                max_len=4, min_len=4, batch_size=2, num_steps=201,
                wandb_run=run, sharpness_interval=100,
                sharpness_num_inputs=7, sharpness_num_perturbations=3,
                sharpness_rho=0.04,
            )

        self.assertEqual(sharpness.call_count, 2)
        self.assertEqual(
            [step for step, metrics in zip(run.steps, run.logged_metrics)
             if "sharpness" in metrics],
            [100, 200],
        )
        self.assertEqual(sharpness.call_args.kwargs["num_inputs"], 7)
        self.assertEqual(sharpness.call_args.kwargs["num_perturbations"], 3)
        self.assertEqual(sharpness.call_args.kwargs["rho"], 0.04)


if __name__ == "__main__":
    unittest.main()
