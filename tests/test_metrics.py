import unittest

import torch

from src.functions import Majority, Mean, Parity, Parity_n, SelectorFunction
from src.metrics import batch_metrics
from src.model import TransformerModel
from src.training import sample_batch, train


class CapturingRun:
    def __init__(self) -> None:
        self.logged_metrics: list[dict[str, float]] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        del step
        self.logged_metrics.append(metrics)


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


if __name__ == "__main__":
    unittest.main()
