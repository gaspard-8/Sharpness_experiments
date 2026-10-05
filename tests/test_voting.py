import unittest
from itertools import product
from unittest.mock import patch

import torch

from src.functions import (
    First, Isordered, Isrepeating, Majority_n, Mean, MeanOfFunctions,
    Parity, SelectorFunction, Voting_Function, WeightedVoting,
)
from src.metrics import batch_metrics
from src.model import TransformerModel
from src.training import train


class VotingWrapperTest(unittest.TestCase):
    def test_continuous_unsigned_scores_and_midpoint_ties(self):
        inputs = torch.tensor(list(product((0, 1), repeat=4)))
        expected = (inputs.sum(dim=-1) > 2).long()
        function = Voting_Function(Mean())
        torch.testing.assert_close(function(inputs), expected)
        self.assertTrue(function.bool_output)
        self.assertEqual(function.metric_name, "voting_mean")

    def test_signed_scores_and_zero_ties(self):
        inputs = torch.tensor([[1, 0], [0, 0], [1, 1], [0, 1]])
        torch.testing.assert_close(
            Voting_Function(Mean(negative_value=-1))(inputs),
            torch.tensor([-1, -1, 1, -1]),
        )
        torch.testing.assert_close(
            Voting_Function(Isordered())(inputs), torch.tensor([-1, 1, 1, 1])
        )

    def test_binary_functions_are_unchanged_in_both_encodings(self):
        inputs = torch.tensor(list(product((0, 1), repeat=4)))
        for negative_value in (0, -1):
            for function in (First(negative_value), Parity(negative_value)):
                with self.subTest(function=function.metric_name, encoding=negative_value):
                    torch.testing.assert_close(Voting_Function(function)(inputs), function(inputs))

    def test_output_encoding_does_not_change_source_threshold(self):
        inputs = torch.tensor([[0, 0], [0, 1], [1, 1]])
        torch.testing.assert_close(
            Voting_Function(Mean(), negative_value=-1)(inputs), torch.tensor([-1, -1, 1])
        )
        torch.testing.assert_close(
            Voting_Function(Mean(negative_value=-1), negative_value=0)(inputs),
            torch.tensor([0, 0, 1]),
        )

    def test_signed_means_and_custom_callable_boundaries(self):
        inputs = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]])
        average = MeanOfFunctions([First(-1), Parity(-1)])
        torch.testing.assert_close(
            Voting_Function(average)(inputs), torch.tensor([-1, -1, -1, 1])
        )
        function = Voting_Function(lambda x: x.sum(dim=-1) - 1, threshold=0,
                                   negative_value=-1, name="sum_vote")
        torch.testing.assert_close(function(inputs), torch.tensor([-1, -1, 1, -1]))
        self.assertEqual(function.metric_name, "sum_vote")

    def test_wrapped_windows_preserve_input_shape_and_bounds(self):
        inputs = torch.tensor([[[0, 1, 1, 0], [1, 0, 1, 1]]])
        function = Voting_Function(Isrepeating(1, 3))
        torch.testing.assert_close(function(inputs), torch.tensor([[1, 0]]))
        torch.testing.assert_close(function(inputs[0, 0]), torch.tensor(1))
        self.assertEqual(function(torch.empty(0, 4)).shape, (0,))

    def test_invalid_wrapper_parameters(self):
        with self.assertRaises(TypeError):
            Voting_Function(None)
        with self.assertRaises(ValueError):
            Voting_Function(Mean(), negative_value=-2)
        for threshold in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                Voting_Function(Mean(), threshold=threshold)


class WeightedVotingTest(unittest.TestCase):
    def test_truth_table_with_negative_weights_and_ignored_tail(self):
        rows = list(product((0, 1), repeat=4))
        weights = [2, -1, 0.5]
        expected = torch.tensor([
            int(sum(a * (2 * bit - 1) for a, bit in zip(weights, row)) > 0)
            for row in rows
        ])
        for negative_value in (0, -1):
            function = WeightedVoting(3, weights, negative_value=negative_value)
            labels = expected if negative_value == 0 else 2 * expected - 1
            torch.testing.assert_close(function(torch.tensor(rows)), labels)
            torch.testing.assert_close(
                WeightedVoting(3, weights, negative_value=negative_value,
                               input_negative_value=-1)(2 * torch.tensor(rows) - 1), labels
            )

    def test_equal_weights_reproduce_majority_including_ties(self):
        for n in range(1, 6):
            inputs = torch.tensor(list(product((0, 1), repeat=n)))
            torch.testing.assert_close(WeightedVoting(n, [1] * n)(inputs), Majority_n(n)(inputs))
        torch.testing.assert_close(
            WeightedVoting(2, [0, 0], negative_value=-1)(torch.tensor([[0, 0], [1, 1]])),
            torch.tensor([-1, -1]),
        )

    def test_shapes_dtypes_and_device(self):
        inputs = torch.tensor([[0, 1, 1], [1, 0, 0]])
        function = WeightedVoting(3, [3, 1, 1])
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        if torch.backends.mps.is_available():
            devices.append(torch.device("mps"))
        for device in devices:
            for dtype in (torch.long, torch.float32, torch.bool):
                with self.subTest(device=device, dtype=dtype):
                    labels = function(inputs.to(device=device, dtype=dtype))
                    torch.testing.assert_close(labels, torch.tensor([0, 1], device=device))
        torch.testing.assert_close(function(inputs[0]), torch.tensor(0))
        torch.testing.assert_close(function(inputs.expand(2, -1, -1)), torch.tensor([[0, 1]] * 2))
        torch.testing.assert_close(function(inputs.to(torch.float64)), torch.tensor([0, 1]))
        strided = inputs.repeat_interleave(2, dim=-1)[:, ::2]
        torch.testing.assert_close(function(strided), torch.tensor([0, 1]))
        self.assertEqual(function(torch.empty(0, 3)).shape, (0,))

    def test_parameter_validation_and_short_inputs(self):
        for n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                WeightedVoting(n, [])
        for weights in ([], [1], [1, 2, 3], [1, float("nan")], [1, float("inf")]):
            with self.assertRaises(ValueError):
                WeightedVoting(2, weights)
        for kwargs in ({"negative_value": -2}, {"input_negative_value": -2}):
            with self.assertRaises(ValueError):
                WeightedVoting(2, [1, 1], **kwargs)
        for inputs in (torch.zeros(2, 1), torch.tensor(1)):
            with self.assertRaisesRegex(ValueError, "at least n"):
                WeightedVoting(2, [1, 1])(inputs)


class VotingIntegrationTest(unittest.TestCase):
    def test_selector_accuracy_uses_binary_labels_and_strict_boundaries(self):
        function = SelectorFunction([
            Voting_Function(Mean(), name="mean_vote"),
            WeightedVoting(2, [2, 1], negative_value=-1, name="weighted_vote"),
        ])
        inputs = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        labels = function(inputs)
        torch.testing.assert_close(labels, torch.tensor([0., 1., -1., 1.]))
        metrics = batch_metrics(
            function, torch.nn.MSELoss(), inputs, torch.tensor([0.5, 0.6, 0.0, -0.1]), labels
        )
        self.assertEqual(metrics["accuracy/mean_vote"], 1.0)
        self.assertEqual(metrics["accuracy/weighted_vote"], 0.5)
        self.assertEqual(metrics["accuracy"], 0.75)

    def test_training_and_evaluation_log_voting_accuracy(self):
        function = SelectorFunction([Voting_Function(Mean()), WeightedVoting(3, [2, 1, 1])])
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4)

        class CapturingRun:
            def log(self, metrics, step):
                self.metrics = metrics

        run = CapturingRun()
        train(model, function, torch.nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.01),
              max_len=4, min_len=4, batch_size=4, num_steps=1, balanced_selectors=True,
              eval_interval=1, eval_num_inputs_per_function=4, eval_batch_size=4, wandb_run=run)
        for prefix in ("", "eval/"):
            for task in ("voting_mean", "weighted_voting_3"):
                self.assertIn(f"{prefix}accuracy/{task}", run.metrics)

    def test_default_experiment_has_only_binary_tasks(self):
        import test as experiment

        with patch("sys.argv", ["test.py", "--smoke-test"]), \
                patch.dict("os.environ", {"SHARPNESS_DEVICE": "cpu"}), \
                patch.object(experiment.wandb, "init"), \
                patch.object(experiment, "train") as training:
            experiment.main()
        mix = training.call_args.kwargs["function"]
        self.assertEqual(len(mix.functions), 8)
        inputs = torch.randint(0, 2, (64, 21))
        for function in mix.functions:
            self.assertTrue(function.bool_output)
            labels = function(inputs)
            self.assertTrue(((labels == function.negative_value) | (labels == 1)).all())


if __name__ == "__main__":
    unittest.main()
