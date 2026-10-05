import unittest
from itertools import product

import torch

from src.functions import (
    First,
    Majority_n,
    Majority_nm,
    Mean,
    MeanOfFunctions,
    Parity,
    SelectorFunction,
)
from src.metrics import batch_metrics
from src.model import TransformerModel
from src.training import train


class MajorityRangeTest(unittest.TestCase):
    def test_only_bits_in_the_half_open_interval_affect_majority(self):
        inputs = torch.tensor(list(product((0, 1), repeat=6)))
        expected = torch.tensor([
            int(sum(row[1:4]) >= 2) for row in inputs.tolist()
        ])

        torch.testing.assert_close(Majority_nm(1, 4)(inputs), expected)

    def test_ties_are_negative_with_both_output_encodings(self):
        inputs = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 0], [1, 0, 0, 1]])

        torch.testing.assert_close(
            Majority_nm(1, 3)(inputs), torch.tensor([0, 1, 0])
        )
        torch.testing.assert_close(
            Majority_nm(1, 3, negative_value=-1)(inputs),
            torch.tensor([-1, 1, -1]),
        )

    def test_prefix_range_matches_majority_n(self):
        inputs = torch.tensor(list(product((0, 1), repeat=4)))
        for end in range(1, 5):
            with self.subTest(end=end):
                torch.testing.assert_close(
                    Majority_nm(0, end)(inputs), Majority_n(end)(inputs)
                )

    def test_invalid_or_incomplete_ranges_are_rejected(self):
        for start, end in ((-1, 2), (0, 0), (3, 2)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                Majority_nm(start, end)
        with self.assertRaisesRegex(ValueError, "at least m"):
            Majority_nm(1, 4)(torch.zeros((2, 3), dtype=torch.long))


class MeanOfFunctionsTest(unittest.TestCase):
    def test_mean_uses_the_same_full_input_for_every_function(self):
        inputs = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 1]])
        function = MeanOfFunctions([First(), Parity(), Majority_nm(1, 3)])

        torch.testing.assert_close(
            function(inputs), torch.tensor([1 / 3, 1 / 3, 1.0])
        )
        self.assertFalse(function.bool_output)
        self.assertNotIsInstance(function, SelectorFunction)

    def test_child_output_encodings_and_continuous_values_are_preserved(self):
        inputs = torch.tensor([[0, 1], [1, 1], [0, 0]])
        function = MeanOfFunctions([First(negative_value=-1), Mean()])

        torch.testing.assert_close(
            function(inputs), torch.tensor([-0.25, 1.0, -0.5])
        )

    def test_single_and_repeated_functions(self):
        inputs = torch.tensor([[0, 1], [1, 1]])
        torch.testing.assert_close(
            MeanOfFunctions([First()])(inputs), torch.tensor([0.0, 1.0])
        )
        torch.testing.assert_close(
            MeanOfFunctions([First(), First(), Parity()])(inputs),
            torch.tensor([1 / 3, 2 / 3]),
        )

    def test_empty_function_list_is_rejected(self):
        with self.assertRaises(ValueError):
            MeanOfFunctions([])

    def test_accuracy_rounds_to_mean_levels_using_the_number_of_functions(self):
        function = MeanOfFunctions([First(), Parity()])
        inputs = torch.tensor([
            [0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0],
            [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
        ])
        outputs = torch.tensor([-0.2, 0.26, 1.2, 0.74, 0.24, 0.76])
        labels = function(inputs)

        torch.testing.assert_close(
            function.accuracy_mask(outputs, labels, inputs),
            torch.tensor([True, True, True, True, False, False]),
        )
        metrics = batch_metrics(function, torch.nn.MSELoss(), inputs, outputs, labels)
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)

    def test_accuracy_respects_signed_and_mixed_encodings(self):
        inputs = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]])
        for first_negative, parity_negative, predictions in (
            (-1, -1, [-1.2, 0.49, 0.51, 1.2]),
            (-1, 0, [-0.6, 0.24, 0.24, 1.2]),
        ):
            with self.subTest(encodings=(first_negative, parity_negative)):
                function = MeanOfFunctions([
                    First(negative_value=first_negative),
                    Parity(negative_value=parity_negative),
                ])
                torch.testing.assert_close(
                    function.accuracy_mask(torch.tensor(predictions), function(inputs), inputs),
                    torch.tensor([True, True, False, True]),
                )

    def test_accuracy_supports_other_function_counts(self):
        inputs = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]])
        for children, predictions in (
            ([First()], [0.49, 0.51, 0.51, 0.49]),
            ([First(), First(), Parity()], [0.1, 0.4, 0.8, 0.8]),
        ):
            with self.subTest(count=len(children)):
                function = MeanOfFunctions(children)
                expected = ([True, False, True, False] if len(children) == 1
                            else [True, True, True, False])
                torch.testing.assert_close(
                    function.accuracy_mask(torch.tensor(predictions), function(inputs), inputs),
                    torch.tensor(expected),
                )

    def test_accuracy_midpoint_ties_match_mean_rounding(self):
        function = MeanOfFunctions([First(), Parity()])
        inputs = torch.tensor([[0, 0], [0, 1], [0, 1], [1, 0]])
        outputs = torch.tensor([0.25, 0.25, 0.75, 0.75])

        torch.testing.assert_close(
            function.accuracy_mask(outputs, function(inputs), inputs),
            torch.tensor([True, False, False, True]),
        )

    def test_nonbinary_children_do_not_use_binary_mean_accuracy(self):
        function = MeanOfFunctions([First(), Mean()])
        inputs = torch.tensor([[0, 1, 1]])
        labels = function(inputs)

        self.assertIsNone(function.accuracy_mask(labels, labels, inputs))

    def test_selector_treats_mean_as_one_task_and_strips_only_its_prefix(self):
        average = MeanOfFunctions([First(), Majority_nm(1, 3)], name="average")
        function = SelectorFunction([average, Parity()])
        inputs = torch.tensor([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
        ])
        labels = function(inputs)

        self.assertEqual(function.selector_size, 1)
        torch.testing.assert_close(labels, torch.tensor([0.5, 0.5, 0.0, 1.0]))
        outputs = torch.tensor([0.1, 0.6, 0.0, 1.0])
        metrics = batch_metrics(function, torch.nn.MSELoss(), inputs, outputs, labels)
        self.assertEqual(
            set(metrics),
            {"loss", "loss/average", "accuracy/average", "loss/parity", "accuracy/parity", "accuracy"},
        )
        self.assertAlmostEqual(metrics["loss/average"], 0.085)
        self.assertEqual(metrics["accuracy/average"], 0.5)
        self.assertEqual(metrics["accuracy/parity"], 1.0)
        self.assertEqual(metrics["accuracy"], 0.75)

    def test_training_logs_accuracy_for_each_mean_task(self):
        function = SelectorFunction([
            MeanOfFunctions([First(), Parity()]),
            MeanOfFunctions([First(), Majority_nm(1, 3)]),
        ])
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4)

        class CapturingRun:
            def log(self, metrics, step):
                self.metrics = metrics

        run = CapturingRun()
        train(
            model, function, torch.nn.MSELoss(),
            torch.optim.SGD(model.parameters(), lr=0.01),
            max_len=4, min_len=4, batch_size=4, num_steps=1,
            balanced_selectors=True, wandb_run=run,
        )

        for key in ("accuracy", "accuracy/mean_of_functions_0", "accuracy/mean_of_functions_1"):
            self.assertIn(key, run.metrics)
            self.assertTrue(0.0 <= run.metrics[key] <= 1.0)


if __name__ == "__main__":
    unittest.main()
