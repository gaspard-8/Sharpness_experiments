import unittest
from itertools import product

import torch

from src.functions import MeanOfFunctions, Parity, SelectorFunction, Tribe_ws
from src.metrics import batch_metrics
from src.model import TransformerModel
from src.training import train


class TribeTest(unittest.TestCase):
    def test_truth_tables_and_ignored_extra_bits(self):
        for w, s in ((1, 1), (1, 3), (3, 1), (2, 3), (3, 2)):
            with self.subTest(w=w, s=s):
                rows = list(product((0, 1), repeat=w * s + 1))
                expected = torch.tensor([
                    int(any(all(row[i:i + w]) for i in range(0, w * s, w)))
                    for row in rows
                ])
                for negative_value in (0, -1):
                    function = Tribe_ws(w, s, negative_value=negative_value)
                    labels = expected if negative_value == 0 else 2 * expected - 1
                    torch.testing.assert_close(function(torch.tensor(rows)), labels)

    def test_shapes_dtypes_and_noncontiguous_inputs(self):
        inputs = torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]])
        function = Tribe_ws(2, 2)
        for dtype in (torch.long, torch.float32, torch.bool):
            with self.subTest(dtype=dtype):
                torch.testing.assert_close(function(inputs.to(dtype)), torch.tensor([0, 1]))
        torch.testing.assert_close(function(inputs[0]), torch.tensor(0))
        torch.testing.assert_close(
            function(inputs.expand(3, -1, -1)), torch.tensor([[0, 1]] * 3)
        )
        strided = inputs.repeat_interleave(2, dim=-1)[:, ::2]
        self.assertFalse(strided.is_contiguous())
        torch.testing.assert_close(function(strided), torch.tensor([0, 1]))
        self.assertEqual(function(torch.empty(0, 4)).shape, (0,))

    def test_invalid_parameters_and_short_inputs_are_rejected(self):
        for invalid in (0, -1, 1.5, True, "2", None):
            for w, s in ((invalid, 2), (2, invalid)):
                with self.subTest(w=w, s=s), self.assertRaisesRegex(ValueError, "positive integers"):
                    Tribe_ws(w, s)
        for inputs in (torch.zeros(2, 3), torch.tensor(1)):
            with self.assertRaisesRegex(ValueError, "at least w \\* s"):
                Tribe_ws(2, 2)(inputs)

    def test_accuracy_thresholds_for_both_encodings(self):
        inputs = torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]] * 3)
        for negative_value, predictions in (
            (0, [0.49, 0.5, 0.5, 0.49, -2.0, 2.0]),
            (-1, [-0.01, 0.0, 0.0, -0.01, -2.0, 2.0]),
        ):
            with self.subTest(negative_value=negative_value):
                function = Tribe_ws(2, 2, negative_value=negative_value)
                outputs = torch.tensor(predictions)
                labels = function(inputs)
                torch.testing.assert_close(
                    function.accuracy_mask(outputs, labels, inputs),
                    torch.tensor([True, True, False, False, True, True]),
                )
                metrics = batch_metrics(function, torch.nn.MSELoss(), inputs, outputs, labels)
                self.assertAlmostEqual(metrics["accuracy"], 2 / 3)

    def test_selector_strips_prefix_and_reports_named_and_overall_accuracy(self):
        function = SelectorFunction([
            Tribe_ws(2, 2), Tribe_ws(2, 2, negative_value=-1, name="signed_tribes")
        ])
        inputs = torch.tensor([
            [0, 1, 1, 0, 0], [0, 0, 1, 1, 0],
            [1, 1, 1, 0, 0], [1, 0, 1, 1, 0],
        ])
        labels = function(inputs)
        torch.testing.assert_close(labels, torch.tensor([1.0, 0.0, 1.0, -1.0]))
        metrics = batch_metrics(
            function, torch.nn.MSELoss(), inputs, torch.tensor([0.8, 0.6, 0.1, -0.1]), labels
        )
        self.assertEqual(metrics["accuracy/tribe_2_2"], 0.5)
        self.assertEqual(metrics["accuracy/signed_tribes"], 1.0)
        self.assertEqual(metrics["accuracy"], 0.75)

    def test_mean_of_functions_includes_tribes_in_accuracy(self):
        function = MeanOfFunctions([Tribe_ws(2, 2), Parity()])
        inputs = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0], [1, 1, 1, 0]])
        labels = function(inputs)
        torch.testing.assert_close(labels, torch.tensor([0.5, 0.0, 1.0]))
        metrics = batch_metrics(
            function, torch.nn.MSELoss(), inputs, torch.tensor([0.6, 0.4, 0.9]), labels
        )
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)

    def test_training_logs_accuracy_for_each_tribe_configuration(self):
        function = SelectorFunction([Tribe_ws(2, 2), Tribe_ws(1, 4)])
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=5)

        class CapturingRun:
            def log(self, metrics, step):
                self.metrics = metrics

        run = CapturingRun()
        train(
            model, function, torch.nn.MSELoss(),
            torch.optim.SGD(model.parameters(), lr=0.01),
            max_len=5, min_len=5, batch_size=4, num_steps=1,
            balanced_selectors=True, wandb_run=run,
        )
        for key in ("accuracy", "accuracy/tribe_2_2", "accuracy/tribe_1_4"):
            self.assertIn(key, run.metrics)
            self.assertTrue(0.0 <= run.metrics[key] <= 1.0)


if __name__ == "__main__":
    unittest.main()
