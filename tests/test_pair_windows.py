import unittest
from itertools import product

import torch

from src.functions import Isordered, Isrepeating, SelectorFunction
from src.metrics import batch_metrics
from src.model import TransformerModel
from src.training import train


def reference_labels(function_type, rows, start, end):
    scores = ({(0, 0): 0, (0, 1): 0, (1, 0): 2, (1, 1): 0}
              if function_type is Isordered else
              {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1})
    labels = []
    for row in rows:
        window = row[start:end]
        mean = sum(scores[pair] for pair in zip(window, window[1:])) / (len(window) - 1)
        labels.append(1 - mean if function_type is Isordered else mean)
    return torch.tensor(labels, dtype=torch.float64)


def nearest_level_accuracy(outputs, labels):
    # Every possible window occurs in the exhaustive input set. Find the
    # nearest attainable output directly, independently of count quantization.
    levels = labels.unique(sorted=True)
    distances = (outputs[:, None] - levels).abs()
    nearest = levels[distances.argmin(dim=-1)]
    return nearest.eq(labels)


class PairWindowTest(unittest.TestCase):
    def test_exhaustive_labels_and_accuracy_use_only_the_window(self):
        rows = list(product((0, 1), repeat=8))
        inputs = torch.tensor(rows)
        for function_type in (Isordered, Isrepeating):
            for start, end in ((0, 2), (1, 3), (1, 4), (2, 6), (0, 8), (3, 8), (2, None)):
                with self.subTest(function=function_type.__name__, start=start, end=end):
                    function = function_type(n=start, m=end)
                    expected = reference_labels(function_type, rows, start, end)
                    labels = function(inputs)
                    torch.testing.assert_close(labels, expected.float())
                    # Includes predictions beyond both ends of the output range.
                    outputs = torch.linspace(-1.913, 2.117, len(rows), dtype=torch.float64)
                    correct = nearest_level_accuracy(outputs, expected)
                    torch.testing.assert_close(function.accuracy_mask(outputs, labels, inputs), correct)
                    metrics = batch_metrics(function, torch.nn.MSELoss(), inputs, outputs, labels)
                    self.assertAlmostEqual(metrics["accuracy"], correct.double().mean().item())

    def test_fixed_window_ignores_extra_suffix_and_open_end_tracks_input_length(self):
        inputs = torch.tensor([[1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 0]])
        for function_type in (Isordered, Isrepeating):
            with self.subTest(function=function_type.__name__):
                fixed = function_type(1, 4)
                torch.testing.assert_close(fixed(inputs), fixed(inputs[:, :4]))
                torch.testing.assert_close(
                    function_type(2, None)(inputs), function_type()(inputs[:, 2:])
                )
                torch.testing.assert_close(
                    fixed(inputs.expand(2, -1, -1)), fixed(inputs).expand(2, -1)
                )

    def test_invalid_bounds_and_incomplete_windows_are_rejected(self):
        for function_type in (Isordered, Isrepeating):
            for start, end in ((-1, 3), (0, 0), (0, 1), (3, 2), (2, 3),
                               (True, 4), (0, False), (1.5, 4), (0, 3.5), (None, 4)):
                with self.subTest(function=function_type.__name__, start=start, end=end):
                    with self.assertRaises(ValueError):
                        function_type(start, end)
            for function in (function_type(1, 5), function_type(3, None)):
                inputs = torch.zeros(2, 4, dtype=torch.long)
                with self.subTest(function=function.metric_name):
                    with self.assertRaises(ValueError):
                        function(inputs)
                    with self.assertRaises(ValueError):
                        function.accuracy_mask(torch.zeros(2), torch.zeros(2), inputs)

    def test_names_distinguish_windows_and_preserve_custom_names(self):
        for function_type in (Isordered, Isrepeating):
            base = function_type.__name__.lower()
            self.assertEqual(function_type().metric_name, base)
            self.assertEqual(function_type(1, 5).metric_name, f"{base}_1_5")
            self.assertEqual(function_type(2).metric_name, f"{base}_2_end")
            self.assertEqual(function_type(1, 5, name="custom").metric_name, "custom")

    def test_midpoint_ties_use_window_counts(self):
        ordered = Isordered(1, 6)
        ordered_inputs = torch.tensor([
            [1, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1],
        ])
        repeating = Isrepeating(1, 4)
        repeating_inputs = torch.tensor([
            [1, 0, 1, 0, 1], [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1], [1, 0, 0, 1, 1],
        ])
        for function, inputs, outputs in (
            (ordered, ordered_inputs, [0.75, 0.75, 0.25, 0.25]),
            (repeating, repeating_inputs, [0.25, 0.25, 0.75, 0.75]),
        ):
            torch.testing.assert_close(
                function.accuracy_mask(torch.tensor(outputs), function(inputs), inputs),
                torch.tensor([True, False, True, False]),
            )

    def test_selector_window_offsets_and_aggregate_accuracy(self):
        children = [Isordered(1, 4), Isordered(2, 6), Isrepeating(1, 3),
                    Isrepeating(2, 5, name="repetition")]
        function = SelectorFunction(children)
        rows = list(product((0, 1), repeat=6))
        inputs = torch.tensor([
            [index >> 1, index & 1] + list(row)
            for index in range(len(children)) for row in rows
        ])
        expected = torch.cat([
            reference_labels(type(child), rows, child.n, child.m) for child in children
        ])
        labels = function(inputs)
        torch.testing.assert_close(labels, expected.float())
        outputs = torch.linspace(-0.813, 1.917, len(inputs), dtype=torch.float64)
        metrics = batch_metrics(function, torch.nn.MSELoss(), inputs, outputs, labels)
        correct_by_task = []
        for index, child in enumerate(children):
            selection = slice(index * len(rows), (index + 1) * len(rows))
            correct = nearest_level_accuracy(outputs[selection], expected[selection])
            correct_by_task.append(correct)
            self.assertAlmostEqual(metrics[f"accuracy/{child.metric_name}"], correct.double().mean().item())
        self.assertAlmostEqual(metrics["accuracy"], torch.cat(correct_by_task).double().mean().item())

    def test_training_and_held_out_evaluation_log_window_accuracy(self):
        function = SelectorFunction([Isordered(1, 4), Isrepeating(2, 6)])
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=7)

        class CapturingRun:
            def log(self, metrics, step):
                self.metrics = metrics

        run = CapturingRun()
        train(
            model, function, torch.nn.MSELoss(),
            torch.optim.SGD(model.parameters(), lr=0.01),
            max_len=7, min_len=7, batch_size=4, num_steps=1,
            balanced_selectors=True, wandb_run=run,
            eval_interval=1, eval_num_inputs_per_function=4, eval_batch_size=3,
        )
        for prefix in ("", "eval/"):
            for key in ("accuracy", "accuracy/isordered_1_4", "accuracy/isrepeating_2_6"):
                self.assertIn(prefix + key, run.metrics)
                self.assertTrue(0.0 <= run.metrics[prefix + key] <= 1.0)


if __name__ == "__main__":
    unittest.main()
