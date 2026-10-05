import unittest
from itertools import product

import torch

from src.functions import Isrepeating, SelectorFunction
from src.metrics import batch_metrics
from src.model import TransformerModel
from src.training import train


class IsrepeatingTest(unittest.TestCase):
    def test_all_binary_sequences_match_the_pair_score_table(self):
        scores = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1}
        function = Isrepeating()
        for length in range(2, 8):
            with self.subTest(length=length):
                rows = list(product((0, 1), repeat=length))
                expected = torch.tensor([
                    sum(scores[pair] for pair in zip(row, row[1:])) / (length - 1)
                    for row in rows
                ])
                torch.testing.assert_close(function(torch.tensor(rows)), expected)
        self.assertFalse(function.bool_output)

    def test_constant_alternating_and_mixed_examples(self):
        inputs = torch.tensor([
            [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1],
            [1, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 0],
        ])
        torch.testing.assert_close(
            Isrepeating()(inputs), torch.tensor([1, 1, 0, 0, 2 / 3, 1 / 3])
        )

    def test_shapes_dtypes_and_noncontiguous_inputs(self):
        function = Isrepeating()
        inputs = torch.tensor([[0, 0, 1], [1, 1, 1], [1, 0, 1]])
        expected = torch.tensor([0.5, 1.0, 0.0])
        for dtype in (torch.long, torch.uint8, torch.bool, torch.float32):
            with self.subTest(dtype=dtype):
                result = function(inputs.to(dtype))
                torch.testing.assert_close(result, expected)
                self.assertEqual(result.device, inputs.device)
        torch.testing.assert_close(function(inputs[0]), expected[0])
        torch.testing.assert_close(function(inputs.expand(2, -1, -1)), expected.expand(2, -1))
        strided = inputs.repeat_interleave(2, dim=-1)[:, ::2]
        self.assertFalse(strided.is_contiguous())
        torch.testing.assert_close(function(strided), expected)
        self.assertEqual(function(torch.empty(0, 3)).shape, (0,))

    def test_inputs_without_pairs_are_rejected(self):
        function = Isrepeating()
        for inputs in (torch.empty(2, 0), torch.zeros(2, 1), torch.tensor(1)):
            with self.subTest(shape=inputs.shape):
                with self.assertRaisesRegex(ValueError, "at least two"):
                    function(inputs)
                with self.assertRaisesRegex(ValueError, "at least two"):
                    function.accuracy_mask(torch.zeros(2), torch.zeros(2), inputs)

    def test_accuracy_uses_pair_count_and_clamps_to_attainable_levels(self):
        function = Isrepeating()
        for length in (2, 3, 5, 10):
            with self.subTest(length=length):
                # k repeated zero pairs followed by alternating bits.
                rows = [[0] * (k + 1) + [(j + 1) % 2 for j in range(length - k - 1)]
                        for k in range(length)]
                inputs = torch.tensor(rows * 2)
                counts = torch.arange(length, dtype=torch.float32)
                nearby_counts = counts + 0.49
                nearby_counts[0] = -10.0
                nearby_counts[-1] = length + 10.0
                wrong_counts = (counts + 1) % length
                outputs = torch.cat((nearby_counts, wrong_counts)) / (length - 1)
                labels = function(inputs)
                torch.testing.assert_close(labels, counts.repeat(2) / (length - 1))
                torch.testing.assert_close(
                    function.accuracy_mask(outputs, labels, inputs),
                    torch.tensor([True] * length + [False] * length),
                )
                metrics = batch_metrics(function, torch.nn.MSELoss(), inputs, outputs, labels)
                self.assertEqual(metrics["accuracy"], 0.5)

    def test_accuracy_midpoint_ties_round_to_even_counts(self):
        function = Isrepeating()
        inputs = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 1]])
        torch.testing.assert_close(
            function.accuracy_mask(torch.tensor([0.25, 0.25, 0.75, 0.75]), function(inputs), inputs),
            torch.tensor([True, False, True, False]),
        )

    def test_selector_excludes_prefix_and_logs_named_and_overall_accuracy(self):
        function = SelectorFunction([Isrepeating(), Isrepeating(name="repetition")])
        inputs = torch.tensor([
            [0, 1, 1, 1], [0, 0, 1, 0],
            [1, 0, 0, 1], [1, 1, 0, 0],
        ])
        labels = function(inputs)
        torch.testing.assert_close(labels, torch.tensor([1.0, 0.0, 0.5, 0.5]))
        metrics = batch_metrics(
            function, torch.nn.MSELoss(), inputs, torch.tensor([0.9, 0.4, 0.6, 0.4]), labels
        )
        self.assertEqual(metrics["accuracy/isrepeating"], 0.5)
        self.assertEqual(metrics["accuracy/repetition"], 1.0)
        self.assertEqual(metrics["accuracy"], 0.75)

    def test_training_logs_repeating_accuracy(self):
        function = SelectorFunction([Isrepeating(), Isrepeating(name="repetition")])
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
        for key in ("accuracy", "accuracy/isrepeating", "accuracy/repetition"):
            self.assertIn(key, run.metrics)
            self.assertTrue(0.0 <= run.metrics[key] <= 1.0)


if __name__ == "__main__":
    unittest.main()
