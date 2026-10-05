import unittest
from itertools import product

import torch

from src.functions import Isordered, SelectorFunction
from src.metrics import batch_metrics
from src.model import TransformerModel
from src.training import train


class IsorderedTest(unittest.TestCase):
    def test_every_pair_and_longer_sequences_match_the_score_table(self):
        scores = {(0, 0): 0, (0, 1): 0, (1, 0): 2, (1, 1): 0}
        function = Isordered()
        for length in range(2, 8):
            with self.subTest(length=length):
                rows = list(product((0, 1), repeat=length))
                expected = torch.tensor([
                    1 - sum(scores[pair] for pair in zip(row, row[1:])) / (length - 1)
                    for row in rows
                ])
                torch.testing.assert_close(function(torch.tensor(rows)), expected)
        self.assertFalse(function.bool_output)

    def test_only_unordered_pairs_are_penalized_including_interior_pairs(self):
        inputs = torch.tensor([
            [0, 0, 1, 1], [1, 1, 0, 0],
            [0, 1, 0, 1], [1, 0, 1, 0],
            [0, 1, 1, 0], [1, 0, 0, 1],
            [0, 0, 0, 0], [1, 1, 1, 1],
        ])
        torch.testing.assert_close(
            Isordered()(inputs), torch.tensor([1, 1 / 3, 1 / 3, -1 / 3, 1 / 3, 1 / 3, 1, 1])
        )

    def test_alternating_sequences_keep_the_exact_mean_without_clamping(self):
        function = Isordered()
        example = torch.tensor([int(bit) for bit in "101010101010101010101010"])
        torch.testing.assert_close(function(example), torch.tensor(-1 / 23))
        torch.testing.assert_close(function(1 - example), torch.tensor(1 / 23))
        torch.testing.assert_close(function(example[:-1]), torch.tensor(0.0))

    def test_shapes_dtypes_and_noncontiguous_inputs(self):
        function = Isordered()
        inputs = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        expected = torch.tensor([1.0, 0.0, 0.0])
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

    def test_inputs_without_a_pair_are_rejected(self):
        function = Isordered()
        for inputs in (torch.empty(2, 0), torch.zeros(2, 1), torch.tensor(1)):
            with self.subTest(shape=inputs.shape):
                with self.assertRaisesRegex(ValueError, "at least two"):
                    function(inputs)
                with self.assertRaisesRegex(ValueError, "at least two"):
                    function.accuracy_mask(torch.zeros(2), torch.zeros(2), inputs)

    def test_accuracy_rounds_to_all_attainable_counts_at_each_length(self):
        function = Isordered()
        for length in (2, 3, 5, 10):
            with self.subTest(length=length):
                max_count = length // 2
                rows = [[1, 0] * k + [0] * (length - 2 * k) for k in range(max_count + 1)]
                inputs = torch.tensor(rows * 2)
                counts = torch.arange(max_count + 1, dtype=torch.float32)
                nearby_counts = counts + 0.49
                nearby_counts[0] = -10.0
                nearby_counts[-1] = max_count + 10.0
                wrong_counts = (counts + 1) % (max_count + 1)
                outputs = 1 - 2 * torch.cat((nearby_counts, wrong_counts)) / (length - 1)
                labels = function(inputs)
                expected_labels = 1 - 2 * counts.repeat(2) / (length - 1)
                torch.testing.assert_close(labels, expected_labels)
                torch.testing.assert_close(
                    function.accuracy_mask(outputs, labels, inputs),
                    torch.tensor([True] * (max_count + 1) + [False] * (max_count + 1)),
                )
                metrics = batch_metrics(function, torch.nn.MSELoss(), inputs, outputs, labels)
                self.assertEqual(metrics["accuracy"], 0.5)

    def test_accuracy_midpoint_ties_round_to_even_counts(self):
        inputs = torch.tensor([
            [0, 0, 0, 0, 0], [1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0], [1, 0, 0, 0, 0],
        ])
        function = Isordered()
        torch.testing.assert_close(
            function.accuracy_mask(torch.tensor([0.75, 0.75, 0.25, 0.25]), function(inputs), inputs),
            torch.tensor([True, False, True, False]),
        )

    def test_selector_excludes_prefix_and_logs_named_and_overall_accuracy(self):
        function = SelectorFunction([Isordered(), Isordered(name="order_score")])
        inputs = torch.tensor([
            [0, 1, 0, 0], [0, 0, 1, 0],
            [1, 0, 0, 1], [1, 1, 0, 1],
        ])
        labels = function(inputs)
        torch.testing.assert_close(labels, torch.tensor([0.0, 0.0, 1.0, 0.0]))
        metrics = batch_metrics(
            function, torch.nn.MSELoss(), inputs, torch.tensor([0.2, 0.8, 0.9, -0.1]), labels
        )
        self.assertEqual(metrics["accuracy/isordered"], 0.5)
        self.assertEqual(metrics["accuracy/order_score"], 1.0)
        self.assertEqual(metrics["accuracy"], 0.75)

    def test_training_logs_isordered_accuracy(self):
        function = SelectorFunction([Isordered(), Isordered(name="order_score")])
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
        for key in ("accuracy", "accuracy/isordered", "accuracy/order_score"):
            self.assertIn(key, run.metrics)
            self.assertTrue(0.0 <= run.metrics[key] <= 1.0)


if __name__ == "__main__":
    unittest.main()
