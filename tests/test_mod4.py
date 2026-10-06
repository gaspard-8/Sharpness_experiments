import unittest
from itertools import product

import torch

from src.functions import Mod4SquareWave, SelectorFunction
from src.metrics import batch_metrics


class Mod4SquareWaveTest(unittest.TestCase):
    def test_truth_tables_match_parity_of_all_pairs(self):
        for n in range(1, 8):
            with self.subTest(n=n):
                rows = list(product((0, 1), repeat=n))
                # Independent characterization: parity of the number of 11 pairs.
                expected = torch.tensor([
                    sum(row[i] * row[j] for i in range(n) for j in range(i + 1, n)) % 2
                    for row in rows
                ])
                torch.testing.assert_close(Mod4SquareWave(n)(torch.tensor(rows)), expected)

    def test_count_pattern_and_ignored_suffix(self):
        inputs = torch.tensor([[1] * count + [0] * (8 - count) for count in range(9)])
        expected = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0])
        function = Mod4SquareWave(8)
        for suffix in (torch.zeros(9, 3), torch.ones(9, 3)):
            torch.testing.assert_close(function(torch.cat((inputs, suffix), dim=-1)), expected)

    def test_encodings_shapes_dtypes_and_noncontiguous_inputs(self):
        inputs = torch.tensor([[1, 0, 0, 1], [1, 1, 1, 0]])
        for negative_value in (0, -1):
            function = Mod4SquareWave(3, negative_value=negative_value)
            expected = torch.tensor([negative_value, 1])
            for dtype in (torch.long, torch.float32, torch.bool):
                with self.subTest(negative_value=negative_value, dtype=dtype):
                    torch.testing.assert_close(function(inputs.to(dtype)), expected)
            torch.testing.assert_close(function(inputs[0]), expected[0])
            torch.testing.assert_close(function(inputs.expand(2, -1, -1)), expected.expand(2, -1))
            strided = inputs.repeat_interleave(2, dim=-1)[:, ::2]
            self.assertFalse(strided.is_contiguous())
            torch.testing.assert_close(function(strided), expected)
            self.assertEqual(function(torch.empty(0, 4)).shape, (0,))

    def test_average_sensitivity_matches_exhaustive_bit_flips(self):
        for n in range(1, 8):
            with self.subTest(n=n):
                inputs = torch.tensor(list(product((0, 1), repeat=n + 1)))
                function = Mod4SquareWave(n)
                labels = function(inputs)
                pivotal = 0
                for i in range(n + 1):
                    flipped = inputs.clone()
                    flipped[:, i] = 1 - flipped[:, i]
                    pivotal += function(flipped).ne(labels).sum().item()
                self.assertEqual(function.avg_sensitivity(n + 1), pivotal / len(inputs))

    def test_invalid_parameters_and_short_inputs_are_rejected(self):
        for invalid in (0, -1, 1.5, True, "2", None):
            with self.subTest(n=invalid), self.assertRaisesRegex(ValueError, "positive integer"):
                Mod4SquareWave(invalid)
        with self.assertRaisesRegex(ValueError, "negative_value"):
            Mod4SquareWave(3, negative_value=2)
        for inputs in (torch.zeros(2, 2), torch.tensor(1)):
            with self.assertRaisesRegex(ValueError, "at least n"):
                Mod4SquareWave(3)(inputs)
        with self.assertRaisesRegex(ValueError, "at least n"):
            Mod4SquareWave(3).avg_sensitivity(2)

    def test_selector_payload_and_accuracy_for_both_encodings(self):
        function = SelectorFunction([
            Mod4SquareWave(3), Mod4SquareWave(3, negative_value=-1, name="signed_mod4")
        ])
        inputs = torch.tensor([
            [0, 1, 0, 0, 1], [0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1], [1, 1, 1, 1, 0],
        ])
        labels = function(inputs)
        torch.testing.assert_close(labels, torch.tensor([0.0, 1.0, -1.0, 1.0]))
        metrics = batch_metrics(
            function, torch.nn.MSELoss(), inputs, torch.tensor([0.2, 0.8, -0.8, -0.2]), labels
        )
        self.assertEqual(metrics["accuracy/mod4_square_wave_3"], 1.0)
        self.assertEqual(metrics["accuracy/signed_mod4"], 0.5)
        self.assertEqual(metrics["accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
