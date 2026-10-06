import unittest
from itertools import product

import torch

from src.functions import HammingWeightSquareWave, Majority, Mod4SquareWave, Parity, SelectorFunction
from src.metrics import batch_metrics


class HammingWeightSquareWaveTest(unittest.TestCase):
    def test_width_three_pattern_and_ignored_suffix(self):
        inputs = torch.tensor([[1] * k + [0] * (12 - k) for k in range(13)])
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0])
        for suffix in (torch.zeros(13, 2), torch.ones(13, 2)):
            torch.testing.assert_close(
                HammingWeightSquareWave(12, 3)(torch.cat((inputs, suffix), dim=-1)), expected
            )

    def test_parity_mod4_majority_and_constant_special_cases(self):
        for n in range(1, 8):
            inputs = torch.tensor(list(product((0, 1), repeat=n)))
            for negative_value in (0, -1):
                with self.subTest(n=n, negative_value=negative_value):
                    torch.testing.assert_close(
                        HammingWeightSquareWave(n, 1, negative_value)(inputs),
                        Parity(negative_value=negative_value)(inputs),
                    )
                    torch.testing.assert_close(
                        HammingWeightSquareWave(n, 2, negative_value)(inputs),
                        Mod4SquareWave(n, negative_value)(inputs),
                    )
                    torch.testing.assert_close(
                        HammingWeightSquareWave(n, n + 1, negative_value)(inputs),
                        torch.full((len(inputs),), negative_value),
                    )
                    if n % 2:
                        torch.testing.assert_close(
                            HammingWeightSquareWave(n, (n + 1) // 2, negative_value)(inputs),
                            Majority(negative_value=negative_value)(inputs),
                        )

    def test_exact_sensitivity_matches_exhaustive_flips(self):
        for n in range(1, 8):
            inputs = torch.tensor(list(product((0, 1), repeat=n + 1)))
            for L in range(1, n + 2):
                with self.subTest(n=n, L=L):
                    function = HammingWeightSquareWave(n, L)
                    labels = function(inputs)
                    pivotal = 0
                    for i in range(n + 1):
                        flipped = inputs.clone()
                        flipped[:, i] = 1 - flipped[:, i]
                        pivotal += function(flipped).ne(labels).sum().item()
                    self.assertEqual(function.avg_sensitivity(n + 1), pivotal / len(inputs))

    def test_shapes_dtypes_and_noncontiguous_inputs(self):
        inputs = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 0]])
        function = HammingWeightSquareWave(3, 3, negative_value=-1)
        expected = torch.tensor([-1, 1])
        for dtype in (torch.long, torch.float32, torch.bool):
            with self.subTest(dtype=dtype):
                torch.testing.assert_close(function(inputs.to(dtype)), expected)
        torch.testing.assert_close(function(inputs[0]), expected[0])
        torch.testing.assert_close(function(inputs.expand(2, -1, -1)), expected.expand(2, -1))
        strided = inputs.repeat_interleave(2, dim=-1)[:, ::2]
        self.assertFalse(strided.is_contiguous())
        torch.testing.assert_close(function(strided), expected)
        self.assertEqual(function(torch.empty(0, 4)).shape, (0,))

    def test_invalid_parameters_and_short_inputs(self):
        for invalid in (0, -1, 1.5, True, "2", None):
            for n, L in ((invalid, 3), (3, invalid)):
                with self.subTest(n=n, L=L), self.assertRaisesRegex(ValueError, "positive integers"):
                    HammingWeightSquareWave(n, L)
        with self.assertRaisesRegex(ValueError, "negative_value"):
            HammingWeightSquareWave(3, 2, negative_value=2)
        for inputs in (torch.zeros(2, 2), torch.tensor(1)):
            with self.assertRaisesRegex(ValueError, "at least n"):
                HammingWeightSquareWave(3, 2)(inputs)
        with self.assertRaisesRegex(ValueError, "at least n"):
            HammingWeightSquareWave(3, 2).avg_sensitivity(2)

    def test_large_n_sensitivity_avoids_float_overflow(self):
        self.assertEqual(HammingWeightSquareWave(1100, 1).avg_sensitivity(1100), 1100.0)
        self.assertEqual(HammingWeightSquareWave(1100, 2).avg_sensitivity(1100), 550.0)

    def test_selector_payload_metric_names_and_accuracy(self):
        function = SelectorFunction([
            HammingWeightSquareWave(3, 3),
            HammingWeightSquareWave(3, 1, negative_value=-1, name="signed_wave"),
        ])
        inputs = torch.tensor([
            [0, 1, 1, 0, 1], [0, 1, 1, 1, 0],
            [1, 1, 1, 0, 1], [1, 1, 1, 1, 0],
        ])
        labels = function(inputs)
        torch.testing.assert_close(labels, torch.tensor([0.0, 1.0, -1.0, 1.0]))
        metrics = batch_metrics(
            function, torch.nn.MSELoss(), inputs, torch.tensor([0.2, 0.8, -0.8, -0.2]), labels
        )
        self.assertEqual(metrics["accuracy/hamming_weight_square_wave_3_3"], 1.0)
        self.assertEqual(metrics["accuracy/signed_wave"], 0.5)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(Mod4SquareWave(3).metric_name, "mod4_square_wave_3")


if __name__ == "__main__":
    unittest.main()
