import copy
import unittest
from unittest.mock import patch

import torch

from src.evaluation import evaluate_fixed_set, make_evaluation_set
from src.functions import First, Mean, Parity, SelectorFunction
from src.metrics import batch_metrics
from src.model import TransformerModel
from src.training import sample_batch, train


class CapturingRun:
    def __init__(self):
        self.logged = []

    def log(self, metrics, step):
        self.logged.append((step, metrics))


class EvaluationTest(unittest.TestCase):
    def test_set_is_unique_balanced_reproducible_and_rng_independent(self):
        function = SelectorFunction([First(), Parity(), Mean()])
        for length, count in ((5, 7), (48, 320)):
            with self.subTest(length=length):
                before = torch.get_rng_state().clone()
                held_out = make_evaluation_set(function, length, count, seed=17)
                torch.testing.assert_close(torch.get_rng_state(), before)
                repeated = make_evaluation_set(function, length, count, seed=17)
                different = make_evaluation_set(function, length, count, seed=18)
                torch.testing.assert_close(held_out.inputs, repeated.inputs)
                self.assertFalse(torch.equal(held_out.inputs, different.inputs))
                self.assertEqual(torch.unique(held_out.inputs, dim=0).size(0), 3 * count)
                torch.testing.assert_close(
                    torch.bincount(function.function_ids(held_out.inputs)),
                    torch.full((3,), count),
                )
                torch.testing.assert_close(held_out.labels, function(held_out.inputs))

    def test_training_rejects_collisions_without_changing_selector_counts(self):
        function = SelectorFunction([First(), Parity(), Mean()])
        # Hold out seven of the eight payloads per task, forcing collisions.
        held_out = make_evaluation_set(function, 5, 7)
        for probabilities, balanced in (([0., 1., 0.], False), (None, True)):
            inputs, labels = sample_batch(
                function, batch_size=60, seq_len=5, min_seq_len=5,
                selector_probabilities=probabilities, balanced_selectors=balanced,
                held_out=held_out,
            )
            self.assertEqual(held_out.overlapping_rows(inputs), [])
            expected_counts = torch.tensor([0, 60, 0] if probabilities else [20, 20, 20])
            torch.testing.assert_close(
                torch.bincount(function.function_ids(inputs), minlength=3), expected_counts,
            )
            torch.testing.assert_close(labels, function(inputs))

        shorter, _ = sample_batch(function, 10, seq_len=4, min_seq_len=4, held_out=held_out)
        self.assertEqual(held_out.overlapping_rows(shorter), [])

    def test_single_function_and_single_selector_task_are_supported(self):
        for function in (Parity(), SelectorFunction([Parity()])):
            held_out = make_evaluation_set(function, 3, 7)
            inputs, labels = sample_batch(function, 20, seq_len=3, min_seq_len=3, held_out=held_out)
            self.assertEqual(held_out.overlapping_rows(inputs), [])
            self.assertEqual(torch.unique(inputs, dim=0).size(0), 1)
            torch.testing.assert_close(labels, function(inputs))

    def test_invalid_sizes_fail_before_training(self):
        for count in (0, -1, True, 1.5, 8, 9):
            with self.subTest(count=count), self.assertRaises(ValueError):
                make_evaluation_set(SelectorFunction([First(), Parity()]), 4, count)
        with self.assertRaises(ValueError):
            make_evaluation_set(SelectorFunction([First(), Parity()]), 1, 1)
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4)
        for config in ({"eval_interval": 0}, {"eval_interval": True}, {"eval_interval": 1, "eval_batch_size": 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                train(model, First(), torch.nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=.01), **config)

    def test_no_grad_without_inference_preserves_modes_state_gradients_and_rng(self):
        torch.manual_seed(5)
        function = SelectorFunction([First(), First(), Mean()])
        held_out = make_evaluation_set(function, 5, 3)
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=5, dropout=.3)
        model.train()
        model.positional_encoding.eval()
        modes = [module.training for module in model.modules()]
        state = copy.deepcopy(model.state_dict())
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        gradients = [parameter.grad for parameter in model.parameters()]
        observed = []

        def record(module, args, output):
            observed.append((module.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled(), output.requires_grad))

        handle = model.register_forward_hook(record)
        rng = torch.get_rng_state().clone()
        try:
            for reduction in ("mean", "sum", "none"):
                criterion = torch.nn.MSELoss(reduction=reduction)
                actual = evaluate_fixed_set(model, function, criterion, held_out, batch_size=4)
                self.assertEqual([module.training for module in model.modules()], modes)
                # Full-set scoring must match chunked scoring, including the
                # last short batch and tasks split between multiple chunks.
                model.eval()
                with torch.no_grad():
                    expected = batch_metrics(function, criterion, held_out.inputs, model(held_out.inputs), held_out.labels)
                for module, training in zip(model.modules(), modes):
                    module.training = training
                self.assertEqual(set(actual), set(expected))
                for key in actual:
                    self.assertAlmostEqual(actual[key], expected[key], places=5)
                self.assertIn("accuracy/first_0", actual)
                self.assertIn("accuracy/first_1", actual)
        finally:
            handle.remove()
        self.assertTrue(observed)
        self.assertTrue(all(flags == (False, False, False, False) for flags in observed))
        self.assertEqual([module.training for module in model.modules()], modes)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name])
        for parameter, gradient in zip(model.parameters(), gradients):
            self.assertIs(parameter.grad, gradient)
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))

    def test_modes_are_restored_after_evaluation_failure(self):
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4)
        model.train()
        model.positional_encoding.eval()
        modes = [module.training for module in model.modules()]
        with patch.object(model, "forward", side_effect=RuntimeError("failed forward")):
            with self.assertRaisesRegex(RuntimeError, "failed forward"):
                evaluate_fixed_set(model, First(), torch.nn.MSELoss(), make_evaluation_set(First(), 4, 3))
        self.assertEqual([module.training for module in model.modules()], modes)

    def test_training_evaluates_every_ten_steps_and_continues_with_gradients(self):
        torch.manual_seed(9)
        function = SelectorFunction([First(), Parity()])
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4, dropout=.2)
        initial = copy.deepcopy(model)
        held_out = make_evaluation_set(function, 4, 3, seed=12)
        run = CapturingRun()
        seen_training = []
        seen_evaluation = []

        def observe(module, args, output):
            self.assertFalse(torch.is_inference_mode_enabled())
            if module.training:
                self.assertTrue(torch.is_grad_enabled())
                self.assertTrue(output.requires_grad)
                self.assertEqual(held_out.overlapping_rows(args[0]), [])
                seen_training.append(args[0].clone())
            else:
                self.assertFalse(torch.is_grad_enabled())
                seen_evaluation.append(args[0].clone())

        handle = model.register_forward_hook(observe)
        config = dict(
            max_len=4, min_len=4, batch_size=6, num_steps=21,
            balanced_selectors=True, eval_interval=10, eval_num_inputs_per_function=3,
            eval_batch_size=4, eval_seed=12, gradient_interval=100, curvature_interval=100,
        )
        try:
            torch.manual_seed(21)
            losses = train(model, function, torch.nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=.01), wandb_run=run, **config)
        finally:
            handle.remove()
        final_rng = torch.get_rng_state().clone()
        self.assertEqual(len(seen_training), 21)
        self.assertEqual([step for step, metrics in run.logged if "eval/accuracy" in metrics], [10, 20])
        self.assertEqual(len(seen_evaluation), 4)
        for offset in (0, 2):
            torch.testing.assert_close(torch.cat(seen_evaluation[offset:offset + 2]), held_out.inputs)
        for step, metrics in run.logged:
            self.assertIn("accuracy", metrics)
            if step % 10 == 0:
                self.assertIn("eval/loss", metrics)
                self.assertIn("eval/accuracy/first", metrics)
                self.assertIn("eval/accuracy/parity", metrics)
        self.assertTrue(model.training)

        # Reserving the same set but skipping evaluation must yield identical
        # training updates and dropout randomness.
        torch.manual_seed(21)
        with patch("src.training.evaluate_fixed_set", side_effect=AssertionError("unexpected evaluation")):
            no_logger_losses = train(initial, function, torch.nn.MSELoss(), torch.optim.SGD(initial.parameters(), lr=.01), **config)
        self.assertEqual(losses, no_logger_losses)
        torch.testing.assert_close(torch.get_rng_state(), final_rng)
        for actual, expected in zip(model.parameters(), initial.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
