import copy
from collections import Counter
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.func import functional_call

from src.arithmetic import (
    ANSWER, EOS, MINUS, PAD, AbsoluteDifference, Addition, AdditionMod10, ArithmeticFunction,
    ArithmeticSelector, ArithmeticTransformer, FunctionGroup, IntegerDivision,
    Majority, Multiplication, Remainder, SequenceCrossEntropy, Subtraction, decode_answer,
    encode_answers, evaluate_fixed_set, make_evaluation_set, sample_batch,
    sequence_metrics, teacher_tokens,
)
from src.arithmetic_training import (
    DiagnosticAdapter, TrainingConfig, average_direction_sharpness, train,
)
from src.metrics import gradient_metrics, multi_task_curvature, task_affinity_metrics
from src.model import TransformerModel
from src.spaces import loss_spaces


class ArithmeticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, selector, output_tokens=5, dropout=0.):
        return ArithmeticTransformer(d_model=8, num_attn_heads=2, num_layers=1,
                                     max_len=selector.prompt_len + output_tokens, dropout=dropout)

    def test_operations_signed_results_and_equal_width_arguments(self):
        inputs = torch.tensor([[1, 2, 0, 3], [0, 0, 9, 9], [9, 9, 0, 1]])
        expected = ([15, 99, 100], [9, -99, 98], [36, 0, 99], [4, 0, 99], [0, 0, 0])
        for function, values in zip((Addition(), Subtraction(), Multiplication(), IntegerDivision(), Remainder()), expected):
            self.assertEqual(function(inputs), values)
        payload = torch.tensor([[0, 2, 1, 2, 0, 3]])
        self.assertEqual(Addition(arity=3)(payload), [17])
        self.assertEqual(Multiplication(arity=3)(payload), [72])
        self.assertEqual(Addition(digits_per_argument=2)(payload), [14])
        self.assertEqual(AbsoluteDifference()(inputs), [9, 99, 98])
        with self.assertRaises(TypeError):
            ArithmeticFunction()
        for task in (IntegerDivision(), Remainder()):
            with self.assertRaisesRegex(ValueError, "zero divisor"):
                task(torch.tensor([[1, 2, 0, 0]]))
        with self.assertRaises(ValueError):
            Addition(arity=3)(inputs)
        with self.assertRaises(ValueError):
            Subtraction(arity=3)

    def test_large_integer_arithmetic_does_not_overflow(self):
        payloads = torch.full((1, 40), 9, dtype=torch.long)
        self.assertEqual(Multiplication()(payloads), [(10 ** 20 - 1) ** 2])

    def test_unary_digit_tasks_preserve_zeros_ties_and_argument_width(self):
        payloads = torch.tensor([[9, 8, 7, 0], [0, 0, 1, 2], [9, 9, 1, 1],
                                 [7, 7, 2, 7], [0, 0, 0, 0], [8, 4, 8, 4]])
        self.assertEqual(AdditionMod10()(payloads), [4, 3, 0, 3, 0, 4])
        self.assertEqual(Majority()(payloads), [0, 0, 1, 7, 0, 4])
        self.assertEqual(AdditionMod10(digits_per_argument=2)(payloads), [7, 0, 8, 4, 0, 2])
        self.assertEqual(Majority(digits_per_argument=2)(payloads), [8, 0, 9, 7, 0, 4])
        for task in (AdditionMod10(), Majority()):
            self.assertEqual(task.arity, 1)
            self.assertEqual(task.required_output_tokens(100), 2)
            self.assertEqual(task(torch.arange(10)[:, None]), list(range(10)))
            with self.assertRaisesRegex(ValueError, "arity 1"):
                type(task)(arity=2)
            with self.assertRaisesRegex(ValueError, "too short"):
                type(task)(digits_per_argument=5)(payloads)
        self.assertEqual(Majority(digits_per_argument=4).evaluate((12,)), 0)

    def test_unary_digit_tasks_exhaustive_three_digit_domain(self):
        strings = [f"{value:03d}" for value in range(1000)]
        payloads = torch.tensor([list(map(int, value)) for value in strings])
        expected_majority = [int(min(Counter(value), key=lambda digit: (-value.count(digit), digit)))
                             for value in strings]
        # Removing decimal place values gives this independent digit-sum formula.
        expected_mod10 = [(value // 100 + value // 10 % 10 + value % 10) % 10
                          for value in range(1000)]
        self.assertEqual(Majority()(payloads), expected_majority)
        self.assertEqual(AdditionMod10()(payloads), expected_mod10)

    def test_unary_sensitivity_preserves_leading_zeros_and_rng(self):
        self.assertEqual(AdditionMod10().avg_sensitivity(6), 6.)
        self.assertEqual(AdditionMod10(digits_per_argument=2).avg_sensitivity(6), 2.)
        self.assertEqual(AdditionMod10(sensitivity=.25).avg_sensitivity(6), .25)
        self.assertEqual(Majority(sensitivity=.5).avg_sensitivity(6), .5)
        state = torch.get_rng_state().clone()
        self.assertEqual(Majority().avg_sensitivity(1, 32, 5), 1.)
        self.assertEqual(Majority().avg_sensitivity(4, 32, 5),
                         Majority().avg_sensitivity(4, 32, 5))
        # Changing one of four zero digits cannot change the majority.
        with patch('src.arithmetic.sample_payloads', return_value=torch.zeros(32, 4, dtype=torch.long)):
            self.assertEqual(Majority().avg_sensitivity(4, 32, 5), 0.)
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)

    def test_signed_encoding_eos_canonical_zero_and_limit(self):
        values = [0, 7, 100, -12]
        targets = encode_answers(values, 6)
        self.assertEqual(targets.tolist()[3], [MINUS, 1, 2, EOS, PAD, PAD])
        self.assertEqual([decode_answer(row) for row in targets.tolist()], values)
        for invalid in ([EOS], [0, 1, EOS], [MINUS, 0, EOS], [1, 2], [1, EOS, 2], [MINUS, EOS]):
            self.assertIsNone(decode_answer(invalid))
        with self.assertRaises(ValueError):
            encode_answers([-12], 3)
        self.assertEqual(ArithmeticSelector([Subtraction()], payload_len=4).required_output_tokens, 4)

    def test_grouped_and_decimal_selectors_duplicates_and_invalid_codes(self):
        selector = ArithmeticSelector(groups=[FunctionGroup("additive", [Addition(), Subtraction()]),
                                              FunctionGroup("multiplicative", [Multiplication(), IntegerDivision()])])
        self.assertEqual(selector.codes, [(0, 0), (0, 1), (1, 0), (1, 1)])
        payloads = torch.tensor([[0, 3, 1, 2]])
        inputs = torch.cat([selector.inputs_for(index, payloads) for index in range(4)])
        self.assertEqual(selector(inputs), [15, -9, 36, 0])
        self.assertEqual(selector.function_ids(inputs).tolist(), [0, 1, 2, 3])
        inputs[0, :2] = 9
        with self.assertRaises(ValueError):
            selector(inputs)
        flat = ArithmeticSelector([Addition()] * 12, payload_len=2)
        self.assertEqual(flat.codes[-1], (1, 1))
        self.assertEqual(flat.names, [f"addition_2_{index}" for index in range(12)])
        self.assertEqual(flat(flat.inputs_for(11, torch.tensor([[3, 8]]))), [11])
        with self.assertRaises(ValueError):
            ArithmeticSelector(groups=[FunctionGroup("empty", [])])

    def test_sensitivity_exact_formulas_override_and_private_rng(self):
        self.assertEqual(Addition(arity=3).avg_sensitivity(6), 6.)
        self.assertEqual(Addition(digits_per_argument=1).avg_sensitivity(6), 2.)
        self.assertAlmostEqual(Multiplication().avg_sensitivity(2), 1.8)
        self.assertEqual(Addition(sensitivity=0.125).avg_sensitivity(4), .125)
        state = torch.get_rng_state().clone()
        first = Subtraction().avg_sensitivity(4, num_inputs=32, seed=13)
        self.assertEqual(first, 4.)
        self.assertEqual(first, Subtraction().avg_sensitivity(4, num_inputs=32, seed=13))
        self.assertEqual(IntegerDivision().avg_sensitivity(2, 64, 2),
                         IntegerDivision().avg_sensitivity(2, 64, 2))
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)

    def test_balanced_sampling_valid_divisors_unique_held_out_and_reproducibility(self):
        selector = ArithmeticSelector([Addition(), IntegerDivision()], payload_len=2)
        state = torch.get_rng_state().clone()
        reserved = make_evaluation_set(selector, 4, 20, seed=9)
        again = make_evaluation_set(selector, 4, 20, seed=9)
        torch.testing.assert_close(reserved.inputs, again.inputs)
        self.assertEqual(len({tuple(row) for row in reserved.inputs.tolist()}), 40)
        inputs, _ = sample_batch(selector, 200, 4, balanced_selectors=True, held_out=reserved,
                                generator=torch.Generator().manual_seed(5))
        self.assertEqual(reserved.overlapping_rows(inputs), [])
        self.assertEqual(torch.bincount(selector.function_ids(inputs)).tolist(), [100, 100])
        divisor_rows = inputs[selector.function_ids(inputs) == 1]
        self.assertTrue(bool((divisor_rows[:, -1] != 0).all()))
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            make_evaluation_set(selector, 4, 90)
        for probabilities in ([.3], [1., 1.], [-1., 2.], [float("nan"), 0.]):
            with self.assertRaises(ValueError):
                sample_batch(selector, 8, 4, selector_probabilities=probabilities)
        inputs, _ = sample_batch(selector, 8, 4, selector_probabilities=[0., 1.])
        self.assertEqual(selector.function_ids(inputs).tolist(), [1] * 8)

    def test_architecture_reuse_and_no_future_answer_leakage(self):
        selector = ArithmeticSelector([Addition()])
        model = self.model(selector).eval()
        self.assertIsInstance(model, TransformerModel)
        self.assertEqual(model.embedding.embedding_dim, 4)
        self.assertEqual(model.positional_encoding.embedding_dim, 4)
        self.assertEqual(model.encoder.layers[0].linear1.out_features, 8)
        inputs = torch.tensor([[0, 1, 2, 3, 4]])
        targets = encode_answers([46], 5)
        changed = targets.clone()
        changed[0, 1] = 9
        with torch.no_grad():
            a = model.teacher_logits(inputs, targets)
            b = model.teacher_logits(inputs, changed)
        self.assertEqual(a.shape, (1, 5, 12))
        # Target token 1 only enters the input at answer position 2.
        torch.testing.assert_close(a[:, :2], b[:, :2], rtol=0, atol=0)
        self.assertFalse(torch.equal(a[:, 2], b[:, 2]))
        self.assertEqual(teacher_tokens(inputs, targets)[0, inputs.size(1)].item(), ANSWER)

    def test_loss_averages_sequences_masks_padding_and_includes_eos(self):
        targets = encode_answers([1, -12], 6)
        logits = torch.randn((2, 6, 12), requires_grad=True)
        criterion = SequenceCrossEntropy()
        actual = criterion(logits, targets)
        expected = torch.stack([-logits[row].log_softmax(-1)[positions, targets[row, positions]].mean()
                                for row in range(2)
                                for positions in [(targets[row] != PAD).nonzero(as_tuple=True)[0]]]).mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, criterion(logits, targets.float()))
        actual.backward()
        self.assertTrue(bool((logits.grad[targets == PAD] == 0).all()))
        self.assertTrue(bool((logits.grad[targets == EOS].abs().sum(-1) > 0).all()))

    def test_generation_stops_per_example_and_at_cap_and_restores_modes(self):
        selector = ArithmeticSelector([Addition()], payload_len=2)

        class Scripted(ArithmeticTransformer):
            def forward(self, tokens, prompt_len=0):
                position = tokens.size(1) - prompt_len - 1
                logits = torch.full((tokens.size(0), tokens.size(1), 12), -100.)
                scripts = ([4, EOS, 8, 8], [MINUS, 2, EOS, 8], [5, 5, 5, 5])
                for row, script in enumerate(scripts):
                    logits[row, -1, script[position]] = 100.
                return logits

        model = Scripted(d_model=8, num_attn_heads=2, num_layers=1, max_len=7)
        model.embedding.eval()
        modes = [module.training for module in model.modules()]
        inputs = torch.tensor([[0, 1, 2]] * 3)
        generated = model.generate(inputs, 4)
        self.assertEqual(generated.tolist(), [[4, EOS, PAD, PAD], [MINUS, 2, EOS, PAD], [5, 5, 5, 5]])
        self.assertEqual(modes, [module.training for module in model.modules()])
        targets = encode_answers([4, -2, 5], 4)
        metrics = sequence_metrics(selector, inputs, torch.zeros(3, 4, 12), targets, generated)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["truncation_rate"], 1 / 3)

    def test_gradient_and_affinity_adapters_match_direct_arithmetic_loss(self):
        selector = ArithmeticSelector(groups=[FunctionGroup("one", [Addition()]),
                                              FunctionGroup("two", [Subtraction()])], payload_len=2)
        torch.manual_seed(2)
        model = self.model(selector).double().eval()
        probe = DiagnosticAdapter(model, selector, 5, 3, 9)
        criterion = SequenceCrossEntropy()
        metrics = gradient_metrics(probe, probe.function, criterion, probe.payloads)
        direct_gradients = []
        for index, name in enumerate(selector.names):
            loss = criterion(model.teacher_logits(probe.probe_inputs[index], probe.probe_targets[index]),
                             probe.probe_targets[index])
            grads = torch.autograd.grad(loss, tuple(model.parameters()))
            direct_gradients.append(grads)
            norm = float(torch.cat([gradient.flatten() for gradient in grads]).norm())
            self.assertAlmostEqual(metrics[f"gradient_norm/{name}"], norm, places=10)
        lr = .003
        affinities = task_affinity_metrics(probe, probe.function, criterion, probe.payloads,
                                          learning_rate=lr, eval_batch_size=2)
        for source, grads in enumerate(direct_gradients):
            parameters = {name: parameter - lr * gradient
                          for (name, parameter), gradient in zip(model.named_parameters(), grads)}
            for target, (inputs, labels) in enumerate(zip(probe.probe_inputs, probe.probe_targets)):
                baseline = criterion(model.teacher_logits(inputs, labels), labels)
                logits = functional_call(model, parameters, (teacher_tokens(inputs, labels), inputs.size(1)))[:, inputs.size(1):]
                expected = float(1 - criterion(logits, labels) / baseline)
                self.assertAlmostEqual(affinities[f"affinity/{selector.names[source]}_to_{selector.names[target]}"],
                                       expected, places=10)

    def test_diagnostics_evaluation_preserve_state_gradients_modes_rng(self):
        selector = ArithmeticSelector([Addition(), Subtraction()], payload_len=2)
        model = self.model(selector, dropout=.3)
        model.embedding.eval()
        probe = DiagnosticAdapter(model, selector, 5, 2, 0)
        state = copy.deepcopy(model.state_dict())
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 7.)
        grad_refs = [parameter.grad for parameter in model.parameters()]
        modes = [module.training for module in model.modules()]
        rng = torch.get_rng_state().clone()
        criterion = SequenceCrossEntropy()
        gradient_metrics(probe, probe.function, criterion, probe.payloads)
        task_affinity_metrics(probe, probe.function, criterion, probe.payloads)
        before = {name: parameter.detach().clone() for name, parameter in probe.named_parameters()}
        curvature = multi_task_curvature(probe, probe.function, criterion, probe.payloads, before, num_points=1)
        self.assertTrue(math.isfinite(curvature["curvature/multi_task"]))
        self.assertEqual(average_direction_sharpness(probe, 1, rho=0)["sharpness"], 0.)
        self.assertGreater(average_direction_sharpness(probe, 1, rho=.02)["sharpness"], 0.)
        spaces, directions = loss_spaces(probe, probe.function, criterion, probe.payloads,
                                        delta=.02, epsilon=.002, num_directions=1, hvp_budget=2)
        self.assertIn("spaces/parameter_count", spaces)
        self.assertTrue(directions)
        evaluation = make_evaluation_set(selector, 5, 3)
        metrics = evaluate_fixed_set(model, selector, evaluation, 5, batch_size=2)
        self.assertIn("accuracy/addition_2", metrics)
        self.assertEqual(modes, [module.training for module in model.modules()])
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, state[name], rtol=0, atol=0)
        for parameter, reference in zip(model.parameters(), grad_refs):
            self.assertIs(parameter.grad, reference)
            self.assertTrue(bool((parameter.grad == 7).all()))

    def test_training_diagnostics_do_not_change_adamw_updates(self):
        selector = ArithmeticSelector([Addition(), Subtraction()], payload_len=2)
        initial = self.model(selector, dropout=.2)
        outcomes = []
        for interval in (None, 1):
            model = copy.deepcopy(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            torch.manual_seed(11)
            config = TrainingConfig(num_steps=2, batch_size=4, max_output_tokens=5,
                                    eval_interval=1, eval_num_inputs_per_function=3, eval_batch_size=2,
                                    gradient_interval=interval, affinity_interval=interval,
                                    curvature_interval=interval, sharpness_interval=interval,
                                    gradient_num_inputs=2, affinity_num_inputs=2, sharpness_num_inputs=2,
                                    curvature_num_points=1, sharpness_num_perturbations=1,
                                    sensitivity_num_inputs=8)
            with tempfile.TemporaryDirectory() as directory:
                result = train(model, selector, optimizer, config, output_dir=Path(directory))
                self.assertEqual(len((Path(directory) / "metrics.jsonl").read_text().splitlines()), 2)
            outcomes.append((result.losses, model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        for name, parameter in outcomes[0][1].items():
            torch.testing.assert_close(parameter, outcomes[1][1][name], rtol=0, atol=0)
        for index, values in outcomes[0][2]["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(value, outcomes[1][2]["state"][index][key], rtol=0, atol=0)
        torch.testing.assert_close(outcomes[0][3], outcomes[1][3], rtol=0, atol=0)

    def test_config_rejects_truncating_targets_and_cli_builds_requested_layout(self):
        import test_arithmetic as experiment
        args = experiment.build_parser().parse_args(["--grouped-selectors", "--digits-per-argument", "3"])
        selector = experiment.build_selector(args)
        self.assertEqual(selector.payload_len, 6)
        self.assertEqual(selector.codes, [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)])
        self.assertEqual(selector.required_output_tokens, 7)
        with self.assertRaisesRegex(ValueError, "fit every answer"):
            TrainingConfig(max_output_tokens=3).validate(selector, self.model(selector))
        args = experiment.build_parser().parse_args(["--payload-len", "6", "--arity", "3"])
        selector = experiment.build_selector(args)
        self.assertEqual([task.argument_width(6) for task in selector.functions], [2, 3, 2, 3, 3, 6, 6])

    def test_cli_mix_includes_unary_tasks_in_sampling_and_evaluation(self):
        import test_arithmetic as experiment
        for flags in ([], ["--grouped-selectors"]):
            with self.subTest(flags=flags):
                selector = experiment.build_selector(experiment.build_parser().parse_args(flags))
                self.assertEqual(len(selector.functions), 7)
                self.assertEqual(selector.names[-2:], ["addition_mod_10_1", "majority_1"])
                self.assertEqual([task.arity for task in selector.functions], [2, 2, 2, 2, 2, 1, 1])
                self.assertEqual([task.argument_width(4) for task in selector.functions], [2, 2, 2, 2, 2, 4, 4])
                payloads = torch.tensor([[9, 9, 1, 1], [0, 0, 7, 7]])
                for index, answers in ((5, [0, 4]), (6, [1, 0])):
                    targets = selector.targets(selector.inputs_for(index, payloads), 5)
                    self.assertEqual([decode_answer(row) for row in targets.tolist()], answers)
                held_out = make_evaluation_set(selector, 5, 4)
                inputs, targets = sample_batch(selector, 28, 5, balanced_selectors=True, held_out=held_out,
                                                generator=torch.Generator().manual_seed(4))
                self.assertEqual(torch.bincount(selector.function_ids(inputs)).tolist(), [4] * 7)
                self.assertEqual(held_out.overlapping_rows(inputs), [])
                self.assertEqual([decode_answer(row) for row in targets.tolist()], selector(inputs))
        args = experiment.build_parser().parse_args(
            ["--operations", "addition_mod_10", "majority", "--payload-len", "4", "--grouped-selectors"])
        selector = experiment.build_selector(args)
        self.assertEqual(selector.group_names, ["digit"])
        self.assertEqual(selector.required_output_tokens, 2)
        payloads = torch.tensor([[9, 8, 7, 0], [0, 0, 1, 2]])
        self.assertEqual(selector(selector.inputs_for(0, payloads)), [4, 3])
        self.assertEqual(selector(selector.inputs_for(1, payloads)), [0, 0])


if __name__ == "__main__":
    unittest.main()
