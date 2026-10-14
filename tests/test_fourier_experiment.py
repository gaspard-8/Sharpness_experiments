"""Fourier identities, cumulative-loss semantics, and real single-task updates."""

import contextlib
import copy
from itertools import product
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import sweep_fourier as sweep
import test_fourier as experiment
from src.evaluation import EvaluationSet
from src.fourier import FUNCTION_NAMES, FourierFunction, make_function
from src.fourier_training import (
    FourierEvaluation, TrainingConfig, evaluate_cumulative, output_sharpness, prepare_evaluation, train,
)
from src.model import TransformerModel


ROOT = Path(__file__).resolve().parents[1]


class CapturedRun:
    def __init__(self):
        self.records = []
        self.finished = False

    def log(self, metrics, step):
        self.records.append((step, dict(metrics)))

    def finish(self, **kwargs):
        self.finished = True


class FourierTargetTest(unittest.TestCase):
    def test_exact_transform_matches_specified_coefficients_and_decreasing_mass(self):
        inputs = torch.tensor(list(product((0, 1), repeat=5)))
        # Enumerate ALL characters, including unsupported ones and the constant.
        supports = [tuple(i for i, included in enumerate(mask) if included)
                    for mask in product((0, 1), repeat=5)]
        characters = torch.stack([
            (1 - 2 * inputs[:, subset]).prod(dim=1) if subset else torch.ones(len(inputs))
            for subset in supports
        ], dim=1).double()
        for name in FUNCTION_NAMES:
            with self.subTest(name=name):
                function = make_function(name, input_len=5, sparse_terms=3)
                values = function.cumulative(inputs, dtype=torch.float64)[:, -1]
                actual_coefficients = characters.T @ values / len(inputs)
                specified = dict(zip(function.subsets, function.coefficients))
                expected = torch.tensor([specified.get(s, 0.) for s in supports], dtype=torch.float64)
                torch.testing.assert_close(actual_coefficients, expected, atol=1e-14, rtol=1e-14)
                self.assertAlmostEqual(float(values.mean()), 0.)
                self.assertAlmostEqual(float(values.square().mean()), 1.)
                measured_mass = [float(actual_coefficients[[len(s) == degree for s in supports]].square().sum())
                                 for degree in range(1, 6)]
                for degree, mass in enumerate(measured_mass):
                    self.assertAlmostEqual(mass, function.degree_mass[degree])
                self.assertTrue(all(a > b > 0 for a, b in zip(measured_mass, measured_mass[1:])))
                for degree in range(6):
                    approximation = (torch.zeros_like(values) if degree == 0 else
                                     function.cumulative(inputs, dtype=torch.float64)[:, degree - 1])
                    self.assertAlmostEqual(float((values - approximation).square().mean()), function.tail_mass[degree])
                    self.assertEqual(function.approximation_degree(function.tail_mass[degree]), degree)
                self.assertEqual(function.approximation_degree(0), 5)
                self.assertEqual(function.approximation_degree(2), 0)

    def test_32_bit_targets_match_mass_across_layouts_and_are_reproducible(self):
        before = torch.get_rng_state().clone()
        for spectrum in ("powerlaw", "geometric"):
            concentrated = FourierFunction(spectrum=spectrum)
            sparse = FourierFunction(layout="sparse", spectrum=spectrum)
            repeated = FourierFunction(layout="sparse", spectrum=spectrum)
            changed = FourierFunction(layout="sparse", spectrum=spectrum, seed=18)
            self.assertEqual(concentrated.degree_mass, sparse.degree_mass)
            self.assertEqual(sparse.metadata(), repeated.metadata())
            self.assertNotEqual(sparse.subsets, changed.subsets)
            self.assertEqual(len(concentrated.subsets), 32)
            for degree in range(1, 33):
                subset = concentrated.subsets[degree - 1]
                self.assertEqual(subset, tuple(range(degree)))
                start, end = sparse.offsets[degree - 1:degree + 1]
                self.assertEqual(end - start, min(16, math.comb(32, degree)))
                self.assertEqual(len(set(sparse.subsets[start:end])), end - start)
                self.assertAlmostEqual(sum(c * c for c in sparse.coefficients[start:end]), sparse.degree_mass[degree - 1])
        torch.testing.assert_close(torch.get_rng_state(), before)

    def test_degree_cap_and_invalid_configurations(self):
        function = FourierFunction(input_len=32, max_degree=12)
        self.assertEqual(function.max_degree, 12)
        self.assertEqual(function.tail_mass[-1], 0.)
        self.assertEqual(function(torch.zeros(2, 32, dtype=torch.long)).shape, (2,))
        for kwargs in ({"input_len": 0}, {"max_degree": 33}, {"max_degree": True},
                       {"sparse_terms": 0}, {"power": 0}, {"geometric_decay": 1},
                       {"geometric_decay": float("nan")}, {"power": float("inf")},
                       {"layout": "dense"}, {"spectrum": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FourierFunction(**kwargs)
        with self.assertRaises(ValueError):
            function(torch.zeros(2, 31, dtype=torch.long))
        with self.assertRaises(ValueError):
            function.approximation_degree(-0.1)


class FourierEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_one_forward_bank_scores_all_cumulative_targets_and_matches_tail_identity(self):
        function = make_function("sparse_geometric", input_len=5, sparse_terms=3)
        inputs = torch.tensor(list(product((0, 1), repeat=5)))
        targets = function.cumulative(inputs, dtype=torch.float64)
        evaluation = FourierEvaluation(EvaluationSet(inputs, targets[:, -1]), targets)

        class PerfectModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                self.assert_eval = not self.training and not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
                return function.cumulative(x, dtype=torch.float64)[:, -1]

        model = PerfectModel()
        metrics = evaluate_cumulative(model, evaluation, batch_size=7)
        self.assertEqual(model.calls, math.ceil(len(inputs) / 7))
        self.assertTrue(model.assert_eval)
        self.assertTrue(model.training)
        for degree in range(1, 6):
            self.assertAlmostEqual(metrics[f"eval/loss_degree_{degree}"], function.tail_mass[degree])
        self.assertEqual(metrics["eval/loss"], 0.)
        self.assertEqual(metrics["eval/loss_degree_5"], 0.)

    def test_chunked_evaluation_and_sharpness_preserve_rng_modes_parameters_and_gradients(self):
        torch.manual_seed(12)
        function = make_function("sparse_powerlaw", input_len=32, max_degree=8)
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=32, dropout=0.2)
        model.positional_encoding.eval()
        modes = [module.training for module in model.modules()]
        state = copy.deepcopy(model.state_dict())
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        gradients = [p.grad for p in model.parameters()]
        rng = torch.get_rng_state().clone()
        evaluation = prepare_evaluation(function, num_inputs=9)
        metrics = evaluate_cumulative(model, evaluation, batch_size=4)
        whole = evaluate_cumulative(model, evaluation, batch_size=9)
        for key in metrics:
            self.assertAlmostEqual(metrics[key], whole[key], places=6)
        sharpness = output_sharpness(model, function, num_inputs=3, num_perturbations=2)
        self.assertTrue(math.isfinite(sharpness["sharpness"]))
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertEqual([module.training for module in model.modules()], modes)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name])
        for parameter, gradient in zip(model.parameters(), gradients):
            self.assertIs(parameter.grad, gradient)
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        with patch.object(model, "forward", side_effect=RuntimeError("broken")):
            with self.assertRaises(RuntimeError):
                evaluate_cumulative(model, evaluation)
        self.assertEqual([module.training for module in model.modules()], modes)

    def test_real_updates_use_only_full_labels_exclude_holdout_and_evaluate_post_update_without_logger(self):
        function = make_function("concentrated_powerlaw", input_len=32, max_degree=5)
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=32)
        config = TrainingConfig(num_steps=3, batch_size=4, eval_num_inputs=5,
                                eval_batch_size=3, sharpness_interval=None)
        evaluation = prepare_evaluation(function, config.eval_num_inputs, config.eval_seed)
        seen_training, seen_evaluation = [], []

        def observe(module, args, output):
            if module.training:
                seen_training.append(args[0].detach().clone())
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(evaluation.held_out.overlapping_rows(args[0]), [])
            else:
                seen_evaluation.append(args[0].detach().clone())
                self.assertFalse(torch.is_grad_enabled())

        handle = model.register_forward_hook(observe)
        initial = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            try:
                result = train(model, function, optimizer, config, output_dir=Path(directory))
            finally:
                handle.remove()
            records = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in records], [0, 1, 2, 3])
            self.assertEqual(len(seen_training), 3)
            self.assertEqual(len(seen_evaluation), 8)  # Baseline + 3 updates, two batches each.
            for offset in range(0, 8, 2):
                torch.testing.assert_close(torch.cat(seen_evaluation[offset:offset + 2]), evaluation.held_out.inputs)
            final = evaluate_cumulative(model, evaluation, config.eval_batch_size)
            for key, value in final.items():
                self.assertAlmostEqual(records[-1][key], value)
            self.assertEqual(len(result.losses), 3)
            self.assertTrue(any(not torch.equal(value, initial[name]) for name, value in model.state_dict().items()))
            stored = torch.load(Path(directory) / "evaluation.pt", weights_only=True)
            torch.testing.assert_close(stored["cumulative_targets"], evaluation.cumulative_targets)


class FourierSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_grid_is_independent_one_task_one_layer_and_listing_does_not_train(self):
        grid = sweep.build_grid()
        self.assertEqual(len(grid), 4)
        self.assertEqual({row["function"] for row in grid}, set(FUNCTION_NAMES))
        self.assertEqual(len({tuple(row.items()) for row in grid}), 4)
        for index, row in enumerate(grid):
            args = sweep.build_parser().parse_args(["--run-index", str(index)])
            options = experiment.build_parser().parse_args(sweep.experiment_arguments(args))
            config = experiment.build_model_config(options)
            self.assertEqual(options.function, row["function"])
            self.assertEqual(options.batch_size, row["batch_size"])
            self.assertEqual(config["num_layers"], 1)
            self.assertGreater(config["max_len"], 24)
            self.assertEqual(config["d_model"], 2 * row["embedding_size"])
            self.assertEqual(config["d_model"], 256)
            self.assertEqual(config["num_attn_heads"], 8)
            self.assertEqual(options.batch_size, 320)
            self.assertEqual(options.function_seed, 17)
            self.assertEqual(options.eval_seed, 23)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(experiment, "main") as run:
            sweep.main(["--list"])
        run.assert_not_called()
        self.assertEqual(len(json.loads(output.getvalue())), 4)
        for index in (-1, 4):
            with self.assertRaises(ValueError):
                sweep.experiment_arguments(sweep.build_parser().parse_args(["--run-index", str(index)]))

    def test_each_of_the_four_targets_has_a_real_smoke_run_with_every_degree_logged(self):
        for index, name in enumerate(FUNCTION_NAMES):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                run = CapturedRun()
                wandb = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs)
                with patch.dict(sys.modules, {"wandb": wandb}), \
                     patch.object(wandb, "init", return_value=run) as initialize, \
                     contextlib.redirect_stdout(io.StringIO()):
                    output = sweep.main(["--run-index", str(index), "--smoke-test", "--device", "cpu",
                                         "--output-dir", directory])
                self.assertTrue(run.finished)
                self.assertEqual([step for step, _ in run.records], [0, 1, 2])
                config = json.loads((output / "config.json").read_text())
                self.assertEqual(config["num_tasks"], 1)
                self.assertEqual(config["selector_size"], 0)
                self.assertEqual(config["model_num_layers"], 1)
                self.assertEqual(config["model_d_model"], 256)
                self.assertEqual(config["model_num_attn_heads"], 8)
                self.assertEqual(config["train_batch_size"], 320)
                self.assertEqual(config["function"]["name"], name)
                self.assertEqual(config["function"]["max_degree"], 32)
                self.assertEqual(json.loads(json.dumps(initialize.call_args.kwargs["config"])), config)
                for step, metrics in run.records:
                    for degree in range(1, 33):
                        self.assertTrue(math.isfinite(metrics[f"eval/loss_degree_{degree}"]))
                    self.assertEqual(metrics["eval/loss"], metrics["eval/loss_degree_32"])
                    if step:
                        self.assertIn("loss", metrics)
                        self.assertIn("sharpness", metrics)
                checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
                model = TransformerModel(**checkpoint["model_config"])
                self.assertEqual(len(model.encoder.layers), 1)
                model.load_state_dict(checkpoint["model_state_dict"])
                self.assertEqual(checkpoint["completed_steps"], 2)
                self.assertTrue((output / "target.json").is_file())

    def test_submission_queues_four_targets_and_preserves_overrides(self):
        submit = (ROOT / "sweep_fourier.sub").read_text()
        self.assertIn("queue 4", submit)
        self.assertIn("SHARPNESS_ENTRYPOINT=sweep_fourier.py", submit)
        self.assertIn("$(project_dir)/test_fourier.py", submit)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "submit-fourier-sweep.sh").write_text((ROOT / "submit-fourier-sweep.sh").read_text())
            (project / "submit-wandb.sh").write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$SHARPNESS_SUBMIT_FILE" "$@"\n')
            captured = subprocess.run(["bash", str(project / "submit-fourier-sweep.sh"), "-append",
                                       "experiment_args=--steps 1000 --seed 19"], check=True, capture_output=True)
            self.assertEqual(captured.stdout.decode().rstrip("\0").split("\0"), [
                str(project / "sweep_fourier.sub"), "-append", f"project_dir={project}",
                "-append", "experiment_args=--steps 1000 --seed 19"])

    def test_invalid_architectures_lengths_and_existing_results_are_rejected(self):
        for flags in (["--input-len", "24"], ["--embedding-size", "7", "--num-attn-heads", "8"]):
            with self.assertRaises(ValueError):
                experiment.build_model_config(experiment.build_parser().parse_args(flags))
        function = FourierFunction(input_len=32)
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=2, max_len=32)
        with self.assertRaisesRegex(ValueError, "one transformer layer"):
            TrainingConfig().validate(function, model)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "keep.txt").write_text("previous result")
            with self.assertRaisesRegex(ValueError, "preserve previous results"):
                experiment.main(["--smoke-test", "--no-wandb", "--device", "cpu", "--output-dir", directory])
            self.assertEqual((Path(directory) / "keep.txt").read_text(), "previous result")


if __name__ == "__main__":
    unittest.main()
