"""Input conditional expectations, controlled spectra, and real sweep updates."""

import contextlib
from itertools import product
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch._dynamo  # Load C++ registrations before the sys.modules W&B mocks.

import sweep_fourier
import sweep_input_tower as sweep
import test_fourier as experiment
from src.fourier_training import evaluate_cumulative, prepare_evaluation
from src.input_tower import INPUT_TOWER_FUNCTION_NAMES, InputTowerFunction, make_input_tower


ROOT = Path(__file__).resolve().parents[1]


class InputTowerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(7)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_exact_conditional_expectations_and_tail_errors_on_complete_cube(self):
        inputs = torch.tensor(list(product((0, 1), repeat=5)))
        for name in INPUT_TOWER_FUNCTION_NAMES:
            with self.subTest(name=name):
                function = make_input_tower(name, input_len=5, core_size=2, terms_per_stage_degree=3,
                                           input_order="permuted")
                values = function.cumulative(inputs, dtype=torch.float64)[:, -1]
                input_targets = function.input_cumulative(inputs, dtype=torch.float64)
                self.assertAlmostEqual(float(values.mean()), 0., places=13)
                self.assertAlmostEqual(float(values.square().mean()), 1., places=13)
                for count in range(1, 6):
                    expected = []
                    positions = list(function.input_order[:count])
                    for row in inputs:
                        same_prefix = (inputs[:, positions] == row[positions]).all(dim=1)
                        expected.append(values[same_prefix].mean())
                    torch.testing.assert_close(input_targets[:, count - 1], torch.stack(expected), atol=1e-13, rtol=1e-13)
                    residual = (values-input_targets[:, count - 1]).square().mean()
                    self.assertAlmostEqual(float(residual), function.input_tail_mass[count], places=13)
                    self.assertEqual(function.approximation_inputs(function.input_tail_mass[count]), count)
                for count in range(2, 6):
                    before = input_targets[:, count - 2] if count > 2 else torch.zeros_like(values)
                    # The initial stage includes the whole core; later stages
                    # add exactly the prescribed variance for that new input.
                    if count == 2:
                        before = torch.zeros_like(values)
                    contribution = (input_targets[:, count - 1] - before).square().mean()
                    self.assertAlmostEqual(float(contribution), function.stage_mass[count-2], places=13)
                for degree in range(1, 3):
                    component = function.components(inputs, dtype=torch.float64)[:, degree - 1]
                    self.assertAlmostEqual(float(component.square().mean()), function.degree_mass[degree-1], places=13)

    def test_default_profiles_share_input_stage_mass_support_and_signs(self):
        functions = [make_input_tower(name) for name in INPUT_TOWER_FUNCTION_NAMES]
        reference = functions[0]
        for function in functions:
            self.assertEqual(function.subsets, reference.subsets)
            self.assertEqual(function.stage_mass, reference.stage_mass)
            self.assertEqual([math.copysign(1, a) for a in function.coefficients],
                             [math.copysign(1, a) for a in reference.coefficients])
            self.assertEqual(len(set(function.subsets)), len(function.subsets))
            self.assertEqual(function.metadata()["relevant_inputs"], list(range(32)))
            self.assertEqual(function.metadata()["degree_profile_scope"], "global")
            for count in range(8, 33):
                self.assertAlmostEqual(function.input_tail_mass[count], reference.input_tail_mass[count], places=13)
            for actual, requested in zip(function.degree_mass, function.requested_degree_mass):
                self.assertAlmostEqual(actual, requested, places=13)
            self.assertTrue(all(a > b for a, b in zip(function.stage_mass, function.stage_mass[1:])))
        increasing, randomized, top_heavy = functions
        self.assertTrue(all(a < b for a, b in zip(increasing.degree_mass, increasing.degree_mass[1:])))
        deltas = [b-a for a, b in zip(randomized.degree_mass, randomized.degree_mass[1:])]
        self.assertTrue(any(delta > 0 for delta in deltas) and any(delta < 0 for delta in deltas))
        self.assertAlmostEqual(sum(top_heavy.degree_mass[-2:]), .9)

    def test_permutation_only_renames_inputs_and_rng_is_independent(self):
        inputs = torch.tensor(list(product((0, 1), repeat=5)))
        rng = torch.get_rng_state().clone()
        python_rng = random.getstate()
        prefix = InputTowerFunction(input_len=5, core_size=3, terms_per_stage_degree=4)
        permuted = InputTowerFunction(input_len=5, core_size=3, terms_per_stage_degree=4, input_order="permuted")
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertEqual(random.getstate(), python_rng)
        reordered = inputs[:, list(permuted.input_order)]
        torch.testing.assert_close(permuted.cumulative(inputs, dtype=torch.float64),
                                   prefix.cumulative(reordered, dtype=torch.float64), atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(permuted.input_cumulative(inputs, dtype=torch.float64),
                                   prefix.input_cumulative(reordered, dtype=torch.float64), atol=1e-13, rtol=1e-13)

    def test_one_input_core_reports_actual_conditional_degree_spectrum(self):
        function = InputTowerFunction(input_len=5, core_size=1, max_degree=5)
        self.assertEqual(function.metadata()["degree_profile_scope"], "conditional_on_available_degrees_per_stage")
        self.assertAlmostEqual(function.input_mass[0], function.stage_mass[0], places=13)
        self.assertGreater(function.degree_mass[0], function.degree_mass[-1])
        self.assertNotEqual(function.degree_mass, function.requested_degree_mass)
        for stage in range(5):
            self.assertTrue(all(mass == 0 for mass in function.stage_degree_mass[stage][stage+1:]))
            self.assertAlmostEqual(sum(function.stage_degree_mass[stage]), function.stage_mass[stage])

    def test_one_prediction_bank_scores_both_axes(self):
        function = InputTowerFunction(input_len=5, core_size=2)
        evaluation = prepare_evaluation(function, num_inputs=9)

        class PerfectModel(torch.nn.Module):
            calls = 0

            def forward(self, inputs):
                self.calls += 1
                self.assert_eval = not self.training and not torch.is_grad_enabled()
                return function.cumulative(inputs, dtype=torch.float64)[:, -1]

        model = PerfectModel()
        metrics = evaluate_cumulative(model, evaluation, batch_size=4)
        self.assertEqual(model.calls, 3)
        self.assertTrue(model.assert_eval)
        self.assertTrue(model.training)
        self.assertEqual(metrics["eval/loss"], metrics["eval/loss_degree_2"])
        self.assertEqual(metrics["eval/loss"], metrics["eval/loss_input_5"])
        for count in range(1, 6):
            expected = (evaluation.cumulative_targets[:, -1] - evaluation.input_targets[:, count - 1]).square().mean()
            self.assertAlmostEqual(metrics[f"eval/loss_input_{count}"], float(expected))

    def test_invalid_inputs_and_spectra_are_rejected(self):
        for kwargs in ({"core_size": 33}, {"core_size": 0}, {"max_degree": 33},
                       {"degree_profile": "missing"}, {"input_spectrum": "missing"},
                       {"input_power": 0}, {"degree_power": float("nan")},
                       {"input_decay": 1}, {"top_mass": 1}, {"top_band": 9},
                       {"top_mass": .2}, {"input_order": "missing"}, {"terms_per_stage_degree": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                InputTowerFunction(**kwargs)

    def test_three_real_sweep_smokes_keep_model_fixed_and_log_every_input_and_degree(self):
        class CapturedRun:
            def __init__(self):
                self.records = []
                self.finished = False

            def log(self, metrics, step):
                self.records.append((step, dict(metrics)))

            def finish(self, **kwargs):
                self.finished = True

        for index, name in enumerate(INPUT_TOWER_FUNCTION_NAMES):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                run = CapturedRun()
                wandb = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs)
                with patch.dict(sys.modules, {"wandb": wandb}), contextlib.redirect_stdout(io.StringIO()):
                    output = sweep.main(["--run-index", str(index), "--smoke-test", "--device", "cpu", "--output-dir", directory])
                self.assertTrue(run.finished)
                config = json.loads((output / "config.json").read_text())
                self.assertEqual(config["experiment"], "single_task_input_tower")
                self.assertEqual(config["function"]["name"], name)
                self.assertEqual(config["model_num_layers"], 1)
                self.assertEqual(config["model_d_model"], 256)
                self.assertEqual(config["model_num_attn_heads"], 8)
                self.assertEqual(config["train_batch_size"], 64)
                self.assertEqual(config["num_tasks"], 1)
                self.assertEqual([step for step, _ in run.records], [0, 1, 2])
                for _, metrics in run.records:
                    for degree in range(1, 9):
                        self.assertTrue(math.isfinite(metrics[f"eval/loss_degree_{degree}"]))
                    for count in range(1, 33):
                        self.assertTrue(math.isfinite(metrics[f"eval/loss_input_{count}"]))
                    self.assertEqual(metrics["eval/loss_degree_8"], metrics["eval/loss_input_32"])
                stored = torch.load(output / "evaluation.pt", weights_only=True)
                self.assertEqual(stored["cumulative_targets"].shape, (8, 8))
                self.assertEqual(stored["input_targets"].shape, (8, 32))
                torch.testing.assert_close(stored["input_order"], torch.arange(32))
                self.assertEqual(torch.load(output / "checkpoint.pt", weights_only=True)["completed_steps"], 2)

    def test_sweep_has_three_functions_legacy_grid_is_unchanged_and_submission_forwards_overrides(self):
        self.assertEqual(len(sweep.build_grid()), 3)
        self.assertEqual(len(sweep_fourier.build_grid()), 4)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), patch.object(experiment, "main") as run:
            sweep.main(["--list"])
        run.assert_not_called()
        rows = json.loads(captured.getvalue())
        self.assertEqual([row["function"] for row in rows], list(INPUT_TOWER_FUNCTION_NAMES))
        for index in (-1, 3):
            with self.assertRaises(ValueError):
                sweep.experiment_arguments(sweep.build_parser().parse_args(["--run-index", str(index)]))
        submit = (ROOT / "sweep_input_tower.sub").read_text()
        self.assertIn("queue 3", submit)
        self.assertIn("SHARPNESS_ENTRYPOINT=sweep_input_tower.py", submit)
        self.assertIn("$(project_dir)/test_fourier.py", submit)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "submit-input-tower-sweep.sh").write_text((ROOT / "submit-input-tower-sweep.sh").read_text())
            (project / "submit-wandb.sh").write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$SHARPNESS_SUBMIT_FILE" "$@"\n')
            result = subprocess.run(["bash", str(project / "submit-input-tower-sweep.sh"), "-append",
                                     "experiment_args=--input-order permuted"], check=True, capture_output=True)
            self.assertEqual(result.stdout.decode().rstrip("\0").split("\0"), [
                str(project / "sweep_input_tower.sub"), "-append", f"project_dir={project}",
                "-append", "experiment_args=--input-order permuted"])


if __name__ == "__main__":
    unittest.main()
