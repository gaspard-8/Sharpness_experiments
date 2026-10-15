"""Product-alphabet identities, multiplication digit semantics, and large models."""

import contextlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import sweep_multiplication_digit as sweep
import test_multiplication_digit as experiment
from src import multiplication_digit as training
from src.multiplication_digit import MultiplicationDigitTransformer, TrainingConfig, hundreds_digit, inputs_from_codes
from src.multiplication_fourier import (alphabet_basis, cumulative_projections, grouped_mass,
                                       load_manifest, product_transform, support_degrees)


ROOT = Path(__file__).resolve().parents[1]


class MultiplicationDigitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.targets = tempfile.TemporaryDirectory()
        for name in ("manifest.json", "multiplication_hundreds_spectrum.npz",
                     "evaluation-n1024-seed23.npz", "evaluation-n8-seed23.npz"):
            path = ROOT / "multiplication_fourier_targets" / name
            if path.exists():
                shutil.copy2(path, Path(cls.targets.name) / name)

    @classmethod
    def tearDownClass(cls):
        cls.targets.cleanup()

    def test_uniform_alphabet_basis_transform_projections_and_basis_invariance(self):
        basis = alphabet_basis(3)
        np.testing.assert_allclose(basis @ basis.T / 3, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(alphabet_basis() @ alphabet_basis().T / 10, np.eye(10), atol=1e-14)
        inputs = np.indices((3,) * 3).reshape(3, -1).T
        labels = (inputs[:, 0] * inputs[:, 1] + inputs[:, 2]) % 3
        values = np.eye(3)[labels].reshape(3, 3, 3, 3)
        coefficients = product_transform(values, basis, 3)
        np.testing.assert_allclose(product_transform(coefficients, basis, 3, inverse=True), values, atol=1e-14)
        characters = np.ones((27, 27))
        for axis in range(3):
            characters *= basis[inputs[:, axis][None, :], inputs[:, axis][:, None]]
        np.testing.assert_allclose(coefficients.reshape(27, 3), characters.T @ values.reshape(27, 3) / 27, atol=1e-14)
        degrees = support_degrees(3, 3)
        projections = cumulative_projections(values, inputs, 3)
        for cutoff in range(4):
            retained = coefficients * (degrees <= cutoff)[..., None]
            expected = product_transform(retained, basis, 3, inverse=True).reshape(27, 3)
            np.testing.assert_allclose(projections[:, cutoff], expected, atol=1e-14)
        rotation = np.array([[1., 0., 0.], [0., .6, -.8], [0., .8, .6]])
        rotated = product_transform(values, rotation @ basis, 3)
        np.testing.assert_allclose(grouped_mass(coefficients, degrees, 3), grouped_mass(rotated, degrees, 3), atol=1e-14)
        self.assertAlmostEqual(float(grouped_mass(coefficients, degrees, 3).sum()), 1.)

    def test_hundreds_digit_leading_zeros_input_order_and_commutativity(self):
        operands = [(0, 999), (1, 100), (123, 321), (109, 205), (999, 1), (999, 999), (159, 253)]
        codes = torch.tensor([a * 1000 + b for a, b in operands])
        inputs = inputs_from_codes(codes)
        expected = torch.tensor([0, 1, 4, 3, 9, 0, 2])
        torch.testing.assert_close(hundreds_digit(inputs), expected)
        torch.testing.assert_close(hundreds_digit(torch.cat((inputs[:, 3:], inputs[:, :3]), dim=1)), expected)
        torch.testing.assert_close(training.codes_from_inputs(inputs), codes)
        with self.assertRaises(ValueError):
            hundreds_digit(torch.zeros(2, 5, dtype=torch.long))
        with self.assertRaises(ValueError):
            hundreds_digit(torch.full((2, 6), 10, dtype=torch.long))

    def test_full_spectrum_and_prepared_evaluation_match_the_exact_task(self):
        manifest = load_manifest(self.targets.name, verify=True)
        self.assertAlmostEqual(sum(manifest["indicator_degree_mass"]), 1.)
        self.assertAlmostEqual(sum(manifest["scalar_degree_mass"][1:]), manifest["scalar_variance"])
        self.assertGreater(sum(manifest["indicator_fraction_of_variance_by_degree"][4:]), .8)
        self.assertGreater(sum(manifest["scalar_fraction_of_variance_by_degree"][4:]), .83)
        self.assertLess(manifest["reconstruction_max_absolute_error"], 1e-10)
        config = TrainingConfig()
        evaluation = training.prepare_evaluation(self.targets.name, config, torch.device("cpu"))
        torch.testing.assert_close(hundreds_digit(evaluation.inputs), evaluation.labels)
        torch.testing.assert_close(evaluation.indicator_targets[:, -1], torch.nn.functional.one_hot(evaluation.labels, 10).double())
        torch.testing.assert_close(evaluation.scalar_targets[:, -1], evaluation.labels.double())
        codes = training.codes_from_inputs(evaluation.inputs)
        self.assertEqual(len(codes.unique()), 1024)
        self.assertTrue(bool(evaluation.excluded_codes[codes].all()))
        self.assertTrue(bool(evaluation.excluded_codes[(codes % 1000) * 1000 + codes // 1000].all()))
        for _ in range(8):
            inputs, labels = training.sample_batch(64, evaluation)
            self.assertFalse(bool(evaluation.excluded_codes[training.codes_from_inputs(inputs)].any()))
            torch.testing.assert_close(labels, hundreds_digit(inputs))

    def test_all_nine_requested_architectures_have_six_inputs_and_ten_output_logits(self):
        self.assertEqual([(row["num_layers"], row["d_model"]) for row in sweep.build_grid()],
                         [(layers, width) for layers in (1, 2, 4) for width in (512, 1024, 2048)])
        for index, row in enumerate(sweep.build_grid()):
            options = experiment.build_parser().parse_args(sweep.experiment_arguments(
                sweep.build_parser().parse_args(["--run-index", str(index)])))
            self.assertEqual(experiment.model_config(options), row)
            # Validate the actual full-size parameter graph without allocating
            # nine large optimizers or requiring a GPU in the unit tests.
            with torch.device("meta"):
                model = MultiplicationDigitTransformer(**row)
                self.assertEqual(model(torch.zeros((2, 6), dtype=torch.long)).shape, (2, 10))
            width, layers = row["d_model"], row["num_layers"]
            self.assertEqual(sum(p.numel() for p in model.parameters()), layers * (6 * width ** 2 + 10 * width) + 18 * width + 10)
            self.assertEqual(model.embedding.num_embeddings, 10)
            self.assertEqual(model.positional_encoding.num_embeddings, 6)
        self.assertEqual(TrainingConfig().num_steps, 50_000)
        self.assertEqual(TrainingConfig().batch_size, 64)

    def test_one_prediction_bank_scores_ce_accuracy_and_both_fourier_representations(self):
        config = TrainingConfig(eval_num_inputs=8)
        evaluation = training.prepare_evaluation(self.targets.name, config, torch.device("cpu"))

        class PerfectModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, inputs):
                self.calls += 1
                return torch.zeros(len(inputs), 10).scatter_(1, hundreds_digit(inputs)[:, None], 80.)

        model = PerfectModel()
        metrics = training.evaluate(model, evaluation, 3)
        self.assertEqual(model.calls, 3)
        self.assertTrue(model.training)
        self.assertAlmostEqual(metrics["eval/loss"], 0.)
        self.assertEqual(metrics["eval/accuracy"], 1.)
        self.assertAlmostEqual(metrics["eval/brier"], 0.)
        self.assertAlmostEqual(metrics["eval/scalar_mse"], 0.)
        self.assertEqual(metrics["eval/brier"], metrics["eval/loss_degree_6"])
        self.assertEqual(metrics["eval/scalar_mse"], metrics["eval/scalar_loss_degree_6"])
        self.assertGreater(metrics["eval/loss_degree_5"], .01)

    def test_real_training_with_one_two_and_four_layers_saves_metrics_and_preserves_settings(self):
        for index in (0, 4, 6):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                with contextlib.redirect_stdout(io.StringIO()):
                    output = sweep.main(["--run-index", str(index), "--smoke-test", "--no-wandb", "--device", "cpu",
                                         "--steps", "1", "--batch-size", "2", "--eval-inputs", "8", "--sharpness-interval", "0",
                                         "--target-dir", self.targets.name, "--output-dir", directory])
                config = json.loads((output / "config.json").read_text())
                self.assertEqual(config["model_d_model"], sweep.build_grid()[index]["d_model"])
                self.assertEqual(config["model_num_layers"], sweep.build_grid()[index]["num_layers"])
                self.assertEqual(config["input_vocab_size"], 10)
                self.assertEqual(config["input_len"], 6)
                self.assertTrue(config["excludes_reversed_operand_pairs"])
                rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual([row["step"] for row in rows], [0, 1])
                for row in rows:
                    self.assertTrue(math.isfinite(row["eval/loss"]))
                    for degree in range(1, 7):
                        self.assertTrue(math.isfinite(row[f"eval/loss_degree_{degree}"]))
                        self.assertTrue(math.isfinite(row[f"eval/scalar_loss_degree_{degree}"]))
                checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
                self.assertEqual(checkpoint["completed_steps"], 1)
                torch.manual_seed(7)
                initial = MultiplicationDigitTransformer(**checkpoint["model_config"])
                self.assertFalse(torch.equal(initial.linear_head.weight, checkpoint["model_state_dict"]["linear_head.weight"]))
                self.assertTrue((output / "target_spectrum.npz").is_file())

    def test_listing_and_submission_queue_exactly_nine_and_archive_the_executed_runner(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(experiment, "main") as run:
            sweep.main(["--list"])
        run.assert_not_called()
        self.assertEqual(len(json.loads(output.getvalue())), 9)
        for index in (-1, 9):
            with self.assertRaises(ValueError):
                sweep.experiment_arguments(sweep.build_parser().parse_args(["--run-index", str(index)]))
        submit = (ROOT / "sweep_multiplication_digit.sub").read_text()
        self.assertIn("\nqueue 9\n", submit)
        self.assertIn("$(project_dir)/multiplication_fourier_targets", submit)
        self.assertIn("SHARPNESS_ENTRYPOINT=sweep_multiplication_digit.py", submit)
        self.assertIn("cp test_multiplication_digit.py results/code/", (ROOT / "run-test.sh").read_text())
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "multiplication_fourier_targets").mkdir()
            (project / "multiplication_fourier_targets" / "manifest.json").write_text("{}")
            (project / "submit-multiplication-digit-sweep.sh").write_text((ROOT / "submit-multiplication-digit-sweep.sh").read_text())
            (project / "submit-wandb.sh").write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$SHARPNESS_SUBMIT_FILE" "$WANDB_PROJECT" "$@"\n')
            result = subprocess.run(["bash", str(project / "submit-multiplication-digit-sweep.sh"), "-append",
                                     "experiment_args=--steps 1000"], capture_output=True, check=True)
            self.assertEqual(result.stdout.decode().rstrip("\0").split("\0"), [
                str(project / "sweep_multiplication_digit.sub"), "multiplication digit fourier analysis",
                "-append", f"project_dir={project}", "-append", "experiment_args=--steps 1000"])


if __name__ == "__main__":
    unittest.main()
