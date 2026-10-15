"""Exact classic spectra, full supervision, and four real transformer smoke runs."""

import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import sweep_classic_fourier as sweep
import test_fourier as experiment
from src import fourier_training
from src.classic_fourier import fwht, make_classic_function, spectrum_path
from src.evaluation import EvaluationSet
from src.fourier_training import FourierEvaluation, evaluate_cumulative


ROOT = Path(__file__).resolve().parents[1]


def cube(length):
    return ((torch.arange(2 ** length)[:, None] >> torch.arange(length)) & 1).long()


class ClassicFourierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.targets = tempfile.TemporaryDirectory()
        cls.manifest = sweep.prepare_targets(Path(cls.targets.name))

    @classmethod
    def tearDownClass(cls):
        cls.targets.cleanup()

    def test_exact_transform_and_cumulative_projections_match_explicit_characters(self):
        for name, length in (("classic_majority", 3), ("classic_weighted_majority", 5),
                             ("classic_tribes", 6), ("classic_mean", 4)):
            with self.subTest(name=name):
                function = make_classic_function(name, length, max_degree=length)
                inputs = cube(length)
                masks = torch.arange(2 ** length)
                incidence = ((masks[:, None] >> torch.arange(length)) & 1).double()
                characters = 1 - 2 * (inputs.double() @ incidence.T).remainder(2)
                values = function(inputs).double()
                actual = characters.T @ values / len(inputs)
                specified = torch.zeros(len(inputs), dtype=torch.float64)
                specified[torch.from_numpy(function.masks.astype(np.int64))] = torch.from_numpy(function.coefficients)
                torch.testing.assert_close(actual, specified, atol=1e-14, rtol=1e-14)
                degrees = incidence.sum(dim=1)
                for cutoff in range(1, length + 1):
                    expected = characters[:, degrees <= cutoff] @ actual[degrees <= cutoff]
                    torch.testing.assert_close(function.cumulative(inputs, dtype=torch.float64)[:, cutoff - 1], expected)
                torch.testing.assert_close(function.cumulative(inputs, dtype=torch.float64)[:, -1], values)
                self.assertAlmostEqual(float(actual.square().sum()), float(values.square().mean()))
                torch.testing.assert_close(torch.from_numpy(fwht(fwht(values.numpy())) / len(values)), values)

    def test_prepared_sensitivities_match_input_flips_and_justify_selected_lengths(self):
        measured = []
        generator = torch.Generator().manual_seed(99)
        for name, length in sweep.TARGET_LENGTHS:
            function = make_classic_function(name, length, target_dir=self.targets.name)
            inputs = cube(length) if length < 20 else torch.randint(0, 2, (64, length), generator=generator)
            values = function(inputs).double()
            sensitivity = 0.
            for bit in range(length):
                flipped = inputs.clone()
                flipped[:, bit] = 1 - flipped[:, bit]
                sensitivity += float((values - function(flipped).double()).square().mean())
            self.assertAlmostEqual(sensitivity, function.squared_flip_sensitivity)
            measured.append(sensitivity)
            self.assertTrue(bool(((values >= 0) & (values <= 1)).all()))
            self.assertEqual(function.max_degree, 8)
        self.assertTrue(all(left > right for left, right in zip(measured, measured[1:])))
        self.assertEqual([length for _, length in sweep.TARGET_LENGTHS], [11, 13, 15, 32])
        majority, weighted, tribes, mean = self.manifest["targets"]
        self.assertEqual(weighted["definition"]["parameters"]["weights"], [2, 4, 1, 8, 13, 10, 12, 3, 11, 6, 5, 7, 9])
        self.assertAlmostEqual(tribes["squared_flip_sensitivity"], 15 * 2 ** -2 * (7 / 8) ** 4)
        self.assertTrue(all(target["tail_mass_above_degree"][8] > 0 for target in (majority, weighted, tribes)))
        self.assertEqual(mean["full_target_degree"], 1)
        self.assertEqual(mean["tail_mass_above_degree"][1], 0.)
        self.assertAlmostEqual(mean["variance"], 1 / 128)

    def test_full_target_loss_is_distinct_from_degree_eight_and_uses_one_prediction_bank(self):
        function = make_classic_function("classic_majority", 11, target_dir=self.targets.name)
        inputs = cube(11)
        values = function(inputs).double()
        evaluation = FourierEvaluation(EvaluationSet(inputs, values), function.cumulative(inputs, dtype=torch.float64),
                                       full_targets=values, target_variance=function.variance)

        class PerfectModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, inputs):
                self.calls += 1
                return function(inputs)

        model = PerfectModel()
        metrics = evaluate_cumulative(model, evaluation, batch_size=320)
        self.assertEqual(model.calls, math.ceil(len(inputs) / 320))
        self.assertTrue(model.training)
        self.assertEqual(metrics["eval/loss"], 0.)
        self.assertAlmostEqual(metrics["eval/loss_degree_8"], function.metadata()["tail_mass_above_degree"][8])
        self.assertGreater(metrics["eval/loss_degree_8"], 0)
        self.assertEqual(metrics["eval/loss_degree_8_normalized"], metrics["eval/loss_degree_8"] / function.variance)
        for odd in (1, 3, 5, 7):
            self.assertEqual(metrics[f"eval/loss_degree_{odd}"], metrics[f"eval/loss_degree_{odd + 1}"])

    def test_invalid_caps_missing_or_corrupted_spectra_are_rejected(self):
        with self.assertRaises(ValueError):
            make_classic_function("classic_majority", 11, max_degree=9)
        with self.assertRaises(ValueError):
            make_classic_function("classic_tribes", 14)
        with self.assertRaises(ValueError):
            fwht(np.zeros(7))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "prepare"):
                make_classic_function("classic_majority", 11, target_dir=directory)
            function = make_classic_function("classic_majority", 3, max_degree=3)
            path = spectrum_path(directory, function.spec)
            function.save_spectrum(path)
            with np.load(path, allow_pickle=False) as data:
                payload = dict(data)
            payload["coefficients"][0] += .1
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "checksum"):
                make_classic_function("classic_majority", 3, max_degree=3, target_dir=directory)

    def test_four_real_smokes_use_full_natural_labels_and_evaluate_only_cutoffs_one_through_eight(self):
        sample_batch = fourier_training.sample_batch
        for index, row in enumerate(sweep.build_grid()):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                seen = []

                def capture_batch(*args, **kwargs):
                    inputs, labels = sample_batch(*args, **kwargs)
                    self.assertEqual(inputs.shape, (64, row["input_len"]))
                    self.assertEqual(kwargs["held_out"].overlapping_rows(inputs), [])
                    torch.testing.assert_close(labels, args[0].source(inputs).float())
                    seen.append(True)
                    return inputs, labels

                with patch.object(fourier_training, "sample_batch", side_effect=capture_batch), \
                     contextlib.redirect_stdout(io.StringIO()):
                    output = sweep.main(["--run-index", str(index), "--smoke-test", "--no-wandb", "--device", "cpu",
                                         "--classic-target-dir", self.targets.name, "--output-dir", directory])
                self.assertEqual(len(seen), 2)
                config = json.loads((output / "config.json").read_text())
                self.assertEqual(config["model_d_model"], 256)
                self.assertEqual(config["model_num_attn_heads"], 8)
                self.assertEqual(config["model_num_layers"], 1)
                self.assertEqual(config["train_batch_size"], 64)
                self.assertEqual(config["train_eval_num_inputs"], row["eval_inputs"])
                self.assertEqual(config["experiment"], "single_task_classic_fourier")
                self.assertEqual(config["wandb_project"], "sparse functions fourier analysis")
                self.assertTrue((output / "target_spectrum.npz").is_file())
                records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual([record["step"] for record in records], [0, 1, 2])
                for record in records:
                    self.assertNotIn("eval/loss_degree_9", record)
                    for degree in range(1, 9):
                        self.assertTrue(math.isfinite(record[f"eval/loss_degree_{degree}"]))
                    self.assertTrue(math.isfinite(record["eval/loss_normalized"]))
                if row["function"] == "classic_mean":
                    for record in records:
                        for degree in range(1, 9):
                            self.assertAlmostEqual(record[f"eval/loss_degree_{degree}"], record["eval/loss"])
                stored = torch.load(output / "evaluation.pt", weights_only=True)
                self.assertEqual(stored["cumulative_targets"].shape, (row["eval_inputs"], 8))
                self.assertEqual(stored["full_targets"].shape, (row["eval_inputs"],))
                self.assertEqual(torch.load(output / "checkpoint.pt", weights_only=True)["completed_steps"], 2)

    def test_grid_listing_and_submission_preserve_setup_and_precomputed_artifacts(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(experiment, "main") as run:
            sweep.main(["--list"])
        run.assert_not_called()
        self.assertEqual(len(json.loads(output.getvalue())), 4)
        for index in range(4):
            options = experiment.build_parser().parse_args(sweep.experiment_arguments(
                sweep.build_parser().parse_args(["--run-index", str(index)])))
            self.assertEqual((options.steps, options.max_degree, options.batch_size), (50_000, 8, 64))
        for index in (-1, 4):
            with self.assertRaises(ValueError):
                sweep.experiment_arguments(sweep.build_parser().parse_args(["--run-index", str(index)]))
        submit = (ROOT / "sweep_classic_fourier.sub").read_text()
        self.assertIn("\nqueue 4\n", submit)
        self.assertIn("$(project_dir)/classic_fourier_targets", submit)
        self.assertIn("SHARPNESS_ENTRYPOINT=sweep_classic_fourier.py", submit)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "classic_fourier_targets").mkdir()
            (project / "classic_fourier_targets" / "manifest.json").write_text("{}")
            (project / "submit-classic-fourier-sweep.sh").write_text((ROOT / "submit-classic-fourier-sweep.sh").read_text())
            (project / "submit-wandb.sh").write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$SHARPNESS_SUBMIT_FILE" "$@"\n')
            result = subprocess.run(["bash", str(project / "submit-classic-fourier-sweep.sh"), "-append",
                                     "experiment_args=--smoke-test"], check=True, capture_output=True)
            self.assertEqual(result.stdout.decode().rstrip("\0").split("\0"), [
                str(project / "sweep_classic_fourier.sub"), "-append", f"project_dir={project}",
                "-append", "experiment_args=--smoke-test"])


if __name__ == "__main__":
    unittest.main()
