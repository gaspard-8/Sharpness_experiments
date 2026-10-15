"""The two sparse targets use 20 inputs and strictly decreasing weights through degree 6."""

import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

import sweep_fourier as original
import sweep_sparse_long as sweep
import test_fourier as experiment
from src.fourier import make_function
from src.model import TransformerModel


ROOT = Path(__file__).resolve().parents[1]


class LongSparseSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_two_runs_match_the_original_sparse_construction_at_length_20_degree_6(self):
        self.assertEqual(sweep.build_grid(), original.build_grid()[2:])
        for index, function in enumerate(("sparse_powerlaw", "sparse_geometric")):
            with self.subTest(function=function):
                args = sweep.build_parser().parse_args(["--run-index", str(index)])
                old_args = original.build_parser().parse_args([
                    "--run-index", str(index + 2), "--input-len", "20", "--max-degree", "6"])
                options = experiment.build_parser().parse_args(sweep.experiment_arguments(args))
                old_options = experiment.build_parser().parse_args(original.experiment_arguments(old_args))
                actual, expected = vars(options).copy(), vars(old_options).copy()
                actual.pop("wandb_group")
                expected.pop("wandb_group")
                self.assertEqual(actual, expected)
                target = experiment.build_function(options)
                self.assertEqual(target.metadata(), make_function(function, input_len=20, max_degree=6).metadata())
                self.assertEqual((target.input_len, target.max_degree), (20, 6))
                self.assertEqual(len(target.subsets), 96)
                self.assertEqual(set().union(*map(set, target.subsets)), set(range(20)))
                magnitudes = [[abs(coefficient) for coefficient in target.coefficients[start:end]]
                              for start, end in zip(target.offsets, target.offsets[1:])]
                self.assertTrue(all(len(level) == 16 for level in magnitudes))
                self.assertTrue(all(min(lower) > max(higher)
                                    for lower, higher in zip(magnitudes, magnitudes[1:])))
                masses = [math.fsum(coefficient ** 2 for coefficient in level) for level in magnitudes]
                self.assertAlmostEqual(math.fsum(masses), 1.)
                self.assertTrue(all(lower > higher > 0 for lower, higher in zip(masses, masses[1:])))

    def test_listing_is_read_only_and_invalid_indices_are_rejected(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(experiment, "main") as run:
            sweep.main(["--list"])
        run.assert_not_called()
        listed = json.loads(output.getvalue())
        self.assertEqual(len(listed), 2)
        self.assertEqual([row["function"] for row in listed], ["sparse_powerlaw", "sparse_geometric"])
        self.assertTrue(all(row["input_len"] == 20 and row["max_degree"] == 6 for row in listed))
        for index in (-1, 2):
            with self.assertRaises(ValueError):
                sweep.experiment_arguments(sweep.build_parser().parse_args(["--run-index", str(index)]))

    def test_two_real_smokes_preserve_architecture_and_evaluate_every_degree_every_step(self):
        for index, function in enumerate(sweep.FUNCTION_NAMES):
            with self.subTest(function=function), tempfile.TemporaryDirectory() as directory:
                with contextlib.redirect_stdout(io.StringIO()):
                    output = sweep.main(["--run-index", str(index), "--smoke-test", "--no-wandb",
                                         "--device", "cpu", "--output-dir", directory])
                config = json.loads((output / "config.json").read_text())
                self.assertEqual(config["function"], make_function(function, input_len=20, max_degree=6).metadata())
                self.assertEqual(config["model_num_layers"], 1)
                self.assertEqual(config["model_d_model"], 256)
                self.assertEqual(config["model_num_attn_heads"], 8)
                self.assertEqual(config["model_max_len"], 20)
                self.assertEqual(config["train_batch_size"], 64)
                self.assertEqual((config["optimizer_lr"], config["optimizer_weight_decay"]), (1e-4, 1e-2))
                self.assertEqual(config["num_tasks"], 1)
                records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual([row["step"] for row in records], [0, 1, 2])
                degree_keys = {f"eval/loss_degree_{degree}" for degree in range(1, 7)}
                for row in records:
                    self.assertEqual({key for key in row if key.startswith("eval/loss_degree_")}, degree_keys)
                    self.assertTrue(all(math.isfinite(row[key]) for key in degree_keys))
                    self.assertEqual(row["eval/loss"], row["eval/loss_degree_6"])
                checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
                self.assertEqual(checkpoint["completed_steps"], 2)
                model = TransformerModel(**checkpoint["model_config"])
                model.load_state_dict(checkpoint["model_state_dict"])
                self.assertEqual(len(model.encoder.layers), 1)

    def test_submission_queues_two_and_forwards_overrides(self):
        submit = (ROOT / "sweep_sparse_long.sub").read_text()
        self.assertIn("\nqueue 2\n", submit)
        self.assertIn("SHARPNESS_ENTRYPOINT=sweep_sparse_long.py", submit)
        self.assertIn("$(project_dir)/test_fourier.py", submit)
        self.assertIn("$(project_dir)/src", submit)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "submit-sparse-long-sweep.sh").write_text((ROOT / "submit-sparse-long-sweep.sh").read_text())
            (project / "submit-wandb.sh").write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$SHARPNESS_SUBMIT_FILE" "$@"\n')
            captured = subprocess.run(["bash", str(project / "submit-sparse-long-sweep.sh"), "-append",
                                       "experiment_args=--steps 1000 --seed 19"], check=True, capture_output=True)
            self.assertEqual(captured.stdout.decode().rstrip("\0").split("\0"), [
                str(project / "sweep_sparse_long.sub"), "-append", f"project_dir={project}",
                "-append", "experiment_args=--steps 1000 --seed 19"])


if __name__ == "__main__":
    unittest.main()
