"""The six small-domain runs keep the original targets and training setup."""

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

import sweep_fourier
import sweep_input_tower
import sweep_sparse_small as sweep
import test_fourier as experiment
from src import fourier_training
from src.fourier import make_function
from src.fourier_training import TrainingConfig


ROOT = Path(__file__).resolve().parents[1]


class SmallSparseSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_six_configurations_preserve_targets_and_all_training_defaults(self):
        grid = sweep.build_grid()
        self.assertEqual([(row["input_len"], row["function"]) for row in grid], [
            (length, name) for length in (5, 7, 10)
            for name in ("sparse_powerlaw", "sparse_geometric")])
        names = set()
        for index, row in enumerate(grid):
            with self.subTest(index=index):
                args = sweep.build_parser().parse_args(["--run-index", str(index)])
                options = experiment.build_parser().parse_args(sweep.experiment_arguments(args))
                length = row["input_len"]
                self.assertEqual(options.max_degree, length)
                self.assertEqual(options.input_len, length)
                self.assertEqual(options.eval_inputs, {5: 8, 7: 32, 10: 256}[length])
                self.assertEqual(options.batch_size, 64)
                self.assertEqual(options.embedding_size, 128)
                self.assertEqual(experiment.build_model_config(options), dict(
                    d_model=256, num_attn_heads=8, num_layers=1, vocab_size=2, dropout=0., max_len=length))
                self.assertEqual((options.lr, options.weight_decay), (1e-4, 1e-2))
                self.assertEqual((options.seed, options.function_seed, options.eval_seed), (7, 17, 23))
                self.assertEqual((options.sparse_terms, options.power, options.geometric_decay), (16, 2., .9))
                self.assertEqual(options.eval_batch_size, 320)
                self.assertIsNone(options.steps)
                self.assertIsNone(options.sharpness_interval)
                self.assertFalse(options.smoke_test)
                self.assertTrue(options.wandb)
                names.add(options.run_name)
                target = experiment.build_function(options)
                self.assertEqual(target.metadata(), make_function(row["function"], input_len=length).metadata())
        self.assertEqual(len(names), 6)
        self.assertEqual(TrainingConfig().num_steps, 30_000)
        self.assertEqual(TrainingConfig().eval_interval, 1)
        self.assertEqual(TrainingConfig().sharpness_interval, 100)
        self.assertEqual(len(sweep_fourier.build_grid()), 4)
        self.assertEqual(len(sweep_input_tower.build_grid()), 3)

    def test_listing_is_read_only_and_out_of_range_indices_are_rejected(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(experiment, "main") as run:
            sweep.main(["--list"])
        run.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), [
            dict(run_index=index, **row) for index, row in enumerate(sweep.build_grid())])
        for index in (-1, 6):
            with self.assertRaises(ValueError):
                sweep.experiment_arguments(sweep.build_parser().parse_args(["--run-index", str(index)]))

    def test_every_run_trains_with_disjoint_evaluation_and_logs_all_degrees_every_step(self):
        sample_batch = fourier_training.sample_batch
        for index, row in enumerate(sweep.build_grid()):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                training_batches = []

                def capture_batch(*args, **kwargs):
                    inputs, labels = sample_batch(*args, **kwargs)
                    self.assertEqual(kwargs["held_out"].overlapping_rows(inputs), [])
                    self.assertEqual(inputs.shape, (64, row["input_len"]))
                    torch.testing.assert_close(labels, args[0](inputs))
                    training_batches.append(inputs.cpu().clone())
                    return inputs, labels

                with patch.object(fourier_training, "sample_batch", side_effect=capture_batch), \
                     contextlib.redirect_stdout(io.StringIO()):
                    output = sweep.main(["--run-index", str(index), "--smoke-test", "--no-wandb",
                                         "--device", "cpu", "--output-dir", directory])
                self.assertEqual(len(training_batches), 2)
                metadata = json.loads((output / "config.json").read_text())
                self.assertEqual(metadata["function"]["input_len"], row["input_len"])
                self.assertEqual(metadata["function"]["max_degree"], row["max_degree"])
                self.assertEqual(metadata["function"]["name"], row["function"])
                self.assertEqual(metadata["train_eval_num_inputs"], row["eval_inputs"])
                self.assertEqual(metadata["train_batch_size"], 64)
                self.assertEqual(metadata["model_num_layers"], 1)
                self.assertEqual(metadata["model_d_model"], 256)
                self.assertEqual(metadata["model_num_attn_heads"], 8)
                self.assertEqual(metadata["num_tasks"], 1)
                self.assertEqual(metadata["selector_size"], 0)
                records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual([record["step"] for record in records], [0, 1, 2])
                keys = {f"eval/loss_degree_{degree}" for degree in range(1, row["max_degree"] + 1)}
                for record in records:
                    self.assertEqual({key for key in record if key.startswith("eval/loss_degree_")}, keys)
                    self.assertTrue(all(math.isfinite(record[key]) for key in keys))
                    self.assertEqual(record["eval/loss"], record[f"eval/loss_degree_{row['max_degree']}"])
                saved = torch.load(output / "evaluation.pt", weights_only=True)
                self.assertEqual(saved["inputs"].shape, (row["eval_inputs"], row["input_len"]))
                self.assertEqual(len(saved["inputs"].unique(dim=0)), row["eval_inputs"])
                self.assertEqual(saved["cumulative_targets"].shape, (row["eval_inputs"], row["max_degree"]))
                function = make_function(row["function"], input_len=row["input_len"])
                torch.testing.assert_close(saved["cumulative_targets"],
                                           function.cumulative(saved["inputs"], dtype=torch.float64))
                checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
                self.assertEqual(checkpoint["completed_steps"], 2)

    def test_cluster_submission_queues_exactly_six_and_forwards_shared_overrides(self):
        submit = (ROOT / "sweep_sparse_small.sub").read_text()
        self.assertIn("\nqueue 6\n", submit)
        self.assertIn("SHARPNESS_ENTRYPOINT=sweep_sparse_small.py", submit)
        self.assertIn("$(project_dir)/test_fourier.py", submit)
        self.assertIn("$(project_dir)/src", submit)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "submit-sparse-small-sweep.sh").write_text((ROOT / "submit-sparse-small-sweep.sh").read_text())
            (project / "submit-wandb.sh").write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$SHARPNESS_SUBMIT_FILE" "$@"\n')
            captured = subprocess.run(["bash", str(project / "submit-sparse-small-sweep.sh"), "-append",
                                       "experiment_args=--steps 1000 --seed 19"], check=True, capture_output=True)
            self.assertEqual(captured.stdout.decode().rstrip("\0").split("\0"), [
                str(project / "sweep_sparse_small.sub"), "-append", f"project_dir={project}",
                "-append", "experiment_args=--steps 1000 --seed 19"])


if __name__ == "__main__":
    unittest.main()
