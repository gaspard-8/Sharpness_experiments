"""Sixteen bounded length/degree experiments with 50,000 updates per model."""

from collections import Counter
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

import sweep_sparse_grid as sweep
import test_fourier as experiment
from src import fourier_training
from src.fourier import make_function


ROOT = Path(__file__).resolve().parents[1]
PAIRS = ((10, 3), (10, 6), (12, 5), (12, 6), (15, 4), (15, 5), (20, 3), (20, 4))


class SparseGridSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sixteen_runs_balance_lengths_and_degrees_and_preserve_the_targets(self):
        grid = sweep.build_grid()
        self.assertEqual(len(grid), 16)
        self.assertEqual([(row["input_len"], row["max_degree"], row["function"]) for row in grid], [
            (length, degree, name) for length, degree in PAIRS
            for name in ("sparse_powerlaw", "sparse_geometric")])
        self.assertEqual(Counter(row["function"] for row in grid),
                         {"sparse_powerlaw": 8, "sparse_geometric": 8})
        self.assertEqual(Counter(length for length, _ in PAIRS), {10: 2, 12: 2, 15: 2, 20: 2})
        self.assertEqual(Counter(degree for _, degree in PAIRS), {3: 2, 4: 2, 5: 2, 6: 2})
        self.assertNotIn((20, 6), PAIRS)
        names = set()
        for index, row in enumerate(grid):
            options = experiment.build_parser().parse_args(sweep.experiment_arguments(
                sweep.build_parser().parse_args(["--run-index", str(index)])))
            self.assertEqual(options.steps, 50_000)
            self.assertEqual(options.eval_inputs, 256 if row["input_len"] == 10 else 1024)
            self.assertEqual(options.batch_size, 64)
            self.assertEqual(experiment.build_model_config(options), dict(
                d_model=256, num_attn_heads=8, num_layers=1, vocab_size=2, dropout=0., max_len=row["input_len"]))
            self.assertEqual((options.lr, options.weight_decay), (1e-4, 1e-2))
            self.assertEqual((options.seed, options.function_seed, options.eval_seed), (7, 17, 23))
            self.assertTrue(options.wandb)
            self.assertEqual(options.wandb_project, "sparse functions fourier analysis")
            names.add(options.run_name)
            function = experiment.build_function(options)
            self.assertEqual(function.metadata(), make_function(
                row["function"], input_len=row["input_len"], max_degree=row["max_degree"]).metadata())
            magnitudes = [[abs(value) for value in function.coefficients[start:end]]
                          for start, end in zip(function.offsets, function.offsets[1:])]
            self.assertAlmostEqual(math.fsum(value ** 2 for value in function.coefficients), 1.)
            self.assertTrue(all(min(low) > max(high) for low, high in zip(magnitudes, magnitudes[1:])))
            self.assertTrue(all(low > high > 0 for low, high in zip(function.degree_mass, function.degree_mass[1:])))
        self.assertEqual(len(names), 16)
        for powerlaw, geometric in zip(grid[::2], grid[1::2]):
            left = make_function(powerlaw["function"], input_len=powerlaw["input_len"], max_degree=powerlaw["max_degree"])
            right = make_function(geometric["function"], input_len=geometric["input_len"], max_degree=geometric["max_degree"])
            self.assertEqual(left.subsets, right.subsets)
            self.assertEqual([value > 0 for value in left.coefficients], [value > 0 for value in right.coefficients])

    def test_listing_does_not_train_and_smoke_or_explicit_steps_are_resolved_correctly(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(experiment, "main") as run:
            sweep.main(["--list"])
        run.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), [dict(run_index=i, **row) for i, row in enumerate(sweep.build_grid())])
        for index in (-1, 16):
            with self.assertRaises(ValueError):
                sweep.experiment_arguments(sweep.build_parser().parse_args(["--run-index", str(index)]))
        for flags, steps in ((["--smoke-test"], 2), (["--steps", "7"], 7),
                             (["--smoke-test", "--steps", "3"], 3)):
            options = experiment.build_parser().parse_args(sweep.experiment_arguments(
                sweep.build_parser().parse_args(["--run-index", "0", *flags])))
            self.assertEqual(options.steps, steps)

    def test_all_sixteen_real_smokes_exclude_heldout_inputs_and_log_all_degrees_every_step(self):
        sample_batch = fourier_training.sample_batch
        for index, row in enumerate(sweep.build_grid()):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                seen = []

                def capture_batch(*args, **kwargs):
                    inputs, labels = sample_batch(*args, **kwargs)
                    self.assertEqual(inputs.shape, (64, row["input_len"]))
                    self.assertEqual(kwargs["held_out"].overlapping_rows(inputs), [])
                    torch.testing.assert_close(labels, args[0](inputs))
                    seen.append(True)
                    return inputs, labels

                with patch.object(fourier_training, "sample_batch", side_effect=capture_batch), \
                     contextlib.redirect_stdout(io.StringIO()):
                    output = sweep.main(["--run-index", str(index), "--smoke-test", "--no-wandb",
                                         "--device", "cpu", "--output-dir", directory])
                self.assertEqual(len(seen), 2)
                config = json.loads((output / "config.json").read_text())
                self.assertEqual(config["train_num_steps"], 2)
                self.assertEqual(config["train_batch_size"], 64)
                self.assertEqual(config["train_eval_num_inputs"], row["eval_inputs"])
                self.assertEqual(config["model_num_layers"], 1)
                self.assertEqual(config["model_d_model"], 256)
                self.assertEqual(config["model_num_attn_heads"], 8)
                self.assertEqual(config["num_tasks"], 1)
                self.assertEqual(config["wandb_project"], "sparse functions fourier analysis")
                self.assertEqual(config["function"]["name"], row["function"])
                self.assertEqual(config["function"]["max_degree"], row["max_degree"])
                records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual([record["step"] for record in records], [0, 1, 2])
                degree_keys = {f"eval/loss_degree_{d}" for d in range(1, row["max_degree"] + 1)}
                for record in records:
                    self.assertEqual({key for key in record if key.startswith("eval/loss_degree_")}, degree_keys)
                    self.assertTrue(all(math.isfinite(record[key]) for key in degree_keys))
                    self.assertEqual(record["eval/loss"], record[f"eval/loss_degree_{row['max_degree']}"])
                saved = torch.load(output / "evaluation.pt", weights_only=True)
                self.assertEqual(saved["inputs"].shape, (row["eval_inputs"], row["input_len"]))
                self.assertEqual(len(saved["inputs"].unique(dim=0)), row["eval_inputs"])
                self.assertEqual(saved["cumulative_targets"].shape, (row["eval_inputs"], row["max_degree"]))
                self.assertEqual(torch.load(output / "checkpoint.pt", weights_only=True)["completed_steps"], 2)

    def test_submission_queues_sixteen_and_preserves_overrides(self):
        submit = (ROOT / "sweep_sparse_grid.sub").read_text()
        self.assertIn("\nqueue 16\n", submit)
        self.assertIn("SHARPNESS_ENTRYPOINT=sweep_sparse_grid.py", submit)
        self.assertIn("$(project_dir)/test_fourier.py", submit)
        self.assertIn("$(project_dir)/src", submit)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "submit-sparse-grid-sweep.sh").write_text((ROOT / "submit-sparse-grid-sweep.sh").read_text())
            (project / "submit-wandb.sh").write_text(
                '#!/usr/bin/env bash\nprintf "%s\\0" "$SHARPNESS_SUBMIT_FILE" "$WANDB_PROJECT" "$@"\n')
            result = subprocess.run(["bash", str(project / "submit-sparse-grid-sweep.sh"), "-append",
                                     "experiment_args=--steps 1000 --seed 19"], check=True, capture_output=True)
            self.assertEqual(result.stdout.decode().rstrip("\0").split("\0"), [
                str(project / "sweep_sparse_grid.sub"), "sparse functions fourier analysis",
                "-append", f"project_dir={project}",
                "-append", "experiment_args=--steps 1000 --seed 19"])


if __name__ == "__main__":
    unittest.main()
