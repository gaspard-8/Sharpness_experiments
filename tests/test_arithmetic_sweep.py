import contextlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import sweep_arithmetic as sweep
import test_arithmetic as experiment
from src.arithmetic import ArithmeticTransformer, SequenceCrossEntropy, sample_batch


ROOT = Path(__file__).resolve().parents[1]


class ArithmeticSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def run_options(self, index, *extra):
        options = sweep.build_parser().parse_args(["--run-index", str(index), *extra])
        return experiment.build_parser().parse_args(sweep.experiment_arguments(options))

    def test_complete_grid_matches_condor_queue_and_architecture(self):
        grid = sweep.build_grid()
        self.assertEqual(len(grid), 36)
        self.assertEqual(len({tuple(config.values()) for config in grid}), 36)
        submit = (ROOT / "sweep_arithmetic.sub").read_text()
        self.assertEqual(int(re.search(r"^queue (\d+)$", submit, re.MULTILINE)[1]), len(grid))
        self.assertIn("--run-index $(Process)", submit)
        self.assertIn("SHARPNESS_ENTRYPOINT=sweep_arithmetic.py", submit)
        self.assertIn("$(project_dir)/test_arithmetic.py", submit)
        for index, config in enumerate(grid):
            with self.subTest(index=index, config=config):
                args = self.run_options(index)
                self.assertEqual(args.steps, 30_000)
                self.assertTrue(args.wandb)
                self.assertFalse(args.smoke_test)
                self.assertFalse(args.loss_spaces)
                self.assertEqual(args.payload_len, 4)
                selector = experiment.build_selector(args)
                model_config = experiment.build_model_config(args, selector, selector.required_output_tokens)
                model = ArithmeticTransformer(**model_config)
                self.assertEqual(model.embedding.embedding_dim, config["embedding_size"])
                self.assertEqual(model.positional_encoding.embedding_dim, config["embedding_size"])
                self.assertEqual(len(model.encoder.layers), config["num_layers"])
                self.assertEqual(model.encoder.layers[0].self_attn.num_heads, config["num_attn_heads"])
                self.assertEqual(args.batch_size, config["batch_size"])
                self.assertEqual([task.argument_width(4) for task in selector.functions], [2] * 5 + [4, 4])
                # Exercise the actual configured batch size and architecture,
                # including a differentiable loss, for every grid combination.
                inputs, targets = sample_batch(selector, args.batch_size, 5, balanced_selectors=True,
                                                generator=torch.Generator().manual_seed(7))
                loss = SequenceCrossEntropy()(model.teacher_logits(inputs, targets), targets)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())

    def test_model_options_validation_and_smoke_preserves_requested_shape(self):
        for index in (0, 17, 35):
            args = self.run_options(index, "--smoke-test")
            selector = experiment.build_selector(args)
            config = experiment.build_model_config(args, selector, 5)
            self.assertEqual(config["d_model"], 2 * sweep.build_grid()[index]["embedding_size"])
            self.assertEqual(config["num_layers"], sweep.build_grid()[index]["num_layers"])
            self.assertEqual(args.steps, 2)
        for flags in (["--d-model", "33"], ["--embedding-size", "17", "--num-attn-heads", "8"]):
            args = experiment.build_parser().parse_args(flags)
            with self.assertRaisesRegex(ValueError, "even and divisible"):
                experiment.build_model_config(args, experiment.build_selector(args), 5)
        for flag in ("--embedding-size", "--d-model", "--num-layers", "--num-attn-heads"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                experiment.build_parser().parse_args([flag, "0"])
        for index in (-1, 36):
            with self.assertRaisesRegex(ValueError, "run-index"):
                self.run_options(index)

    def test_list_is_read_only_and_covers_every_combination(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(experiment, "main") as train:
            sweep.main(["--list"])
        train.assert_not_called()
        rows = json.loads(output.getvalue())
        self.assertEqual([row["run_index"] for row in rows], list(range(36)))
        for key, expected in (("embedding_size", {16, 32, 64}), ("num_layers", {2, 4}),
                              ("num_attn_heads", {2, 4, 8}), ("batch_size", {64, 320})):
            self.assertEqual({row[key] for row in rows}, expected)

    def test_sweep_submission_selects_its_own_file_and_preserves_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "submit-sweep.sh").write_text((ROOT / "submit-sweep.sh").read_text())
            (project / "submit-wandb.sh").write_text(
                '#!/usr/bin/env bash\nprintf "%s\\0" "$SHARPNESS_SUBMIT_FILE" "$@"\n')
            result = subprocess.run(["bash", str(project / "submit-sweep.sh"), "-append",
                                     "experiment_args=--steps 1000"], check=True, capture_output=True)
            self.assertEqual(result.stdout.decode().rstrip("\0").split("\0"), [
                str(project / "sweep_arithmetic.sub"), "-append", f"project_dir={project}",
                "-append", "experiment_args=--steps 1000"])

    def test_sweep_smoke_records_real_model_and_logs_all_tasks_without_spaces(self):
        class CapturedRun:
            def __init__(self):
                self.records = []
                self.finished = False

            def log(self, metrics, step):
                self.records.append((step, metrics))

            def finish(self):
                self.finished = True

        for index in (0, 35):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                run = CapturedRun()
                wandb = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs,
                                        Table=lambda **kwargs: kwargs)
                with patch.dict(sys.modules, {"wandb": wandb}), \
                     patch.object(wandb, "init", return_value=run) as initialize, \
                     patch.object(experiment, "configure_cpu_environment", return_value=None), \
                     patch("src.arithmetic_training.loss_spaces", side_effect=AssertionError("spaces disabled")), \
                     contextlib.redirect_stdout(io.StringIO()):
                    output = sweep.main(["--run-index", str(index), "--smoke-test", "--device", "cpu",
                                         "--output-dir", directory, "--wandb-group", "test-sweep"])
                config = json.loads((output / "config.json").read_text())
                chosen = sweep.build_grid()[index]
                self.assertEqual(config["model_embedding_size"], chosen["embedding_size"])
                self.assertEqual(config["model_d_model"], 2 * chosen["embedding_size"])
                self.assertEqual(config["model_num_layers"], chosen["num_layers"])
                self.assertEqual(config["model_num_attn_heads"], chosen["num_attn_heads"])
                self.assertEqual(config["train_batch_size"], chosen["batch_size"])
                self.assertEqual(config["train_num_steps"], 2)
                self.assertFalse(config["train_space_enabled"])
                self.assertEqual([f["argument_width"] for f in config["functions"]], [2] * 5 + [4, 4])
                self.assertEqual(initialize.call_args.kwargs["group"], "test-sweep")
                self.assertEqual(json.loads(json.dumps(initialize.call_args.kwargs["config"])), config)
                self.assertIn(f"e{chosen['embedding_size']}-l{chosen['num_layers']}", initialize.call_args.kwargs["name"])
                self.assertTrue(run.finished)
                self.assertEqual(len(run.records), 2)
                for step, metrics in run.records:
                    self.assertIn("eval/accuracy/addition_mod_10_1", metrics)
                    self.assertIn("eval/accuracy/majority_1", metrics)
                    self.assertIn("affinity/matrix", metrics)
                    self.assertFalse(any(key.startswith("spaces/") for key in metrics))
                self.assertTrue((output / "checkpoint.pt").is_file())
                self.assertFalse(list(output.glob("spaces_step_*.pt")))


if __name__ == "__main__":
    unittest.main()
