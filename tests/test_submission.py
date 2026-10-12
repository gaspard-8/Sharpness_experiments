"""Regression checks for the documented HTCondor -append precedence contract."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

import test_arithmetic


ROOT = Path(__file__).resolve().parents[1]


class ArithmeticSubmissionTest(unittest.TestCase):
    def submission_arguments(self, *overrides):
        # Capture the real launcher's argument boundaries without credentials,
        # contacting W&B, or submitting anything to a scheduler.
        with tempfile.TemporaryDirectory(prefix="arithmetic-submit-") as directory:
            project = Path(directory)
            shutil.copy2(ROOT / "submit.sh", project / "submit.sh")
            (project / "submit-wandb.sh").write_text(
                '#!/usr/bin/env bash\nprintf "%s\\0" "$@"\n'
            )
            result = subprocess.run(
                ["bash", str(project / "submit.sh"), *overrides],
                cwd=project.parent, env=dict(os.environ), check=True, capture_output=True,
            )
            return result.stdout.decode().rstrip("\0").split("\0"), str(project)

    def appended_settings(self, arguments):
        # Bare assignments precede test.sub; only -append settings override it.
        # Verify every built-in setting uses that documented interface.
        self.assertEqual(len(arguments) % 2, 0)
        settings = {}
        for option, assignment in zip(arguments[::2], arguments[1::2]):
            self.assertEqual(option, "-append")
            key, value = assignment.split("=", 1)
            settings[key.strip()] = value.strip()
        return settings

    def test_arithmetic_entrypoint_arguments_and_logs_override_submit_defaults(self):
        arguments, project = self.submission_arguments()
        settings = self.appended_settings(arguments)
        self.assertEqual(settings["project_dir"], project)
        self.assertEqual(settings["experiment_script"], "test_arithmetic.py")
        parsed = test_arithmetic.build_parser().parse_args(shlex.split(settings["experiment_args"]))
        self.assertTrue(parsed.wandb)
        self.assertFalse(parsed.smoke_test)
        self.assertEqual(parsed.steps, 30_000)
        self.assertTrue(parsed.grouped_selectors)
        self.assertFalse(parsed.loss_spaces)
        self.assertEqual(len(parsed.operations), 7)
        self.assertIn("addition_mod_10", parsed.operations)
        self.assertIn("majority", parsed.operations)
        self.assertEqual(settings["output"], "$(log_dir)/arithmetic.$(ClusterId).$(ProcId).out")
        self.assertEqual(settings["error"], "$(log_dir)/arithmetic.$(ClusterId).$(ProcId).err")
        self.assertEqual(settings["log"], "$(log_dir)/arithmetic.$(ClusterId).log")

    def test_user_append_overrides_follow_launcher_defaults(self):
        arguments, _ = self.submission_arguments(
            "-append", "experiment_args=--steps 1000 --grouped-selectors --digits-per-argument 3 --no-loss-spaces",
            "-append", "request_memory=12GB",
        )
        settings = self.appended_settings(arguments)
        self.assertEqual(settings["experiment_script"], "test_arithmetic.py")
        self.assertEqual(settings["request_memory"], "12GB")
        parsed = test_arithmetic.build_parser().parse_args(shlex.split(settings["experiment_args"]))
        self.assertFalse(parsed.smoke_test)
        self.assertEqual(parsed.steps, 1000)
        self.assertFalse(parsed.loss_spaces)
        self.assertEqual(parsed.digits_per_argument, 3)

    def test_condor_options_are_forwarded_unchanged(self):
        arguments, _ = self.submission_arguments("-dry-run", "job-preview.ad")
        self.assertEqual(arguments[-2:], ["-dry-run", "job-preview.ad"])
        self.appended_settings(arguments[:-2])


if __name__ == "__main__":
    unittest.main()
