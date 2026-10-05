import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.runtime import DiagnosticProgress, configure_cpu_environment


class RuntimeTest(unittest.TestCase):
    def test_thread_limits_are_applied_before_numerical_libraries_import(self):
        environment = dict(os.environ, SHARPNESS_CPUS="2", OPENBLAS_NUM_THREADS="32")
        result = subprocess.run(
            [sys.executable, "-c", "import json, os; import test; import torch; "
             "print(json.dumps({'intra': torch.get_num_threads(), 'interop': torch.get_num_interop_threads(), "
             "'openblas': os.environ['OPENBLAS_NUM_THREADS']}))"],
            env=environment, check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(result.stdout), {"intra": 2, "interop": 1, "openblas": "2"})

    def test_invalid_cpu_allocation_is_rejected_before_pool_configuration(self):
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ, {"SHARPNESS_CPUS": value}):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    configure_cpu_environment()

    def test_progress_flushes_parseable_records_without_using_training_rng(self):
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()) as output:
            progress = DiagnosticProgress(Path(temporary), torch.device("cpu"))
            progress({"stage": "hvp", "hvp_count": 500, "task": "parity"})
            progress({"stage": "task_complete", "task": "parity"})
            lines = (Path(temporary) / "diagnostic-progress.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in lines]
        self.assertEqual([row["stage"] for row in records], ["hvp", "task_complete"])
        self.assertEqual(records[0]["hvp_count"], 500)
        self.assertGreater(records[0]["process_peak_rss_mib"], 0)
        self.assertEqual(records[0]["torch_cpu_threads"], torch.get_num_threads())
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        torch.testing.assert_close(torch.get_rng_state(), rng)

    def test_progress_survives_disk_write_failure_and_still_logs_to_stdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "file"
            path.write_text("not a directory")
            progress = DiagnosticProgress(path, torch.device("cpu"))
            with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()) as error:
                progress({"stage": "artifact_saved"})
        self.assertIn('"stage": "artifact_saved"', output.getvalue())
        self.assertIn("Could not save diagnostic progress", error.getvalue())


if __name__ == "__main__":
    unittest.main()
