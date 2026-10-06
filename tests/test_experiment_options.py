import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class ExperimentOptionsTest(unittest.TestCase):
    def test_loss_spaces_default_off_and_explicit_switches_are_logged(self):
        import test as experiment

        for flags, enabled in (
            ([], False), (["--smoke-test"], False),
            (["--loss-spaces"], True),
            (["--smoke-test", "--loss-spaces"], True),
            (["--loss-spaces", "--no-loss-spaces"], False),
        ):
            with self.subTest(flags=flags), redirect_stdout(io.StringIO()), \
                    patch("sys.argv", ["test.py", *flags]), \
                    patch.dict("os.environ", {"SHARPNESS_DEVICE": "cpu"}), \
                    patch.object(experiment.wandb, "init") as init, \
                    patch.object(experiment, "train") as training, \
                    patch.object(experiment, "DiagnosticProgress"):
                experiment.main()
                self.assertIs(training.call_args.kwargs["space_enabled"], enabled)
                self.assertIs(init.call_args.kwargs["config"]["train_space_enabled"], enabled)
                self.assertEqual(training.call_args.kwargs["space_delta"], .02)
                affinity_interval = 1 if "--smoke-test" in flags else 100
                self.assertEqual(training.call_args.kwargs["affinity_interval"], affinity_interval)
                self.assertEqual(init.call_args.kwargs["config"]["train_affinity_interval"], affinity_interval)
                self.assertNotIn("affinity_radius", training.call_args.kwargs)
                self.assertNotIn("train_affinity_radius", init.call_args.kwargs["config"])


if __name__ == "__main__":
    unittest.main()
