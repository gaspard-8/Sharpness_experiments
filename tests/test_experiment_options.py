import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class ExperimentOptionsTest(unittest.TestCase):
    def test_only_retained_diagnostics_are_configured(self):
        import test as experiment

        for flags in ([], ["--smoke-test"]):
            with self.subTest(flags=flags), redirect_stdout(io.StringIO()), \
                    patch("sys.argv", ["test.py", *flags]), \
                    patch.dict("os.environ", {"SHARPNESS_DEVICE": "cpu"}), \
                    patch.object(experiment.wandb, "init") as init, \
                    patch.object(experiment, "train") as training, \
                    patch.object(experiment, "DiagnosticProgress"):
                experiment.main()
                kwargs = training.call_args.kwargs
                config = init.call_args.kwargs["config"]
                self.assertFalse(any(key.startswith(("space_", "gradient_", "curvature_")) for key in kwargs))
                self.assertFalse(any(key.startswith(("train_space_", "train_gradient_", "train_curvature_")) for key in config))
                affinity_interval = 1 if "--smoke-test" in flags else 100
                self.assertEqual(kwargs["affinity_interval"], affinity_interval)
                self.assertEqual(config["train_affinity_interval"], affinity_interval)
                self.assertNotIn("affinity_radius", kwargs)


if __name__ == "__main__":
    unittest.main()
