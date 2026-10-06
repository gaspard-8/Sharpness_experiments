import csv
import html
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.summarize_spaces import load_report, main, observations, tables, write_report


class SpacesReportTest(unittest.TestCase):
    def make_artifact(self, directory):
        eye = torch.eye(1000)
        artifact = {
            "format_version": 4, "parameter_names": ["weights"], "parameter_shapes": [(1000,)],
            "delta": .02, "epsilon": .01, "intersection_cosine": .999,
            "tasks": {"A": {"tangent": eye[:, :16], "sharpness": eye[:, 30:32]},
                      "B_to_C": {"tangent": eye[:, :1], "sharpness": eye[:, 31:34]}},
        }
        path = directory / "spaces_step_10000.pt"
        torch.save(artifact, path)
        return path

    def make_run(self, directory, run_id="abc123"):
        files = directory / f"run-20261005_100000-{run_id}" / "files"
        files.mkdir(parents=True)
        config = {
            "selector_functions": {"value": ["A", "B_to_C"]},
            "train_space_epsilon": {"value": 0.01},
            "train_space_delta": {"value": 0.02},
            "train_space_num_inputs": {"value": 64},
        }
        (files / "config.json").write_text(json.dumps(config))
        metrics = {
            "_step": 10000,
            "spaces/parameter_count": 1000,
            "spaces/search_dim": 1000,
            "spaces/direction_cap": 16,
            "spaces/hvp_budget": 500,
            "spaces/tangent_dim/A": 16,
            "spaces/tangent_cap_reached/A": 1,
            "spaces/sharpness_dim/A": 2,
            "spaces/sharpness_stop_harm/A": 0.005,
            "spaces/max_harm/A": 0.05,
            "spaces/min_harm/A": -0.0001,
            "spaces/hvp_count/A": 500,
            "spaces/tangent_dim/B_to_C": 1,
            "spaces/sharpness_dim/B_to_C": 3,
            "spaces/tangent_candidate_exhausted/B_to_C": 1,
            "spaces/hvp_count/B_to_C": 500,
            "spaces/interference_dim/A_to_B_to_C": 1,
        }
        path = files / "wandb-summary.json"
        path.write_text(json.dumps(metrics))
        return path

    def test_pair_orientation_and_missing_values_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.make_run(Path(temporary))
            report = load_report(summary)
        self.assertEqual([row["task"] for row in report["tasks"]], ["A", "B_to_C"])
        pairs = {(row["safe_task"], row["harmed_task"]): row["directions"] for row in report["pairs"]}
        self.assertEqual(pairs[("B_to_C", "A")], 1)
        self.assertIsNone(pairs[("A", "B_to_C")])
        matrix = tables(report)[2]
        self.assertEqual(matrix[3][1][1], 1)
        self.assertIsNone(matrix[3][0][2])
        self.assertEqual(matrix[3][0][1], "—")
        self.assertEqual(report["tasks"][0]["first_sharp_over_epsilon"], 5)

    def test_caps_stopping_and_zero_overlap_interpretation(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = load_report(self.make_run(Path(temporary)))
        self.assertEqual(report["tasks"][0]["safe_stop_reason"], "Direction cap reached")
        self.assertEqual(report["tasks"][0]["sharp_stop_reason"], "Next candidate within epsilon")
        self.assertEqual(report["tasks"][1]["safe_stop_reason"], "Approximation candidates exhausted")
        readings = "\n".join(observations(report))
        self.assertIn("1/2 tasks reached", readings)
        self.assertIn("cannot rank the full tangent dimensions", readings)
        self.assertIn("1/1 recorded ordered pairs", readings)
        self.assertIn("1 pairs missing", readings)
        self.assertIn("other safe-for-one/harmful-for-another directions may exist", readings)

    def test_csv_round_trip_exact_numbers_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self.make_run(root)
            before = summary.read_bytes()
            report = load_report(summary)
            output = root / "report"
            write_report(report, output)
            self.assertEqual(summary.read_bytes(), before)
            with (output / "metrics.csv").open() as stream:
                raw = list(csv.DictReader(stream))
            self.assertEqual(len(raw), len(report["catalog"]))
            original = json.loads(before)
            for row in raw:
                self.assertEqual(float(row["value"]), original[row["metric"]])
            with (output / "interference_matrix.csv").open() as stream:
                matrix = list(csv.DictReader(stream))
            self.assertEqual(matrix[1]["safe_task"], "B_to_C")
            self.assertEqual(matrix[1]["A"], "1")
            self.assertEqual(matrix[0]["B_to_C"], "")
            data = json.loads((output / "report.json").read_text())
            self.assertIsNone(data["pairs"][0]["directions"])
            self.assertIn("16 (cap)", (output / "report.md").read_text())

    def test_html_escapes_task_names_and_keeps_every_raw_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self.make_run(root)
            malicious = "<script>alert(1)</script>"
            summary.write_text(summary.read_text().replace("A", malicious))
            config = summary.parent / "config.json"
            config.write_text(config.read_text().replace("A", malicious))
            report = load_report(summary)
            output = root / "report"
            write_report(report, output)
            page = (output / "report.html").read_text()
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)
        self.assertNotIn("<script>alert(1)</script>", page)
        for row in report["catalog"]:
            self.assertIn(html.escape(row["metric"]), page)

    def test_missing_config_does_not_invent_thresholds_or_zero_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self.make_run(root)
            (summary.parent / "config.json").unlink()
            data = json.loads(summary.read_text())
            data.pop("spaces/sharpness_dim/A")
            summary.write_text(json.dumps(data))
            report = load_report(summary)
        self.assertIsNone(report["settings"]["epsilon"])
        row = next(row for row in report["tasks"] if row["task"] == "A")
        self.assertIsNone(row["sharp_directions"])
        self.assertIsNone(row["first_sharp_over_epsilon"])

    def test_cli_discovers_multiple_runs_and_preserves_the_recorded_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.make_run(root, "first")
            self.make_run(root, "second")
            output = root / "output"
            self.assertEqual(main([str(root), "--output-dir", str(output)]), 0)
            self.assertTrue((output / "first" / "report.html").exists())
            self.assertTrue((output / "second" / "report.html").exists())
            with self.assertRaisesRegex(ValueError, "differs from the saved"):
                load_report(first, epsilon=0.5)

    def test_restarted_run_distinguishes_total_products_and_active_storage(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.make_run(Path(temporary))
            config_path = summary.parent / "config.json"
            config = json.loads(config_path.read_text())
            config["train_space_epsilon"]["value"] = 0.002
            config_path.write_text(json.dumps(config))
            metrics = json.loads(summary.read_text())
            metrics.update({
                "spaces/hvp_budget": 10000,
                "spaces/direction_cap": 100,
                "spaces/krylov_max_dim": 512,
                "spaces/hvp_count/A": 10000,
                "spaces/krylov_dim/A": 272,
                "spaces/lanczos_restarts/A": 31,
                "spaces/full_space_covered/A": 0,
            })
            summary.write_text(json.dumps(metrics))
            report = load_report(summary)
        self.assertEqual(report["settings"]["epsilon"], 0.002)
        self.assertEqual(report["settings"]["krylov_max_dim"], 512)
        self.assertEqual(report["quality"][0]["hvp_count"], 10000)
        self.assertEqual(report["quality"][0]["krylov_dim"], 272)
        self.assertEqual(report["quality"][0]["lanczos_restarts"], 31)
        self.assertEqual(report["quality"][0]["full_space_covered"], 0)
        self.assertFalse(any(row["group"] == "Other" for row in report["catalog"]))
        self.assertIn("total computation across restart cycles", "\n".join(observations(report)))
        quality_table = tables(report)[3]
        self.assertIn("Thick restarts", quality_table[2])
        self.assertEqual(quality_table[3][0][1:3], [10000, 272])

    def test_same_kind_logged_pairs_are_symmetric_and_not_discovered_as_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.make_run(Path(temporary))
            metrics = json.loads(summary.read_text())
            metrics.update({"spaces/sharpness_overlap_dim/A_and_B_to_C": 1,
                            "spaces/sharpness_overlap_verified_dim/A_and_B_to_C": 0,
                            "spaces/sharpness_overlap_max_cosine/A_and_B_to_C": .9999,
                            "spaces/tangent_overlap_dim/A_and_B_to_C": 0})
            summary.write_text(json.dumps(metrics))
            report = load_report(summary)
            write_report(report, Path(temporary) / "output")
            with (Path(temporary) / "output" / "sharpness_overlap_matrix.csv").open() as stream:
                matrix = list(csv.DictReader(stream))
        self.assertEqual([row["task"] for row in report["tasks"]], ["A", "B_to_C"])
        self.assertEqual(matrix[0]["B_to_C"], "1")
        self.assertEqual(matrix[1]["A"], "1")
        self.assertEqual(report["overlaps"]["sharpness"][0]["loss_verified_directions"], 0)
        self.assertIsNone(report["overlaps"]["tangent"][0]["loss_verified_directions"])
        self.assertFalse(any(row["group"] == "Other" for row in report["catalog"]))

    def test_saved_artifact_backfills_geometric_overlaps_and_exports_full_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self.make_run(root)
            before = summary.read_bytes()
            report = load_report(summary, directions_artifact=self.make_artifact(root))
            output = root / "output"
            write_report(report, output)
            exported = torch.load(output / "overlap_directions.pt", weights_only=True)
            with (output / "overlap_directions.csv").open() as stream:
                directions = list(csv.DictReader(stream))
            for row in directions:
                value = exported["overlaps"][row["vector_kind"]][row["vector_pair"]]
                vector = value["directions"][:, int(row["vector_column"])]
                expected = torch.zeros(1000)
                expected[31 if row["kind"] == "sharpness" else 0] = 1
                torch.testing.assert_close(vector, expected)
                self.assertEqual(row["loss_verified"], "")
            self.assertEqual(len(directions), 2)
            self.assertEqual(summary.read_bytes(), before)
            self.assertNotIn("_overlap_vectors", json.loads((output / "report.json").read_text()))
            with (output / "metrics.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(report["catalog"]))
        for kind in ("sharpness", "tangent"):
            self.assertEqual(report["overlaps"][kind][0]["directions"], 1)
            self.assertIsNone(report["overlaps"][kind][0]["loss_verified_directions"])

    def test_artifact_checks_run_and_measurement_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self.make_run(root)
            path = self.make_artifact(root)
            artifact = torch.load(path, weights_only=True)
            artifact["epsilon"] = .002
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, "epsilon does not match"):
                load_report(summary, directions_artifact=path)
            artifact["epsilon"] = .01
            artifact["run_id"] = "another-run"
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, "run ID does not match"):
                load_report(summary, directions_artifact=path)

    def test_cli_finds_cached_artifact_by_run_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self.make_run(root)
            cache = root / "artifacts" / "abc123"
            cache.mkdir(parents=True)
            self.make_artifact(cache)
            output = root / "output"
            self.assertEqual(main([str(summary), "--output-dir", str(output),
                                   "--artifact-dir", str(cache.parent)]), 0)
            report = json.loads((output / "abc123" / "report.json").read_text())
            self.assertEqual(report["overlaps"]["tangent"][0]["directions"], 1)

    def test_summary_without_artifact_keeps_missing_overlap_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = load_report(self.make_run(Path(temporary)))
        for kind in ("sharpness", "tangent"):
            self.assertIsNone(report["overlaps"][kind][0]["directions"])
        self.assertEqual(report["overlap_directions"], [])


if __name__ == "__main__":
    unittest.main()
