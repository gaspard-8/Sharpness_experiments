import unittest
from unittest.mock import patch

import torch

from src.functions import Parity, SelectorFunction
from src.model import TransformerModel
from src.spaces import _lanczos, _maximize_quadratic, loss_spaces
from src.training import train


class AxisModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor([2.0, 3.0, 4.0]))

    def forward(self, inputs):
        return torch.where(inputs[:, 0].bool(), self.weights[1], self.weights[0]).expand(inputs.size(0))


class MeanOutputLoss(torch.nn.Module):
    def forward(self, outputs, labels):
        return outputs.mean()


class QuadraticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        return (self.weight.square()).expand(inputs.size(0))


class QuarticModel(QuadraticModel):
    def forward(self, inputs):
        return self.weight.pow(4).expand(inputs.size(0))


class HiddenCurvatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(20))

    def forward(self, inputs):
        return self.weights[-1].square().expand(inputs.size(0))


class DiagonalLossModel(torch.nn.Module):
    def __init__(self, diagonal, linear=None):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros_like(diagonal))
        self.register_buffer("diagonal", diagonal)
        self.register_buffer("linear", torch.zeros_like(diagonal) if linear is None else linear)

    def forward(self, inputs):
        loss = (self.weights.square() * self.diagonal).sum() / 2 + self.weights @ self.linear
        return loss.expand(inputs.size(0))


class CapturingRun:
    def __init__(self):
        self.id = "test-run"
        self.logged = []
        self.artifacts = []

    def log(self, metrics, step):
        self.logged.append((step, metrics))

    def log_artifact(self, artifact, aliases):
        self.artifacts.append((artifact, aliases))


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.saved = None

    def add_file(self, path):
        self.saved = torch.load(path, weights_only=False)


class LossSpacesTest(unittest.TestCase):
    def test_progress_reports_restarts_without_spending_extra_products(self):
        diagonal = torch.linspace(-3, 3, 40, dtype=torch.float64)
        events = []
        calls = []

        def product(vector):
            calls.append(1)
            return diagonal * vector

        _, _, _, solver = _lanczos(
            product, 40, 100, [], 3, torch.Generator().manual_seed(2), torch.float64,
            max_basis_dim=16, progress=events.append,
        )
        self.assertEqual(len(calls), 100)
        self.assertEqual(solver["hvp_count"], 100)
        self.assertEqual(events[0]["hvp_count"], 1)
        self.assertEqual(events[-1]["hvp_count"], 100)
        self.assertTrue(any(event["stage"] == "lanczos_restart" for event in events))

    def test_progress_identifies_task_checks_interference_and_completion(self):
        events = []
        model = AxisModel()
        metrics, artifact = loss_spaces(
            model, SelectorFunction([Parity(), Parity()]), MeanOutputLoss(),
            torch.tensor([[0], [1]]), delta=0.02, epsilon=0.002,
            hvp_budget=4, num_directions=2, progress=events.append,
        )
        stages = [event["stage"] for event in events]
        for stage in ("baseline_gradients", "hvp", "eigensystem", "direction_checks",
                      "interference_start", "artifact_cpu_export", "measurement_complete"):
            self.assertIn(stage, stages)
        self.assertEqual(stages.count("task_start"), 2)
        self.assertEqual(stages.count("task_complete"), 2)
        self.assertEqual(stages[-1], "measurement_complete")
        for task, result in artifact["tasks"].items():
            self.assertEqual(metrics[f"spaces/hvp_count/{task}"], 3)
            self.assertEqual(result["hvp_count"], 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_lanczos_keeps_products_and_restarted_algebra_on_device(self):
        diagonal = torch.linspace(-5, 5, 40, dtype=torch.float64, device="cuda")
        seed = torch.ones_like(diagonal)
        calls = 0

        def product(vector):
            nonlocal calls
            self.assertEqual(vector.device, diagonal.device)
            calls += 1
            return diagonal * vector

        basis, images, reduced, solver = _lanczos(
            product, 40, 1000, [seed], 2,
            torch.Generator(device="cuda").manual_seed(19), torch.float64,
            max_basis_dim=16,
        )
        self.assertEqual(calls, 1000)
        self.assertGreater(solver["lanczos_restarts"], 0)
        for tensor in (basis, images, reduced):
            self.assertEqual(tensor.device, diagonal.device)
        torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1], dtype=torch.float64, device="cuda"))
        torch.testing.assert_close(images, diagonal[:, None] * basis, atol=1e-10, rtol=1e-10)
        values = torch.linalg.eigvalsh(reduced)
        torch.testing.assert_close(values[:2], diagonal[:2], atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(values[-2:], diagonal[-2:], atol=1e-8, rtol=1e-8)

    def assert_accelerator_spaces_match_cpu(self, device):
        function = SelectorFunction([Parity(), Parity()])
        payloads = torch.tensor([[0], [1]])
        model = AxisModel().to(device)
        model.train()
        original = model.weights.detach().clone()
        model.weights.grad = torch.ones_like(model.weights)
        previous_grad = model.weights.grad
        cpu_rng = torch.get_rng_state().clone()
        accelerator_rng = torch.cuda.get_rng_state() if device == "cuda" else torch.mps.get_rng_state()
        expected, expected_artifact = loss_spaces(
            AxisModel(), function, MeanOutputLoss(), payloads,
            delta=1.0, epsilon=0.5, num_directions=3,
        )

        original_lanczos = _lanczos

        def checked(hvp, *args, **kwargs):
            def product(vector):
                self.assertEqual(vector.device.type, "cuda" if device == "cuda" else "cpu")
                image = hvp(vector)
                self.assertEqual(image.device, vector.device)
                return image

            return original_lanczos(product, *args, **kwargs)

        with patch("src.spaces._lanczos", side_effect=checked):
            actual, artifact = loss_spaces(
                model, function, MeanOutputLoss(), payloads.to(device),
                delta=1.0, epsilon=0.5, num_directions=3,
            )
        self.assertEqual(artifact["solver_device"].split(":")[0], "cuda" if device == "cuda" else "cpu")
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for name, result in artifact["tasks"].items():
            for kind in ("tangent", "sharpness"):
                basis = result[kind]
                reference = expected_artifact["tasks"][name][kind]
                torch.testing.assert_close(basis @ basis.T, reference @ reference.T, atol=1e-5, rtol=1e-5)
        for key, result in artifact["interference"].items():
            basis = result["directions"]
            reference = expected_artifact["interference"][key]["directions"]
            torch.testing.assert_close(basis @ basis.T, reference @ reference.T, atol=1e-5, rtol=1e-5)

        def assert_cpu(value):
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device.type, "cpu")
            elif isinstance(value, dict):
                for item in value.values():
                    assert_cpu(item)

        assert_cpu(artifact)
        torch.testing.assert_close(model.weights, original)
        self.assertIs(model.weights.grad, previous_grad)
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng)
        current_rng = torch.cuda.get_rng_state() if device == "cuda" else torch.mps.get_rng_state()
        torch.testing.assert_close(current_rng, accelerator_rng)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_spaces_match_cpu_and_export_portable_artifacts(self):
        self.assert_accelerator_spaces_match_cpu("cuda")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_spaces_use_cpu_algebra_and_preserve_device_state(self):
        self.assert_accelerator_spaces_match_cpu("mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_transformer_spaces_preserve_state_and_verify_actual_harms(self):
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4, dropout=0.2).cuda()
        model.train()
        parameters = [p.detach().clone() for p in model.parameters()]
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        gradients = [p.grad for p in model.parameters()]
        rng = torch.cuda.get_rng_state().clone()
        _, artifact = loss_spaces(
            model, Parity(), torch.nn.MSELoss(),
            torch.tensor([[0, 0, 0, 1], [1, 0, 1, 1]], device="cuda"),
            delta=0.02, epsilon=0.01, num_directions=3, hvp_budget=8,
        )
        result = artifact["tasks"]["parity"]
        for kind in ("tangent", "sharpness"):
            basis = result[kind]
            torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1]), atol=1e-5, rtol=1e-5)
            harms = result[f"{kind}_loss_changes"].amax(dim=1)
            self.assertTrue(bool((harms <= 0.01).all()) if kind == "tangent" else bool((harms > 0.01).all()))
        for p, previous, gradient in zip(model.parameters(), parameters, gradients):
            torch.testing.assert_close(p, previous)
            self.assertIs(p.grad, gradient)
            torch.testing.assert_close(p.grad, torch.ones_like(p))
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng)

    def test_thick_restarts_preserve_extreme_modes_images_and_gradient_seed(self):
        diagonal = torch.linspace(-5, 5, 160, dtype=torch.float64)
        diagonal[:3], diagonal[-3:] = -7, 7
        generator = torch.Generator().manual_seed(19)
        seed = torch.randn(160, generator=generator, dtype=torch.float64)
        calls = 0

        def product(vector):
            nonlocal calls
            calls += 1
            return diagonal * vector

        basis, images, reduced, solver = _lanczos(
            product, 160, 10_000, [seed], 6, generator, torch.float64,
            max_basis_dim=64,
        )
        self.assertEqual(calls, 10_000)
        self.assertEqual(solver["hvp_count"], calls)
        self.assertGreater(solver["lanczos_restarts"], 0)
        self.assertLessEqual(basis.shape[1], 64)
        self.assertEqual(solver["max_basis_dim"], 64)
        self.assertFalse(solver["full_space_covered"])
        torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1], dtype=torch.float64))
        torch.testing.assert_close(images, diagonal[:, None] * basis, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(basis @ (basis.T @ seed), seed, atol=1e-10, rtol=1e-10)
        eigenvalues = torch.linalg.eigvalsh(reduced)
        torch.testing.assert_close(eigenvalues[:6], diagonal[:6], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(eigenvalues[-6:], diagonal[-6:], atol=1e-9, rtol=1e-9)

    def test_new_budget_and_100_direction_cap_with_actual_task_hessian(self):
        model = DiagonalLossModel(torch.cat((torch.zeros(550), torch.full((50,), 20.0))))
        model.weights.grad = torch.ones_like(model.weights)
        gradients = model.weights.grad
        parameters = model.weights.detach().clone()
        torch.manual_seed(23)
        rng = torch.random.get_rng_state().clone()
        metrics, artifact = loss_spaces(
            model, Parity(), MeanOutputLoss(), torch.tensor([[0]]),
            delta=0.02, epsilon=0.002,
        )
        result = artifact["tasks"]["parity"]
        self.assertEqual(metrics["spaces/hvp_budget"], 10_000)
        self.assertEqual(metrics["spaces/hvp_count/parity"], 10_000)
        self.assertEqual(metrics["spaces/direction_cap"], 100)
        self.assertEqual(metrics["spaces/tangent_dim/parity"], 100)
        self.assertEqual(metrics["spaces/sharpness_dim/parity"], 50)
        self.assertEqual(metrics["spaces/tangent_cap_reached/parity"], 1)
        self.assertGreater(metrics["spaces/lanczos_restarts/parity"], 0)
        self.assertLessEqual(metrics["spaces/krylov_dim/parity"], 512)
        self.assertEqual(metrics["spaces/krylov_max_dim"], 512)
        for kind in ("tangent", "sharpness"):
            vectors = result[kind].double()
            torch.testing.assert_close(
                vectors.T @ vectors, torch.eye(vectors.shape[1], dtype=torch.float64),
                atol=1e-5, rtol=1e-5,
            )
            harms = result[f"{kind}_loss_changes"].max(dim=1).values
            self.assertTrue(bool((harms <= 0.002).all()) if kind == "tangent" else bool((harms > 0.002).all()))
        torch.testing.assert_close(model.weights, parameters)
        self.assertIs(model.weights.grad, gradients)
        torch.testing.assert_close(model.weights.grad, torch.ones_like(model.weights))
        torch.testing.assert_close(torch.random.get_rng_state(), rng)

    def test_lanczos_spends_500_products_and_recovers_repeated_eigenvalues(self):
        diagonal = torch.cat((torch.zeros(512, dtype=torch.float64), torch.full((8,), 4.0, dtype=torch.float64)))
        calls = []

        def product(vector):
            calls.append(vector.clone())
            return diagonal * vector

        basis, images, reduced, solver = _lanczos(
            product, 520, 500, [], 8, torch.Generator().manual_seed(17), torch.float64,
        )
        self.assertEqual(len(calls), 500)
        self.assertEqual(basis.shape, (520, 500))
        self.assertGreater(solver["random_restarts"], 0)
        torch.testing.assert_close(basis.T @ basis, torch.eye(500, dtype=torch.float64))
        torch.testing.assert_close(images, diagonal[:, None] * basis)
        eigenvalues, eigenvectors = torch.linalg.eigh(reduced)
        torch.testing.assert_close(eigenvalues[-8:], torch.full((8,), 4.0, dtype=torch.float64))
        vectors = basis @ eigenvectors[:, -8:]
        torch.testing.assert_close(diagonal[:, None] * vectors, 4 * vectors, atol=1e-10, rtol=1e-10)

    def test_budget_is_shared_between_both_kinds_for_each_task(self):
        model = DiagonalLossModel(torch.cat((torch.zeros(16), torch.arange(1.0, 5.0))))
        function = SelectorFunction([Parity(), Parity()])
        counts = []
        original_lanczos = _lanczos

        def counted(hvp, *args):
            counts.append(0)

            def product(vector):
                counts[-1] += 1
                return hvp(vector)

            return original_lanczos(product, *args)

        with patch("src.spaces._lanczos", side_effect=counted):
            metrics, artifact = loss_spaces(
                model, function, MeanOutputLoss(), torch.tensor([[0], [1]]),
                delta=0.2, epsilon=0.01, num_directions=2, hvp_budget=12,
            )
        self.assertEqual(counts, [12, 12])
        self.assertEqual(metrics["spaces/hvp_budget"], 12)
        for name in ("parity_0", "parity_1"):
            self.assertEqual(metrics[f"spaces/hvp_count/{name}"], 12)
            self.assertEqual(metrics[f"spaces/tangent_dim/{name}"], 2)
            self.assertEqual(metrics[f"spaces/sharpness_dim/{name}"], 2)
        self.assertEqual(artifact["format_version"], 4)
        self.assertNotIn("search_steps", artifact)
        self.assertNotIn("search_restarts", artifact)

    def test_quadratic_modes_are_selected_in_curvature_order(self):
        model = DiagonalLossModel(torch.tensor([-3.0, 0.0, 2.0, 6.0], dtype=torch.float64))
        metrics, artifact = loss_spaces(
            model, Parity(), MeanOutputLoss(), torch.tensor([[0], [1]]),
            delta=1.0, epsilon=0.1, num_directions=4,
        )
        result = artifact["tasks"]["parity"]
        self.assertEqual(metrics["spaces/hvp_count/parity"], 4)
        self.assertEqual(metrics["spaces/hvp_budget_exhausted/parity"], 0)
        torch.testing.assert_close(result["tangent_curvatures"], torch.tensor([-3.0, 0.0], dtype=torch.float64))
        torch.testing.assert_close(result["sharpness_curvatures"], torch.tensor([6.0, 2.0], dtype=torch.float64))
        torch.testing.assert_close(result["tangent_predicted_harms"], result["tangent_loss_changes"][:, 0].double())
        torch.testing.assert_close(result["sharpness_predicted_harms"], result["sharpness_loss_changes"][:, 0].double())
        self.assertLess(float(result["ritz_residual_norms"].max()), 1e-10)

    def test_nonzero_gradient_is_used_for_sharpness_and_removed_for_tangent(self):
        model = DiagonalLossModel(
            torch.tensor([0.0, 2.0, 3.0], dtype=torch.float64),
            torch.tensor([-10.0, 0.0, 0.0], dtype=torch.float64),
        )
        _, artifact = loss_spaces(
            model, Parity(), MeanOutputLoss(), torch.tensor([[0]]),
            delta=0.1, epsilon=0.02, num_directions=3,
        )
        result = artifact["tasks"]["parity"]
        # The strongest direction is the gradient, despite its zero curvature.
        torch.testing.assert_close(result["sharpness"], torch.tensor([[1.0], [0.0], [0.0]]))
        # Orientation changes must swap the actual +/- loss-change columns.
        torch.testing.assert_close(result["sharpness_loss_changes"], torch.tensor([[-1.0, 1.0]]))
        torch.testing.assert_close(result["tangent"].T @ result["loss_gradient"], torch.zeros(2))
        self.assertTrue(bool((result["tangent_loss_changes"].max(dim=1).values <= 0.02).all()))

    def test_trust_region_solution_includes_the_hard_case(self):
        hessian = torch.diag(torch.tensor([0.0, 4.0], dtype=torch.float64))
        linear = torch.tensor([1.0, 0.0], dtype=torch.float64)
        direction = _maximize_quadratic(hessian, linear)
        self.assertAlmostEqual(float(direction.norm()), 1.0)
        self.assertAlmostEqual(float(direction[0]), 0.25)
        self.assertAlmostEqual(float(linear @ direction + direction @ hessian @ direction / 2), 2.125)

    def test_known_axes_and_ordered_interference(self):
        model = AxisModel()
        model.train()
        original = model.weights.detach().clone()
        model.weights.grad = torch.tensor([7.0, 8.0, 9.0])
        function = SelectorFunction([Parity(), Parity()])
        metrics, artifact = loss_spaces(
            model, function, MeanOutputLoss(), torch.tensor([[0], [1]]),
            delta=1.0, epsilon=0.5, num_directions=3,
        )

        self.assertEqual(metrics["spaces/search_dim"], 3)
        for name in ("parity_0", "parity_1"):
            self.assertEqual(metrics[f"spaces/tangent_dim/{name}"], 2)
            self.assertEqual(metrics[f"spaces/sharpness_dim/{name}"], 1)
            for kind in ("tangent", "sharpness"):
                coordinates = artifact["tasks"][name][kind]
                torch.testing.assert_close(
                    coordinates.T @ coordinates, torch.eye(coordinates.shape[1]),
                    atol=1e-5, rtol=1e-5,
                )
        for pair in ("parity_0_to_parity_1", "parity_1_to_parity_0"):
            self.assertEqual(metrics[f"spaces/interference_dim/{pair}"], 1)
            vector = artifact["interference"][pair]["directions"]
            self.assertAlmostEqual(float(vector.norm()), 1.0, places=5)
        self.assertTrue(model.training)
        torch.testing.assert_close(model.weights, original)
        torch.testing.assert_close(model.weights.grad, torch.tensor([7.0, 8.0, 9.0]))

    def test_threshold_uses_both_signs_of_actual_loss_change(self):
        model = AxisModel()
        function = SelectorFunction([Parity(), Parity()])
        metrics, _ = loss_spaces(
            model, function, MeanOutputLoss(), torch.tensor([[0]]),
            delta=0.02, epsilon=0.01, num_directions=3,
        )
        self.assertEqual(metrics["spaces/tangent_dim/parity_0"], 2)
        self.assertEqual(metrics["spaces/sharpness_dim/parity_0"], 1)

    def test_finite_radius_detects_harm_at_zero_loss_gradient(self):
        metrics, artifact = loss_spaces(
            QuadraticModel().double(), Parity(), MeanOutputLoss(), torch.tensor([[0], [1]]),
            delta=0.2, epsilon=0.01, num_directions=1,
        )
        self.assertEqual(metrics["spaces/tangent_dim/parity"], 0)
        self.assertEqual(metrics["spaces/sharpness_dim/parity"], 1)
        self.assertAlmostEqual(metrics["spaces/max_harm/parity"], 0.04, places=5)
        self.assertEqual(artifact["tasks"]["parity"]["sharpness"].shape, (1, 1))

    def test_finite_radius_detects_harm_when_gradient_and_hessian_are_zero(self):
        metrics, artifact = loss_spaces(
            QuarticModel(), Parity(), MeanOutputLoss(), torch.tensor([[0], [1]]),
            delta=0.5, epsilon=0.01, num_directions=1,
        )
        self.assertEqual(metrics["spaces/tangent_dim/parity"], 0)
        self.assertEqual(metrics["spaces/sharpness_dim/parity"], 1)
        self.assertAlmostEqual(metrics["spaces/max_harm/parity"], 0.0625, places=5)
        torch.testing.assert_close(
            artifact["tasks"]["parity"]["sharpness_loss_changes"],
            torch.tensor([[0.0625, 0.0625]]),
        )

    def test_one_direction_cap_still_searches_all_parameter_coordinates(self):
        metrics, artifact = loss_spaces(
            HiddenCurvatureModel(), Parity(), MeanOutputLoss(), torch.tensor([[0], [1]]),
            delta=1.0, epsilon=0.5, num_directions=1, hvp_budget=20,
        )
        self.assertEqual(metrics["spaces/search_dim"], 20)
        self.assertEqual(metrics["spaces/direction_cap"], 1)
        self.assertEqual(metrics["spaces/tangent_dim/parity"], 1)
        self.assertEqual(metrics["spaces/sharpness_dim/parity"], 1)
        self.assertEqual(metrics["spaces/tangent_cap_reached/parity"], 1)
        self.assertEqual(metrics["spaces/sharpness_cap_reached/parity"], 1)
        self.assertGreater(metrics["spaces/max_harm/parity"], 0.9)
        directions = artifact["tasks"]["parity"]["sharpness"]
        self.assertEqual(directions.shape, (20, 1))
        self.assertGreater(abs(float(directions[-1, 0])), 0.95)
        self.assertNotIn("search_basis", artifact)

    def test_transformer_preserves_parameters_gradients_modes_and_rng(self):
        torch.manual_seed(14)
        model = TransformerModel(
            d_model=8, num_attn_heads=2, num_layers=1, max_len=4, dropout=0.2,
        )
        model.train()
        model.encoder.layers[0].linear1.eval()
        modes = [module.training for module in model.modules()]
        original = [p.detach().clone() for p in model.parameters()]
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        gradients = [p.grad for p in model.parameters()]
        rng = torch.random.get_rng_state().clone()
        metrics, artifact = loss_spaces(
            model, Parity(), torch.nn.MSELoss(),
            torch.tensor([[0, 0, 0, 1], [1, 0, 1, 1]]),
            delta=0.02, epsilon=0.01, num_directions=3,
            hvp_budget=8,
        )
        self.assertEqual(metrics["spaces/search_dim"], sum(p.numel() for p in model.parameters()))
        self.assertEqual(metrics["spaces/direction_cap"], 3)
        self.assertEqual([m.training for m in model.modules()], modes)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        for p, before, gradient in zip(model.parameters(), original, gradients):
            torch.testing.assert_close(p, before)
            self.assertIs(p.grad, gradient)
            torch.testing.assert_close(p.grad, torch.ones_like(p))
        for kind in ("tangent", "sharpness"):
            actual = artifact["tasks"]["parity"][kind].T
            torch.testing.assert_close(
                actual @ actual.T, torch.eye(actual.shape[0]), atol=1e-5, rtol=1e-5,
            )

    def test_training_measures_only_at_end(self):
        model = AxisModel()
        run = CapturingRun()
        with patch("src.training.loss_spaces", return_value=({"spaces/search_dim": 2.0}, {"directions": [1, 2]})) as measured, patch("wandb.Artifact", FakeArtifact):
            train(
                model, SelectorFunction([Parity(), Parity()]), torch.nn.MSELoss(),
                torch.optim.SGD(model.parameters(), lr=0.01), max_len=2,
                min_len=2, batch_size=2, num_steps=5, wandb_run=run,
                sharpness_interval=100, gradient_interval=100,
                curvature_interval=100, space_delta=0.02,
                space_epsilon=0.01, space_num_inputs=3, space_seed=17,
            )
        self.assertEqual(measured.call_count, 1)
        self.assertEqual([aliases for _, aliases in run.artifacts], [["step-5"]])
        self.assertEqual([artifact.metadata["step"] for artifact, _ in run.artifacts], [5])
        self.assertEqual([artifact.saved for artifact, _ in run.artifacts], [{"directions": [1, 2]}])
        self.assertEqual(
            [step for step, metrics in run.logged if "spaces/search_dim" in metrics],
            [5],
        )
        self.assertEqual(measured.call_args.kwargs["hvp_budget"], 10_000)
        self.assertEqual(measured.call_args.kwargs["num_directions"], 100)
        self.assertEqual(run.artifacts[0][0].metadata["hvp_budget"], 10_000)

    def test_training_logs_real_lanczos_results_and_serializes_directions(self):
        model = AxisModel()
        run = CapturingRun()
        with patch("wandb.Artifact", FakeArtifact):
            train(
                model, SelectorFunction([Parity(), Parity()]), torch.nn.MSELoss(),
                torch.optim.SGD(model.parameters(), lr=0.01), max_len=2,
                min_len=2, batch_size=2, num_steps=2, wandb_run=run,
                sharpness_interval=100, gradient_interval=100,
                curvature_interval=100, space_delta=0.02,
                space_epsilon=0.01, space_num_inputs=3,
            )
        self.assertEqual(len(run.artifacts), 1)
        artifact, aliases = run.artifacts[0]
        self.assertEqual(aliases, ["step-2"])
        self.assertEqual(artifact.saved["format_version"], 4)
        self.assertEqual(artifact.saved["hvp_budget"], 10_000)
        self.assertEqual(artifact.saved["parameter_names"], ["weights"])
        final_step, metrics = run.logged[-1]
        self.assertEqual(final_step, 2)
        self.assertNotIn("spaces/hvp_budget", run.logged[0][1])
        for name in ("parity_0", "parity_1"):
            self.assertEqual(metrics[f"spaces/hvp_count/{name}"], 3)
            result = artifact.saved["tasks"][name]
            self.assertEqual(result["hvp_count"], 3)
            for kind in ("tangent", "sharpness"):
                self.assertEqual(result[kind].shape[0], 3)
                changes = result[f"{kind}_loss_changes"].max(dim=1).values
                self.assertTrue(bool((changes <= 0.01).all()) if kind == "tangent" else bool((changes > 0.01).all()))


if __name__ == "__main__":
    unittest.main()
