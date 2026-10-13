"""Analytic checks for the forward-step loss residual and clean logging."""
import copy
import math
import unittest
from unittest.mock import patch

import torch

from src.functions import First, SelectorFunction
from src.metrics import task_affinity_metrics
from src.model import TransformerModel
from src.training import train


class MeanPrediction(torch.nn.Module):
    reduction = "mean"

    def forward(self, outputs, labels):
        return outputs.mean()


class QuadraticTasks(torch.nn.Module):
    def __init__(self, hessians, slopes, dtype=torch.float64):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(2, dtype=dtype))
        self.register_buffer("hessians", torch.tensor(hessians, dtype=dtype))
        self.register_buffer("slopes", torch.tensor(slopes, dtype=dtype))

    def forward(self, inputs):
        ids = inputs[:, 0]
        return (100 + (self.slopes[ids] * self.theta).sum(-1)
                + .5 * torch.einsum('i,nij,j->n', self.theta, self.hessians[ids], self.theta))


class LocalCurvatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def measure(self, model, rate):
        return task_affinity_metrics(model, SelectorFunction([First()] * 2),
                                     MeanPrediction(), torch.ones((3, 1), dtype=torch.long),
                                     learning_rate=rate, eval_batch_size=2)

    def test_matches_quadratic_hessian_for_every_source_target_and_step_size(self):
        model = QuadraticTasks([[[2, 0], [0, 8]], [[8, 3], [3, 4]]], [[-2, 1], [1, -3]])
        previous = None
        for rate in (.01, .02):
            # Curvature reuses exactly the one existing gradient per task.
            with patch('torch.autograd.grad', wraps=torch.autograd.grad) as backward:
                metrics = self.measure(model, rate)
            self.assertEqual(backward.call_count, 2)
            for i in range(2):
                delta = -rate * model.slopes[i]
                for j in range(2):
                    residual = float(.5 * delta @ model.hessians[j] @ delta)
                    linear = float(model.slopes[j] @ delta)
                    pair = f'first_{i}_to_first_{j}'
                    self.assertAlmostEqual(metrics[f'local_curvature/{pair}'], residual, places=12)
                    self.assertAlmostEqual(metrics[f'affinity/{pair}'], -(linear + residual) / 100, places=12)
                    if previous is not None:
                        self.assertAlmostEqual(metrics[f'local_curvature/{pair}'],
                                               4 * previous[f'local_curvature/{pair}'], places=12)
            previous = metrics

    def test_positive_affinity_can_coexist_with_positive_or_negative_curvature(self):
        model = QuadraticTasks([[[2, 0], [0, 2]], [[-2, 0], [0, -2]]], [[-2, 0], [-2, 0]])
        metrics = self.measure(model, .1)
        self.assertGreater(metrics['affinity/first_0_to_first_0'], 0)
        self.assertGreater(metrics['affinity/first_0_to_first_1'], 0)
        self.assertAlmostEqual(metrics['local_curvature/first_0_to_first_0'], .04, places=12)
        self.assertAlmostEqual(metrics['local_curvature/first_0_to_first_1'], -.04, places=12)

    def test_float32_model_resolves_small_curvature_without_changing_training_dtype(self):
        model = QuadraticTasks([[[2, 0], [0, 2]]] * 2, [[-2, 0]] * 2, dtype=torch.float32)
        before = copy.deepcopy(model.state_dict())
        metrics = self.measure(model, 1e-4)
        self.assertAlmostEqual(metrics['local_curvature/first_0_to_first_1'] / 4e-8, 1., places=5)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertEqual(model.theta.dtype, torch.float32)
        self.assertIsNone(model.theta.grad)

    def test_affine_target_has_no_curvature_despite_nonzero_affinity(self):
        model = QuadraticTasks([[[0, 0], [0, 0]]] * 2, [[-2, 1], [1, -3]])
        metrics = self.measure(model, .1)
        self.assertNotEqual(metrics['affinity/first_0_to_first_1'], 0)
        for key, value in metrics.items():
            if key.startswith('local_curvature/'):
                self.assertAlmostEqual(value, 0., places=12)

    def test_nonfinite_target_loss_is_undefined_even_for_a_zero_step(self):
        model = QuadraticTasks([[[0, 0], [0, 0]]] * 2, [[0, 0]] * 2)
        original_forward = model.forward

        def forward(inputs):
            return original_forward(inputs) + torch.where(inputs[:, 0] == 1, float('inf'), 0.)

        with patch.object(model, 'forward', side_effect=forward):
            metrics = self.measure(model, .1)
        self.assertEqual(metrics['affinity/first_0_to_first_0'], 0.)
        self.assertEqual(metrics['local_curvature/first_0_to_first_0'], 0.)
        self.assertTrue(math.isnan(metrics['affinity/first_0_to_first_1']))
        self.assertTrue(math.isnan(metrics['local_curvature/first_0_to_first_1']))

    def test_boolean_training_logs_only_requested_metrics_and_two_matching_matrices(self):
        model = TransformerModel(d_model=8, num_attn_heads=2, num_layers=1, max_len=4)
        records = []

        class Run:
            def log(self, metrics, step):
                records.append((step, metrics))

        with patch('src.metrics.gradient_metrics', side_effect=AssertionError('removed diagnostic')), \
             patch('src.metrics.multi_task_curvature', side_effect=AssertionError('removed diagnostic')), \
             patch('src.spaces.loss_spaces', side_effect=AssertionError('removed diagnostic')):
            train(model, SelectorFunction([First()] * 2), torch.nn.MSELoss(),
                  torch.optim.SGD(model.parameters(), lr=.001), max_len=4, min_len=4,
                  num_steps=1, batch_size=4, balanced_selectors=True, wandb_run=Run(),
                  affinity_interval=1, affinity_num_inputs=2, eval_interval=1,
                  eval_num_inputs_per_function=2, sharpness_interval=1,
                  sharpness_num_inputs=2, sharpness_num_perturbations=1)
        self.assertEqual(len(records), 1)
        metrics = records[0][1]
        allowed = {'loss', 'accuracy', 'eval', 'sharpness', 'affinity', 'local_curvature'}
        self.assertTrue(all(key.split('/')[0] in allowed for key in metrics))
        for kind in ('affinity', 'local_curvature'):
            self.assertEqual({key for key in metrics if key.startswith(kind + '/')},
                             {f'{kind}/first_{i}_to_first_{j}' for i in range(2) for j in range(2)}
                             | {f'{kind}/matrix'})
            table = metrics[f'{kind}/matrix']
            self.assertEqual(table.columns, ['source / target', 'first_0', 'first_1'])
            self.assertEqual(table.data, [[f'first_{i}', *[metrics[f'{kind}/first_{i}_to_first_{j}']
                                                          for j in range(2)]] for i in range(2)])

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is unavailable')
    def test_mps_uses_precise_cpu_probes_and_restores_state(self):
        model = QuadraticTasks([[[2, 0], [0, 2]]] * 2, [[-2, 0]] * 2, dtype=torch.float32)
        expected = self.measure(model, 1e-4)
        model = model.to('mps')
        actual = task_affinity_metrics(model, SelectorFunction([First()] * 2), MeanPrediction(),
                                       torch.ones((3, 1), dtype=torch.long, device='mps'))
        self.assertEqual(actual, expected)
        self.assertEqual(model.theta.device.type, 'mps')


if __name__ == '__main__':
    unittest.main()
