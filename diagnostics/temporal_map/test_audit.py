import unittest

import torch

from audit import audit_temporal_map


class TemporalAuditTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.arange(24, dtype=torch.float64).reshape(1, 3, 1, 2, 4) / 10
        self.mask = torch.ones(1, 2, 4, dtype=torch.bool)

    def anchor(self, x):
        return x[:, -1:].expand(-1, 2, -1, -1, -1)

    def run_audit(self, model, **kwargs):
        return audit_temporal_map(model, self.x, self.anchor, self.mask, [(0, 0, 0)], **kwargs)

    def test_zero_residual_does_not_look_like_learned_history(self):
        result = self.run_audit(self.anchor)
        self.assertTrue(result['all_probed_residual_gradients_zero'])
        for row in result['rows']:
            self.assertEqual(row['total_history_l1'], [0, 0, 1])
            self.assertEqual(row['residual_history_l1'], [0, 0, 0])
        self.assertTrue(result['all_finite_differences_pass'])

    def test_known_linear_map_and_no_parameter_updates(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = torch.nn.Parameter(torch.tensor([[2., -3., 0.], [0., 4., 5.]], dtype=torch.float64))

            def forward(self, x):
                return x[:, -1:].expand(-1, 2, -1, -1, -1) + torch.einsum('lt,btchw->blchw', self.weights, x)
        model = Linear().eval()
        before = model.weights.detach().clone()
        result = self.run_audit(model)
        self.assertEqual(result['rows'][0]['residual_history_signed_sum'], [2., -3., 0.])
        self.assertEqual(result['rows'][0]['residual_history_l1'], [2., 3., 0.])
        self.assertEqual(result['rows'][1]['residual_history_signed_sum'], [0., 4., 5.])
        self.assertTrue(result['all_finite_differences_pass'])
        self.assertLess(max(abs(r['total_affine_defect']) for r in result['rows']), 1e-12)
        self.assertTrue(torch.equal(model.weights, before))
        self.assertIsNone(model.weights.grad)

    def test_signed_cancellation_keeps_absolute_sensitivity(self):
        def model(x):
            delta = x[:, :1, :, :, :1] - x[:, :1, :, :, 1:2]
            return self.anchor(x) + delta.expand(-1, 2, -1, -1, 4)
        row = self.run_audit(model)['rows'][0]
        self.assertEqual(row['residual_history_signed_sum'], [0., 0., 0.])
        self.assertEqual(row['residual_history_l1'], [2., 0., 0.])

    def test_nonlinear_map_refutes_affine_equivalence(self):
        result = self.run_audit(lambda x: self.anchor(x) + x[:, :1].square().expand(-1, 2, -1, -1, -1))
        self.assertTrue(result['all_finite_differences_pass'])
        self.assertGreater(abs(result['rows'][0]['residual_affine_defect']), 1e-5)

    def test_constant_disconnected_output(self):
        result = self.run_audit(lambda x: torch.zeros(1, 2, 1, 2, 4, dtype=x.dtype))
        self.assertEqual(result['rows'][0]['total_history_l1'], [0., 0., 0.])
        self.assertTrue(result['all_finite_differences_pass'])

    def test_stop_gradient_is_flagged_by_finite_difference(self):
        # Forward dependence can remain even when autograd deliberately detaches it.
        # An attribution report must not silently equate these two derivatives.
        result = self.run_audit(lambda x: self.anchor(x) + x[:, :1].detach().expand(-1, 2, -1, -1, -1))
        self.assertFalse(result['all_finite_differences_pass'])
        self.assertTrue(result['all_probed_residual_gradients_zero'])

    def test_invalid_dimensions_dtype_nonfinite_and_training_mode(self):
        for x in (self.x.float(), self.x[0], self.x.repeat(2, 1, 1, 1, 1), self.x * float('nan')):
            with self.assertRaises(ValueError):
                audit_temporal_map(self.anchor, x, self.anchor, self.mask, [(0, 0, 0)])
        with self.assertRaises(ValueError):
            self.run_audit(torch.nn.Identity())
        with self.assertRaises(ValueError):
            self.run_audit(self.anchor, epsilon=0)

    def test_dry_outside_and_empty_probes(self):
        self.mask[0, 0, 0] = False
        for probes in ([(0, 0, 0)], [(0, -1, 0)], [(0, 5, 0)], []):
            with self.assertRaises(ValueError):
                audit_temporal_map(self.anchor, self.x, self.anchor, self.mask, probes)

    def test_anchor_must_match_output_shape(self):
        with self.assertRaises(ValueError):
            audit_temporal_map(self.anchor, self.x, lambda x: x[:, -1:], self.mask, [(0, 0, 0)])


if __name__ == '__main__':
    unittest.main()
