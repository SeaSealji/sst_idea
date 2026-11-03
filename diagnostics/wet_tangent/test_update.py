import copy
import unittest

import torch
from torch.func import functional_call

from update import WetMetric, TangentUpdate, project

torch.set_num_threads(2)


class TangentTests(unittest.TestCase):
    def metric(self):
        return WetMetric([.01, 1., .3, .7], [[0, 1], [2, 3]])

    def test_weighted_orthogonality_each_sample_component(self):
        torch.manual_seed(11)
        metric = self.metric()
        g, m = [torch.randn(3, 4, 5, dtype=torch.float64) for _ in range(2)]
        t = project(metric, m, g)
        torch.testing.assert_close(metric.inner(t, g), torch.zeros(3, 2, dtype=torch.float64), atol=1e-14, rtol=0)

    def test_small_large_gradient_and_zero(self):
        for dtype, scales in [(torch.float32, [0., 1e-30, 1., 1e30]),
                              (torch.float64, [0., 1e-150, 1., 1e150])]:
            metric = self.metric().to(dtype=dtype)
            torch.manual_seed(18)
            m = torch.randn(2, 4, 3, dtype=dtype)
            base = torch.randn_like(m)
            reference = project(metric, m, base)
            for scale in scales:
                t = project(metric, m, base * scale)
                self.assertTrue(torch.isfinite(t).all())
                torch.testing.assert_close(t, m if scale == 0 else reference, atol=1e-6, rtol=1e-5)

    def test_regularized_denominator_is_not_exact_projection(self):
        g = torch.tensor([[[1e-10, 0.]]], dtype=torch.float64)
        m = torch.ones_like(g)
        biased = m - ((m*g).sum() / (g.square().sum() + 1e-8))*g
        self.assertGreater(float((biased*g).sum().abs()), 9e-11)
        metric = WetMetric([1.], [])
        self.assertEqual(float(metric.inner(project(metric, m, g), g)), 0.)

    def test_v1_formula_counterexample(self):
        g = torch.tensor([2., 0.], dtype=torch.float64)
        m = torch.tensor([1., 1.], dtype=torch.float64)
        v1 = m - torch.dot(m, g/g.norm()) * g
        self.assertEqual(float(torch.dot(v1, g)), -2.)
        final = m - torch.dot(m, g/g.norm()) * (g/g.norm())
        self.assertEqual(float(torch.dot(final, g)), 0.)

    def test_metric_gradient_and_chain_rule(self):
        metric = self.metric()
        model = TangentUpdate(metric, 3).double()
        x = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
        _, euclidean, g, _, t, d = model.directions(x, True)
        diagonal = torch.nn.functional.softplus(model.log_diagonal) + .1
        torch.testing.assert_close(g, diagonal*x)
        derivative = metric.reduce((euclidean*d).sum(-1))
        expected = -torch.sigmoid(model.alpha_logit)*metric.inner(g, g)
        torch.testing.assert_close(derivative, expected, atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(metric.inner(t, g), torch.zeros_like(expected), atol=1e-14, rtol=0)

    def test_wrong_metric_can_increase_continuous_energy(self):
        metric = WetMetric([.01, 1.], [[0, 1]])
        euclidean = torch.tensor([[[1.], [1.]]], dtype=torch.float64)
        m = torch.tensor([[[1.], [0.]]], dtype=torch.float64)
        wrong_t = project(metric, m, euclidean)
        self.assertGreater(float((euclidean*wrong_t).sum()), .9)

    def test_euler_can_increase_despite_negative_derivative(self):
        # V=.5||x||^2, g=(1,0), T=(0,1), alpha=.1 beta=1.
        x = torch.tensor([1., 0.], dtype=torch.float64)
        d = torch.tensor([-.1, 1.], dtype=torch.float64)
        self.assertLess(float(x.dot(d)), 0.)
        delta = lambda step: .5*((x + step*d).square().sum() - x.square().sum())
        self.assertLess(float(delta(.05)), 0.)
        self.assertGreater(float(delta(1.)), 0.)
        self.assertGreater(float(.5*((x+torch.tensor([0., .1])).square().sum()-x.square().sum())), 0.)

    def test_second_order_input_and_parameter_gradcheck(self):
        torch.manual_seed(19)
        model = TangentUpdate(self.metric(), 2).double()
        x = torch.randn(1, 4, 2, dtype=torch.float64, requires_grad=True)
        names, params = zip(*model.named_parameters())
        fn = lambda values, *p: functional_call(model, dict(zip(names, p)), (values,))
        self.assertTrue(torch.autograd.gradcheck(fn, (x, *params), fast_mode=True))
        model(x).square().sum().backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0.)

    def test_no_grad_eval_matches_and_inference_mode_rejected(self):
        model = TangentUpdate(self.metric(), 2).double().eval()
        x = torch.randn(1, 4, 2, dtype=torch.float64)
        y = model(x)
        with torch.no_grad():
            z = model(x)
        torch.testing.assert_close(y, z)
        self.assertFalse(z.requires_grad)
        with torch.inference_mode(), self.assertRaises(RuntimeError): model(x)

    def test_disconnected_components_and_batch_isolation(self):
        model = TangentUpdate(self.metric(), 2).double()
        x = torch.randn(2, 4, 2, dtype=torch.float64, requires_grad=True)
        y = model(x)
        grad, = torch.autograd.grad(y[0, :2].sum(), x)
        self.assertEqual(float(grad[0, 2:].abs().sum()), 0.)
        self.assertEqual(float(grad[1].abs().sum()), 0.)

    def test_projection_is_nonlocal_within_component(self):
        metric = WetMetric([1., 1., 1.], [[0, 1], [1, 2]])
        g = torch.ones(1, 3, 1, dtype=torch.float64)
        m = torch.tensor([[[1.], [0.], [0.]]], dtype=torch.float64, requires_grad=True)
        t = project(metric, m, g)
        grad, = torch.autograd.grad(t[0, 2, 0], m)
        self.assertAlmostEqual(float(grad[0, 0, 0]), -1/3)

    def test_projected_and_unprojected_equal_parameter_sets(self):
        model = TangentUpdate(self.metric(), 2).double()
        plain = copy.deepcopy(model)
        plain.projection = False
        self.assertEqual(list(dict(model.named_parameters())), list(dict(plain.named_parameters())))
        x = torch.randn(1, 4, 2, dtype=torch.float64, requires_grad=True)
        a = model.directions(x, True)
        b = plain.directions(x, True)
        for i in (0, 1, 2, 3): torch.testing.assert_close(a[i], b[i])
        self.assertGreater(float((a[5]-b[5]).abs().max()), 0.)

    def test_component_area_rescaling_invariance(self):
        a = WetMetric([.01, 1., .3, .7], [[0, 1], [2, 3]])
        b = WetMetric([1., 100., .003, .007], [[0, 1], [2, 3]])
        torch.testing.assert_close(a.weight, b.weight)

    def test_fixed_normal_global_control_same_parameters(self):
        torch.manual_seed(31)
        model = TangentUpdate(self.metric(), 3).double()
        control = copy.deepcopy(model)
        control.projection = 'fixed_normal'
        x = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
        _, _, gradient, _, tangent, _ = control.directions(x, True)
        torch.testing.assert_close(control.metric.inner(tangent, control.fixed_normal.expand_as(tangent)),
                                   torch.zeros(2, 2, dtype=torch.float64), atol=1e-14, rtol=0)
        self.assertGreater(float(control.metric.inner(tangent, gradient).abs().max()), 1e-4)
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in control.parameters()))
        control(x).square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                            for p in control.parameters()))

    def test_shapes_weights_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError): WetMetric([0., 1.], [])
        with self.assertRaises(ValueError): WetMetric([1.], [[0, 1]])
        with self.assertRaises(ValueError): self.metric().validate(torch.zeros(1, 3, 2))
        with self.assertRaises(ValueError): self.metric().validate(torch.full((1, 4, 2), float('nan')))
        with self.assertRaises(ValueError): TangentUpdate(self.metric(), 2, step=-1.)


if __name__ == '__main__':
    unittest.main()
