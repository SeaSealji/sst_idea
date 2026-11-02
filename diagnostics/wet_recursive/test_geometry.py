import unittest

import torch

from geometry import WetHierarchy, RecursiveContext


torch.set_num_threads(2)


class GeometryTests(unittest.TestCase):
    def make(self, mask, depth=3, periodic=False):
        mask = torch.tensor(mask, dtype=torch.bool)
        area = torch.arange(1, mask.numel() + 1, dtype=torch.float64).reshape(mask.shape)
        return WetHierarchy(mask, area, depth, periodic)

    def test_all_nonempty_2x2_masks_area_and_constant(self):
        for bits in range(1, 16):
            h = self.make([[bits & 1, bits & 2], [bits & 4, bits & 8]])
            x = torch.ones(2, len(h.graphs[0].area), 3, dtype=torch.float64)
            for k in range(3):
                y = h.down(x, k)
                torch.testing.assert_close(y, torch.ones_like(y))
                torch.testing.assert_close(h.up(y, k), x)
                self.assertAlmostEqual(sum(h.graphs[k].area), sum(h.graphs[k + 1].area))
                x = y

    def test_area_integral_and_down_up_identity(self):
        h = self.make([[1, 1, 0, 1, 1], [1, 0, 1, 1, 0], [1, 1, 1, 0, 1]])
        torch.manual_seed(12)
        x = torch.randn(2, len(h.graphs[0].area), 4, dtype=torch.float64)
        for k in range(3):
            y = h.down(x, k)
            a, b = h.levels[k].area, h.levels[k + 1].area
            torch.testing.assert_close((x * a[None, :, None]).sum(1), (y * b[None, :, None]).sum(1))
            torch.testing.assert_close(h.down(h.up(y, k), k), y)
            x = y

    def test_induced_connectivity_not_global_basin_label(self):
        # Diagonal nodes in a bin are joined via a right/bottom detour.
        h = self.make([[0, 1, 1], [1, 0, 1], [1, 1, 1]])
        a = h.graphs[0].coords.index((0, 1))
        b = h.graphs[0].coords.index((1, 0))
        self.assertNotEqual(h.levels[0].parent[a], h.levels[0].parent[b])
        self.assertGreater(h.summary()[1]['extra_component_slots'], 0)

    def test_every_coarse_edge_has_child_witness(self):
        h = self.make([[1, 1, 0, 1, 1], [1, 0, 1, 1, 0], [1, 1, 1, 0, 1]], periodic=True)
        for k in range(3):
            p = h.levels[k].parent.tolist()
            expected = {tuple(sorted((p[a], p[b]))) for a, b in h.graphs[k].edges if p[a] != p[b]}
            self.assertEqual(expected, h.graphs[k + 1].edges)
            # Each aggregate is connected by child edges, not just same bin.
            for target in set(p):
                members = {i for i, group in enumerate(p) if group == target}
                reached = {min(members)}
                while True:
                    grown = reached | {b for a, b in h.levels[k].edges.tolist() if a in reached and b in members}
                    if grown == reached:
                        break
                    reached = grown
                self.assertEqual(reached, members)

    def test_periodic_seam_odd_width_and_no_latitude_wrap(self):
        mask = [[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]]
        closed, periodic = self.make(mask), self.make(mask, periodic=True)
        self.assertEqual(len(closed.graphs[0].edges), 0)
        self.assertEqual(periodic.graphs[0].edges, {(0, 1), (2, 3)})
        self.assertEqual(len(periodic.graphs[-1].area), 2)

    def test_disconnected_basins_forward_and_backward(self):
        h = self.make([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        model = RecursiveContext(h, 2).double()
        torch.manual_seed(7)
        x = torch.randn(1, 2, 3, 3, dtype=torch.float64, requires_grad=True)
        y = model(x)
        g, = torch.autograd.grad(y[:, :, :, 0].sum(), x)
        torch.testing.assert_close(g[:, :, :, 2], torch.zeros_like(g[:, :, :, 2]), rtol=0, atol=0)
        changed = x.detach().clone()
        changed[:, :, :, 2] += 100
        torch.testing.assert_close(model(changed)[:, :, :, 0], y[:, :, :, 0], rtol=0, atol=0)

    def test_naive_single_slot_pool_mixes_diagonal(self):
        h = self.make([[1, 0], [0, 1]])
        x = torch.tensor([[[1.], [9.]]], dtype=torch.float64)
        torch.testing.assert_close(h.up(h.down(x, 0), 0), x)
        a = h.levels[0].area
        naive = (x * a[None, :, None]).sum(1) / a.sum()
        self.assertNotEqual(float(naive), 1.)
        self.assertNotEqual(float(naive), 9.)

    def test_pool_is_lossy_not_reconstruction(self):
        h = self.make([[1, 1], [1, 1]])
        x = torch.arange(4, dtype=torch.float64).reshape(1, 4, 1)
        self.assertGreater(float((h.up(h.down(x, 0), 0) - x).abs().max()), .1)

    def test_pool_analytic_gradient(self):
        h = self.make([[1, 1], [1, 1]])
        x = torch.randn(1, 4, 2, dtype=torch.float64, requires_grad=True)
        h.down(x, 0).sum().backward()
        expected = (h.levels[0].area / h.levels[0].area.sum())[None, :, None].expand_as(x)
        torch.testing.assert_close(x.grad, expected)

    def test_shared_identity_calls_gradients_and_capacity(self):
        h = self.make([[1] * 5 for _ in range(3)])
        shared = RecursiveContext(h, 3).double()
        untied = RecursiveContext(h, 3, shared=False).double()
        self.assertEqual(len({id(shared.operator_at(k)) for k in range(4)}), 1)
        self.assertEqual(len({id(untied.operator_at(k)) for k in range(4)}), 4)
        calls = []
        handle = shared.blocks[0].register_forward_hook(lambda m, i, o: calls.append(o.shape[1]))
        x = torch.randn(1, 3, 3, 5, dtype=torch.float64, requires_grad=True)
        shared(x).square().sum().backward()
        handle.remove()
        self.assertEqual(calls, [len(g.area) for g in h.graphs])
        for p in shared.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0)
        self.assertGreater(sum(p.numel() for p in untied.parameters()), sum(p.numel() for p in shared.parameters()))

    def test_recursive_double_gradcheck(self):
        h = self.make([[1, 0, 1], [1, 1, 1]], depth=2)
        model = RecursiveContext(h, 2).double()
        x = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(model, (x,), fast_mode=True))

    def test_dry_nan_and_input_contract(self):
        h = self.make([[1, 0, 1], [1, 1, 0]], depth=2)
        model = RecursiveContext(h, 2)
        x = torch.randn(1, 2, 2, 3).masked_fill(~h.mask, float('nan')).requires_grad_(True)
        y = model(x)
        self.assertTrue(torch.isfinite(y).all())
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(float(x.grad.masked_select(~h.mask[None, None]).abs().sum()), 0.)
        with self.assertRaises(ValueError): h.pack(torch.zeros(1, 2, 3, 2))
        with self.assertRaises(ValueError): h.pack(torch.full_like(x, float('nan')))
        with self.assertRaises(ValueError): WetHierarchy([[1]], [[1.]])
        with self.assertRaises(ValueError): WetHierarchy(torch.ones(1, 1, dtype=torch.bool), [[0.]])
        with self.assertRaises(ValueError): WetHierarchy(torch.zeros(1, 1, dtype=torch.bool), [[1.]])


if __name__ == '__main__':
    unittest.main()
