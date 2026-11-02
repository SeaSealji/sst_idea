import unittest

import torch
from torch.nn import functional as F

from transform import HAAR, WetBlockTransform, basis_table, components, pack


class WetBlockTests(unittest.TestCase):
    def test_all_16_patterns_orthogonality_support_and_dc(self):
        table, supports, dc = basis_table()
        for p in range(16):
            wet = torch.tensor([bool(p & (1 << i)) for i in range(4)])
            torch.testing.assert_close(table[p].T @ table[p], torch.diag(wet.double()), rtol=0, atol=1e-14)
            self.assertEqual(int(dc[p].sum()), len(components(p)))
            self.assertEqual(int((supports[p] != 0).sum()), int(wet.sum()))
            for k in range(4):
                # No nonzero coefficient row touches two local wet components.
                touched = [g for g in components(p) if table[p, k, g].abs().sum() > 1e-14]
                self.assertLessEqual(len(touched), 1)
                if touched:
                    self.assertEqual(int(supports[p, k]), sum(1 << i for i in touched[0]))
                if not dc[p, k]:
                    self.assertLess(abs(float(table[p, k].sum())), 1e-14)

    def test_float32_reconstruction_energy_and_dry_nan_all_patterns(self):
        generator = torch.Generator().manual_seed(42)
        for p in range(16):
            mask = torch.tensor([bool(p & (1 << i)) for i in range(4)]).reshape(2, 2)
            op = WetBlockTransform(mask)
            x = torch.randn(2, 3, 2, 2, generator=generator).masked_fill(~mask, float('nan'))
            coeff = op(x)
            restored = op.inverse(coeff)
            clean = torch.where(mask, x, 0.)
            torch.testing.assert_close(restored, clean, rtol=0, atol=1e-6)
            torch.testing.assert_close(coeff.square().sum(), clean.square().sum(), rtol=2e-6, atol=2e-6)
            self.assertTrue((restored[..., ~mask] == 0).all())

    def test_full_wet_is_standard_haar(self):
        x = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
        op = WetBlockTransform(torch.ones(4, 4, dtype=torch.bool))
        ref = torch.einsum('ki,bchwi->bckhw', HAAR, pack(x))
        torch.testing.assert_close(op(x), ref, rtol=0, atol=1e-14)

    def test_diagonal_impulse_has_no_other_component_coefficient(self):
        for p in (6, 9):
            mask = torch.tensor([bool(p & (1 << i)) for i in range(4)]).reshape(2, 2)
            op = WetBlockTransform(mask)
            sites = mask.flatten().nonzero().flatten().tolist()
            for site in sites:
                x = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
                x.flatten()[site] = 1
                coeff = op(x)[0, 0, :, 0, 0]
                support = op.component_support[:, 0, 0]
                self.assertTrue((coeff[support != (1 << site)] == 0).all())
                self.assertEqual(int(op.dc_slots.sum()), 2)

    def test_local_disconnection_is_not_global_disconnection(self):
        # Diagonal sites in an interior block connect by a path outside it.
        mask = torch.ones(6, 6, dtype=torch.bool)
        mask[2, 3] = mask[3, 2] = False
        self.assertEqual(len(components(9)), 2)
        # Global topology cannot be inferred from these per-block labels.
        self.assertEqual(WetBlockTransform(mask).summary()['locally_disconnected_blocks'], 1)

    def test_odd_sizes_dry_padding_and_no_latitude_wrap(self):
        for h, w in ((1, 1), (3, 5), (5, 4)):
            mask = torch.ones(h, w, dtype=torch.bool)
            op = WetBlockTransform(mask)
            x = torch.randn(2, 3, h, w, dtype=torch.float64)
            torch.testing.assert_close(op.inverse(op(x)), x, rtol=0, atol=1e-14)
            self.assertEqual(op.summary()['active_coefficients'], h*w)
        op = WetBlockTransform(torch.ones(3, 4, dtype=torch.bool))
        self.assertTrue((op.patterns[-1] == 3).all())

    def test_longitude_seam_pairing_and_inverse(self):
        mask = torch.ones(2, 6, dtype=torch.bool)
        op = WetBlockTransform(mask, longitude_offset=1)
        x = torch.zeros(1, 1, 2, 6, dtype=torch.float64)
        x[..., -1] = 2
        x[..., 0] = 4
        self.assertEqual(float(op(x)[0, 0, 0, 0, 0]), 6.)
        torch.testing.assert_close(op.inverse(op(x)), x, rtol=0, atol=1e-14)

    def test_gradient_and_identity_backward(self):
        mask = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 0, 0]], dtype=torch.bool)
        op = WetBlockTransform(mask)
        x = torch.randn(1, 2, 3, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z: op.inverse(op(z)), (x,), fast_mode=True))
        op.inverse(op(x)).square().sum().backward()
        torch.testing.assert_close(x.grad, 2*x.detach()*mask, rtol=0, atol=1e-14)

    def test_constant_component_has_no_detail(self):
        for p in range(16):
            mask = torch.tensor([bool(p & (1 << i)) for i in range(4)]).reshape(2, 2)
            op = WetBlockTransform(mask)
            coeff = op(torch.full((1, 1, 2, 2), 7., dtype=torch.float64))
            self.assertLess(float(coeff[0, 0][~op.dc_slots].abs().max()), 1e-13)

    def test_arbitrary_slot_mixing_breaks_isolation(self):
        mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
        op = WetBlockTransform(mask)
        x = torch.tensor([[[[1., 0], [0, 0]]]], dtype=torch.float64)
        coeff = op(x)
        mixed = coeff.clone()
        mixed[:, :, 1] = coeff[:, :, 0]
        self.assertEqual(float(op.inverse(mixed)[0, 0, 1, 1]), 1.)

    def test_binary_normalized_interpolation_does_not_encode_components(self):
        # Two coarse valid features with different component identities. Binary
        # validity lacks those identities; upsampling still averages them.
        coarse = torch.tensor([[[[1., 9.]]]])
        valid = torch.ones_like(coarse)
        up = F.interpolate(coarse*valid, (4, 4), mode='bilinear', align_corners=False)
        norm = F.interpolate(valid, (4, 4), mode='bilinear', align_corners=False)
        wet = torch.zeros(4, 4, dtype=torch.bool)
        wet[:2, :2] = wet[2:, 2:] = True  # disconnected four-neighbor regions
        masked = (up/norm)*wet
        self.assertEqual(float(masked[0, 0, 1, 1]), 3.)  # component value was 1

    def test_invalid_shapes_masks_and_nonfinite_wet(self):
        for mask in (torch.ones(2, 2), torch.ones(0, 2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)):
            with self.assertRaises(ValueError): WetBlockTransform(mask)
        with self.assertRaises(ValueError): WetBlockTransform(torch.ones(2, 3, dtype=torch.bool), longitude_offset=1)
        op = WetBlockTransform(torch.ones(2, 2, dtype=torch.bool))
        for x in (torch.ones(2, 2), torch.ones(1, 1, 3, 2), torch.full((1, 1, 2, 2), float('inf'))):
            with self.assertRaises(ValueError): op(x)
        with self.assertRaises(ValueError): op.inverse(torch.zeros(1, 1, 3, 1, 1))
        with self.assertRaises(ValueError): op.inverse(torch.full((1, 1, 4, 1, 1), float('nan')))


if __name__ == '__main__':
    unittest.main()
