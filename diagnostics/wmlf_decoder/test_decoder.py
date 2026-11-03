import unittest
import torch

from geometry import WetHierarchy
from decoder import WetMix, LinearScaleDecoder, error_partition


def hierarchy():
    return WetHierarchy(torch.ones(2,4,dtype=torch.bool),
        torch.tensor([[1.,2,3,4],[2,1,4,3]],dtype=torch.float64),depth=1)


class DecoderTests(unittest.TestCase):
    def test_sum_not_consistent_project_is(self):
        h=hierarchy()
        f=torch.ones(2,8,7,dtype=torch.float64)
        c=torch.full((2,2,7),3.,dtype=torch.float64)
        self.assertEqual(float((h.down(WetMix(h,'sum')(f,c),0)-c).abs().max()),1.)
        torch.testing.assert_close(h.down(WetMix(h,'project')(f,c),0),c,atol=1e-14,rtol=0)

    def test_projection_preserves_only_fine_anomalies(self):
        h=hierarchy();f=torch.randn(2,8,7,dtype=torch.float64);c=torch.randn(2,2,7,dtype=torch.float64)
        out=WetMix(h,'project')(f,c)
        torch.testing.assert_close(out-h.up(h.down(out,0),0),f-h.up(h.down(f,0),0),atol=1e-14,rtol=1e-14)

    def test_linear_output_sum_equals_feature_concat_forward_and_backward(self):
        torch.manual_seed(42);h=hierarchy();m=LinearScaleDecoder(h,3,7).double()
        f=torch.randn(2,8,3,dtype=torch.float64,requires_grad=True)
        c=torch.randn(2,2,3,dtype=torch.float64,requires_grad=True)
        a,b=m(f,c),m.concatenated_sum(f,c)
        torch.testing.assert_close(a,b,atol=1e-14,rtol=1e-14)
        args=(f,c,*m.parameters());probe=torch.randn_like(a)
        ga=torch.autograd.grad((a*probe).sum(),args,retain_graph=True)
        gb=torch.autograd.grad((b*probe).sum(),args)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=1e-13,rtol=1e-13)

    def test_projection_parameter_and_input_gradcheck(self):
        torch.manual_seed(12);h=hierarchy();m=LinearScaleDecoder(h,2,3,'project').double()
        f=torch.randn(1,8,2,dtype=torch.float64,requires_grad=True)
        c=torch.randn(1,2,2,dtype=torch.float64,requires_grad=True)
        wf,wc=m.fine.weight,m.coarse.weight
        fn=lambda x,z,a,b:m.mix(torch.nn.functional.linear(x,a),torch.nn.functional.linear(z,b))
        self.assertTrue(torch.autograd.gradcheck(fn,(f,c,wf,wc),atol=1e-5,rtol=1e-4))
        m(f,c).square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters()))

    def test_equal_effective_capacity_and_no_grad(self):
        h=hierarchy();a=LinearScaleDecoder(h,32);b=LinearScaleDecoder(h,32,mode='project')
        self.assertEqual(sum(p.numel() for p in a.parameters()),448)
        self.assertEqual(sum(p.numel() for p in b.parameters()),448)
        x=torch.randn(1,8,32);c=h.down(x,0)
        for m in [a,b]:
            out=m(x,c)
            with torch.no_grad():replay=m(x,c)
            torch.testing.assert_close(out,replay,atol=0,rtol=0)

    def test_weighted_whole_support_error_decomposition(self):
        h=hierarchy();e=torch.randn(3,8,7,dtype=torch.float64);p=error_partition(h,e)
        torch.testing.assert_close(p['total'],p['background']+p['detail'],atol=1e-12,rtol=1e-12)
        self.assertLess(float(p['cross'].abs().max()),1e-13)

    def test_subregion_mask_breaks_orthogonality(self):
        h=hierarchy();e=torch.arange(8,dtype=torch.float64).reshape(1,8,1)
        bg=h.up(h.down(e,0),0);de=e-bg
        # Scoring a single coastal child with full-group pooling is not orthogonal.
        self.assertGreater(float((bg[:,:1]*de[:,:1]).abs().sum()),0.)

    def test_standardized_consistency_does_not_imply_celsius_consistency(self):
        h=WetHierarchy(torch.ones(1,2,dtype=torch.bool),torch.ones(1,2),depth=1)
        f=torch.tensor([[[1.],[-1.]]],dtype=torch.float64);c=torch.zeros(1,1,1,dtype=torch.float64)
        y=WetMix(h,'project')(f,c)
        self.assertEqual(float(h.down(y,0)),0.)
        sigma=torch.tensor([[[1.],[3.]]],dtype=torch.float64)
        self.assertEqual(float(h.down(sigma*y,0)),-1.)

    def test_raw_anchor_is_added_exactly_once(self):
        h=hierarchy();anchor=torch.randn(2,8,1);f=torch.randn(2,8,7);c=torch.randn(2,2,7)
        delta=WetMix(h,'project')(f,c);out=anchor+delta
        torch.testing.assert_close(out-anchor,delta,atol=2e-7,rtol=1e-6)
        torch.testing.assert_close(h.down(out,0),h.down(anchor,0)+c,atol=2e-7,rtol=1e-6)

    def test_local_support_and_dry_nan(self):
        mask=torch.tensor([[1,0],[0,1]],dtype=torch.bool)
        h=WetHierarchy(mask,torch.ones(2,2),depth=1)
        x=torch.randn(2,3,2,2).masked_fill(~mask,float('nan')).requires_grad_()
        m=LinearScaleDecoder(h,3,7,'sum');wet=h.pack(x)
        c=h.down(wet,0);out=m(wet,c)
        g=torch.autograd.grad(out[0,0].sum(),x)[0]
        self.assertTrue(torch.isfinite(out).all() and torch.isfinite(g).all())
        self.assertEqual(float(g[1].abs().sum()),0.)
        self.assertEqual(float(g[0,:,1,1].abs().sum()),0.)
        self.assertEqual(float(g.masked_select(~mask[None,None]).abs().sum()),0.)

    def test_zero_mean_fine_bias_is_unidentifiable(self):
        h=hierarchy();f=torch.randn(1,8,7,dtype=torch.float64);c=torch.randn(1,2,7,dtype=torch.float64)
        bias=torch.randn(1,1,7,dtype=torch.float64,requires_grad=True)
        out=WetMix(h,'project')(f+bias,c)
        grad=torch.autograd.grad(out.square().sum(),bias)[0]
        self.assertLess(float(grad.abs().max()),1e-12)

    def test_invalid_inputs(self):
        h=hierarchy();m=WetMix(h)
        with self.assertRaises(ValueError):m(torch.ones(1,8,7),torch.ones(1,3,7))
        with self.assertRaises(ValueError):m(torch.full((1,8,7),float('nan')),torch.ones(1,2,7))
        with self.assertRaises(ValueError):WetMix(h,'attention')


if __name__=='__main__':unittest.main()
