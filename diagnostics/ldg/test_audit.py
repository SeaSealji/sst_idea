import math
import unittest
import numpy as np
from scipy import special
import torch
from torch.nn import functional as F

from audit import ldg_kernel, row_normalized, path_heat, apply_time, fold_weights


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_bessel_values_and_scale_derivatives_against_scipy(self):
        distance=np.abs(np.arange(7)[:,None]-np.arange(7))
        for value in (.0001,.01,math.log(2),1.,5.,10.,20.):
            for dtype,atol in ((torch.float64,2e-14),(torch.float32,2e-7)):
                scale=torch.full((7,),value,dtype=dtype,requires_grad=True)
                k=ldg_kernel(scale)
                n=np.arange(7);s=scale.detach().numpy().astype('float64')
                expected=special.ive(n,s)
                np.testing.assert_allclose(k.detach().numpy(),expected[distance],rtol=2e-6,atol=atol)
                gradient=torch.autograd.grad(k.sum(),scale)[0]
                deriv=.5*(special.ive(n-1,s)+special.ive(n+1,s))-expected
                multiplicity=np.bincount(distance.ravel(),minlength=7)
                np.testing.assert_allclose(gradient.numpy(),deriv*multiplicity,rtol=2e-4,atol=atol*100)

    def test_gradcheck(self):
        s=torch.linspace(.2,1.4,7,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(ldg_kernel,(s,)))
        t=torch.tensor(.9,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(path_heat,(t,)))

    def test_unrestricted_distance_parameters_can_expand(self):
        k=ldg_kernel(torch.tensor([.01,1,1,1,1,1,1],dtype=torch.float64))
        u,s,vh=torch.linalg.svd(k)
        self.assertGreater(float(s[0]),1.45)
        self.assertGreater(float((k@vh[0]).norm()/vh[0].norm()),1.45)
        self.assertGreater(float(k.sum(1).max()),1.52)

    def test_equal_scale_finite_window_loses_constant_at_edges(self):
        k=ldg_kernel(torch.full((7,),math.log(2),dtype=torch.float64))
        rows=k.sum(1)
        self.assertLess(float(rows.max()),1.)
        self.assertLess(float(rows[0]),float(rows[3]))
        self.assertLess(float(torch.linalg.matrix_norm(k,2)),1.)

    def test_row_normalization_is_not_l2_guarantee(self):
        k=row_normalized(ldg_kernel(torch.tensor([.01,1,1,1,1,1,1],dtype=torch.float64)))
        self.assertTrue(torch.allclose(k.sum(1),torch.ones(7,dtype=k.dtype)))
        self.assertGreater(float(torch.linalg.matrix_norm(k,2)),1.0004)
        self.assertGreater(float((k-k.T).abs().max()),.02)

    def test_row_vector_orientation_requires_transposed_normalized_kernel(self):
        k=row_normalized(ldg_kernel(torch.tensor([.01,1,1,1,1,1,1],dtype=torch.float64)))
        x=torch.ones(2,3,7,dtype=torch.float64)
        # This audit defines output Kx, whereas upstream uses x_ct @ K.
        # Symmetric LDG hides the distinction; row normalization does not.
        torch.testing.assert_close(x@k.T,x,atol=2e-15,rtol=0)
        self.assertGreater(float((x@k-x).abs().max()),.03)

    def test_path_heat_constants_contractivity_semigroup(self):
        a=path_heat(torch.tensor(.4,dtype=torch.float64))
        b=path_heat(torch.tensor(.7,dtype=torch.float64))
        c=path_heat(torch.tensor(1.1,dtype=torch.float64))
        self.assertTrue(torch.allclose(c,c.T,atol=1e-13,rtol=0))
        self.assertGreaterEqual(float(c.min()),-1e-14)
        self.assertTrue(torch.allclose(c.sum(1),torch.ones(7,dtype=c.dtype),atol=1e-13,rtol=0))
        self.assertAlmostEqual(float(torch.linalg.matrix_norm(c,2)),1.,places=12)
        self.assertTrue(torch.allclose(a@b,c,atol=1e-13,rtol=0))

    def test_fold_forward_and_all_gradients(self):
        torch.manual_seed(42)
        s=torch.linspace(.3,1.5,7,dtype=torch.float64,requires_grad=True)
        k=ldg_kernel(s)
        x=torch.randn(2,7,11,13,dtype=torch.float64,requires_grad=True)
        ws=torch.randn(4,7,3,3,dtype=torch.float64,requires_grad=True)
        wr=torch.randn_like(ws,requires_grad=True)
        bs=torch.randn(4,dtype=torch.float64,requires_grad=True)
        br=torch.randn_like(bs,requires_grad=True)
        z=apply_time(k,x)
        a=F.conv2d(z,ws,bs,stride=2,padding=1)+F.conv2d(x-z,wr,br,stride=2,padding=1)
        b=F.conv2d(x,fold_weights(ws,wr,k),bs+br,stride=2,padding=1)
        torch.testing.assert_close(a,b,rtol=0,atol=2e-14)
        probe=torch.randn_like(a)
        params=(x,s,ws,wr,bs,br)
        ga=torch.autograd.grad((probe*a).sum(),params,retain_graph=True)
        gb=torch.autograd.grad((probe*b).sum(),params)
        for u,v in zip(ga,gb):torch.testing.assert_close(u,v,rtol=0,atol=2e-13)

    def test_nonlinearity_blocks_simple_folding(self):
        # A nonzero second difference cannot be represented by one affine conv.
        k=path_heat(torch.tensor(1.,dtype=torch.float64))
        x=torch.linspace(-2,2,7,dtype=torch.float64)[None,:,None,None]
        z=apply_time(k,x)
        f=lambda a:F.silu(apply_time(k,a))+F.silu(a-apply_time(k,a))
        self.assertGreater(float((f(x)+f(-x)-2*f(torch.zeros_like(x))).abs().max()),.1)
        # The original pair is invertible by summation; no new observations.
        torch.testing.assert_close(z+(x-z),x)

    def test_history_only_shape_and_input_domain(self):
        x=torch.randn(2,7,9,11)
        k=ldg_kernel(torch.ones(7))
        self.assertEqual(apply_time(k,x).shape,x.shape)
        for s in (torch.ones(6),torch.zeros(7),torch.full((7,),21.),torch.full((7,),float('nan'))):
            with self.assertRaises(ValueError):ldg_kernel(s)
        with self.assertRaises(ValueError):ldg_kernel(torch.ones(7,dtype=torch.float16))


if __name__=='__main__':
    unittest.main(verbosity=2)
