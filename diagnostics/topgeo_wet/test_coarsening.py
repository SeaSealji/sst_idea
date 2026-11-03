import math
import random
import unittest

import torch

from coarsening import SphereGraph, angular_distance, coarsen
from context import WetHierarchy, RecursiveContext


class CoarseningTests(unittest.TestCase):
    def test_sphere_seam_pole_antipode(self):
        self.assertAlmostEqual(angular_distance((0,359),(0,1)), math.radians(2), places=14)
        self.assertAlmostEqual(angular_distance((0,0),(0,180)), math.pi, places=14)
        self.assertLess(angular_distance((90,0),(90,180)), 1e-14)
        self.assertLess(angular_distance((80,0),(80,1)), angular_distance((0,0),(0,1)))

    def test_edge_order_direction_duplicates(self):
        edges = [(0,1),(1,2),(2,3),(3,4),(1,4)]
        graph = SphereGraph([(0,i) for i in range(5)], [1.]*5, edges)
        ref = coarsen(graph)
        shuffled = edges + [(b,a) for a,b in edges] + edges
        random.Random(123).shuffle(shuffled)
        alt = coarsen(SphereGraph(graph.latlon, graph.area, shuffled))
        self.assertEqual(ref, alt)

    def test_connected_stars_and_edge_witnesses(self):
        # Exhaust all nonempty undirected graphs on four vertices.
        possible = [(i,j) for i in range(4) for j in range(i+1,4)]
        for bits in range(64):
            edges = {e for k,e in enumerate(possible) if bits & (1<<k)}
            g = SphereGraph([(0,i*3) for i in range(4)], [1,2,3,4], edges)
            c,p,a = coarsen(g)
            for i,r in enumerate(a['parent_representative']):
                self.assertTrue(i==r or tuple(sorted((i,r))) in edges)
            expected = {tuple(sorted((p[i],p[j]))) for i,j in edges if p[i]!=p[j]}
            self.assertEqual(expected,c.edges)
            self.assertAlmostEqual(sum(c.area),10.)

    def test_isolates_and_disconnected_nearby_coordinates(self):
        g = SphereGraph([(0,0),(0,.01),(0,.02),(0,.03),(0,.04)], [1]*5, [(0,2),(1,3)])
        c,p,a = coarsen(g)
        self.assertNotEqual(p[0],p[1])
        self.assertNotIn(p[4],p[:4])
        self.assertEqual(a['isolated_nodes'],1)
        self.assertEqual(len(c.area),3)

    def test_analytic_geometry_ranking_is_not_front_detection(self):
        # Two equal-degree components with different physical edge spacing.
        g=SphereGraph([(0,0),(0,1),(80,0),(80,1)], [1]*4, [(0,1),(2,3)])
        _,_,a=coarsen(g)
        self.assertLess(a['score'][2],a['score'][0])
        # No SST/features are accepted by coarsen: this priority is latitude.
        _,_,d=coarsen(g,ranking='degree')
        self.assertEqual(d['score'],[1]*4)

    def test_constant_integral_and_du(self):
        mask=torch.ones(5,7,dtype=torch.bool)
        area=torch.linspace(.1,2,35,dtype=torch.float64).reshape(5,7)
        h=WetHierarchy(mask,area,[-70,-30,0,30,70],list(range(7)),depth=3)
        x=torch.randn(2,35,3,dtype=torch.float64)
        for k in range(3):
            y=h.down(x,k)
            torch.testing.assert_close((x*h.levels[k].area[None,:,None]).sum(1),
                (y*h.levels[k+1].area[None,:,None]).sum(1),atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(h.down(torch.ones_like(x),k),torch.ones_like(y),atol=1e-14,rtol=1e-14)
            torch.testing.assert_close(h.down(h.up(y,k),k),y,atol=1e-14,rtol=1e-14)
            x=y

    def test_down_up_loses_subaggregate_detail(self):
        h=WetHierarchy(torch.ones(1,3,dtype=torch.bool),torch.ones(1,3),[0],[0,1,2],depth=1)
        x=torch.tensor([[[1.],[7.],[13.]]])
        self.assertFalse(torch.equal(h.up(h.down(x,0),0),x))

    def test_periodic_longitude_no_latitude_wrap(self):
        mask=torch.tensor([[1,0,0,1],[0,0,0,0],[1,0,0,0]],dtype=torch.bool)
        h=WetHierarchy(mask,torch.ones(3,4),[-60,0,60],[.5,90.5,180.5,359.5],depth=1,periodic_longitude=True)
        self.assertEqual(h.graphs[0].edges,{(0,1)})
        self.assertEqual(h.levels[0].parent.tolist()[0],h.levels[0].parent.tolist()[1])
        self.assertNotEqual(h.levels[0].parent.tolist()[0],h.levels[0].parent.tolist()[2])

    def test_context_gradcheck_and_all_parameters(self):
        torch.manual_seed(17)
        h=WetHierarchy(torch.ones(2,3,dtype=torch.bool),torch.ones(2,3),[-20,20],[0,1,2],depth=2)
        net=RecursiveContext(h,2).double()
        x=torch.randn(1,2,2,3,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(net,(x,),atol=1e-5,rtol=1e-4))
        net(x).square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in net.parameters()))

    def test_context_component_batch_gradient_isolation(self):
        mask=torch.tensor([[1,1,0,1,1]],dtype=torch.bool)
        h=WetHierarchy(mask,torch.ones(1,5),[0],[0,1,2,3,4],depth=2)
        net=RecursiveContext(h,3).double()
        x=torch.randn(2,3,1,5,dtype=torch.float64).masked_fill(~mask,float('nan')).requires_grad_()
        y=net(x)
        self.assertTrue(torch.isfinite(y).all())
        g=torch.autograd.grad(y[0,:,:,:2].sum(),x)[0]
        self.assertEqual(float(g[1].abs().sum()),0.)
        self.assertEqual(float(g[0,:,:,2:].abs().sum()),0.)

    def test_rank_controls_same_learning_parameters_and_replay(self):
        mask=torch.ones(3,5,dtype=torch.bool)
        nets=[RecursiveContext(WetHierarchy(mask,torch.ones_like(mask),[-50,0,50],list(range(5)),depth=2,ranking=r),4)
              for r in ['geometry','degree','index']]
        for net in nets[1:]:
            # Copy learned parameters only: different geometry buffers are intentional.
            for p,q in zip(net.parameters(),nets[0].parameters()):p.data.copy_(q.data)
        self.assertEqual({sum(p.numel() for p in n.parameters()) for n in nets},{45})
        x=torch.randn(2,4,3,5)
        for net in nets:
            y=net(x)
            with torch.no_grad():z=net(x)
            torch.testing.assert_close(y,z,atol=0,rtol=0)
            y.square().mean().backward()
            self.assertTrue(all(p.grad is not None and p.grad.abs().sum()>0 for p in net.parameters()))

    def test_reject_invalid_geometry_and_features(self):
        for g in [SphereGraph([],[],[]),SphereGraph([(91,0)],[1],[]),
                  SphereGraph([(0,0)],[0],[]),SphereGraph([(0,float('nan'))],[1],[]),
                  SphereGraph([(0,0)],[1],[(0,1)]),SphereGraph([(0,0)],[1],[(0,0)])]:
            with self.assertRaises(ValueError):coarsen(g)
        h=WetHierarchy(torch.ones(1,2,dtype=torch.bool),torch.ones(1,2),[0],[0,1],depth=1)
        with self.assertRaises(ValueError):h.pack(torch.full((1,1,1,2),float('nan')))
        with self.assertRaises(ValueError):h.pack(torch.ones(1,1,2,2))


if __name__=='__main__':
    unittest.main()
