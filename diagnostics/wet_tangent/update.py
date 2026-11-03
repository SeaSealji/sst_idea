"""CPU float32/64 reference for weighted tangent feature updates.

Wet nodes only, not an SST forecaster or TANGO reproduction. Projection is
component-wide, hence nonlocal within each connected component.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class WetMetric(nn.Module):
    def __init__(self, area, edges):
        super().__init__()
        a = torch.as_tensor(area, dtype=torch.float64, device='cpu')
        e = torch.as_tensor(edges, dtype=torch.long, device='cpu').reshape(-1, 2)
        if a.ndim != 1 or not a.numel() or not torch.isfinite(a).all() or not (a > 0).all():
            raise ValueError('positive finite areas [N] required')
        if e.numel() and (int(e.min()) < 0 or int(e.max()) >= a.numel()):
            raise ValueError('edge index outside wet nodes')
        # Treat input as undirected support, deduplicate then store both directions.
        pairs = {tuple(sorted((i, j))) for i, j in e.tolist() if i != j}
        parent = list(range(a.numel()))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in sorted(pairs):
            ri, rj = find(i), find(j)
            parent[max(ri, rj)] = min(ri, rj)
        labels, roots = [], {}
        for i in range(a.numel()):
            root = find(i)
            roots.setdefault(root, len(roots))
            labels.append(roots[root])
        self.groups = len(roots)
        labels = torch.tensor(labels, dtype=torch.long)
        # Normalize separately per component; changing component size must not
        # change the energy gradient solely through a raw total-area factor.
        normalized = torch.empty_like(a)
        for k in range(self.groups):
            select = labels == k
            scaled = a[select] / a[select].max()
            normalized[select] = scaled / scaled.sum()
        if not (normalized > 0).all():
            raise ValueError('area ratios outside representable range')
        self.register_buffer('weight', normalized)
        self.register_buffer('labels', labels)
        directed = [(i, j) for a, b in sorted(pairs) for i, j in ((a, b), (b, a))]
        self.register_buffer('edges', torch.tensor(directed, dtype=torch.long).reshape(-1, 2))

    def validate(self, x):
        if x.ndim != 3 or x.shape[1] != self.weight.numel() or x.dtype not in (torch.float32, torch.float64):
            raise ValueError('expected float32/64 [B,Nwet,C]')
        if x.device != self.weight.device or not torch.isfinite(x).all():
            raise ValueError('device mismatch or nonfinite wet input')
        if not (self.weight.to(x) > 0).all():
            raise ValueError('metric weights underflowed in feature dtype')

    def reduce(self, node_values):
        out = node_values.new_zeros(node_values.shape[0], self.groups)
        return out.index_add(1, self.labels, node_values)

    def inner(self, x, y):
        return self.reduce((x * y * self.weight.to(x)[None, :, None]).sum(-1))

    def neighbors(self, x):
        out = torch.zeros_like(x)
        degree = x.new_zeros(x.shape[1])
        if self.edges.numel():
            src, dst = self.edges.unbind(1)
            out.index_add_(1, dst, x.index_select(1, src))
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        return torch.where(degree[None, :, None] > 0,
                           out / degree.clamp_min(1)[None, :, None], x)


def project(metric, proposal, gradient):
    """W-orthogonal projection, exact-zero branch; no additive epsilon bias.

    Normalize gradient by detached component maximum before squaring to avoid
    norm underflow/overflow. This is algebraically the same projection away
    from g=0. Derivatives at exactly g=0 are not asserted to be continuous.
    """
    metric.validate(proposal)
    metric.validate(gradient)
    if proposal.shape != gradient.shape:
        raise ValueError('proposal/gradient shape mismatch')
    outputs = []
    for k in range(metric.groups):
        select = metric.labels == k
        m, g = proposal[:, select], gradient[:, select]
        w = metric.weight.to(g)[select][None, :, None]
        scale = g.detach().abs().amax((1, 2), keepdim=True)
        q = g / torch.where(scale > 0, scale, torch.ones_like(scale))
        denominator = (q.square() * w).sum((1, 2), keepdim=True)
        coefficient = (m * q * w).sum((1, 2), keepdim=True) / torch.where(
            denominator > 0, denominator, torch.ones_like(denominator))
        outputs.append((select, m - coefficient * q))
    result = torch.zeros_like(proposal)
    for select, value in outputs:
        result[:, select] = value
    return result


class TangentUpdate(nn.Module):
    """One feature step with a learned positive diagonal quadratic potential.

    Simpler than TANGO's EnergyGNN: this bounded prototype audits metric and
    higher-order autograd, not graph benchmark results or physical energy.
    """
    def __init__(self, metric, width, step=.05, projection=True):
        super().__init__()
        if not math.isfinite(step) or step < 0:
            raise ValueError('finite nonnegative step required')
        if projection not in (True, False, 'fixed_normal'):
            raise ValueError('projection must be True, False or fixed_normal')
        self.metric = metric
        self.step = step
        self.projection = projection
        self.log_diagonal = nn.Parameter(torch.zeros(width))
        self.self_map = nn.Linear(width, width)
        self.neighbor_map = nn.Linear(width, width, bias=False)
        self.alpha_logit = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.ones(()))
        self.register_buffer('fixed_normal', torch.randn(1, 1, width, generator=torch.Generator().manual_seed(0)))

    def energy(self, x):
        diagonal = F.softplus(self.log_diagonal) + .1
        density = .5 * (diagonal * x.square()).sum(-1)
        return self.metric.reduce(density * self.metric.weight.to(x)[None])

    def directions(self, x, create_graph):
        energy = self.energy(x)
        euclidean, = torch.autograd.grad(energy.sum(), x, create_graph=create_graph)
        gradient = euclidean / self.metric.weight.to(x)[None, :, None]
        proposal = torch.tanh(self.self_map(x) + self.neighbor_map(self.metric.neighbors(x)))
        if self.projection == 'fixed_normal':
            # Same component-wide reduction and parameter set; unrelated normal
            # controls for extra global receptive field, not Lyapunov geometry.
            tangent = project(self.metric, proposal, self.fixed_normal.expand_as(proposal))
        else:
            tangent = project(self.metric, proposal, gradient) if self.projection else proposal
        direction = -torch.sigmoid(self.alpha_logit) * gradient + self.beta * tangent
        return energy, euclidean, gradient, proposal, tangent, direction

    def forward(self, x):
        if torch.is_inference_mode_enabled():
            raise RuntimeError('reference requires autograd for feature energy; use no_grad, not inference_mode')
        self.metric.validate(x)
        outer_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            h = x if x.requires_grad else x.detach().requires_grad_(True)
            *_, direction = self.directions(h, create_graph=outer_grad)
            out = h + self.step * direction
        return out if outer_grad else out.detach()
