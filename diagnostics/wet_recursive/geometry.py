"""CPU reference: connected agglomerates in spatial bins, not basin labels.

Every coarse edge has an actual child-edge witness. No physical metric or
finite-volume claim: areas are user supplied, edges are binary four-neighbors.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class Graph:
    coords: list
    area: list
    edges: set  # undirected (min, max), no self edges


def coarsen(graph):
    """Connected components of each 2x2 bin's induced child graph."""
    bins = [(r // 2, c // 2) for r, c in graph.coords]
    parent = list(range(len(bins)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in sorted(graph.edges):
        if bins[a] == bins[b]:
            ra, rb = find(a), find(b)
            parent[max(ra, rb)] = min(ra, rb)
    roots = {}
    mapping = []
    for i in range(len(bins)):
        root = find(i)
        roots.setdefault(root, len(roots))
        mapping.append(roots[root])
    coords, areas = [None] * len(roots), [0.] * len(roots)
    for i, p in enumerate(mapping):
        coords[p] = bins[i]
        areas[p] += graph.area[i]
    edges = set()
    for a, b in graph.edges:
        a, b = mapping[a], mapping[b]
        if a != b:
            edges.add(tuple(sorted((a, b))))
    return Graph(coords, areas, edges), mapping


class Level(nn.Module):
    def __init__(self, graph, mapping=None):
        super().__init__()
        self.register_buffer('area', torch.tensor(graph.area, dtype=torch.float64))
        directed = [(a, b) for a, b in sorted(graph.edges) for a, b in ((a, b), (b, a))]
        self.register_buffer('edges', torch.tensor(directed, dtype=torch.long).reshape(-1, 2))
        self.register_buffer('parent', None if mapping is None else torch.tensor(mapping, dtype=torch.long))
        self.coords = graph.coords

    def neighbors(self, x):
        # Isolated nodes use their own value. No normalization across nodes.
        out = torch.zeros_like(x)
        degree = x.new_zeros(x.shape[1])
        if self.edges.numel():
            src, dst = self.edges.unbind(1)
            out.index_add_(1, dst, x.index_select(1, src))
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        return torch.where(degree[None, :, None] > 0,
                           out / degree.clamp_min(1)[None, :, None], x)


class WetHierarchy(nn.Module):
    """Ragged [B,N,C] graphs retain multiple components in one coarse cell.

    Build on CPU. Longitude wrap only applies to the physical core supplied by
    the caller, never to a halo/tile. No north-south wrap or pole-half-turn.
    """
    def __init__(self, mask, area, depth=3, periodic_longitude=False):
        super().__init__()
        mask = torch.as_tensor(mask, device='cpu')
        area = torch.as_tensor(area, dtype=torch.float64, device='cpu')
        if mask.dtype != torch.bool or mask.ndim != 2 or area.shape != mask.shape:
            raise ValueError('Boolean [H,W] mask and same-shape area required')
        if type(depth) is not int or depth < 0 or not mask.any():
            raise ValueError('nonempty mask and nonnegative integer depth required')
        if not torch.isfinite(area[mask]).all() or not (area[mask] > 0).all():
            raise ValueError('wet areas must be finite and positive')
        self.register_buffer('mask', mask.clone())
        self.register_buffer('wet_indices', mask.flatten().nonzero().flatten())
        self.shape = tuple(mask.shape)
        coords = [tuple(rc) for rc in mask.nonzero().tolist()]
        lookup = {tuple(rc): i for i, rc in enumerate(coords)}
        edges = set()
        h, w = mask.shape
        for (r, c), i in lookup.items():
            candidates = [(r + 1, c), (r, c + 1)]
            if periodic_longitude and c == w - 1:
                candidates.append((r, 0))
            for rc in candidates:
                j = lookup.get(rc)
                if j is not None and i != j:
                    edges.add(tuple(sorted((i, j))))
        graphs = [Graph(coords, area[mask].tolist(), edges)]
        maps = []
        for _ in range(depth):
            g, mapping = coarsen(graphs[-1])
            maps.append(mapping)
            graphs.append(g)
        self.graphs = graphs  # CPU audit metadata, not used by forward
        self.levels = nn.ModuleList([Level(g, maps[k] if k < depth else None)
                                     for k, g in enumerate(graphs)])

    def pack(self, x):
        if x.ndim != 4 or tuple(x.shape[-2:]) != self.shape:
            raise ValueError('expected [B,C,H,W] matching geometry')
        if x.dtype not in (torch.float32, torch.float64):
            raise ValueError('reference supports float32/64 only')
        if x.device != self.mask.device:
            raise ValueError('geometry/input device mismatch')
        wet = x.flatten(2).index_select(2, self.wet_indices).transpose(1, 2)
        if not torch.isfinite(wet).all():
            raise ValueError('wet features must be finite')
        return wet

    def unpack(self, x):
        if x.ndim != 3 or x.shape[1] != self.wet_indices.numel():
            raise ValueError('expected [B,Nwet,C]')
        out = x.new_zeros(x.shape[0], x.shape[2], math.prod(self.shape))
        out.index_copy_(2, self.wet_indices, x.transpose(1, 2))
        return out.reshape(x.shape[0], x.shape[2], *self.shape)

    def down(self, x, k):
        level, coarse = self.levels[k:k + 2]
        if x.ndim != 3 or x.shape[1] != level.area.numel():
            raise ValueError('wrong graph node count')
        out = x.new_zeros(x.shape[0], coarse.area.numel(), x.shape[2])
        out.index_add_(1, level.parent, x * level.area.to(x)[None, :, None])
        return out / coarse.area.to(x)[None, :, None]

    def up(self, x, k):
        if x.ndim != 3 or x.shape[1] != self.levels[k + 1].area.numel():
            raise ValueError('wrong coarse node count')
        return x.index_select(1, self.levels[k].parent)

    def summary(self):
        return [{'nodes': len(g.area), 'edges': len(g.edges),
                 'occupied_bins': len(set(g.coords)),
                 'extra_component_slots': len(g.coords) - len(set(g.coords)),
                 'total_area': sum(g.area)} for g in self.graphs]


class SharedGraphBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_map = nn.Linear(width, width)
        self.neighbor_map = nn.Linear(width, width, bias=False)

    def forward(self, x, level):
        return x + torch.tanh(self.self_map(x) + self.neighbor_map(level.neighbors(x)))


class RecursiveContext(nn.Module):
    """Synthetic graph context MVP, NOT a drop-in T1 replacement or SSM.

    Fixed width at every level; one actual operator and gate instance when
    shared. Learned messages/gates do not preserve constants or area means.
    """
    def __init__(self, hierarchy, width, shared=True):
        super().__init__()
        self.hierarchy = hierarchy
        self.shared = shared
        count = 1 if shared else len(hierarchy.levels)
        self.blocks = nn.ModuleList([SharedGraphBlock(width) for _ in range(count)])
        gate_count = min(1, len(hierarchy.levels) - 1) if shared else len(hierarchy.levels) - 1
        self.gates = nn.ModuleList([nn.Linear(2 * width, 1) for _ in range(gate_count)])

    def operator_at(self, k):
        return self.blocks[0 if self.shared else k]

    def _run(self, x, k):
        local = self.operator_at(k)(x, self.hierarchy.levels[k])
        if k == len(self.hierarchy.levels) - 1:
            return local
        coarse = self._run(self.hierarchy.down(local, k), k + 1)
        lifted = self.hierarchy.up(coarse, k)
        gate = torch.sigmoid(self.gates[0 if self.shared else k](torch.cat((local, lifted), -1)))
        return local + gate * (lifted - local)

    def forward(self, x):
        return self.hierarchy.unpack(self._run(self.hierarchy.pack(x), 0))
