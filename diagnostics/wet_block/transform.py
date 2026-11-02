"""Single-level orthogonal wet-component block transform; no model/data I/O.

This is a local graph-Haar-style diagnostic, not a multilevel ocean wavelet.
Only the unmodified transform has the component isolation guarantee below.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


HAAR = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1],
                     [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.float64) / 2
EDGES = ((0, 1), (0, 2), (1, 3), (2, 3))


def components(pattern):
    """Local four-neighbor components of TL/TR/BL/BR bits 0/1/2/3."""
    if type(pattern) is not int or not 0 <= pattern < 16:
        raise ValueError('pattern must be an integer in [0,15]')
    remaining = {i for i in range(4) if pattern & (1 << i)}
    groups = []
    while remaining:
        stack, group = [min(remaining)], []
        remaining.remove(stack[0])
        while stack:
            i = stack.pop()
            group.append(i)
            for a, b in EDGES:
                j = b if a == i else a if b == i else None
                if j in remaining:
                    remaining.remove(j)
                    stack.append(j)
        groups.append(sorted(group))
    return groups


def basis_table():
    """Return Q[pattern,slot,pixel], component support bits, and DC flags.

    Q.T Q is diag(wet), not the full 4x4 identity on partially wet blocks.
    Each component has its OWN constant/DC coefficient, followed by details.
    All-wet blocks use standard orthonormal Haar. No area weighting is implied.
    """
    table = torch.zeros(16, 4, 4, dtype=torch.float64)
    support = torch.zeros(16, 4, dtype=torch.int64)
    dc = torch.zeros(16, 4, dtype=torch.bool)
    for pattern in range(16):
        slot = 0
        for group in components(pattern):
            dc_vector = torch.zeros(4, dtype=torch.float64)
            dc_vector[group] = 1 / math.sqrt(len(group))
            rows = [dc_vector]
            # Deterministic modified Gram-Schmidt on coordinate vectors.
            candidates = list(HAAR[1:]) if len(group) == 4 else list(torch.eye(4, dtype=torch.float64)[group])
            for candidate in candidates:
                v = candidate.clone()
                for row in rows:
                    v = v - torch.dot(v, row) * row
                norm = torch.linalg.vector_norm(v)
                if norm > 1e-12:
                    rows.append(v / norm)
                if len(rows) == len(group):
                    break
            assert len(rows) == len(group)
            bits = sum(1 << i for i in group)
            for j, row in enumerate(rows):
                table[pattern, slot] = row
                support[pattern, slot] = bits
                dc[pattern, slot] = j == 0
                slot += 1
    return table, support, dc


def pack(x):
    """[...,H,W] even spatial sizes -> [...,H/2,W/2,4]."""
    h, w = x.shape[-2:]
    return x.reshape(*x.shape[:-2], h//2, 2, w//2, 2).transpose(-3, -2).reshape(*x.shape[:-2], h//2, w//2, 4)


def unpack(x):
    """Inverse of pack, with the four-pixel dimension last."""
    h, w = x.shape[-3:-1]
    return x.reshape(*x.shape[:-3], h, w, 2, 2).transpose(-3, -2).reshape(*x.shape[:-3], 2*h, 2*w)


class WetBlockTransform(nn.Module):
    """Fixed-mask [B,C,H,W] <-> [B,C,4,ceil(H/2),ceil(W/2)].

    Bottom/right odd-size padding is DRY, never latitude-periodic. With even W,
    longitude_offset=1 pairs (last,first) columns at the seam by a cyclic roll.
    Offset changes the entire pairing alignment; it is not a multilevel graph.
    Dry NaNs are ignored by where before arithmetic. Wet nonfinite values fail.
    """
    def __init__(self, mask, *, longitude_offset=0):
        super().__init__()
        mask = torch.as_tensor(mask, device='cpu')
        if mask.dtype != torch.bool or mask.ndim != 2 or min(mask.shape) < 1:
            raise ValueError('mask must be nonempty Boolean [H,W]')
        if longitude_offset not in (0, 1) or type(longitude_offset) is not int:
            raise ValueError('longitude_offset must be 0 or 1')
        self.h, self.w = mask.shape
        if longitude_offset and self.w % 2:
            raise ValueError('Seam pairing requires even longitude size')
        self.longitude_offset = longitude_offset
        self.register_buffer('mask', mask.clone())
        aligned = torch.roll(mask, longitude_offset, -1)
        padded = F.pad(aligned, (0, self.w % 2, 0, self.h % 2), value=False)
        bits = (pack(padded).long() * torch.tensor([1, 2, 4, 8])).sum(-1)
        table, supports, dc = basis_table()
        self.register_buffer('patterns', bits)
        self.register_buffer('basis', table[bits])
        self.register_buffer('component_support', supports[bits].permute(2, 0, 1))
        self.register_buffer('dc_slots', dc[bits].permute(2, 0, 1))
        self.register_buffer('active_slots', (supports[bits] != 0).permute(2, 0, 1))

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[-2:]) != (self.h, self.w):
            raise ValueError('Expected [B,C,H,W] matching static mask')
        if x.dtype not in (torch.float32, torch.float64) or x.device != self.mask.device:
            raise ValueError('Use float32/float64 input on the same device as the transform')
        clean = torch.where(self.mask, x, torch.zeros_like(x))
        if not torch.isfinite(clean).all():
            raise ValueError('Nonfinite wet input')
        aligned = torch.roll(clean, self.longitude_offset, -1)
        blocks = pack(F.pad(aligned, (0, self.w % 2, 0, self.h % 2)))
        coefficients = torch.einsum('hwki,bchwi->bchwk', self.basis.to(x.dtype), blocks)
        return coefficients.permute(0, 1, 4, 2, 3).contiguous()

    def inverse(self, coefficients):
        spatial = ((self.h+1)//2, (self.w+1)//2)
        if coefficients.ndim != 5 or tuple(coefficients.shape[2:]) != (4, *spatial):
            raise ValueError('Expected [B,C,4,ceil(H/2),ceil(W/2)]')
        if coefficients.dtype not in (torch.float32, torch.float64) or coefficients.device != self.mask.device:
            raise ValueError('Coefficient dtype/device mismatch')
        if not torch.isfinite(coefficients).all():
            raise ValueError('Nonfinite coefficients')
        blocks = torch.einsum('hwki,bckhw->bchwi', self.basis.to(coefficients.dtype), coefficients)
        aligned = unpack(blocks)[..., :self.h, :self.w]
        return torch.roll(aligned, -self.longitude_offset, -1)

    def summary(self):
        counts = torch.bincount(self.patterns.flatten(), minlength=16).cpu().tolist()
        disconnected = sum(counts[p] for p in range(16) if len(components(p)) > 1)
        return {'shape': [self.h, self.w], 'longitude_offset': self.longitude_offset,
                'blocks': self.patterns.numel(), 'wet_cells': int(self.mask.sum()),
                'pattern_counts': counts, 'mixed_wet_dry_blocks': sum(counts[1:15]),
                'locally_disconnected_blocks': disconnected,
                'disconnected_wet_cells': sum(counts[p]*p.bit_count() for p in (6, 9)),
                'active_coefficients': int(self.active_slots.sum()),
                'dc_coefficients': int(self.dc_slots.sum()),
                'detail_coefficients': int((self.active_slots & ~self.dc_slots).sum()),
                'parameter_count': sum(p.numel() for p in self.parameters())}
