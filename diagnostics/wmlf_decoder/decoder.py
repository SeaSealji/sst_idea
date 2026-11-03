"""CPU algebra reference for independent coarse/fine residual predictions.

Only predictions/features at initialization time enter this API; no labels.
Area consistency concerns the supplied field coordinates, not physical heat.
"""
import torch
from torch import nn


class WetMix(nn.Module):
    def __init__(self, hierarchy, mode='sum'):
        super().__init__()
        if len(hierarchy.levels) != 2 or mode not in ('sum', 'project'):
            raise ValueError('one-level wet hierarchy and sum/project mode required')
        self.hierarchy = hierarchy
        self.mode = mode

    def forward(self, fine, coarse):
        """[B,Nwet,L], [B,Ncoarse,L] residuals -> [B,Nwet,L]."""
        h = self.hierarchy
        if (fine.ndim != 3 or coarse.ndim != 3 or fine.shape[0] != coarse.shape[0]
                or fine.shape[-1] != coarse.shape[-1]
                or fine.shape[1] != h.levels[0].area.numel()
                or coarse.shape[1] != h.levels[1].area.numel()
                or fine.dtype not in (torch.float32, torch.float64)
                or coarse.dtype != fine.dtype or coarse.device != fine.device
                or fine.device != h.levels[0].area.device):
            raise ValueError('incompatible residual shapes, dtype or device')
        if not torch.isfinite(fine).all() or not torch.isfinite(coarse).all():
            raise ValueError('wet residuals must be finite')
        result = fine + h.up(coarse, 0)
        if self.mode == 'project':
            result = result - h.up(h.down(fine, 0), 0)
        return result


class LinearScaleDecoder(nn.Module):
    """Same 2*C*L effective parameters for sum/project controls.

    Fine bias is omitted: Q=I-UD would annihilate it under projection.
    Coarse bias also omitted to match the two control parameter sets exactly.
    Features must already have the required legal wet support; this decoder
    cannot undo cross-land mixing by upstream convolutions.
    """
    def __init__(self, hierarchy, width=32, leads=7, mode='sum'):
        super().__init__()
        if type(width) is not int or width < 1 or type(leads) is not int or leads < 1:
            raise ValueError('positive integer width/leads required')
        self.width = width
        self.fine = nn.Linear(width, leads, bias=False)
        self.coarse = nn.Linear(width, leads, bias=False)
        self.mix = WetMix(hierarchy, mode)

    def forward(self, fine_features, coarse_features):
        if (fine_features.ndim != 3 or coarse_features.ndim != 3
                or fine_features.shape[-1] != self.width
                or coarse_features.shape[-1] != self.width):
            raise ValueError('expected [B,N,C] fine/coarse features')
        return self.mix(self.fine(fine_features), self.coarse(coarse_features))

    def concatenated_sum(self, fine_features, coarse_features):
        """Exact linear feature-fusion control, only for sum mode.

        Separate output heads followed by addition are one linear head on
        concatenated [F,U(C)]. This does not prove nonlinear heads equivalent.
        """
        if self.mix.mode != 'sum':
            raise ValueError('concatenated control requires sum mode')
        lifted = self.mix.hierarchy.up(coarse_features, 0)
        features = torch.cat((fine_features, lifted), dim=-1)
        weights = torch.cat((self.fine.weight, self.coarse.weight), dim=-1)
        return torch.nn.functional.linear(features, weights)


def error_partition(hierarchy, error):
    """Per sample/lead weighted SSE identity on the whole declared wet support.

    No region-specific mask: orthogonality can fail when scoring only a subset
    of children while retaining full-aggregate pooling weights.
    """
    coarse = hierarchy.down(error, 0)
    background = hierarchy.up(coarse, 0)
    detail = error-background
    area = hierarchy.levels[0].area.to(error)[None, :, None]
    return {'total': (area*error.square()).sum(1),
            'background': (area*background.square()).sum(1),
            'detail': (area*detail.square()).sum(1),
            'cross': (area*background*detail).sum(1)}
