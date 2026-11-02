"""CPU input-gradient probes for a direct SST forecaster; no data/checkpoint I/O."""
import math

import torch


def audit_temporal_map(model, x, anchor, wet_mask, probes, *, seed=17,
                       epsilon=1e-4, perturbation=0.25):
    """Probe [1,T,C,H,W] -> [1,L,C,H,W], separating the supplied anchor.

    Call with an eval-mode CPU float64 model and synthetic or audited training
    input. `anchor(x)` must return the SAME shape as model(x), including masking
    and any land fill. Probes are (channel,row,column) in the output coordinates.
    No labels, optimizer, file loading, automatic split selection or training.
    Sensitivity is local and input dependent; it is not predictive information,
    physical dynamics identification, a full Jacobian or model equivalence.
    """
    if x.device.type != 'cpu' or x.dtype != torch.float64:
        raise ValueError('Use CPU float64 input and matching model for numerical probes')
    if x.ndim != 5 or x.shape[0] != 1 or min(x.shape) < 1:
        raise ValueError('Expected nonempty [1,T,C,H,W] input')
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite input')
    if isinstance(model, torch.nn.Module) and any(m.training for m in model.modules()):
        raise ValueError('Set every model module to eval mode first')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    if not math.isfinite(perturbation) or perturbation <= 0:
        raise ValueError('perturbation must be finite and positive')
    probes = list(probes)
    if not probes:
        raise ValueError('At least one wet output probe is required')
    z = x.detach().clone().requires_grad_(True)

    def outputs(value):
        prediction, base = model(value), anchor(value)
        if (prediction.ndim != 5 or prediction.shape[0] != 1
                or prediction.shape[1] < 1 or prediction.shape[2] != x.shape[2]
                or prediction.shape != base.shape):
            raise ValueError('Prediction and anchor must share [1,L,C,H,W] shape')
        if prediction.device.type != 'cpu' or prediction.dtype != torch.float64:
            raise ValueError('Prediction must remain CPU float64')
        if base.device != prediction.device or base.dtype != prediction.dtype:
            raise ValueError('Anchor device/dtype mismatch')
        if not torch.isfinite(prediction).all() or not torch.isfinite(base).all():
            raise ValueError('Nonfinite prediction or anchor')
        return prediction, prediction - base

    with torch.enable_grad():
        prediction, residual = outputs(z)
        mask = torch.as_tensor(wet_mask, device='cpu')
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(prediction.shape[2:]):
            raise ValueError('wet_mask must be Boolean [C,H,W] in output coordinates')
        for probe in probes:
            if len(probe) != 3 or any(type(i) is not int for i in probe):
                raise ValueError('Probes must contain three integer coordinates')
            if any(i < 0 or i >= n for i, n in zip(probe, mask.shape)) or not mask[probe]:
                raise ValueError('Probe is outside the output wet domain')

        def gradient(scalar):
            if not scalar.requires_grad:
                return torch.zeros_like(z)
            g = torch.autograd.grad(scalar, z, retain_graph=True, allow_unused=True)[0]
            if g is None:
                return torch.zeros_like(z)
            if not torch.isfinite(g).all():
                raise ValueError('Nonfinite input gradient')
            return g

        generator = torch.Generator(device='cpu').manual_seed(seed)
        direction = torch.randn(z.shape, dtype=z.dtype, generator=generator)
        other = torch.randn(z.shape, dtype=z.dtype, generator=generator)
        with torch.no_grad():
            plus = outputs(z + epsilon * direction)
            minus = outputs(z - epsilon * direction)
            fa = outputs(z + perturbation * direction)
            fb = outputs(z + perturbation * other)
            fab = outputs(z + perturbation * (direction + other))
        rows = []
        for probe in probes:
            for lead in range(prediction.shape[1]):
                index = (0, lead, *probe)
                row = {'probe': list(probe), 'lead': lead + 1}
                for k, (name, field) in enumerate((('total', prediction), ('residual', residual))):
                    g = gradient(field[index])
                    row[name + '_history_signed_sum'] = g.sum(dim=(0, 2, 3, 4)).tolist()
                    row[name + '_history_l1'] = g.abs().sum(dim=(0, 2, 3, 4)).tolist()
                    ad = float((g * direction).sum())
                    fd = float((plus[k][index] - minus[k][index]) / (2 * epsilon))
                    row[name + '_directional_ad'] = ad
                    row[name + '_directional_fd'] = fd
                    row[name + '_fd_abs_error'] = abs(ad - fd)
                    row[name + '_finite_difference_pass'] = math.isclose(ad, fd, abs_tol=1e-7, rel_tol=1e-5)
                    # A nonzero value refutes affine behavior at these points;
                    # a zero value does NOT establish global linear equivalence.
                    row[name + '_affine_defect'] = float(
                        fab[k][index] - fa[k][index] - fb[k][index] + field[index].detach())
                rows.append(row)
    return {
        'schema_version': 1,
        'input_shape': list(x.shape), 'output_shape': list(prediction.shape),
        'input_kind': 'caller_supplied_not_split_verified',
        'direction_seed': seed, 'fd_epsilon': epsilon, 'affine_perturbation': perturbation,
        'all_finite_differences_pass': all(
            r[n + '_finite_difference_pass'] for r in rows for n in ('total', 'residual')),
        'all_probed_residual_gradients_zero': all(
            v == 0 for r in rows for v in r['residual_history_l1']),
        'rows': rows,
        'interpretation': 'Local synthetic/training-input diagnostic only; no forecast skill or global equivalence claim',
    }
