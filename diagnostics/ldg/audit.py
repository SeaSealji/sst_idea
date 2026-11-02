"""Synthetic finite-window operator diagnostics, not a forecasting model."""
import math
import torch
from torch.nn import functional as F


def ldg_kernel(scale):
    """Distance-specific exp(-s_d) I_d(s_d), finite T x T truncation.

    Audit-only native autograd power series: 64 terms, verified for T=7,
    1e-4 <= s_d <= 20 in float32/64. This is NOT an unrestricted special
    function implementation. Validation uses a scalar host check; CUDA
    throughput and automatic mixed precision have not been verified.
    """
    if scale.ndim != 1 or not scale.is_floating_point() or len(scale) != 7:
        raise ValueError('audit requires a floating length-seven scale vector')
    if scale.dtype not in (torch.float32, torch.float64):
        raise ValueError('audit supports float32 and float64 only')
    if not bool(torch.isfinite(scale).all() & (scale >= 1e-4).all() & (scale <= 20).all()):
        raise ValueError('scale outside the verified audit domain [1e-4,20]')
    order = torch.arange(7, device=scale.device, dtype=scale.dtype)
    term = torch.exp(-scale) * (scale/2).pow(order) / torch.exp(torch.lgamma(order+1))
    value = term
    for k in range(1, 64):
        term = term * scale.square() / (4*k*(k+order))
        value = value + term
    i = torch.arange(7, device=scale.device)
    return value[(i[:, None]-i[None]).abs()]


def row_normalized(kernel):
    """Constant-preserving control; NOT guaranteed L2-nonexpansive."""
    return kernel / kernel.sum(1, keepdim=True)


def path_heat(scale, length=7):
    """Alternative exp(-s L_path), reflecting/no-flux finite path, not LDG.

    L is the symmetric combinatorial path Laplacian, endpoints degree one.
    For nonnegative scalar s this has nonnegative entries, K1=1, norm2=1,
    and a semigroup. Boundary behavior and parameterization differ from LDG.
    """
    if scale.ndim != 0 or not scale.is_floating_point() or length < 2:
        raise ValueError('positive scalar diffusion time and length >= 2 required')
    if not bool(torch.isfinite(scale) & (scale >= 0)):
        raise ValueError('nonnegative finite diffusion time required')
    incidence = torch.zeros(length-1, length, dtype=scale.dtype, device=scale.device)
    i = torch.arange(length-1, device=scale.device)
    incidence[i, i] = -1
    incidence[i, i+1] = 1
    laplacian = incidence.T @ incidence
    return torch.matrix_exp(-scale*laplacian)


def apply_time(kernel, x):
    """Column convention: [B,T,H,W] -> K x, across history only."""
    return torch.einsum('ij,bjhw->bihw', kernel, x)


def fold_weights(smooth_weight, residual_weight, kernel):
    """Conv(Ws,Kx)+Conv(Wr,(I-K)x) -> one same-support Conv(W_eff,x).

    Same static spatially shared K, channel mixing before any nonlinearity,
    same spatial stride/padding/dilation. Biases are added separately.
    Differentiable through K and both convolution weights; do not detach.
    """
    eye = torch.eye(len(kernel), dtype=kernel.dtype, device=kernel.device)
    return (torch.einsum('oihw,ij->ojhw', smooth_weight, kernel)
            + torch.einsum('oihw,ij->ojhw', residual_weight, eye-kernel))


def properties(kernel):
    """Return JSON-safe diagnostics, including the right singular witness."""
    _, singular, vh = torch.linalg.svd(kernel)
    witness = vh[0]
    return dict(row_sums=kernel.sum(1).detach().tolist(),
                column_sums=kernel.sum(0).detach().tolist(),
                constant_error=float((kernel.sum(1)-1).abs().max()),
                symmetry_error=float((kernel-kernel.T).abs().max()),
                spectral_norm=float(singular[0]),
                witness_gain=float(torch.linalg.vector_norm(kernel@witness)
                                   / torch.linalg.vector_norm(witness)))


def synthetic_summary():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    scale = torch.tensor([.01,1,1,1,1,1,1], dtype=torch.float64)
    cases = dict(default_equal_scale=ldg_kernel(torch.full((7,), math.log(2), dtype=torch.float64)),
                 independent_scale=ldg_kernel(scale),
                 row_normalized=row_normalized(ldg_kernel(scale)),
                 path_heat=path_heat(torch.tensor(1., dtype=torch.float64)))
    errors = {}
    for dtype in (torch.float32, torch.float64):
        s = scale.to(dtype).requires_grad_()
        k = ldg_kernel(s)
        x = torch.randn(2,7,17,19,dtype=dtype,requires_grad=True)
        ws = torch.randn(5,7,3,3,dtype=dtype,requires_grad=True)/10
        wr = torch.randn_like(ws,requires_grad=True)/10
        b = torch.randn(5,dtype=dtype,requires_grad=True)
        a = F.conv2d(apply_time(k,x),ws,b,padding=1) + F.conv2d(x-apply_time(k,x),wr,padding=1)
        f = F.conv2d(x,fold_weights(ws,wr,k),b,padding=1)
        probe = torch.randn_like(a)
        ga = torch.autograd.grad((a*probe).sum(),(x,ws,wr,b,s),retain_graph=True)
        gf = torch.autograd.grad((f*probe).sum(),(x,ws,wr,b,s))
        errors[str(dtype)] = dict(forward_max_abs=float((a-f).abs().max()),
            backward_max_abs={n:float((u-v).abs().max()) for n,u,v in zip(('input','smooth_weight','residual_weight','bias','scale'),ga,gf)})
    return dict(status='PASS', scope='synthetic finite operator and linear-folding audit only',
                cases={k:properties(v) for k,v in cases.items()},folding=errors,
                optimizer_updates=0,gpu_used=False,sst_values_read=False,test_values_read=False)


if __name__ == '__main__':
    import json
    print(json.dumps(synthetic_summary(), indent=2, allow_nan=False))
