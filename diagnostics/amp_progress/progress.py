"""Strict JSON fields for a gradient norm observed AFTER GradScaler.update()."""
import math


def gradient_fields(norm, optimizer_updated):
    """Allow a nonfinite diagnostic only when the AMP step was skipped.

    This does not validate the loss, model or optimizer, and must not sanitize
    metrics. Call every training batch, including batches without progress IO.
    optimizer_updated must come from the scaler outcome, not norm finiteness.
    """
    norm = float(norm)
    if optimizer_updated not in (0, 1):
        raise ValueError('optimizer_updated must be 0 or 1')
    finite = math.isfinite(norm)
    if not finite and optimizer_updated:
        raise FloatingPointError('nonfinite gradient norm without an AMP-skipped update')
    return dict(gradient_norm=norm if finite else None,
                gradient_norm_status='finite' if finite else ('nan' if math.isnan(norm) else 'inf'),
                optimizer_step_skipped=not bool(optimizer_updated))
