"""Offline validation diagnostics, not a forecasting model or training runner.

Each JSONL record represents one origin/lead and flattened spatial fields.
Future truth is used only here for scoring and clearly labelled oracle bounds.
Use the SAME weights for pooling and scoring: all ones for the existing masked
cell-mean SST metric; cosine-latitude weights for a separate area-weighted audit.
No dependencies, GPU access, automatic gain selection or data-path discovery.
"""
import argparse
import datetime as dt
import json
import math


def finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite on wet cells')
    return value


def diagnose(base, truth, guide, region, weights=None, gains=(0., .25, .5, 1.)):
    """Compare predicted regional anchors with pooled base forecasts.

    Land has region=-1, is ignored before arithmetic, and may contain NaNs.
    guide maps integer region ids to predicted future regional means. No guide
    is trained or calibrated here. Oracle scores are diagnostic upper bounds.
    """
    if not (len(base) == len(truth) == len(region)) or not region:
        raise ValueError('Fields and region map must have equal nonzero length')
    if weights is None:
        weights = [1.] * len(region)
    if len(weights) != len(region):
        raise ValueError('Weights must have the same length as the fields')
    groups = {}
    for b, y, r, w in zip(base, truth, region, weights):
        if type(r) is not int or r < -1:
            raise ValueError('Region ids must be integers >= -1')
        if r == -1:
            continue
        b, y, w = finite(b, 'base'), finite(y, 'truth'), finite(w, 'weight')
        if w <= 0:
            raise ValueError('Wet-cell weights must be positive')
        groups.setdefault(r, []).append((b, y, w))
    if not groups:
        raise ValueError('At least one wet region is required')
    if set(guide) != set(groups):
        raise ValueError('Guide region ids must match wet regions exactly')
    guide = {r: finite(value, 'guide') for r, value in guide.items()}
    total = math.fsum(w for cells in groups.values() for _, _, w in cells)
    squared = within = macro = guide_error = 0.
    correction_terms = []
    regions = []
    for r, cells in sorted(groups.items()):
        weight = math.fsum(w for _, _, w in cells)
        bmean = math.fsum(b*w for b, _, w in cells) / weight
        ymean = math.fsum(y*w for _, y, w in cells) / weight
        ebar = bmean-ymean
        delta = guide[r]-bmean
        squared += math.fsum(w*(b-y)**2 for b, y, w in cells)
        within += math.fsum(w*((b-y)-ebar)**2 for b, y, w in cells)
        macro += weight*ebar**2
        guide_error += weight*(guide[r]-ymean)**2
        correction_terms.extend((b-y, delta, w) for b, y, w in cells)
        regions.append(dict(region=r, cells=len(cells), weight=weight,
                            base_mean=bmean, truth_mean=ymean, guide_mean=guide[r]))
    mse = squared/total
    decomposition_error = abs(mse-(within+macro)/total)
    if not math.isclose(mse, (within+macro)/total, rel_tol=1e-10, abs_tol=1e-12):
        raise ArithmeticError('Weighted error decomposition failed')
    scores = []
    for gain in gains:
        gain = finite(gain, 'gain')
        if not 0 <= gain <= 1:
            raise ValueError('Diagnostic gains must be in [0,1]')
        value = math.fsum(w*(e+gain*d)**2 for e, d, w in correction_terms)/total
        derivative = 2*math.fsum(w*d*(e+gain*d) for e, d, w in correction_terms)/total
        scores.append(dict(gain=gain, mse=value, rmse=math.sqrt(value),
                           delta_mse=value-mse, derivative_mse_wrt_gain=derivative))
    return dict(baseline_mse=mse, baseline_rmse=math.sqrt(mse),
                regional_bias_mse=macro/total, within_region_anomaly_mse=within/total,
                base_regional_mean_mse=macro/total, guide_regional_mean_mse=guide_error/total,
                oracle_only_region_mean_replacement_mse=within/total,
                full_guide_replacement_mse=(within+guide_error)/total,
                decomposition_absolute_error=decomposition_error,
                fixed_gain_scores=scores, wet_cells=sum(len(c) for c in groups.values()),
                total_weight=total, regions=regions)


def diagnose_record(record):
    """Enforce an explicit validation-only input contract before scoring.

    Metadata is an assertion by the exporter, not proof against source leakage.
    The exporter must separately audit guide inputs and the frozen checkpoint.
    """
    if record.get('split') != 'validation':
        raise ValueError('Only explicitly labelled validation records are accepted')
    if record.get('guide_source') != 'history_only_prediction':
        raise ValueError('Only history-predicted guides are accepted for deployable scores')
    origin = dt.date.fromisoformat(record['forecast_origin'])
    last_input = dt.date.fromisoformat(record['guide_input_max_date'])
    target = dt.date.fromisoformat(record['target_date'])
    lead = record['lead']
    if type(lead) is not int or lead not in range(1, 8):
        raise ValueError('Lead must be D1-D7')
    if last_input > origin or target != origin+dt.timedelta(days=lead):
        raise ValueError('Guide input or target time alignment is invalid')
    if not dt.date(2017,1,1) <= origin < target <= dt.date(2021,12,31):
        raise ValueError('Only the fixed 2017-2021 validation period is allowed')
    weighting = record['weighting']
    weights = record.get('weights')
    if weighting == 'uniform_wet_cells':
        if weights is not None:
            raise ValueError('Uniform metric must not supply area weights')
    elif weighting == 'explicit_area_weights':
        if weights is None:
            raise ValueError('Area metric requires explicit positive weights')
    else:
        raise ValueError('Unknown weighting protocol')
    guide = {int(k): v for k, v in record['guide'].items()}
    if len(guide) != len(record['guide']):
        raise ValueError('Duplicate normalized guide ids')
    result = diagnose(record['base'], record['truth'], guide, record['region'], weights)
    return dict(split='validation', forecast_origin=str(origin), target_date=str(target),
                lead=lead, weighting=weighting, oracle_is_diagnostic_only=True,
                **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('validation_jsonl', help='Explicit exported validation frames only')
    args = parser.parse_args()
    with open(args.validation_jsonl) as source:
        for line in source:
            if line.strip():
                print(json.dumps(diagnose_record(json.loads(line)), allow_nan=False))


if __name__ == '__main__':
    main()
