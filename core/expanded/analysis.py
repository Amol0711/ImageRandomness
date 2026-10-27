"""Reconstruct moments and rejection counts from saved integer simulations."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from calibration import iid_baci_moments, iid_pair_constants
from fixed_moments import independent_exact_moments

METHODS = {
    'independent_pair_gaussian': 'Independent pairs',
    'within_block_only_gaussian': 'Within-block only',
    'full_covariance_gaussian': 'Full covariance',
    'independent_null_rank': 'Null rank',
    'fixed_interval': 'Fixed interval',
}
SHORT = {
    'independent_pair_gaussian': 'pair',
    'within_block_only_gaussian': 'within',
    'full_covariance_gaussian': 'full',
    'independent_null_rank': 'rank',
    'fixed_interval': 'fixed_interval',
}
PATTERNS = {
    'fixed_zero': ('constant_0', 'Constant 0'),
    'fixed_mid': ('constant_128', 'Constant 128'),
    'fixed_checker': ('checker_0_128', 'Checkerboard'),
    'fixed_split': ('vertical_split_0_128', 'Vertical split'),
    'fixed_ramp': ('column_ramp_0_255', 'Column ramp'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fraction_record(value):
    value = Fraction(value)
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'decimal': float(value)}


def read_fraction(record):
    value = Fraction(record['numerator'], record['denominator'])
    require(math.isclose(float(value), record['decimal'], rel_tol=1e-15,
                         abs_tol=0), 'Incorrect decimal in exact moment record')
    return value


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]) if fields is None else fields,
                            lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(output.getvalue())


def wilson(count, size):
    p = count/size
    z = NormalDist().inv_cdf(.975)
    d = 1+z*z/size
    centre = (p+z*z/(2*size))/d
    radius = z*math.sqrt(p*(1-p)/size+z*z/(4*size*size))/d
    return max(0., centre-radius), min(1., centre+radius)


def exact_scores(totals, raw_mean):
    """Use int64 only after bounding the multiplication and subtraction.

    Arbitrary-precision Python integers are the fallback. No float conversion
    determines ordering or ties in either route.
    """
    p, q = raw_mean.numerator, raw_mean.denominator
    lo, hi = int(totals.min()), int(totals.max())
    limit = np.iinfo(np.int64).max
    safe = max(abs(lo*q), abs(hi*q), abs(p), abs(lo*q-p), abs(hi*q-p)) <= limit
    if safe:
        scores = np.abs(totals.astype(np.int64)*q-p)
        return scores, 'int64 after exact bound check'
    return np.asarray([abs(int(value)*q-p) for value in totals], dtype=object), 'Python arbitrary-precision integer'


def rank_numerators(bank_scores, evaluation_scores):
    ordered = np.sort(bank_scores)
    result = len(ordered)-np.searchsorted(ordered, evaluation_scores, side='left')+1
    # An independent standard-library binary-search recount includes ties.
    independent = sorted(int(v) for v in bank_scores)
    expected = np.asarray([len(independent)-bisect_left(independent, int(v))+1
                           for v in evaluation_scores], dtype=np.int64)
    require(np.array_equal(result, expected), 'Independent rank recount failed')
    return result


def diagnostics(values, mean, variance):
    n = len(values)
    empirical_mean = float(np.mean(values))
    empirical_variance = float(np.var(values, ddof=1))
    fourth = float(np.mean((values-empirical_mean)**4))
    # Descriptive plug-in estimate, not an exact variance confidence interval.
    variance_se = math.sqrt(max(0., (fourth-((n-3)/(n-1))*empirical_variance**2)/n))
    return {
        'size': n, 'empirical_mean': empirical_mean,
        'empirical_variance_ddof1': empirical_variance,
        'theoretical_mean': float(mean), 'theoretical_variance': float(variance),
        'empirical_variance_over_theoretical': empirical_variance/float(variance),
        'mean_standard_error_from_exact_variance': math.sqrt(float(variance)/n),
        'mean_error_in_standard_errors': (empirical_mean-float(mean))/math.sqrt(float(variance)/n),
        'variance_standard_error_plugin': variance_se,
        'variance_ratio_standard_error_plugin': variance_se/float(variance),
        'variance_diagnostic_scope': 'Descriptive empirical fourth-moment plug-in; no exact confidence-interval or testing claim',
    }


def fixed_pattern(mode, rows, columns):
    i, j = np.indices((rows, columns))
    if mode == 'constant0': return np.zeros((rows, columns), dtype=np.uint8)
    if mode == 'constant128': return np.full((rows, columns), 128, dtype=np.uint8)
    if mode == 'checker': return (128*((i+j)%2)).astype(np.uint8)
    if mode == 'split': return (128*(j>=128)).astype(np.uint8)
    if mode == 'ramp': return j.astype(np.uint8)
    raise ValueError('unknown fixed pattern')


def get_moments(design):
    R = design['repeats']
    keys = ('mean', 'variance', 'independent_pair_variance', 'within_block_only_variance')
    if design['mode'] == 'random':
        moments = iid_baci_moments(design['rows'], design['columns'], design['q'], repeats=R)
        require(iid_pair_constants(design['q'], method='sum') ==
                iid_pair_constants(design['q'], method='polynomial'), 'IID constants disagree')
        return {key: getattr(moments, key) for key in keys}
    image = fixed_pattern(design['mode'], design['rows'], design['columns'])
    moments = independent_exact_moments(image, q=design['q'])
    return {key: moments[key] if key == 'mean' else moments[key]/R for key in keys}


def read_design(design, raw_dir):
    name = design['id']
    path = raw_dir / f'{name}_raw.npz'
    L, n, R = design['bank_size'], design['evaluation_size'], design['repeats']
    scale = 6*(design['rows']-1)*(design['columns']-1)*(design['q']-1)
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {'raw_numerators', 'trial_totals', 'bank_trials',
                                     'test_trials', 'repeats', 'per_image_scale'}, f'{name}: fields mismatch')
        raw, totals = archive['raw_numerators'], archive['trial_totals']
        require(raw.dtype == np.dtype('<u4') and raw.shape == (L+n,R), f'{name}: raw schema mismatch')
        require(totals.dtype == np.dtype('<u8') and totals.shape == (L+n,), f'{name}: total schema mismatch')
        for key, value in [('bank_trials', L), ('test_trials', n), ('repeats', R), ('per_image_scale', scale)]:
            require(archive[key].shape == () and int(archive[key]) == value, f'{name}: {key} mismatch')
        require(int(raw.max()) <= scale, f'{name}: raw bound mismatch')
        require(scale*R*(L+n) <= np.iinfo(np.uint64).max, f'{name}: total overflow')
        require(np.array_equal(raw.sum(axis=1, dtype=np.uint64), totals), f'{name}: integer recount failed')
    if design['mode'] != 'random':
        image = fixed_pattern(design['mode'], design['rows'], design['columns'])
        fixed_path = ROOT / 'data' / 'fixed_images' / f'{name}.u8'
        require(fixed_path.read_bytes() == image.tobytes(), f'{name}: synthetic pattern mismatch')
    return raw, totals, scale, {'raw_npz_sha256': digest(path)}


def analyze(protocol, raw_dir):
    rows, moments_rows, designs = [], [], {}
    for design in protocol['designs']:
        name = design['id']
        raw, totals, scale, inputs = read_design(design, raw_dir)
        L, n, R = design['bank_size'], design['evaluation_size'], design['repeats']
        denominator = scale*R
        m = get_moments(design)
        raw_mean = m['mean']*denominator
        scores, score_type = exact_scores(totals, raw_mean)
        bank_scores, test_scores = scores[:L], scores[L:]
        pnum = rank_numerators(bank_scores, test_scores)
        evaluation = totals[L:].astype(np.float64)/denominator
        eval_diagnostics = diagnostics(evaluation, m['mean'], m['variance'])
        design_output = {
            'design': design, 'inputs': inputs, 'normalization': denominator,
            'moments': {key: fraction_record(value) for key, value in m.items()},
            'raw_null_mean': fraction_record(raw_mean), 'rank_score_integer_type': score_type,
            'rank_score_definition': 'abs(trial_total * raw_null_mean.denominator - raw_null_mean.numerator)',
            'rank_pvalue_denominator': L+1,
            'bank': diagnostics(totals[:L].astype(np.float64)/denominator, m['mean'], m['variance']),
            'evaluation': eval_diagnostics,
            'evaluation_single_image': diagnostics(raw[L:].reshape(-1).astype(np.float64)/scale,
                                                   m['mean'], m['variance']*R),
            'variance_full_over_independent_pair': float(m['variance']/m['independent_pair_variance']),
            'variance_full_over_within': float(m['variance']/m['within_block_only_variance']),
            'rejections': [],
        }
        for alpha_float in protocol['levels']:
            alpha = Fraction(str(alpha_float))
            z = NormalDist().inv_cdf(1-float(alpha)/2)
            flags_by_method = {}
            for method, variance_key in [
                ('independent_pair_gaussian', 'independent_pair_variance'),
                ('within_block_only_gaussian', 'within_block_only_variance'),
                ('full_covariance_gaussian', 'variance')]:
                var = m[variance_key]
                threshold = z*math.sqrt(float(var))*denominator*raw_mean.denominator
                flags = test_scores > threshold
                # Recompute each score with Python integers to check overflow/order.
                independent_count = sum(abs(int(v)*raw_mean.denominator-raw_mean.numerator) > threshold
                                        for v in totals[L:])
                require(int(flags.sum()) == independent_count, f'{name}: independent Gaussian recount failed')
                flags_by_method[method] = flags
                gaussian_prediction = 2*(1-NormalDist().cdf(z*math.sqrt(float(var/m['variance']))))
                detail = {
                    'variance': fraction_record(var), 'normal_quantile': z,
                    'interval': [float(m['mean'])-z*math.sqrt(float(var)),
                                 float(m['mean'])+z*math.sqrt(float(var))],
                    'raw_integer_score_threshold': threshold,
                    'minimum_score_distance_from_threshold': float(np.min(np.abs(test_scores.astype(float)-threshold))),
                    'gaussian_reference_rejection_probability': gaussian_prediction,
                    'gaussian_reference_scope': 'Prediction if the centered statistic were Gaussian; not an exact finite-sample size',
                }
                append_result(rows, design_output, name, method, alpha_float, flags, detail)
            flags = pnum*alpha.denominator <= alpha.numerator*(L+1)
            flags_by_method['independent_null_rank'] = flags
            append_result(rows, design_output, name, 'independent_null_rank', alpha_float, flags, {
                'pvalue_rule': '(1 + count(bank_score >= observed_score))/(bank_size+1) <= alpha',
                'ties': 'Included in upper tail; conservative',
                'uncertainty': 'Wilson interval conditional on the realized reused bank; rank guarantee is marginal over independent bank and observation',
                'smallest_observed_pvalue_numerator': int(pnum.min()),
                'largest_observed_pvalue_numerator': int(pnum.max()),
            })
            if design['mode'] == 'random':
                bounds = protocol['fixed_intervals_percent'][name][str(alpha_float)]
                lower, upper = map(lambda x: Fraction(str(x)), bounds)
                # Exact percentage comparisons include both rejection endpoints.
                flags = np.asarray([int(v)*100*lower.denominator <= lower.numerator*denominator or
                                    int(v)*100*upper.denominator >= upper.numerator*denominator
                                    for v in totals[L:]], dtype=bool)
                flags_by_method['fixed_interval'] = flags
                append_result(rows, design_output, name, 'fixed_interval', alpha_float, flags,
                              {'fixed_interval_percent': bounds, 'endpoints': 'Reject <= lower or >= upper, fixed numerical endpoints'})
            design_output.setdefault('paired_disagreements', {})[str(alpha_float)] = {
                method: {
                    'method_reject_full_accept': int(np.sum(flags & ~flags_by_method['full_covariance_gaussian'])),
                    'method_accept_full_reject': int(np.sum(~flags & flags_by_method['full_covariance_gaussian'])),
                } for method, flags in flags_by_method.items() if method != 'full_covariance_gaussian'
            }
        label = f'Random R={R}' if design['mode'] == 'random' else PATTERNS[name][1]
        moments_rows.append({
            'design': name, 'label': label, 'repeats': R, 'mean': float(m['mean']),
            'mean_percent': 100*float(m['mean']), 'full_variance': float(m['variance']),
            'independent_pair_variance': float(m['independent_pair_variance']),
            'within_variance': float(m['within_block_only_variance']),
            'full_over_independent_pair': design_output['variance_full_over_independent_pair'],
            'full_over_within': design_output['variance_full_over_within'],
            'evaluation_mean': eval_diagnostics['empirical_mean'],
            'evaluation_variance': eval_diagnostics['empirical_variance_ddof1'],
            'evaluation_variance_over_full': eval_diagnostics['empirical_variance_over_theoretical'],
            'evaluation_mean_standardized_error': eval_diagnostics['mean_error_in_standard_errors'],
            'evaluation_variance_ratio_se_plugin': eval_diagnostics['variance_ratio_standard_error_plugin'],
        })
        designs[name] = design_output
    require(len(rows) == 60 and len(designs) == 7, 'Campaign cardinality mismatch')
    return rows, moments_rows, designs


def append_result(rows, design_output, design, method, alpha, flags, detail):
    count, n = int(np.sum(flags)), len(flags)
    low, high = wilson(count, n)
    row = {'design': design, 'method': method, 'method_label': METHODS[method],
           'alpha': alpha, 'evaluation_size': n, 'rejection_count': count,
           'rejection_rate': count/n, 'wilson95_lower': low, 'wilson95_upper': high,
           'rejection_percent': 100*count/n, 'wilson95_lower_percent': 100*low,
           'wilson95_upper_percent': 100*high,
           'interval_conditioning': 'realized null bank' if method == 'independent_null_rank' else 'fixed decision rule'}
    rows.append(row)
    design_output['rejections'].append({**row, **detail})

