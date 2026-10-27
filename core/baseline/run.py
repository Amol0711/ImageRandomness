"""Regenerate numerical summaries or independent-uniform-pixel simulations."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm

from numerical import (baci_reference, fixed_reference, reference_moments,
                       simulate_design, wilson, baci, literal_baci, graph_counts)

ROOT = Path(__file__).resolve().parent


def designs(cfg: dict) -> list[dict]:
    items = []
    for shape in cfg['baci_image_shapes']:
        items.append(dict(id=f'BACI_{shape[0]}x{shape[1]}_R1', kind='baci',
                          shape=shape, repeats=1))
    repeated = cfg['baci_repeated_whole_image_design']
    m, n = repeated['shape']
    repeats = repeated['repeats']
    items.append(dict(id=f'BACI_{m}x{n}_R{repeats}', kind='baci',
                      shape=[m, n], repeats=repeats))
    shape = cfg['uaci_fixed_image_shape']
    for name in cfg['uaci_fixed_profiles']:
        if name == 'constant_128':
            profile = np.full(shape, 128, dtype=np.int16)
        elif name == 'checkerboard_0_128':
            profile = ((np.indices(shape).sum(axis=0) % 2) * 128).astype(np.int16)
        else:
            raise ValueError(f'Unknown fixed profile {name}')
        items.append(dict(id=f'UACI_fixed_{name}_{shape[0]}x{shape[1]}',
                          kind='fixed', shape=shape, profile=profile))
    shape = cfg['control_image_shape']
    items.append(dict(id=f'random_random_{shape[0]}x{shape[1]}', kind='controls',
                      shape=shape))
    return items


def evaluate(values: np.ndarray, bank: np.ndarray, ref: dict,
             alphas: list[float]) -> dict:
    dev = np.abs(values - ref['mean'])
    bank_dev = np.sort(np.abs(bank - ref['mean']))
    pvalues = (1 + bank.size - np.searchsorted(bank_dev, dev, side='left')) / (bank.size + 1)
    result = dict(ref)
    result.update(fresh_n=int(values.size), bank_n=int(bank.size),
                  fresh_mean=float(values.mean()),
                  fresh_sample_variance=float(values.var(ddof=1)),
                  empirical_to_corrected_variance_ratio=float(values.var(ddof=1) / ref['corrected_variance']),
                  tests=[])
    for method, variance in ref['reference_variances'].items():
        for alpha in alphas:
            critical = float(norm.isf(alpha / 2) * np.sqrt(variance))
            count = int(np.count_nonzero(dev > critical))
            result['tests'].append(dict(method=method, alpha=alpha, rejections=count,
                                        rate=count / values.size, wilson_95=wilson(count, values.size),
                                        absolute_deviation_threshold=critical,
                                        corrected_to_reference_variance_ratio=ref['corrected_variance'] / variance))
    for alpha in alphas:
        count = int(np.count_nonzero(pvalues <= alpha))
        rank = int(np.floor(alpha * (bank.size + 1) + 1e-10))
        threshold = float(bank_dev[bank.size - rank]) if rank else None
        result['tests'].append(dict(method='independent_rank', alpha=alpha,
                                    rejections=count, rate=count / values.size,
                                    wilson_95=wilson(count, values.size),
                                    absolute_deviation_threshold_strict=threshold,
                                    bank_ties_at_threshold=int(np.count_nonzero(bank_dev == threshold)) if rank else 0))
    return result


def entropy_scores(cfg: dict, trials: int) -> np.ndarray:
    rng = np.random.default_rng(cfg['seed'])
    probabilities = np.full(cfg['alphabet'], 1 / cfg['alphabet'])
    scores = np.empty(trials)
    for first in range(0, trials, cfg['batch_size']):
        size = min(cfg['batch_size'], trials - first)
        counts = rng.multinomial(cfg['pixels_per_block'], probabilities,
                                 size=(size, cfg['blocks_per_trial']))
        p = counts / float(cfg['pixels_per_block'])
        logs = np.zeros_like(p)
        np.log2(p, out=logs, where=p > 0)
        scores[first:first + size] = (-(p * logs).sum(axis=-1)).mean(axis=1)
    return scores


def load_arrays(directory: Path, cfg: dict) -> tuple[dict, np.ndarray]:
    expected = set()
    for design in designs(cfg):
        metrics = ['baci'] if design['kind'] == 'baci' else ['uaci']
        if design['kind'] == 'controls':
            metrics.append('npcr')
        for metric in metrics:
            for kind in ['bank', 'fresh']:
                expected.add(f'{design["id"]}_{metric}_{kind}')
    with np.load(directory / 'null_samples.npz', allow_pickle=False) as stored:
        if set(stored.files) != expected:
            raise ValueError('Unexpected array keys in null_samples.npz')
        arrays = {key: stored[key] for key in stored.files}
    with np.load(directory / 'entropy_null_samples.npz', allow_pickle=False) as stored:
        if stored.files != ['scores']:
            raise ValueError('Unexpected array keys in entropy_null_samples.npz')
        entropy = stored['scores']
    for key, values in list(arrays.items()) + [('scores', entropy)]:
        upper = np.log2(cfg['entropy']['alphabet']) if key == 'scores' else 1
        if (values.dtype != np.float64 or values.ndim != 1 or values.size < 2
                or not np.isfinite(values).all() or (values < 0).any() or (values > upper).any()):
            raise ValueError(f'Invalid numeric score array {key}')
    return arrays, entropy


def summarize(cfg: dict, arrays: dict, entropy: np.ndarray) -> dict:
    moments = reference_moments(cfg['q'])
    result = dict(pixel_reference_moments=moments, designs={})
    for index, design in enumerate(designs(cfg)):
        m, n = design['shape']
        repeats = design.get('repeats', 1)
        item = dict(kind=design['kind'], shape=design['shape'], repeats=repeats,
                    bank_seed_spawn_key=[2 * index], fresh_seed_spawn_key=[2 * index + 1], metrics={})
        metrics = ['baci'] if design['kind'] == 'baci' else ['uaci']
        if design['kind'] == 'controls':
            metrics.append('npcr')
        for metric in metrics:
            if design['kind'] == 'baci':
                ref = baci_reference(m, n, repeats, cfg['q'], moments)
            elif design['kind'] == 'fixed':
                ref = fixed_reference(design['profile'], cfg['q'])
            else:
                variance = moments[f'variance_{metric}_pixel'] / (m * n)
                ref = dict(mean=moments[f'mean_{metric}'], corrected_variance=variance,
                           reference_variances={'standard_gaussian': variance})
            prefix = f'{design["id"]}_{metric}'
            item['metrics'][metric] = evaluate(arrays[f'{prefix}_fresh'], arrays[f'{prefix}_bank'], ref, cfg['alpha'])
        result['designs'][design['id']] = item
    ecfg = cfg['entropy']
    half = float(norm.isf(ecfg['alpha'] / 2) * ecfg['reference_block_sd'] / np.sqrt(ecfg['blocks_per_trial']))
    intervals = {'fixed_interval': ecfg['fixed_interval'],
                 'gaussian_block_mean': [ecfg['reference_mean'] - half, ecfg['reference_mean'] + half]}
    rows = []
    for method, interval in intervals.items():
        count = int(np.count_nonzero((entropy < interval[0]) | (entropy > interval[1])))
        rows.append(dict(method=method, alpha=ecfg['alpha'], interval=interval, rejections=count,
                         n=int(entropy.size), rate=count / entropy.size, wilson_95=wilson(count, entropy.size)))
    result['entropy'] = dict(n=int(entropy.size), empirical_mean=float(entropy.mean()),
                             empirical_sd=float(entropy.std(ddof=1)),
                             reference_mean=ecfg['reference_mean'],
                             reference_average_sd=ecfg['reference_block_sd'] / np.sqrt(ecfg['blocks_per_trial']), tests=rows)
    m, n = cfg['control_image_shape']
    size = m * n
    p = 1 - 1 / cfg['q']
    k = np.arange(size + 1)
    pmf = binom.pmf(k, size, p)
    deviation = np.abs(k / size - p)
    bank = arrays[f'random_random_{m}x{n}_npcr_bank']
    bank_dev = np.sort(np.abs(bank - p))
    rank_p = (1 + bank.size - np.searchsorted(bank_dev, deviation, side='left')) / (bank.size + 1)
    rows = []
    for alpha in cfg['alpha']:
        critical = norm.isf(alpha / 2) * np.sqrt(p * (1 - p) / size)
        rows.extend([dict(alpha=alpha, method='standard_gaussian', probability=float(pmf[deviation > critical].sum())),
                     dict(alpha=alpha, method='independent_rank', probability=float(pmf[rank_p <= alpha].sum()))])
    result['npcr_exact'] = dict(binomial_n=size, binomial_p=p, summed_mass=float(pmf.sum()), tests=rows)
    return result


def write_summaries(output: Path, result: dict) -> None:
    (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    moment_rows, rejection_rows = [], []
    for design, item in result['designs'].items():
        for metric, values in item['metrics'].items():
            moment_rows.append(dict(design=design, metric=metric, n=values['fresh_n'],
                                    exact_mean=values['mean'], sample_mean=values['fresh_mean'],
                                    exact_variance=values['corrected_variance'], sample_variance=values['fresh_sample_variance']))
            for test in values['tests']:
                rejection_rows.append(dict(design=design, metric=metric, method=test['method'],
                                           alpha=test['alpha'], rejections=test['rejections'], n=values['fresh_n'],
                                           rate=test['rate'], wilson_low=test['wilson_95'][0],
                                           wilson_high=test['wilson_95'][1]))
    for test in result['entropy']['tests']:
        rejection_rows.append(dict(design='entropy_uniform_blocks', metric='entropy', method=test['method'],
                                   alpha=test['alpha'], rejections=test['rejections'], n=test['n'],
                                   rate=test['rate'], wilson_low=test['wilson_95'][0], wilson_high=test['wilson_95'][1]))
    for filename, rows in [('moments.csv', moment_rows), ('rejection_rates.csv', rejection_rows)]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output / 'npcr_exact.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['alpha', 'method', 'probability'])
        writer.writeheader()
        writer.writerows(result['npcr_exact']['tests'])


def positive_count(value: str) -> int:
    count = int(value)
    if count < 2:
        raise argparse.ArgumentTypeError('Sample counts must be at least 2')
    return count


def compare_numeric(observed, expected, location='root') -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or observed.keys() != expected.keys():
            raise AssertionError(f'Schema mismatch at {location}')
        for key in expected:
            compare_numeric(observed[key], expected[key], f'{location}.{key}')
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise AssertionError(f'List mismatch at {location}')
        for index, (left, right) in enumerate(zip(observed, expected)):
            compare_numeric(left, right, f'{location}[{index}]')
    elif isinstance(expected, float):
        if not np.isclose(observed, expected, rtol=2e-12, atol=2e-14):
            raise AssertionError(f'Numerical mismatch at {location}')
    elif observed != expected:
        raise AssertionError(f'Value mismatch at {location}')


def verify(cfg: dict) -> None:
    arrays, entropy = load_arrays(ROOT / 'data', cfg)
    expected = json.loads((ROOT / 'expected' / 'results.json').read_text())
    compare_numeric(summarize(cfg, arrays, entropy), expected)
    for name, values in arrays.items():
        count = cfg['independent_calibration_bank_draws'] if name.endswith('_bank') else cfg['fresh_evaluation_draws']
        if values.size != count:
            raise AssertionError(f'Archived sample count mismatch for {name}')
    if entropy.size != cfg['entropy']['trials']:
        raise AssertionError('Archived entropy sample count mismatch')
    binary = np.asarray(list(itertools.product(range(2), repeat=9)), dtype=np.int16).reshape(-1, 3, 3)
    np.testing.assert_array_equal(baci(binary, 2), literal_baci(binary, 2))
    for m, n in [(2, 2), (2, 7), (3, 5), (16, 16), (64, 64), (256, 256)]:
        if graph_counts(m, n) != (10*m*n-12*m-12*n+14, 62*m*n-96*m-96*n+148):
            raise AssertionError('Weighted graph count mismatch')
    # An independently enumerated small-alphabet image law checks the moment formulas.
    q, m, n = 2, 2, 2
    pairs = np.asarray(list(itertools.product(range(q), repeat=2*m*n)), dtype=np.int16)
    differences = np.abs(pairs[:, :m*n] - pairs[:, m*n:]).reshape(-1, m, n)
    values = literal_baci(differences, q)
    ref = baci_reference(m, n, 1, q, reference_moments(q))
    np.testing.assert_allclose([values.mean(), values.var()], [ref['mean'], ref['corrected_variance']], rtol=1e-13)
    constant = fixed_reference(np.full((2, 3), 1, dtype=np.int16), 4)
    mixed = fixed_reference(np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int16), 4)
    if constant['reference_variances']['pooled_pixel_gaussian'] != constant['corrected_variance']:
        raise AssertionError('Constant-profile variances disagree')
    if not mixed['reference_variances']['pooled_pixel_gaussian'] > mixed['corrected_variance']:
        raise AssertionError('Mixed-profile variance gap is absent')
    # Complete ties have p=1; banks smaller than the requested rank cannot reject.
    ref = dict(mean=0.5, corrected_variance=0.1, reference_variances={})
    ties = evaluate(np.full(2, 0.5), np.full(2, 0.5), ref, [0.01, 0.05])
    if any(row['rejections'] != 0 for row in ties['tests']):
        raise AssertionError('Rank tie handling failed')
    print('PASS archived summaries, sample counts, literal scores, graph counts, exhaustive moments, fixed profiles, and rank ties.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    analyze = subparsers.add_parser('analyze', help='Analyze archived or regenerated score arrays')
    analyze.add_argument('--data', type=Path, default=ROOT / 'data')
    analyze.add_argument('--output', type=Path, required=True)
    simulate = subparsers.add_parser('simulate', help='Generate new arrays and analyze them')
    simulate.add_argument('--output', type=Path, required=True)
    simulate.add_argument('--bank-count', type=positive_count)
    simulate.add_argument('--fresh-count', type=positive_count)
    simulate.add_argument('--entropy-trials', type=positive_count)
    subparsers.add_parser('verify', help='Check archived summaries and independent numerical identities')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'config.json').read_text())
    if args.command == 'verify':
        verify(cfg)
        return
    if args.output.exists():
        parser.error('Output directory already exists; choose a new directory')
    if args.output.resolve().is_relative_to(ROOT):
        parser.error('Output must be outside the baseline source and archived data directory')
    if args.command == 'analyze':
        arrays, entropy = load_arrays(args.data, cfg)
    else:
        counts = dict(bank=args.bank_count or cfg['independent_calibration_bank_draws'],
                      fresh=args.fresh_count or cfg['fresh_evaluation_draws'])
        seeds = np.random.SeedSequence(cfg['master_seed']).spawn(2 * len(designs(cfg)))
        arrays = {}
        for index, design in enumerate(designs(cfg)):
            batch = cfg['batch_size']['256_by_256'] if design['shape'][0] == 256 else cfg['batch_size']['default']
            for offset, kind in enumerate(['bank', 'fresh']):
                for metric, values in simulate_design(design, counts[kind], seeds[2 * index + offset], cfg['q'], batch).items():
                    arrays[f'{design["id"]}_{metric}_{kind}'] = values
            print(f'Generated {design["id"]}', flush=True)
        entropy = entropy_scores(cfg['entropy'], args.entropy_trials or cfg['entropy']['trials'])
    result = summarize(cfg, arrays, entropy)
    # Exclusive creation prevents concurrent runs from replacing previous results.
    args.output.mkdir(parents=True, exist_ok=False)
    if args.command == 'simulate':
        np.savez_compressed(args.output / 'null_samples.npz', **arrays)
        np.savez_compressed(args.output / 'entropy_null_samples.npz', scores=entropy)
    write_summaries(args.output, result)
    print('Completed numerical analysis.', flush=True)


if __name__ == '__main__':
    main()
