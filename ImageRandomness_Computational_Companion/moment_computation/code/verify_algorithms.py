"""Independent exact and floating-point validation of the moment algorithms.

The oracle combines directly enumerated edge variances and shared-endpoint
covariances. It does not use the engine's polynomial sweep or edge-residual
decomposition. Selected cases also enumerate complete comparison images.
"""
from collections import Counter
from fractions import Fraction as Q
from itertools import combinations, product
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import random
import sys
import time

import moment_engine as engine
import alphabet_baseline as baseline


def direct_general(laws, edges):
    """Enumerate each marginal support and shared-vertex covariance exactly."""
    supports = [{x: Q(p) for x, p in enumerate(law) if p} for law in laws]
    for support in supports:
        assert sum(support.values()) == 1
    means, seconds, conditional = [], [], []
    for i, j, _ in edges:
        means.append(sum((pi*pj*abs(x-y) for x, pi in supports[i].items()
                          for y, pj in supports[j].items()), Q()))
        seconds.append(sum((pi*pj*(x-y)**2 for x, pi in supports[i].items()
                            for y, pj in supports[j].items()), Q()))
        conditional.append({
            i: {x: sum((pj*abs(x-y) for y, pj in supports[j].items()), Q())
                for x in supports[i]},
            j: {y: sum((pi*abs(x-y) for x, pi in supports[i].items()), Q())
                for y in supports[j]},
        })
    mean = sum((c*m for (_, _, c), m in zip(edges, means)), Q())
    variance = sum((c*c*(s-m*m) for (_, _, c), m, s in zip(edges, means, seconds)), Q())
    for a, b in combinations(range(len(edges)), 2):
        shared = set(edges[a][:2]) & set(edges[b][:2])
        if shared:
            assert len(shared) == 1
            v = shared.pop()
            joint = sum((p*conditional[a][v][x]*conditional[b][v][x]
                         for x, p in supports[v].items()), Q())
            variance += 2*edges[a][2]*edges[b][2]*(joint-means[a]*means[b])
    return mean, variance


def fixed_laws(reference, q):
    laws = []
    for v in reference:
        counts = Counter(abs(v-u) for u in range(q))
        laws.append([Q(counts.get(x, 0), q) for x in range(q)])
    return laws


def exhaustive(reference, q, edges):
    total = total2 = 0
    for image in product(range(q), repeat=len(reference)):
        d = [abs(v-u) for v, u in zip(reference, image)]
        score = sum(c*abs(d[i]-d[j]) for i, j, c in edges)
        total += score
        total2 += score*score
    denominator = q**len(reference)
    mean = Q(total, denominator)
    return mean, Q(total2, denominator)-mean*mean


def exhaustive_general(laws, edges):
    """Enumerate the full finite product law with rational probabilities."""
    total = total2 = Q()
    supports = [[(i, Q(p)) for i, p in enumerate(law) if p] for law in laws]
    for outcome in product(*supports):
        values = [x for x, _ in outcome]
        probability = Q(1)
        for _, p in outcome:
            probability *= p
        score = sum(c*abs(values[i]-values[j]) for i, j, c in edges)
        total += probability*score
        total2 += probability*score*score
    return total, total2-total*total


def windows_edges(M, N, a, b):
    pairs = Counter()
    for r in range(M-a+1):
        for c in range(N-b+1):
            vertices = [i*N+j for i in range(r, r+a) for j in range(c, c+b)]
            pairs.update(combinations(vertices, 2))
    return [(i, j, weight) for (i, j), weight in sorted(pairs.items())]


def pair(result):
    return result.mean, result.variance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_legacy():
    path = Path(__file__).with_name('reference_engine.py')
    spec = importlib.util.spec_from_file_location('legacy_moment_engine', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


class FloatAudit:
    def __init__(self):
        self.rows = []

    def check(self, label, actual, expected, diagnostics=None):
        mean, variance = map(float, actual)
        em, ev = map(float, expected)
        assert math.isfinite(mean) and math.isfinite(variance), label
        mean_error = abs(Q.from_float(mean)-expected[0])
        variance_error = abs(Q.from_float(variance)-expected[1])
        mean_abs, variance_abs = float(mean_error), float(variance_error)
        scale = max(1., em*em)
        near_zero = ev <= 1e-10*scale
        variance_relative = None if expected[1] == 0 else float(variance_error/abs(expected[1]))
        # Near-degenerate variances are evaluated by absolute error explicitly.
        assert mean_abs <= 1e-11*max(1., abs(em)), (label, mean, em)
        if near_zero:
            assert variance_abs <= 1e-12*scale, (label, variance, ev)
        else:
            assert variance_relative <= 1e-9, (label, variance, ev)
        self.rows.append({'case': label, 'mean_absolute_error': mean_abs,
                          'variance_absolute_error': variance_abs,
                          'variance_relative_error': variance_relative,
                          'variance_test': 'absolute' if near_zero else 'relative',
                          'reference_variance': ev, 'reported_variance': variance,
                          'negative_residual_clipped_count': (diagnostics or {}).get('negative_residual_clipped_count', 0),
                          'total_residual_clipping_magnitude': (diagnostics or {}).get('total_residual_clipping_magnitude', 0.),
                          'cancellation_sensitive_pairs': (diagnostics or {}).get('cancellation_sensitive_pairs', 0)})


def run():
    started = time.perf_counter()
    counts = Counter()
    rng = random.Random(20261025)
    legacy, legacy_path = load_legacy()
    float_audit = FloatAudit()

    def check(reference, q, edges, full=False):
        expected = direct_general(fixed_laws(reference, q), edges)
        design = engine.prepare_graph(len(reference), edges, normalizer=Q(7, 3))
        direct = engine.moments_graph(reference, q, edges)
        cached = engine.moments_graph(reference, q, edges, cache=True)
        reused = engine.moments_prepared(reference, q, design)
        cached_reused = engine.moments_prepared(reference, q, design, cache=True)
        for result in [direct, cached, reused, cached_reused]:
            assert pair(result) == expected, (reference, q, edges, pair(result), expected)
            assert result.variance >= 0
        exact_alphabet = baseline.enumerate_exact(reference, q, design)
        assert pair(exact_alphabet) == expected, (reference, q, pair(exact_alphabet), expected)
        counts['exact_alphabet_baseline_cases'] += 1
        floating_alphabet = baseline.enumerate_float(reference, q, design)
        float_audit.check('fixed_'+str(counts['fixed_reference_oracle_cases']), pair(floating_alphabet), expected, floating_alphabet.diagnostics)
        counts['floating_alphabet_baseline_cases'] += 1
        assert not direct.diagnostics['reused_graph_design']
        assert reused.diagnostics['reused_graph_design']
        assert direct.diagnostics['profile_evaluations'] == len(reference)
        assert direct.diagnostics['pair_evaluations'] == len(design.edges)
        assert direct.normalized(design.normalizer) == (expected[0]/design.normalizer,
                                                       expected[1]/design.normalizer**2)
        counts['fixed_reference_oracle_cases'] += 1
        counts['engine_mode_agreements'] += 4
        if legacy:
            assert pair(legacy.moments_graph(reference, q, edges)) == expected
            counts['previous_engine_regressions'] += 1
        if full:
            assert exhaustive(reference, q, edges) == expected
            counts['full_image_enumerations'] += 1
        return direct

    designs = [[], [(0, 1, 1)], [(0, 1, 2), (1, 2, 3)],
               [(0, 1, 2), (0, 2, -3), (1, 2, 1)]]
    for q in range(2, 6):
        for reference in product(range(q), repeat=3):
            for edges in designs:
                check(reference, q, edges, full=True)

    for _ in range(100):
        n, q = rng.randrange(2, 8), rng.randrange(2, 21)
        reference = [rng.randrange(q) for i in range(n)]
        edges = [(i, j, rng.choice([-5, -2, 1, 3, 7]))
                 for i, j in combinations(range(n), 2) if rng.random() < .55]
        check(reference, q, edges)
        counts['random_signed_graph_cases'] += 1

    for M, N in product(range(1, 7), repeat=2):
        for a in range(1, M+1):
            for b in range(1, N+1):
                if a*b < 2:
                    continue
                assert sorted(engine.rectangular_edges(M, N, a, b)) == windows_edges(M, N, a, b)
                counts['rectangular_geometry_cases'] += 1

    for M, N, a, b in [(1, 4, 1, 2), (4, 1, 3, 1), (2, 3, 1, 3),
                        (2, 3, 2, 1), (2, 3, 2, 2), (2, 3, 2, 3),
                        (3, 3, 2, 3), (3, 3, 3, 2)]:
        q = 3
        reference = [rng.randrange(q) for i in range(M*N)]
        edges = windows_edges(M, N, a, b)
        raw = check(reference, q, edges, full=(M*N <= 6))
        normalizer = (M-a+1)*(N-b+1)*(a*b*(a*b-1)//2)*(q-1)
        assert engine.image_normalizer(M, N, a, b, q) == normalizer
        prepared = engine.prepare_image_graph(M, N, a, b, normalizer=normalizer)
        assert pair(engine.moments_prepared(reference, q, prepared)) == pair(raw)
        assert pair(engine.moments_image(reference, q, M, N, a, b)) == pair(raw)
        assert raw.normalized(prepared.normalizer) == (raw.mean/normalizer, raw.variance/normalizer**2)
        counts['image_normalization_cases'] += 1

    reusable = engine.prepare_graph(4, [(0, 1, 2), (1, 3, -1), (2, 3, 3)])
    snapshot = (reusable.edges, reusable.adjacency, reusable.normalizer)
    for q in [2, 3, 7, 256, 65536]:
        for reference in [[0, q-1, q//2, q//3], [q-1, 0, 0, q//2]]:
            original = engine.moments_prepared(reference, q, reusable)
            assert pair(original) == pair(engine.moments_graph(reference, q, reusable.edges, cache=True))
            assert snapshot == (reusable.edges, reusable.adjacency, reusable.normalizer)
            counts['graph_reuse_cases'] += 1
    try:
        reusable.n = 100
    except (AttributeError, TypeError):
        counts['immutability_rejections'] += 1
    else:
        raise AssertionError('GraphDesign must be immutable')

    structural = []
    fixed_reference = [0, 1, 2, 3, 4, 5]
    fixed_edges = [(i, j, 1+i+j) for i, j in combinations(range(6), 2)]
    counter_keys = ['profile_evaluations', 'pair_evaluations', 'projection_integrations',
                    'vertex_intervals', 'vertex_breakpoint_events', 'vertex_event_records',
                    'alphabet_values_enumerated']
    for q in [2**8, 2**12, 2**16, 2**24, 2**40]:
        result = engine.moments_graph(fixed_reference, q, fixed_edges)
        row = {'q': q, **{key: result.diagnostics[key] for key in counter_keys},
               'mean_numerator_bits': result.diagnostics['mean_numerator_bits'],
               'variance_numerator_bits': result.diagnostics['variance_numerator_bits']}
        structural.append(row)
        assert all(row[key] == structural[0][key] for key in counter_keys)
        assert row['alphabet_values_enumerated'] == 0
        counts['large_alphabet_structural_cases'] += 1

    for q in [256, 4096, 65536, 2**24, 2**40]:
        reference = [0, q-1, q//2, q//3, q//7, q-2]
        edges = [(0, 1, 2), (0, 2, -3), (1, 4, 4), (2, 4, 1), (4, 5, -2)]
        result = engine.moments_graph(reference, q, edges)
        reflected = [q-1-v if i%2 else v for i, v in enumerate(reference)]
        assert pair(engine.moments_graph(reflected, q, edges)) == pair(result)
        scaled = engine.moments_graph(reference, q, [(i, j, -7*w) for i, j, w in edges])
        assert pair(scaled) == (-7*result.mean, 49*result.variance)
        permutation = [3, 5, 0, 4, 1, 2]
        inverse = {old: new for new, old in enumerate(permutation)}
        permuted = engine.moments_graph([reference[i] for i in permutation], q,
                                       [(inverse[i], inverse[j], w) for i, j, w in edges])
        assert pair(permuted) == pair(result)
        counts['large_alphabet_symmetries'] += 3

    for reference, edges in [([], []), ([0], []), ([0, 2], [(0, 1, 0)])]:
        assert pair(engine.moments_graph(reference, 3, edges)) == (0, 0)
        assert pair(baseline.enumerate_exact(reference, 3, edges)) == (0, 0)
        float_audit.check('degenerate_'+str(counts['degenerate_graph_cases']),
                          pair(baseline.enumerate_float(reference, 3, edges)), (Q(), Q()))
        counts['degenerate_graph_cases'] += 1

    # The simulation kernel is checked against every outcome of small graphs.
    # This validates score computation only and performs no timed MC study.
    import numpy as np
    import monte_carlo
    simulation_cases = [
        ([0, 1, 2, 1], 3, [(0, 1, 2), (0, 2, -3), (1, 3, 5)]),
        ([0, 1, 2, 0, 2, 1], 3, windows_edges(2, 3, 2, 2)),
        ([0, 1, 0, 1], 2, windows_edges(1, 4, 1, 3)),
        ([0, 1], 2, []),
    ]
    for index, (reference, q, edges) in enumerate(simulation_cases):
        design = engine.prepare_graph(len(reference), edges)
        uniforms = np.asarray(list(product(range(q), repeat=len(reference))), dtype=np.int32)
        expected_scores = []
        for image in uniforms:
            differences = [abs(v-int(u)) for v, u in zip(reference, image)]
            expected_scores.append(sum(c*abs(differences[i]-differences[j]) for i, j, c in edges))
        for chunk_size in [1, 3, 100]:
            ref_array, chunks = monte_carlo.prepare_simulation(reference, q, design, chunk_size)
            scores = monte_carlo.scores_from_uniforms(uniforms, ref_array, chunks)
            assert scores.tolist() == expected_scores
            mean = Q(sum(expected_scores), len(expected_scores))
            variance = Q(sum(s*s for s in expected_scores), len(expected_scores))-mean*mean
            assert (mean, variance) == pair(engine.moments_prepared(reference, q, design))
            counts['simulation_kernel_batches'] += 1
        counts['simulation_distinct_outcomes'] += len(uniforms)

    # Explicit large-alphabet baselines enumerate the alphabet, independently
    # of the sweep, on small graphs. These runs are correctness checks only.
    large_baseline_cases = []
    for q in [256, 4096, 65536]:
        reference = [0, q//3, q//2, q-1]
        edges = [(0, 1, 2), (1, 2, -3), (1, 3, 5)]
        exact = engine.moments_graph(reference, q, edges)
        direct_alphabet = baseline.enumerate_exact(reference, q, edges)
        assert pair(direct_alphabet) == pair(exact)
        floating_alphabet = baseline.enumerate_float(reference, q, edges)
        label = 'large_q_'+str(q)
        float_audit.check(label, pair(floating_alphabet), pair(exact), floating_alphabet.diagnostics)
        large_baseline_cases.append({'q': q, 'vertices': len(reference), 'edges': len(edges),
                                     'mean': str(exact.mean), 'variance': str(exact.variance)})
        counts['large_exact_alphabet_baseline_cases'] += 1
        counts['large_float_alphabet_baseline_cases'] += 1

    # Arbitrary independent PMFs exercise the baseline's broader model only.
    general_cases = [
        ([[Q(1), Q(0)], [Q(0), Q(1)]], [(0, 1, 3)]),
        ([[Q(1)-Q(1, 2**40), Q(1, 2**40)], [Q(0), Q(1)]], [(0, 1, 1)]),
        ([[Q(1, 3), Q(2, 3)], [Q(1, 7), Q(6, 7)], [Q(1, 2), Q(1, 2)]],
         [(0, 1, 5), (1, 2, -7), (0, 2, 2)]),
    ]
    for k in range(5):
        general_cases.append(([[Q(k+1, 10), Q(2, 10), Q(7-k, 10)],
                               [Q(1, 3), Q(0), Q(2, 3)]], [(0, 1, k-2)]))
    for _ in range(50):
        n, q = rng.randrange(2, 7), rng.randrange(2, 9)
        laws = []
        for i in range(n):
            weights = [rng.randrange(0, 10) for x in range(q)]
            if not sum(weights):
                weights[0] = 1
            laws.append([Q(x, sum(weights)) for x in weights])
        edges = [(i, j, rng.choice([-7, -2, 1, 4]))
                 for i, j in combinations(range(n), 2) if rng.random() < .6]
        general_cases.append((laws, edges))
    for k, (laws, edges) in enumerate(general_cases):
        exact = direct_general(laws, edges)
        exact_result = baseline.moments_independent_exact(laws, edges)
        assert pair(exact_result) == exact, (k, pair(exact_result), exact)
        counts['general_pmf_exact_oracle_cases'] += 1
        if k < 8 or (len(laws) <= 3 and len(laws[0]) <= 5):
            assert exhaustive_general(laws, edges) == exact
            counts['general_pmf_full_enumerations'] += 1
        result = baseline.moments_independent([[float(x) for x in law] for law in laws], edges)
        float_audit.check('general_pmf_'+str(k), pair(result), exact, result.diagnostics)
        counts['general_pmf_float_oracle_cases'] += 1

    bad_calls = [lambda: engine.moments_graph([0], 1, []),
                 lambda: engine.moments_graph([2], 2, []),
                 lambda: engine.moments_graph([-1], 3, []),
                 lambda: engine.prepare_graph(2, [(0, 0, 1)]),
                 lambda: engine.prepare_graph(2, [(0, 2, 1)]),
                 lambda: engine.prepare_graph(2, [(0, 1, 1), (1, 0, 2)]),
                 lambda: engine.prepare_graph(2, [(0, 1, 0), (1, 0, 0)]),
                 lambda: engine.prepare_graph(2, [(0, 1, .5)]),
                 lambda: engine.prepare_graph(2, [], normalizer=0),
                 lambda: engine.prepare_graph(2, [], normalizer=.5),
                 lambda: engine.moments_graph([0], 3, reusable),
                 lambda: engine.moments_graph([0], 3, [], cache=1),
                 lambda: engine.moments_prepared([0], 3, []),
                 lambda: engine.rectangular_edges(2, 3, 3, 1),
                 lambda: engine.rectangular_edges(2, 3, 1, 1),
                 lambda: engine.image_normalizer(2, 3, 1, 1, 3)]
    for call in bad_calls:
        try:
            call()
        except (ValueError, TypeError):
            counts['invalid_input_rejections'] += 1
        else:
            raise AssertionError('Expected a rejected invalid input')

    baseline_bad_calls = [lambda: baseline.enumerate_exact([0], 1, []),
                          lambda: baseline.enumerate_float([3], 3, []),
                          lambda: baseline.moments_independent([[.4, .4], [.5, .5]], [(0, 1, 1)]),
                          lambda: baseline.moments_independent([[-.1, 1.1], [.5, .5]], [(0, 1, 1)]),
                          lambda: baseline.moments_independent([[float('nan'), .5], [.5, .5]], [(0, 1, 1)]),
                          lambda: baseline.moments_independent([[float('inf'), 0.], [.5, .5]], [(0, 1, 1)])]
    for call in baseline_bad_calls:
        try:
            call()
        except (ValueError, TypeError):
            counts['baseline_invalid_input_rejections'] += 1
        else:
            raise AssertionError('Expected a rejected invalid baseline input')

    try:
        baseline.enumerate_exact([0, 1], 3, [(0, 1, 1)], max_workspace_bytes=1)
    except MemoryError:
        counts['workspace_guard_rejections'] += 1
    else:
        raise AssertionError('Expected exact-baseline workspace guard')

    relative_rows = [r for r in float_audit.rows if r['variance_test'] == 'relative']
    absolute_rows = [r for r in float_audit.rows if r['variance_test'] == 'absolute']
    float_summary = {
        'cases': len(float_audit.rows),
        'relative_variance_cases': len(relative_rows),
        'absolute_variance_cases': len(absolute_rows),
        'maximum_variance_relative_error_non_degenerate': max(r['variance_relative_error'] for r in relative_rows),
        'maximum_mean_absolute_error': max(r['mean_absolute_error'] for r in float_audit.rows),
        'maximum_variance_absolute_error': max(r['variance_absolute_error'] for r in float_audit.rows),
        'cases_with_negative_residual_clipping': sum(r['negative_residual_clipped_count'] > 0 for r in float_audit.rows),
        'total_clipped_edge_residuals': sum(r['negative_residual_clipped_count'] for r in float_audit.rows),
        'non_degenerate_relative_tolerance': 1e-9,
        'near_degenerate_absolute_tolerance': '1e-12*max(1, exact_mean**2)',
        'near_degenerate_criterion': 'exact_variance <= 1e-10*max(1, exact_mean**2)',
    }

    report = {
        'status': 'PASS',
        'seed': 20261025,
        'counts': dict(sorted(counts.items())),
        'elapsed_seconds': time.perf_counter()-started,
        'sha256': {'moment_engine.py': sha(engine.__file__),
                   'alphabet_baseline.py': sha(baseline.__file__),
                   'monte_carlo.py': sha(monte_carlo.__file__),
                   'verify_algorithms.py': sha(__file__)},
        'oracle': 'Direct alphabet moments, shared-endpoint covariances, and selected full comparison-image enumeration',
        'previous_engine_available': legacy is not None,
        'large_alphabet_structural_counters': structural,
        'large_alphabet_exact_baseline_cases': large_baseline_cases,
        'floating_point_summary': float_summary,
        'floating_point_cases': float_audit.rows,
        'source_inspection': {
            'core_has_alphabet_loop': False,
            'default_core': 'Deterministic uncached evaluation with sorted/coalesced event records',
            'optional_cache': 'Hash-based profile and edge-law-pair caches; lookup cost expected/amortized',
            'graph_preparation': 'Deterministic sorting and adjacency construction; reported separately from prepared evaluations',
        },
    }
    if legacy_path:
        report['sha256']['reference_engine.py'] = sha(legacy_path)
    output = Path(__file__).resolve().parents[1]/'results/verification.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ['large_alphabet_structural_counters', 'floating_point_cases']}, indent=2))


if __name__ == '__main__':
    run()
