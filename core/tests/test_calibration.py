"""Independent deterministic checks, including direct finite-state enumeration."""
from fractions import Fraction as F
from itertools import combinations, permutations, product
from math import comb, prod
import unittest

import numpy as np

import calibration as ref


def enum_grid(rows, m, n, q):
    """Direct original-window statistic; no package edges or moment helpers."""
    denominator = prod(sum(row) for row in rows)
    total = first = second = 0
    scale = 6*(m-1)*(n-1)*(q-1)
    for image in product(range(q), repeat=m*n):
        weight = prod(row[x] for row, x in zip(rows, image))
        if not weight:
            continue
        raw = 0
        for i in range(m-1):
            for j in range(n-1):
                vals = [image[i*n+j], image[i*n+j+1],
                        image[(i+1)*n+j], image[(i+1)*n+j+1]]
                for a in range(4):
                    for b in range(a+1, 4):
                        raw += abs(vals[a]-vals[b])
        total += weight
        first += weight*raw
        second += weight*raw*raw
    assert total == denominator
    mean = F(first, denominator*scale)
    return mean, F(second, denominator*scale*scale)-mean*mean


def enum_graph(rows, edges):
    denominator = prod(sum(row) for row in rows)
    first = second = F()
    for vals in product(range(len(rows[0])), repeat=len(rows)):
        weight = prod(row[x] for row, x in zip(rows, vals))
        raw = sum(w*abs(vals[u]-vals[v]) for (u, v), w in edges.items())
        first += weight*raw
        second += weight*raw*raw
    return first/denominator, second/denominator-(first/denominator)**2


def independent_triangular_counts(q):
    # Enumerate original U,V rather than use the triangular-law formula.
    counts = [0]*q
    for u, v in product(range(q), repeat=2):
        counts[abs(u-v)] += 1
    return counts


class TestBACI(unittest.TestCase):
    def test_original_pair_samples_exhaustive_q2(self):
        # All 256 pairs of 2x2 binary images, so D is formed independently.
        first = second = F()
        for vals in product(range(2), repeat=8):
            u, v = np.array(vals[:4]).reshape(2, 2), np.array(vals[4:]).reshape(2, 2)
            stat = ref.baci_from_images(u, v, q=2, exact=True)
            raw = sum(abs(abs(vals[a]-vals[a+4])-abs(vals[b]-vals[b+4]))
                      for a, b in combinations(range(4), 2))
            self.assertEqual(stat, F(raw, 6))
            first += stat/256
            second += stat*stat/256
        moment = ref.iid_baci_moments(2, 2, q=2)
        self.assertEqual((first, second-first*first), (moment.mean, moment.variance))

    def test_iid_grid_exhaustive(self):
        for q, m, n in [(2, 2, 2), (3, 2, 2), (4, 2, 2),
                         (2, 2, 3), (3, 2, 3), (2, 3, 2), (3, 3, 2), (2, 3, 3)]:
            with self.subTest(q=q, m=m, n=n):
                observed = enum_grid([independent_triangular_counts(q)]*(m*n), m, n, q)
                calculated = ref.iid_baci_moments(m, n, q)
                self.assertEqual(observed, (calculated.mean, calculated.variance))

    def test_statistic_weighted_literal_and_independent_loop(self):
        for q, m, n in [(2, 3, 3), (3, 2, 3), (3, 3, 2)]:
            for values in product(range(q), repeat=m*n):
                a = np.array(values).reshape(m, n)
                literal = ref.baci_statistic(a, q, method='literal', exact=True)
                weighted = ref.baci_statistic(a, q, exact=True)
                raw = sum(abs(values[u]-values[v])
                          for i in range(m-1) for j in range(n-1)
                          for u, v in combinations([i*n+j, i*n+j+1, (i+1)*n+j, (i+1)*n+j+1], 2))
                self.assertEqual(weighted, F(raw, 6*(m-1)*(n-1)*(q-1)))
                self.assertEqual(weighted, literal)

    def test_unsigned_subtraction_and_no_mutation(self):
        u = np.array([[0, 255], [128, 10]], dtype=np.uint8)
        v = np.array([[255, 0], [0, 10]], dtype=np.uint8)
        before_u, before_v = u.copy(), v.copy()
        expected = ref.baci_statistic([[255, 255], [128, 0]], 256, exact=True)
        self.assertEqual(ref.baci_from_images(u, v, exact=True), expected)
        self.assertTrue(np.array_equal(u, before_u))
        self.assertTrue(np.array_equal(v, before_v))

    def test_grid_counts_and_thin_strips(self):
        for m, n in list(product(range(2, 10), repeat=2)) + [(2, 256), (256, 2)]:
            edges = ref.grid_edge_weights(m, n)
            square = sum(w*w for w in edges.values())
            # Independent explicit enumeration of incident edge pairs by vertex.
            incident = {i: [] for i in range(m*n)}
            for (u, v), w in edges.items():
                incident[u].append(w)
                incident[v].append(w)
            wedge = sum(a*b for ws in incident.values() for a, b in combinations(ws, 2))
            self.assertEqual((square, wedge), ref.grid_coefficients(m, n))
            self.assertEqual(sum(edges.values()), 6*(m-1)*(n-1))

    def test_constants_against_discrete_sums(self):
        for q in list(range(2, 18)) + [64, 256]:
            self.assertEqual(ref.iid_pair_constants(q), ref.iid_pair_constants(q, method='sum'))
        # Direct independent triple sum for C, not conditional-mean implementation.
        for q in range(2, 7):
            counts = independent_triangular_counts(q)
            mean = F(sum(counts[x]*counts[y]*abs(x-y) for x, y in product(range(q), repeat=2)), q**4)
            second = F(sum(counts[x]*counts[y]*(x-y)**2 for x, y in product(range(q), repeat=2)), q**4)
            cross = F(sum(counts[x]*counts[y]*counts[z]*abs(x-y)*abs(x-z)
                          for x, y, z in product(range(q), repeat=3)), q**6)
            c = ref.iid_pair_constants(q)
            self.assertEqual((c.mean_distance, c.A, c.C), (mean, second-mean**2, cross-mean**2))

    def test_q2_degeneracy_one_block_and_repetition(self):
        c = ref.iid_pair_constants(2)
        self.assertEqual((c.A, c.C), (F(1, 4), 0))
        for q in (2, 3, 256):
            c = ref.iid_pair_constants(q)
            self.assertEqual(ref.iid_baci_moments(2, 2, q).variance,
                             (c.A/6+F(2, 3)*c.C)/(q-1)**2)
            baseline = ref.iid_baci_moments(2, 256, q)
            average = ref.iid_baci_moments(2, 256, q, repeats=100)
            self.assertGreater(baseline.variance, 0)
            self.assertEqual(average.mean, baseline.mean)
            for name in ('variance', 'independent_pair_variance', 'within_block_only_variance'):
                self.assertEqual(getattr(average, name), getattr(baseline, name)/100)

    def test_headline_exact_constants(self):
        r = ref.iid_baci_moments(256, 256)
        self.assertEqual(r.mean, F(16843009, 62914560))
        self.assertEqual(r.A, F(46118814850137311, 17592186044416))
        self.assertEqual(r.C, F(7686338775514335, 17592186044416))
        self.assertAlmostEqual(float(r.variance/r.independent_pair_variance), 5.093639127500059, places=13)
        floating = ref.iid_baci_moments(256, 256, exact=False)
        self.assertEqual(floating.variance, float(r.variance))


class TestNonidentical(unittest.TestCase):
    def test_general_graph_exact_signed_fractional_weights(self):
        rows = [[2, 1, 0], [0, 1, 2], [1, 1, 1], [1, 2, 1], [0, 0, 1]]
        edges = {(0, 1): 2, (0, 2): -3, (1, 3): F(1, 2), (2, 3): 4,
                 (3, 4): -1}
        expected = enum_graph(rows, edges)
        got = ref.weighted_edge_moments(rows, edges)
        self.assertEqual((got.mean, got.variance), expected)
        self.assertEqual(got.variance, got.vertex_projection_variance+got.edge_residual_variance)
        float_result = ref.weighted_edge_moments(rows, edges, exact=False)
        self.assertAlmostEqual(float_result.mean, float(got.mean), places=13)
        self.assertAlmostEqual(float_result.variance, float(got.variance), places=12)
        repeated = ref.weighted_edge_moments(rows, edges, repeats=7)
        self.assertEqual(repeated.mean, got.mean)
        self.assertEqual(repeated.variance, got.variance/7)

    def test_empty_edges_isolated_vertices_and_degenerate_laws(self):
        for exact in (True, False):
            got = ref.weighted_edge_moments([[1, 0], [0, 1], [1, 0]], {}, exact=exact)
            self.assertEqual((got.mean, got.variance), (0, 0))
            got = ref.weighted_edge_moments([[1, 0], [0, 1], [1, 0]], {(0, 1): -2}, exact=exact)
            self.assertEqual((got.mean, got.variance), (-2, 0))
            one = ref.weighted_edge_moments([[3], [7]], {(0, 1): 9}, exact=exact)
            self.assertEqual((one.mean, one.variance), (0, 0))
        interval = ref.gaussian_interval(-2, 0)
        self.assertEqual((interval.lower, interval.upper), (-2, -2))

    def test_fixed_image_exhaustive_and_float(self):
        for q, m, n in [(2, 2, 2), (3, 2, 2), (4, 2, 2),
                         (2, 2, 3), (3, 2, 3), (3, 3, 2)]:
            values = [(i*2+i//n) % q for i in range(m*n)]
            image = np.array(values).reshape(m, n)
            # Enumerate D counts from fixed pixels and uniform U directly.
            rows = [[sum(abs(v-u) == d for u in range(q)) for d in range(q)] for v in values]
            expected = enum_grid(rows, m, n, q)
            got = ref.fixed_baci_moments(image, q, exact=True)
            self.assertEqual((got.mean, got.variance), expected)
            floating = ref.fixed_baci_moments(image, q)
            self.assertAlmostEqual(floating.mean, float(got.mean), places=13)
            self.assertAlmostEqual(floating.variance, float(got.variance), places=13)
            repeated = ref.fixed_baci_moments(image, q, exact=True, repeats=5)
            self.assertEqual((repeated.mean, repeated.variance), (got.mean, got.variance/5))

    def test_fixed_law_symmetry_and_normalization(self):
        for q in range(2, 12):
            self.assertEqual(sum(ref.triangular_difference_law(q)), 1)
            for v in range(q):
                law = ref.fixed_difference_law(v, q)
                self.assertEqual(sum(law), 1)
                self.assertEqual(law, ref.fixed_difference_law(q-1-v, q))
                self.assertEqual(law, tuple(F(sum(abs(v-u) == d for u in range(q)), q) for d in range(q)))

    def test_float_fixed_large_grid_against_exact_homogeneous_law(self):
        # A 64x64 constant image gives identical nontriangular D laws. Compare
        # scalable vertex accumulation with an independent exact conditional sum.
        q, v, m, n = 256, 128, 64, 64
        counts = [sum(abs(v-u) == d for u in range(q)) for d in range(q)]
        g = [F(sum(counts[y]*abs(x-y) for y in range(q)), q) for x in range(q)]
        mean_h = sum((F(counts[x], q)*g[x] for x in range(q)), F())
        var_h = sum((F(counts[x]*counts[y], q*q)*(x-y)**2
                     for x in range(q) for y in range(q)), F())-mean_h*mean_h
        shared = sum((F(counts[x], q)*(g[x]-mean_h)**2 for x in range(q)), F())
        edges = ref.grid_edge_weights(m, n)
        square = sum(w*w for w in edges.values())
        degrees = [0]*(m*n)
        for (a, b), w in edges.items():
            degrees[a] += w
            degrees[b] += w
        wedge = (sum(d*d for d in degrees)-2*square)//2
        scale = 6*(m-1)*(n-1)*(q-1)
        expected_variance = (var_h*square+2*shared*wedge)/(scale*scale)
        got = ref.fixed_baci_moments(np.full((m, n), v, dtype=np.int64), q)
        self.assertAlmostEqual(got.mean/float(mean_h/(q-1)), 1.0, places=11)
        self.assertAlmostEqual(got.variance/float(expected_variance), 1.0, places=11)


class TestUACIEntropyAndRank(unittest.TestCase):
    def test_fixed_uaci_65536_witness(self):
        got = ref.fixed_uaci_moments([0, 128], q=256)
        one = two = 0
        for u, v in product(range(256), repeat=2):
            raw = abs(u)+abs(128-v)
            one += raw
            two += raw*raw
        mean = F(one, 256**2*510)
        variance = F(two, 256**2*510**2)-mean*mean
        self.assertEqual((got.mean, got.variance), (mean, variance))
        self.assertEqual(got.variance, F(27307, 1040400))
        self.assertEqual(got.pooled_variance, F(23581, 693600))
        self.assertEqual(got.pooled_excess, F(16129, 2080800))
        repeated = ref.fixed_uaci_moments([0, 128], repeats=50)
        self.assertEqual(repeated.mean, got.mean)
        self.assertEqual(repeated.variance, got.variance/50)
        self.assertEqual(repeated.pooled_excess, got.pooled_excess/50)

    def test_fixed_uaci_controls_and_actual_statistics(self):
        self.assertEqual(ref.fixed_uaci_moments([128, 128]).pooled_excess, 0)
        self.assertEqual(ref.fixed_uaci_moments([0, 255]).pooled_excess, 0)
        u, v = np.array([0, 255, 9], dtype=np.uint8), np.array([255, 0, 9], dtype=np.uint8)
        self.assertEqual(ref.uaci_statistic(u, v, exact=True), F(2, 3))
        self.assertEqual(ref.npcr_statistic(u, v, exact=True), F(2, 3))
        self.assertEqual(ref.uaci_statistic([1], [0], q=2, exact=True), 1)

    def test_entropy_values_and_equivalent_count_inputs(self):
        self.assertEqual(ref.count_entropy([1, 1]), 1)
        self.assertEqual(ref.count_entropy([0, 7, 0]), 0)
        self.assertEqual(ref.block_entropy([0, 0, 1, 1], q=2), 1)
        self.assertEqual(ref.local_entropy_mean([[0, 0], [0, 1]], q=2), .5)
        self.assertEqual(ref.local_entropy_from_counts([[2, 0], [1, 1]]), .5)
        blocks = np.array([[[0, 0], [1, 1]], [[0, 0], [0, 0]]])
        self.assertEqual(ref.local_entropy_mean(blocks, q=2), .5)

    def test_entropy_binary_exact_rejection_and_scaling(self):
        correct = ref.local_entropy_interval(.5, .5, 30)
        printed = ref.local_entropy_interval(.5, .5, 30, scaling='linear')
        counts_correct = [j for j in range(31) if correct.lower <= j/30 <= correct.upper]
        counts_printed = [j for j in range(31) if printed.lower <= j/30 <= printed.upper]
        self.assertEqual(counts_correct, list(range(10, 21)))
        self.assertEqual(counts_printed, [15])
        rejection = 1-sum((F(comb(30, j), 2**30) for j in counts_printed), F())
        self.assertEqual(rejection, F(57414019, 67108864))
        expected = ref.local_entropy_interval(7.902469317, .008694225, 30)
        self.assertAlmostEqual(expected.lower, 7.899358185709168, places=12)
        self.assertAlmostEqual(expected.upper, 7.9055804482908325, places=12)

    def test_rank_ties_and_exchangeability_by_enumeration(self):
        self.assertEqual(ref.known_null_rank_pvalue(2, [0, 1, 2], exact=True), F(1, 2))
        self.assertEqual(ref.known_null_rank_pvalue(3, [0, 1, 2], exact=True), F(1, 4))
        self.assertEqual(ref.known_null_rank_pvalue(1, [1, 1], exact=True), 1)
        # Every one of 3! exchangeable orderings of three distinct scores.
        pvals = [ref.known_null_rank_pvalue(x[0], x[1:], exact=True) for x in permutations([0, 1, 2])]
        self.assertEqual(F(sum(p <= F(1, 3) for p in pvals), len(pvals)), F(1, 3))
        # Iid binary scores have ties, hence conservative marginal rejection.
        pvals = [ref.known_null_rank_pvalue(x[0], x[1:], exact=True) for x in product(range(2), repeat=3)]
        self.assertEqual(F(sum(p <= F(1, 3) for p in pvals), len(pvals)), F(1, 8))
        conditional = [ref.known_null_rank_pvalue(x, [0, 0], exact=True) for x in range(2)]
        self.assertGreater(F(sum(p <= F(1, 3) for p in conditional), 2), F(1, 3))
        # Exact ordering survives distinctions beyond float precision.
        self.assertEqual(ref.known_null_rank_pvalue(F(2**54+1), [F(2**54)], exact=True), F(1, 2))


class TestInputValidation(unittest.TestCase):
    def test_invalid_integer_parameters(self):
        for kwargs in ({'q': 1}, {'q': 2.0}, {'q': True}, {'repeats': 0}, {'repeats': 1.0}, {'exact': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                ref.iid_baci_moments(2, 2, **kwargs)
        for shape in [(1, 2), (2, 0), (2.0, 2), (True, 2)]:
            with self.assertRaises((ValueError, TypeError)):
                ref.grid_coefficients(*shape)

    def test_invalid_pixels_and_no_broadcasting(self):
        for a in ([[0., 1.], [0., 1.]], [[False, True], [True, False]],
                  [[0, -1], [0, 0]], [[0, 256], [0, 0]], [0, 1], [], [[[0, 1], [0, 1]]]):
            with self.subTest(image=a), self.assertRaises((TypeError, ValueError)):
                ref.baci_statistic(a)
        for fn in (ref.baci_from_images, ref.uaci_statistic, ref.npcr_statistic):
            with self.assertRaises(ValueError):
                fn([[0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]])
        with self.assertRaises(OverflowError):
            ref.baci_statistic([[0, 2**63-1], [0, 0]], q=2**63)
        with self.assertRaises(ValueError):
            ref.baci_statistic([[0, 1], [0, 1]], method='blocks')

    def test_invalid_laws_and_edges(self):
        cases = [([], {}), ([[0, 0]], {}), ([[1, -1]], {}), ([[1], [1, 0]], {}),
                 ([[1, 1]], {(0, 0): 1}), ([[1, 1]], {(0, 1): 1}),
                 ([[1, 1], [1, 1]], {(0, 1): 1, (1, 0): 2}),
                 ([[.5, .5]], {})]
        for rows, edges in cases:
            with self.subTest(rows=rows, edges=edges), self.assertRaises((ValueError, TypeError)):
                ref.weighted_edge_moments(rows, edges)
        for rows, edges in [([[float('nan'), 1]], {}), ([[1, 1], [1, 1]], {(0, 1): float('inf')})]:
            with self.assertRaises(ValueError):
                ref.weighted_edge_moments(rows, edges, exact=False)

    def test_invalid_intervals_scores_and_entropy(self):
        for args in [(0, -1), (float('nan'), 1), (0, float('inf'))]:
            with self.assertRaises(ValueError):
                ref.gaussian_interval(*args)
        for alpha in (0, 1, float('nan')):
            with self.assertRaises(ValueError):
                ref.gaussian_interval(0, 1, alpha=alpha)
        for scores in ([], [float('nan')], [float('inf')], np.ones((2, 2))):
            with self.assertRaises((ValueError, TypeError)):
                ref.known_null_rank_pvalue(0, scores)
        for counts in ([0, 0], [-1, 2], [1., 1.], [[1, 1]], []):
            with self.assertRaises((ValueError, TypeError)):
                ref.count_entropy(counts)
        with self.assertRaises(ValueError):
            ref.local_entropy_from_counts([[1, 1], [1, 2]])
        with self.assertRaises(ValueError):
            ref.local_entropy_mean([0, 1], q=2)
        with self.assertRaises(ValueError):
            ref.local_entropy_interval(1, 1, 30, scaling='unknown')


if __name__ == '__main__':
    unittest.main(verbosity=2)
