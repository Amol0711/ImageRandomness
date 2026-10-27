"""Exact finite-law moment calculations and deterministic checks."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
import sys

import numpy as np


def geometric_edges(m: int, n: int):
    """Horizontal, vertical, down-right and down-left edges; no windows used."""
    for i in range(m):
        for j in range(n - 1):
            yield i*n+j, i*n+j+1, 1 if i in (0, m-1) else 2
    for i in range(m - 1):
        for j in range(n):
            yield i*n+j, (i+1)*n+j, 1 if j in (0, n-1) else 2
    for i in range(m - 1):
        for j in range(n - 1):
            yield i*n+j, (i+1)*n+j+1, 1
            yield i*n+j+1, (i+1)*n+j, 1


def literal_numerator(d: np.ndarray) -> int:
    """Independent literal six-pair sum; Python integers prevent wraparound."""
    m, n = d.shape
    total = 0
    for i in range(m-1):
        for j in range(n-1):
            block = [int(d[i,j]), int(d[i,j+1]), int(d[i+1,j]), int(d[i+1,j+1])]
            total += sum(abs(x-y) for x,y in combinations(block, 2))
    return total


def independent_exact_moments(image: np.ndarray, q: int = 256) -> dict:
    """Exact raw and normalized moments using edge-pair covariance counting.

    Each count row N_a(d) is the number of uniform comparison values producing
    |a-U|=d. Let g_b(x)=sum_d N_b(d)|x-d|, and M_ab=sum_x N_a(x)g_b(x).
    The pair mean is M_ab/q^2. Edge variance and shared-endpoint covariance
    both have denominator q^4. With central type a and neighbor types b,c,
    the latter numerator is q*sum_x N_a(x)g_b(x)g_c(x)-M_ab*M_ac.
    Disjoint edges have zero covariance by independence.

    Arrays used for local finite sums fit int64 for q<=256. All graph totals
    are accumulated in Python integers, so large-image aggregation cannot
    overflow. No routines from the reference implementation are used here.
    """
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) < 2 or image.dtype.kind not in 'iu':
        raise ValueError('image must be an integer grid with both dimensions >= 2')
    if not isinstance(q, int) or not 2 <= q <= 256:
        raise ValueError('this independent checker supports integer 2 <= q <= 256')
    if int(image.min()) < 0 or int(image.max()) >= q:
        raise ValueError('pixels outside the specified alphabet')
    m,n = image.shape
    values, ids = np.unique(image, return_inverse=True)
    ids = ids.ravel()
    support = np.arange(q, dtype=np.int64)
    counts = np.asarray([np.bincount(np.abs(int(v)-support), minlength=q)
                         for v in values], dtype=np.int64)
    distance = np.abs(support[:,None]-support[None,:])
    g = counts @ distance
    pair_mean_numerator = counts @ g.T
    first = counts @ support
    second = counts @ (support*support)
    pair_second_numerator = q*(second[:,None]+second[None,:])-2*first[:,None]*first[None,:]
    edge_weight = Counter()
    edge_square_weight = Counter()
    shared_weight = Counter()
    within_window_shared_weight = Counter()
    incident = [[] for _ in range(m*n)]
    edge_count = 0
    for u,v,w in geometric_edges(m,n):
        a,b = int(ids[u]), int(ids[v])
        key = (min(a,b), max(a,b))
        edge_weight[key] += w
        edge_square_weight[key] += w*w
        incident[u].append((b,w))
        incident[v].append((a,w))
        edge_count += 1
    for u, edges in enumerate(incident):
        a = int(ids[u])
        for (b,wb),(c,wc) in combinations(edges,2):
            shared_weight[(a,min(b,c),max(b,c))] += wb*wc
    # Each local window has six unit-weight edges and twelve unordered
    # shared-endpoint pairs. This intentionally excludes between-window terms.
    for i in range(m-1):
        for j in range(n-1):
            block = (i*n+j,i*n+j+1,(i+1)*n+j,(i+1)*n+j+1)
            for central in block:
                a = int(ids[central])
                neighbors = [int(ids[v]) for v in block if v != central]
                for b,c in combinations(neighbors,2):
                    within_window_shared_weight[(a,min(b,c),max(b,c))] += 1
    raw_mean_numerator = sum(w*int(pair_mean_numerator[a,b])
                             for (a,b),w in edge_weight.items())
    raw_variance_numerator = 0
    independent_pair_variance_numerator = 0
    for (a,b),w2 in edge_square_weight.items():
        mu = int(pair_mean_numerator[a,b])
        edge_variance = q*q*int(pair_second_numerator[a,b])-mu*mu
        raw_variance_numerator += w2*edge_variance
        independent_pair_variance_numerator += edge_weight[(a,b)]*edge_variance
    shared_numerator = 0
    covariance_cache = {}
    for (a,b,c),ww in shared_weight.items():
        cross = int(np.dot(counts[a], g[b]*g[c]))
        covariance = q*cross-int(pair_mean_numerator[a,b])*int(pair_mean_numerator[a,c])
        covariance_cache[(a,b,c)] = covariance
        shared_numerator += 2*ww*covariance
    within_window_shared_numerator = sum(2*ww*covariance_cache[key]
                                         for key,ww in within_window_shared_weight.items())
    independent_edge_numerator = raw_variance_numerator
    raw_variance_numerator += shared_numerator
    raw_mean = Fraction(raw_mean_numerator,q*q)
    raw_variance = Fraction(raw_variance_numerator,q**4)
    scale = 6*(m-1)*(n-1)*(q-1)
    if raw_variance < 0:
        raise AssertionError('negative exact variance')
    return {
        'raw_mean': raw_mean,
        'raw_variance': raw_variance,
        'mean': raw_mean/scale,
        'variance': raw_variance/(scale*scale),
        'independent_pair_variance': Fraction(independent_pair_variance_numerator,q**4*scale*scale),
        'within_block_only_variance': Fraction(independent_pair_variance_numerator+within_window_shared_numerator,q**4*scale*scale),
        'raw_independent_pair_variance': Fraction(independent_pair_variance_numerator,q**4),
        'raw_within_block_only_variance': Fraction(independent_pair_variance_numerator+within_window_shared_numerator,q**4),
        'raw_independent_edge_variance': Fraction(independent_edge_numerator,q**4),
        'raw_shared_endpoint_contribution': Fraction(shared_numerator,q**4),
        'normalization': scale,
        'vertices': m*n,
        'edges': edge_count,
        'sum_edge_weights': sum(edge_weight.values()),
        'sum_squared_edge_weights': sum(edge_square_weight.values()),
        'shared_endpoint_weight': sum(shared_weight.values()),
        'distinct_intensities': len(values),
        'distinct_edge_type_pairs': len(edge_weight),
        'distinct_shared_endpoint_type_triples': len(shared_weight),
    }


def fraction_record(value: Fraction):
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'decimal': float(value)}


def serialize_moments(result):
    return {key: fraction_record(value) if isinstance(value,Fraction) else value
            for key,value in result.items()}


def exhaustive_moments(fixed: np.ndarray, q: int):
    numerators = []
    flat = fixed.ravel()
    m,n = fixed.shape
    blocks = [(i*n+j,i*n+j+1,(i+1)*n+j,(i+1)*n+j+1)
              for i in range(m-1) for j in range(n-1)]
    window_sums = [0]*len(blocks)
    window_squares = [0]*len(blocks)
    pair_sums = [0]*(6*len(blocks))
    pair_squares = [0]*(6*len(blocks))
    for values in product(range(q), repeat=fixed.size):
        d = np.abs(flat-np.asarray(values,dtype=np.int64)).reshape(fixed.shape)
        numerators.append(literal_numerator(d))
        for k,block in enumerate(blocks):
            pairs = [abs(int(d.flat[u])-int(d.flat[v])) for u,v in combinations(block,2)]
            s = sum(pairs)
            window_sums[k] += s
            window_squares[k] += s*s
            for l,value in enumerate(pairs):
                pair_sums[6*k+l] += value
                pair_squares[6*k+l] += value*value
    total = len(numerators)
    mean = Fraction(sum(numerators),total)
    variance = Fraction(sum(x*x for x in numerators),total)-mean*mean
    independent_pair_variance = sum((Fraction(b,total)-Fraction(a,total)**2
                              for a,b in zip(pair_sums,pair_squares)),Fraction())
    within_block_variance = sum((Fraction(b,total)-Fraction(a,total)**2
                                for a,b in zip(window_sums,window_squares)),Fraction())
    return mean,variance,independent_pair_variance,within_block_variance,total


def fixed_patterns():
    size = 256
    row,col = np.indices((size,size))
    return {
        'constant_0': np.zeros((size,size),dtype=np.int64),
        'constant_128': np.full((size,size),128,dtype=np.int64),
        'checker_0_128': 128*((row+col)%2),
        'vertical_split_0_128': 128*(col>=128),
        'column_ramp_0_255': col,
    }


def check_small_examples(reference):
    geometry_shapes = [(m,n) for m in range(2,7) for n in range(2,7)]
    geometry_shapes += [(2,17),(17,2),(3,19),(19,3)]
    for m,n in geometry_shapes:
        ours = {(min(u,v),max(u,v)):w for u,v,w in geometric_edges(m,n)}
        literal = Counter()
        for i in range(m-1):
            for j in range(n-1):
                literal.update(combinations((i*n+j,i*n+j+1,(i+1)*n+j,(i+1)*n+j+1),2))
        if ours != dict(literal) or ours != reference.grid_edge_weights(m,n):
            raise AssertionError(f'geometry mismatch {(m,n)}')
        incident_weights = [[] for _ in range(m*n)]
        for (u,v),w in ours.items():
            incident_weights[u].append(w)
            incident_weights[v].append(w)
        s2 = sum(w*w for w in ours.values())
        shared = sum(sum(x*y for x,y in combinations(ws,2)) for ws in incident_weights)
        if (s2,shared) != reference.grid_coefficients(m,n):
            raise AssertionError(f'geometry coefficients mismatch {(m,n)}')
    statistic_images = 0
    for shape in [(2,2),(2,3),(3,2),(3,3)]:
        for values in product(range(2),repeat=shape[0]*shape[1]):
            d = np.asarray(values,dtype=np.int64).reshape(shape)
            raw = literal_numerator(d)
            weighted = sum(w*abs(int(d.flat[u])-int(d.flat[v]))
                           for u,v,w in geometric_edges(*shape))
            scale = 6*(shape[0]-1)*(shape[1]-1)
            if raw != weighted or reference.baci_statistic(d,q=2,exact=True) != Fraction(raw,scale):
                raise AssertionError('exhaustive binary statistic disagreement')
            statistic_images += 1
    fixed_cases = []
    for values in product(range(3),repeat=4):
        fixed_cases.append((np.asarray(values,dtype=np.int64).reshape(2,2),3))
    fixed_cases += [(np.asarray(v,dtype=np.int64),3) for v in [
        [[0,1,2],[2,1,0]], [[0,1,0],[1,0,1]],
        [[0,0,1],[0,0,1]], [[2,2,2],[2,2,2]],
        [[0,2],[1,0],[2,1]],
    ]]
    exhaustive_states = 0
    for fixed,q in fixed_cases:
        actual = independent_exact_moments(fixed,q)
        mean,variance,independent_pair,within,states = exhaustive_moments(fixed,q)
        ref = reference.fixed_baci_moments(fixed,q=q,exact=True)
        if (mean,variance) != (actual['raw_mean'],actual['raw_variance']):
            raise AssertionError('exhaustive raw moment disagreement')
        if (ref.mean,ref.variance) != (actual['mean'],actual['variance']):
            raise AssertionError('reference exact moment disagreement')
        if (independent_pair,within) != (actual['raw_independent_pair_variance'],actual['raw_within_block_only_variance']):
            raise AssertionError('exhaustive pair/within-window comparator disagreement')
        exhaustive_states += states
    return {'status':'PASS','geometry_shapes':len(geometry_shapes),
            'exhaustive_binary_statistic_images':statistic_images,
            'fixed_law_cases':len(fixed_cases),
            'enumerated_uniform_comparison_images':exhaustive_states,
            'exhaustive_comparators_per_fixed_law_case':['full','independent_pair','within_block_only']}


def run(reference):
    small = check_small_examples(reference)
    patterns = fixed_patterns()
    output = {}
    maximum_relative_error = 0.0
    for name,image in patterns.items():
        exact = independent_exact_moments(image)
        ref = reference.fixed_baci_moments(image,exact=False)
        mean_error = abs(float(ref.mean)-float(exact['mean']))/float(exact['mean'])
        variance_error = abs(float(ref.variance)-float(exact['variance']))/float(exact['variance'])
        maximum_relative_error = max(maximum_relative_error,mean_error,variance_error)
        if max(mean_error,variance_error)>2e-10:
            raise AssertionError(f'large-grid moment disagreement: {name}')
        # Check transpose independently, including boundary orientation changes.
        transposed = independent_exact_moments(image.T.copy())
        if exact != transposed:
            raise AssertionError(f'transpose symmetry failed: {name}')
        if exact['sum_edge_weights'] != 6*255*255:
            raise AssertionError('incorrect total window multiplicity')
        result = serialize_moments(exact)
        result.update({
            'shape':[256,256], 'q':256, 'repeats':1,
            'fixed_image_sha256_int64_little_endian':sha256(image.astype('<i8').tobytes()).hexdigest(),
            'reference_float_mean':float(ref.mean),
            'reference_float_variance':float(ref.variance),
            'relative_mean_error':mean_error,
            'relative_variance_error':variance_error,
            'exact_transpose_symmetry':'PASS',
            'raw_rank_distance':{
                'definition':'abs(raw_statistic_integer * denominator - numerator)',
                'numerator':exact['raw_mean'].numerator,
                'denominator':exact['raw_mean'].denominator,
                'interpretation':'Exact integer ordering by absolute distance from the exact null mean; no floating tie ambiguity.'
            },
        })
        output[name] = result
    checker,split = patterns['checker_0_128'],patterns['vertical_split_0_128']
    same_histogram = np.array_equal(np.bincount(checker.ravel(),minlength=256),
                                   np.bincount(split.ravel(),minlength=256))
    different_mean = output['checker_0_128']['mean'] != output['vertical_split_0_128']['mean']
    different_variance = output['checker_0_128']['variance'] != output['vertical_split_0_128']['variance']
    if not (same_histogram and different_mean and different_variance):
        raise AssertionError('equal-histogram geometry witness failed')
    return {
        'status':'PASS',
        'random_draws':0,
        'method':'Exact edge-variance plus shared-endpoint covariance aggregation, independently enumerated geometric graph.',
        'small_grid_checks':small,
        'large_grid_checks':output,
        'same_histogram_geometry_witness':{
            'patterns':['checker_0_128','vertical_split_0_128'],
            'counts':{'0':32768,'128':32768},
            'same_histogram':bool(same_histogram),
            'different_exact_mean':different_mean,
            'different_exact_variance':different_variance,
            'mean_difference_checker_minus_split':output['checker_0_128']['mean']['decimal']-output['vertical_split_0_128']['mean']['decimal'],
            'variance_ratio_checker_over_split':output['checker_0_128']['variance']['decimal']/output['vertical_split_0_128']['variance']['decimal'],
        },
        'maximum_reference_relative_error':maximum_relative_error,
        'comparison_tolerance_relative':2e-10,
        'scope':'Deterministic moment and geometry checks only; these checks make no claim about Gaussian rejection rates or cipher security.'
    }

