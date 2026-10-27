"""Numerical scores, finite-alphabet moments, and uniform-pixel simulation."""
from __future__ import annotations

import itertools
import numpy as np
from scipy.stats import norm

def baci(d: np.ndarray, q: int) -> np.ndarray:
    """Exact integer accumulation of six-pair overlapping block numerator."""
    h = np.abs(d[:, :, 1:] - d[:, :, :-1])
    v = np.abs(d[:, 1:, :] - d[:, :-1, :])
    numer = (2 * h.sum(axis=(1, 2), dtype=np.int64)
             - h[:, 0, :].sum(axis=1, dtype=np.int64)
             - h[:, -1, :].sum(axis=1, dtype=np.int64)
             + 2 * v.sum(axis=(1, 2), dtype=np.int64)
             - v[:, :, 0].sum(axis=1, dtype=np.int64)
             - v[:, :, -1].sum(axis=1, dtype=np.int64)
             + np.abs(d[:, 1:, 1:] - d[:, :-1, :-1]).sum(axis=(1, 2), dtype=np.int64)
             + np.abs(d[:, 1:, :-1] - d[:, :-1, 1:]).sum(axis=(1, 2), dtype=np.int64))
    m, n = d.shape[1:]
    return numer / (6 * (m-1) * (n-1) * (q-1))

def literal_baci(d: np.ndarray, q: int) -> np.ndarray:
    corners = (d[:, :-1, :-1], d[:, :-1, 1:], d[:, 1:, :-1], d[:, 1:, 1:])
    numer = np.zeros(d.shape[0], dtype=np.int64)
    for a, b in itertools.combinations(corners, 2):
        numer += np.abs(a-b).sum(axis=(1, 2), dtype=np.int64)
    m, n = d.shape[1:]
    return numer / (6 * (m-1) * (n-1) * (q-1))

def graph_counts(m: int, n: int) -> tuple[int, int]:
    weights: dict[tuple[int, int], int] = {}
    for i in range(m-1):
        for j in range(n-1):
            vertices = [i*n+j, i*n+j+1, (i+1)*n+j, (i+1)*n+j+1]
            for a, b in itertools.combinations(vertices, 2):
                edge = (min(a, b), max(a, b))
                weights[edge] = weights.get(edge, 0) + 1
    degree = [0] * (m*n)
    for (a, b), w in weights.items():
        degree[a] += w
        degree[b] += w
    s2 = sum(w*w for w in weights.values())
    pair = (sum(d*d for d in degree)-2*s2)//2
    return s2, pair

def reference_moments(q: int) -> dict:
    x = np.arange(q, dtype=np.float64)
    p = 2*(q-x)/q**2
    p[0] = 1/q
    kernel = np.abs(x[:, None]-x[None, :])
    g = kernel @ p
    mu = p @ g
    second = p @ ((kernel**2) @ p)
    a = second-mu**2
    bcov = p @ ((g-mu)**2)
    return dict(mean_h=float(mu), A=float(a), B=float(bcov),
                mean_baci=float(mu/(q-1)),
                mean_uaci=float(p@x/(q-1)),
                variance_uaci_pixel=float((p@(x*x)-(p@x)**2)/(q-1)**2),
                mean_npcr=float(1-1/q), variance_npcr_pixel=float((1-1/q)/q))

def baci_reference(m: int, n: int, repeats: int, q: int, mom: dict) -> dict:
    k = (m-1)*(n-1)
    s2 = 10*m*n-12*m-12*n+14
    w = 62*m*n-96*m-96*n+148
    a, b = mom['A'], mom['B']
    return {
        'mean': mom['mean_baci'],
        'corrected_variance': (a*s2+2*b*w)/(36*k*k*(q-1)**2*repeats),
        'reference_variances': {
            'edge_independence_gaussian': a/(6*k*(q-1)**2*repeats),
            'block_independence_gaussian': (a/6+2*b/3)/(k*(q-1)**2*repeats),
            'overlap_gaussian': (a*s2+2*b*w)/(36*k*k*(q-1)**2*repeats)},
        'K': k, 'S2': s2, 'W': w,
    }

def fixed_reference(profile: np.ndarray, q: int) -> dict:
    x = np.arange(q)
    differences = np.abs(profile.reshape(-1, 1)-x)/(q-1)
    means = differences.mean(axis=1)
    variances = differences.var(axis=1)
    n = profile.size
    correct = variances.sum()/n**2
    pooled = correct+means.var()/n
    return {'mean': float(means.mean()), 'corrected_variance': float(correct),
            'reference_variances': {'pooled_pixel_gaussian': float(pooled),
                                    'fixed_position_gaussian': float(correct)},
            'between_position_mean_variance': float(means.var())}

def wilson(k: int, n: int) -> list[float]:
    z = norm.ppf(.975)
    phat = k/n
    den = 1+z*z/n
    mid = (phat+z*z/(2*n))/den
    rad = z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
    return [float(mid-rad), float(mid+rad)]

def simulate_design(design: dict, count: int, seed: np.random.SeedSequence,
                    q: int, batch_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    m, n = design['shape']
    rep = design.get('repeats', 1)
    names = ['baci'] if design['kind'] == 'baci' else ['uaci']
    if design['kind'] == 'controls':
        names = ['uaci', 'npcr']
    out = {name: np.empty(count) for name in names}
    for start in range(0, count, batch_size):
        stop = min(start+batch_size, count)
        size = stop-start
        u = rng.integers(0, q, size=(size*rep, m, n), dtype=np.int16)
        if design['kind'] == 'fixed':
            d = np.abs(u-design['profile'][None, :, :])
        else:
            v = rng.integers(0, q, size=u.shape, dtype=np.int16)
            d = np.abs(u-v)
        if design['kind'] == 'baci':
            out['baci'][start:stop] = baci(d, q).reshape(size, rep).mean(axis=1)
        else:
            out['uaci'][start:stop] = d.sum(axis=(1, 2), dtype=np.int64)/(m*n*(q-1))
            if design['kind'] == 'controls':
                out['npcr'][start:stop] = np.count_nonzero(d, axis=(1, 2))/(m*n)
    return out
