"""Shared statistical primitives for finite-grid calibration and power studies.

All moment summaries concern raw weighted absolute differences. The independent
uniform and fixed-reference laws are distinct. Rank thresholds use randomized
lexicographic ties. A shared threshold has a random conditional rejection level;
pointwise binomial intervals across a fixed bank target that conditional level.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from statistics import NormalDist

import numpy as np
from moment_engine import prepare_graph, moments_prepared
from alphabet_baseline import moments_independent_exact


@dataclass(frozen=True)
class ScoreDesign:
    n: int
    chunks: tuple
    absolute_weight_sum: int


@dataclass(frozen=True)
class MomentSummary:
    mean: Fraction
    variance: Fraction
    naive_variance: Fraction
    diagnostics: dict


@dataclass(frozen=True)
class IidLawMoments:
    first: Fraction
    second: Fraction
    mean: Fraction
    variance: Fraction
    covariance: Fraction
    counts: tuple
    denominator: int


@dataclass(frozen=True)
class RankThreshold:
    value: int | float
    tie: float
    m: int
    rank: int
    k: int
    marginal_level: Fraction
    beta_a: int
    beta_b: int


def _integer(value, name, minimum=0):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return value


def prepare_score(design, edge_chunk_size=8192):
    edge_chunk_size = _integer(edge_chunk_size, 'edge_chunk_size', 1)
    chunks = []
    for offset in range(0, len(design.edges), edge_chunk_size):
        chunk = design.edges[offset:offset+edge_chunk_size]
        chunks.append((np.asarray([i for i,j,c in chunk], dtype=np.intp),
                       np.asarray([j for i,j,c in chunk], dtype=np.intp),
                       np.asarray([c for i,j,c in chunk], dtype=np.int64)))
    return ScoreDesign(design.n, tuple(chunks), design.absolute_weight_sum)


def score_fields(fields, prepared):
    """Return raw S for integer nonnegative difference fields, using int64."""
    values = np.asarray(fields)
    if values.ndim != 2 or values.shape[1] != prepared.n:
        raise ValueError('fields must have shape (batch, graph vertex count)')
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError('fields must contain integer values')
    if values.size:
        low, high = int(values.min()), int(values.max())
        if low < 0:
            raise ValueError('difference field values must be nonnegative')
        if high > np.iinfo(np.int32).max:
            raise ValueError('difference values must fit signed int32')
        if (high-low)*prepared.absolute_weight_sum > np.iinfo(np.int64).max:
            raise OverflowError('possible score exceeds signed int64')
    values = values.astype(np.int32, copy=False)
    scores = np.zeros(len(values), dtype=np.int64)
    for first,second,weights in prepared.chunks:
        contrasts = np.abs(values[:,first]-values[:,second])
        scores += np.sum(contrasts*weights[None,:], axis=1, dtype=np.int64)
    return scores


def reference_for(kind, M, N, q):
    M, N, q = _integer(M,'M',1), _integer(N,'N',1), _integer(q,'q',2)
    values=[]
    for index in range(M*N):
        x=(1664525*(index+271828)+1013904223)&0xffffffff
        x ^= x >> 13
        row,col=divmod(index,N)
        if kind=='palette': value=(x%32)*(q-1)//31
        elif kind=='diverse': value=x%q
        elif kind=='checker': value=0 if (row+col)%2==0 else q//2
        elif kind=='split': value=0 if col<N//2 else q//2
        elif kind in ('constant','zero','constant_zero'): value=0
        elif kind in ('midpoint','constant_mid'): value=q//2
        else: raise ValueError(f'unknown reference kind {kind!r}')
        values.append(value)
    return tuple(values)


def _pmf_array(pmf,n,q):
    array=np.asarray(pmf,dtype=np.float64)
    if array.shape not in ((q,),(n,q)):
        raise ValueError('pmf must have shape (q,) or (n,q)')
    if not np.all(np.isfinite(array)) or np.any(array<0):
        raise ValueError('pmf entries must be finite and nonnegative')
    totals=array.sum(axis=-1)
    if np.any(np.abs(totals-1)>1e-12):
        raise ValueError('pmf rows must sum to one')
    # Only machine-rounding drift in an already normalized law is corrected.
    return array/totals[...,None]


def generate_null(model,rng,count,M,N,q,reference=None,pmf=None):
    count=_integer(count,'count',0);M=_integer(M,'M',1);N=_integer(N,'N',1)
    q=_integer(q,'q',2)
    if q>np.iinfo(np.int32).max:
        raise ValueError('simulation alphabet must fit signed int32')
    n=M*N
    if model=='two_random':
        left=rng.integers(0,q,size=(count,n),dtype=np.int32)
        right=rng.integers(0,q,size=(count,n),dtype=np.int32)
        return np.abs(left-right)
    if model=='fixed':
        if reference is None:
            raise ValueError('fixed model requires an explicit reference')
        ref=np.asarray(reference)
        if ref.shape!=(n,) or not np.issubdtype(ref.dtype,np.integer):
            raise ValueError('reference must be a length-n integer vector')
        if np.any(ref<0) or np.any(ref>=q):
            raise ValueError('reference intensities must lie in range(q)')
        return np.abs(rng.integers(0,q,size=(count,n),dtype=np.int32)-ref.astype(np.int32)[None,:])
    if model=='independent_pmf':
        if pmf is None: raise ValueError('independent_pmf requires explicit probabilities')
        probabilities=_pmf_array(pmf,n,q)
        uniforms=rng.random((count,n))
        cdf=np.cumsum(probabilities,axis=-1)
        cdf[...,-1]=1.0
        if probabilities.ndim==1:
            return np.searchsorted(cdf,uniforms,side='right').astype(np.int32)
        result=np.zeros((count,n),dtype=np.int32)
        for x in range(q-1): result += uniforms>=cdf[None,:,x]
        return result
    raise ValueError(f'unknown null model {model!r}')


def _rational_row(row,q):
    if len(row)!=q: raise ValueError('probability row has incorrect support')
    result=[]
    for p in row:
        if isinstance(p,bool) or not isinstance(p,(int,Fraction)):
            raise TypeError('exact PMF entries must be Python int or Fraction')
        if p<0: raise ValueError('probabilities must be nonnegative')
        result.append(Fraction(p))
    if sum(result,Fraction())!=1:
        raise ValueError('exact PMF rows must sum exactly to one')
    return tuple(result)


@lru_cache(maxsize=32)
def _law_from_counts(counts):
    denominator=sum(counts)
    if denominator<=0 or any(a<0 for a in counts):
        raise ValueError('invalid probability counts')
    first=sum(x*a for x,a in enumerate(counts))
    second=sum(x*x*a for x,a in enumerate(counts))
    prefix_mass=prefix_first=0
    gamma=[]
    for x,a in enumerate(counts):
        prefix_mass+=a;prefix_first+=x*a
        gamma.append(x*(2*prefix_mass-denominator)+first-2*prefix_first)
    mean_num=sum(a*f for a,f in zip(counts,gamma))
    projected_second=sum(a*f*f for a,f in zip(counts,gamma))
    mean=Fraction(mean_num,denominator**2)
    h_second=Fraction(2*denominator*second-2*first*first,denominator**2)
    variance=h_second-mean*mean
    covariance=Fraction(projected_second,denominator**3)-mean*mean
    if variance<0 or covariance<0 or variance-2*covariance<0:
        raise ArithmeticError('invalid exact iid variance decomposition')
    return IidLawMoments(Fraction(first,denominator),Fraction(second,denominator),
                         mean,variance,covariance,counts,denominator)


def iid_law_moments(q,law='triangular',pmf=None):
    q=_integer(q,'q',1)
    if law=='triangular':
        if q<2: raise ValueError('triangular image law requires q>=2')
        counts=(q,)+tuple(2*(q-d) for d in range(1,q))
    elif law=='uniform': counts=(1,)*q
    elif law in ('pmf','independent_pmf'):
        probabilities=_rational_row(pmf,q)
        denominator=math.lcm(*(p.denominator for p in probabilities))
        counts=tuple(int(p*denominator) for p in probabilities)
    else: raise ValueError(f'unknown iid law {law!r}')
    return _law_from_counts(counts)


def iid_moments(q,design,law='triangular',pmf=None):
    result=iid_law_moments(q,law,pmf)
    gamma2=sum(c*c for i,j,c in design.edges)
    weighted_degrees=[sum(c for j,c in adj) for adj in design.adjacency]
    degree2=sum(d*d for d in weighted_degrees)
    variance=(result.variance-2*result.covariance)*gamma2+result.covariance*degree2
    if any(c<0 for i,j,c in design.edges):
        raise ValueError('pair-occurrence diagnostic requires nonnegative multiplicities')
    return MomentSummary(result.mean*design.weight_sum,variance,
                         result.variance*design.weight_sum,
                         {'edge_mean':result.mean,'edge_variance':result.variance,
                          'shared_covariance':result.covariance,'edge_square_sum':gamma2,
                          'weighted_degree_square_sum':degree2,'law':law})


def null_moments(model,design,q,reference=None,pmf=None):
    """Exact raw moments plus the uncombined-pair-occurrence variance diagnostic."""
    if model=='two_random': return iid_moments(q,design)
    if model=='fixed':
        if reference is None: raise ValueError('fixed model requires a reference')
        exact=moments_prepared(reference,q,design,cache=True)
        pair_design=prepare_graph(2,[(0,1,1)])
        cache={}; naive=Fraction()
        for i,j,c in design.edges:
            if c<0: raise ValueError('naive diagnostic requires nonnegative multiplicities')
            key=tuple(sorted((min(reference[i],q-1-reference[i]),min(reference[j],q-1-reference[j]))))
            if key not in cache: cache[key]=moments_prepared(key,q,pair_design,cache=False).variance
            naive+=c*cache[key]
        return MomentSummary(exact.mean,exact.variance,naive,
                             {'law':'fixed reference','naive_class_pairs':len(cache)})
    if model=='independent_pmf':
        if pmf is None: raise ValueError('pmf is required')
        if len(pmf)==q and all(isinstance(p,(int,Fraction)) for p in pmf):
            return iid_moments(q,design,law='pmf',pmf=pmf)
        if len(pmf)!=design.n: raise ValueError('PMF row count differs from graph')
        rows=tuple(_rational_row(row,q) for row in pmf)
        exact=moments_independent_exact(rows,design)
        pair_design=prepare_graph(2,[(0,1,1)]);cache={};naive=Fraction()
        for i,j,c in design.edges:
            if c<0: raise ValueError('naive diagnostic requires nonnegative multiplicities')
            key=tuple(sorted((rows[i],rows[j])))
            if key not in cache: cache[key]=moments_independent_exact(key,pair_design).variance
            naive+=c*cache[key]
        return MomentSummary(exact.mean,exact.variance,naive,
                             {'law':'independent specified PMFs','naive_class_pairs':len(cache)})
    raise ValueError(f'unknown null model {model!r}')


def wilson_interval(successes,trials,confidence=.95):
    trials=_integer(trials,'trials',1);successes=_integer(successes,'successes',0)
    if successes>trials or not 0<confidence<1: raise ValueError('invalid binomial interval input')
    z=NormalDist().inv_cdf((1+confidence)/2)
    phat=successes/trials;den=1+z*z/trials
    center=(phat+z*z/(2*trials))/den
    half=z*math.sqrt(phat*(1-phat)/trials+z*z/(4*trials*trials))/den
    return (0. if successes==0 else max(0.,center-half),
            1. if successes==trials else min(1.,center+half))


def _tie_values(n,rng,ties):
    if ties is None:
        if rng is None: raise ValueError('supply an RNG or explicit tie uniforms')
        ties=rng.random(n)
    result=np.asarray(ties,dtype=np.float64)
    if result.shape!=(n,) or np.any(result<0) or np.any(result>=1) or not np.all(np.isfinite(result)):
        raise ValueError('ties must be finite uniforms in [0,1) with matching length')
    return result


def exact_discrepancy(raw, center):
    """Integer keys proportional to |raw-center|, without rounding the center.

    The returned key is abs(center.denominator*raw-center.numerator). int64
    is used only when both multiplication and subtraction are safe. Otherwise
    Python integers in an object array preserve the complete order and ties.
    """
    values=np.asarray(raw)
    if values.ndim!=1:
        raise ValueError('raw statistics must be a one-dimensional integer vector')
    if values.dtype==object:
        if any(not isinstance(x,(int,np.integer)) or isinstance(x,(bool,np.bool_)) for x in values):
            raise TypeError('raw statistics must be integer-valued')
    elif not np.issubdtype(values.dtype,np.integer):
        raise TypeError('raw statistics must be integer-valued')
    center=Fraction(center);den,num=center.denominator,center.numerator
    if not len(values): return np.empty(0,dtype=np.int64)
    low,high=int(values.min()),int(values.max())
    limit=np.iinfo(np.int64).max
    intermediate=max(abs(den*low),abs(den*high),abs(num))
    final=max(abs(den*low-num),abs(den*high-num))
    if max(intermediate,final)<=limit:
        return np.abs(values.astype(np.int64)*den-num)
    return np.asarray([abs(den*int(x)-num) for x in values],dtype=object)


def _rank_values(discrepancies, nonempty=False):
    values=np.asarray(discrepancies)
    if values.ndim!=1 or (nonempty and not len(values)):
        raise ValueError('discrepancies must be a one-dimensional vector')
    if values.dtype==object:
        if any(not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) for v in values):
            raise TypeError('object discrepancies must contain exact integers')
    elif np.issubdtype(values.dtype,np.integer):
        pass
    elif np.issubdtype(values.dtype,np.floating):
        if not np.all(np.isfinite(values)):
            raise ValueError('discrepancies must be finite')
    else:
        raise TypeError('discrepancies must be integer or floating-point values')
    return values


def fit_rank_threshold(discrepancies,alpha,rng=None,*,ties=None):
    """Fit upper-tail lexicographic order cut with marginal rank level k/(m+1).

    Integer discrepancy keys retain their complete order and ties, including
    arbitrary-size Python integer keys. For randomized scores, the conditional
    null rejection level is Beta(k,m+1-k) across independently simulated banks.
    A reused bank does not give independent unconditional test decisions.
    """
    values=_rank_values(discrepancies,nonempty=True)
    level=alpha if isinstance(alpha,Fraction) else Fraction(str(alpha))
    if not 0<level<1: raise ValueError('alpha must lie strictly between zero and one')
    m=len(values);k=(level.numerator*(m+1))//level.denominator
    if not 1<=k<=m:
        raise ValueError('calibration sample size cannot represent the requested level')
    uniforms=_tie_values(m,rng,ties)
    rank=m+1-k
    order=np.lexsort((uniforms,values));idx=order[rank-1]
    value=values[idx].item() if isinstance(values[idx],np.generic) else values[idx]
    return RankThreshold(value,float(uniforms[idx]),m,rank,k,
                         Fraction(k,m+1),k,m+1-k)


def reject_rank_threshold(discrepancies,threshold,rng=None,*,ties=None):
    values=_rank_values(discrepancies)
    uniforms=_tie_values(len(values),rng,ties)
    return (values>threshold.value)|((values==threshold.value)&(uniforms>threshold.tie))


def rank_size_interval(threshold,confidence=.95):
    """Central distribution interval for random conditional null size across banks."""
    if not 0<confidence<1: raise ValueError('confidence must lie in (0,1)')
    from scipy.stats import beta
    tails=(1-confidence)/2
    return tuple(float(x) for x in beta.ppf([tails,1-tails],threshold.beta_a,threshold.beta_b))
