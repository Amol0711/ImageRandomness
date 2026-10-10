"""Independent exact checks for fixed-image reference moments.

Uses literal windows, direct uniform-symbol enumeration, and edge-pair
covariance expansion. It does not call the tested module to form reference
values. Python integers and fractions are used throughout the reference path.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
import importlib.util
import json
from pathlib import Path
import time


def window_pairs(rows, cols, width=2):
    return [tuple(combinations(tuple((r+i)*cols+c+j for i in range(width)
                                      for j in range(width)), 2))
            for r in range(rows-width+1) for c in range(cols-width+1)]


def literal_numerator(differences, rows, cols, width=2):
    return sum(abs(differences[i]-differences[j])
               for pairs in window_pairs(rows, cols, width) for i,j in pairs)


def exhaustive_oracle(image, q, width=2):
    """Enumerate comparison images, retaining all repeated window pairs."""
    rows, cols = len(image), len(image[0])
    flat = tuple(value for row in image for value in row)
    windows = window_pairs(rows, cols, width)
    window_totals = [0]*len(windows)
    window_squares = [0]*len(windows)
    pair_totals = [[0]*len(pairs) for pairs in windows]
    pair_squares = [[0]*len(pairs) for pairs in windows]
    total = squares = 0
    states = q**len(flat)
    for comparison in product(range(q), repeat=len(flat)):
        d = tuple(abs(a-b) for a,b in zip(flat, comparison))
        raw = 0
        for wi,pairs in enumerate(windows):
            subtotal = 0
            for ei,(i,j) in enumerate(pairs):
                distance = abs(d[i]-d[j])
                subtotal += distance
                pair_totals[wi][ei] += distance
                pair_squares[wi][ei] += distance*distance
            window_totals[wi] += subtotal
            window_squares[wi] += subtotal*subtotal
            raw += subtotal
        total += raw
        squares += raw*raw
    mean = Fraction(total, states)
    variance = Fraction(squares, states)-mean**2
    independent = sum((Fraction(b,states)-Fraction(a,states)**2
                        for aa,bb in zip(pair_totals,pair_squares)
                        for a,b in zip(aa,bb)), Fraction())
    within = sum((Fraction(b,states)-Fraction(a,states)**2
                    for a,b in zip(window_totals,window_squares)), Fraction())
    normalization = len(windows)*len(windows[0])*(q-1)
    return {'mean':mean/normalization, 'variance':variance/normalization**2,
            'independent_pair_variance':independent/normalization**2,
            'within_window_variance':within/normalization**2,
            'raw_mean':mean,'raw_variance':variance, 'states':states,
            'normalization':normalization}


@lru_cache(maxsize=None)
def uniform_endpoint_table(center, neighbor, q):
    """Conditional first moment numerator indexed by the actual uniform draw."""
    return tuple(sum(abs(abs(center-u)-abs(neighbor-v)) for v in range(q))
                 for u in range(q))


@lru_cache(maxsize=None)
def pair_moments(a,b,q):
    values = [abs(abs(a-u)-abs(b-v)) for u in range(q) for v in range(q)]
    mean = Fraction(sum(values),q*q)
    return mean, Fraction(sum(x*x for x in values),q*q)-mean**2


@lru_cache(maxsize=None)
def shared_covariance(a,b,c,q):
    left = uniform_endpoint_table(a,b,q)
    right = uniform_endpoint_table(a,c,q)
    return (Fraction(sum(x*y for x,y in zip(left,right)),q**3)
            -Fraction(sum(left)*sum(right),q**4))


def covariance_oracle(image,q,width=2):
    """Expand distinct-edge covariances; no projection/residual formula."""
    rows,cols = len(image),len(image[0])
    flat = tuple(value for row in image for value in row)
    windows = window_pairs(rows,cols,width)
    weights = Counter(pair for pairs in windows for pair in pairs)
    incident = [[] for _ in flat]
    mean = Fraction()
    variance = Fraction()
    independent = Fraction()
    for (i,j),weight in weights.items():
        mu,var = pair_moments(flat[i],flat[j],q)
        mean += weight*mu
        variance += weight*weight*var
        independent += weight*var
        incident[i].append((j,weight))
        incident[j].append((i,weight))
    shared_total = Fraction()
    negative = 0
    for i,edges in enumerate(incident):
        for (j,wj),(k,wk) in combinations(edges,2):
            covariance = shared_covariance(flat[i],flat[j],flat[k],q)
            shared_total += 2*wj*wk*covariance
            negative += covariance < 0
    variance += shared_total
    within = independent
    for pairs in windows:
        for first,second in combinations(pairs,2):
            common = set(first).intersection(second)
            if len(common)==1:
                i = next(iter(common))
                j = next(x for x in first if x!=i)
                k = next(x for x in second if x!=i)
                within += 2*shared_covariance(flat[i],flat[j],flat[k],q)
    normalization = len(windows)*len(windows[0])*(q-1)
    return {'mean':mean/normalization, 'variance':variance/normalization**2,
            'independent_pair_variance':independent/normalization**2,
            'within_window_variance':within/normalization**2,
            'raw_mean':mean,'raw_variance':variance,
            'normalization':normalization,'negative_shared_covariances':negative,
            'edges':len(weights),'sum_weights':sum(weights.values()),
            'sum_squared_weights':sum(w*w for w in weights.values())}


def uaci_oracle(image,q):
    pixels = [a for row in image for a in row]
    means=[]
    variances=[]
    for a in pixels:
        values=[Fraction(abs(a-u),q-1) for u in range(q)]
        mean = sum(values,Fraction())/q
        means.append(mean)
        variances.append(sum((v*v for v in values),Fraction())/q-mean*mean)
    n=len(pixels)
    mean=sum(means,Fraction())/n
    variance=sum(variances,Fraction())/n**2
    excess=sum(((m-mean)**2 for m in means),Fraction())/n**2
    return {'mean':mean,'variance':variance,'pooled_variance':variance+excess,
            'pooling_excess':excess}


def pack(value):
    if isinstance(value,Fraction):
        return {'numerator':value.numerator,'denominator':value.denominator,
                'decimal':float(value)}
    if isinstance(value,dict): return {k:pack(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [pack(v) for v in value]
    return value


def comparison_cases():
    cases=[]
    for q in (2,3,4):
        for vals in product(range(q),repeat=4):
            cases.append((f'all_{q}_2x2_'+''.join(map(str,vals)),[list(vals[:2]),list(vals[2:])],q,2))
    for q in (2,3):
        cases += [(f'strip_{q}',[[0,q-1,0],[q-1,0,q-1]],q,2),
                  (f'ramp_{q}',[[0,1,0],[1,q-1,0]],q,2),
                  (f'transpose_{q}',[[0,q-1],[1,0],[0,1]],q,2)]
    cases += [('binary_3x3',[[0,1,0],[1,0,1],[0,1,0]],2,2),
              ('width3_single',[[0,1,0],[1,0,1],[0,1,0]],2,3),
              ('width3_overlap',[[0,1,0,1],[1,0,1,0],[0,1,0,1]],2,3),
              ('width3_overlap_transpose',[[0,1,0],[1,0,1],[0,1,0],[1,0,1]],2,3),
              ('width4_single',[[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]],2,4)]
    return cases


def run_oracles_only():
    cases=comparison_cases()
    checked=[]
    states=0
    for name,image,q,width in cases:
        exhaustive=exhaustive_oracle(image,q,width)
        incidence=covariance_oracle(image,q,width)
        for key in ('mean','variance','independent_pair_variance','within_window_variance'):
            assert exhaustive[key]==incidence[key],(name,key)
        states+=exhaustive['states']
        checked.append({'name':name,'q':q,'width':width,'rows':len(image),'cols':len(image[0]),
                        'states':exhaustive['states'],'mean':exhaustive['mean'],
                        'variance':exhaustive['variance']})
    return {'status':'PASS','fixed_cases':len(cases),'uniform_comparison_images':states,
            'comparators':['full','independent_pair','within_window'], 'cases':checked}


def load_module(path):
    import sys
    spec=importlib.util.spec_from_file_location('tested_fixed_image_moments',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def run_api(module):
    import numpy as np
    cases=comparison_cases()
    comparisons=0
    comparison_states=0
    for name,image,q,width in cases:
        if width!=2:
            continue
        expected=exhaustive_oracle(image,q,width)
        actual=module.fixed_baci_moments(image,q=q)
        for key in ('mean','variance'):
            assert getattr(actual,key)==expected[key],(name,key)
        assert actual.pair_occurrence_variance==expected['independent_pair_variance'],name
        assert actual.vertex_projection_variance+actual.edge_residual_variance==actual.variance,name
        u_expected=uaci_oracle(image,q)
        u_actual=module.fixed_uaci_moments(image,q=q)
        for key in ('mean','variance','pooled_variance'):
            assert getattr(u_actual,key)==u_expected[key],(name,'uaci',key)
        assert u_actual.pooled_excess==u_expected['pooling_excess'],name
        comparisons+=1
        comparison_states+=expected['states']
    edge_cases=[('negative_covariances',[[0,54,127,128],[201,255,54,0],[127,54,201,255]],256),
                ('q7_minimal',[[0,1],[3,6]],7),
                ('q16_strip',[[0,1,7,8,14,15],[15,14,8,7,1,0]],16),
                ('q512_int64_path',[[0,255],[256,511]],512),
                ('q512_long_strip_reduction',[[0]*256,[511]*256],512),
                ('q1024_bigint_path',[[512,512],[511,511]],1024)]
    records=[]
    for name,image,q in edge_cases:
        expected=covariance_oracle(image,q)
        actual=module.fixed_baci_moments(np.array(image,dtype=np.int64),q=q)
        assert actual.mean==expected['mean'],name
        assert actual.variance==expected['variance'],name
        assert actual.pair_occurrence_variance==expected['independent_pair_variance'],name
        assert actual.raw_mean==expected['raw_mean'],name
        for transformed in (np.array(image).T,
                            q-1-np.array(image),
                            np.where(np.indices(np.shape(image)).sum(axis=0)%2,
                                     q-1-np.array(image),image)):
            invariant=module.fixed_baci_moments(transformed,q=q)
            assert invariant.mean==actual.mean,(name,'invariant_mean')
            assert invariant.variance==actual.variance,(name,'invariant_variance')
        repeated=module.fixed_baci_moments(image,q=q,repeats=7)
        assert repeated.mean==actual.mean and repeated.raw_mean==actual.raw_mean,name
        for field in ('variance','pair_occurrence_variance','independent_edge_variance',
                      'vertex_projection_variance','edge_residual_variance'):
            assert getattr(repeated,field)==getattr(actual,field)/7,(name,'repeated',field)
        rounded=module.fixed_baci_moments(image,q=q,exact=False)
        assert rounded.mean==float(actual.mean) and rounded.variance==float(actual.variance),name
        u_expected=uaci_oracle(image,q)
        u_actual=module.fixed_uaci_moments(image,q=q,repeats=7)
        assert u_actual.mean==u_expected['mean']
        assert u_actual.variance==u_expected['variance']/7
        assert u_actual.pooled_variance==u_expected['pooled_variance']/7
        assert u_actual.pooled_excess==u_expected['pooling_excess']/7
        if name=='q512_long_strip_reduction':
            one_row_numerator=actual.vertex_projection_variance*actual.normalization**2*q**5/2
            assert one_row_numerator.denominator==1 and one_row_numerator>2**63-1
            assert actual.arithmetic.startswith('checked int64')
        records.append({'name':name,'q':q,'arithmetic':actual.arithmetic,
                        'mean':actual.mean,'variance':actual.variance,
                        'negative_shared_covariances':expected['negative_shared_covariances']})
    image=[[ [0,54,127,128,201,255][(i+2*j)%6] for j in range(64)] for i in range(64)]
    expected=covariance_oracle(image,256)
    actual=module.fixed_baci_moments(np.array(image,dtype=np.uint8),q=256)
    assert actual.mean==expected['mean'] and actual.variance==expected['variance']
    raw_q5=expected['raw_variance']*256**5
    assert raw_q5.denominator==1 and raw_q5>2**63-1
    assert actual.arithmetic.startswith('checked int64')
    records.append({'name':'64x64_totals_beyond_int64','q':256,
                    'arithmetic':actual.arithmetic,'mean':actual.mean,
                    'variance':actual.variance,'raw_variance_numerator_over_q5':raw_q5.numerator,
                    'negative_shared_covariances':expected['negative_shared_covariances']})
    # A UACI equality case has two different reference values but one reflected law.
    for q in (2,3,4,7,256):
        reflected=[[0,q-1],[q-1,0]]
        u=module.fixed_uaci_moments(reflected,q=q)
        assert u.pooled_excess==0 and u.pooled_variance==u.variance
    negative=shared_covariance(54,0,127,256)
    assert negative==Fraction(-151916170599,1073741824)
    invalid=[({'image':[[0,0],[0,0]],'q':True},TypeError),
             ({'image':[[0,0],[0,0]],'q':2.0},TypeError),
             ({'image':[[0,0],[0,0]],'q':1},ValueError),
             ({'image':[[False,True],[True,False]],'q':2},TypeError),
             ({'image':[[0.,1.],[1.,0.]],'q':2},TypeError),
             ({'image':[[0,1]],'q':2},ValueError),
             ({'image':[[0,-1],[1,0]],'q':2},ValueError),
             ({'image':[[0,2],[1,0]],'q':2},ValueError),
             ({'image':[[0,1],[1,0]],'q':2,'repeats':0},ValueError),
             ({'image':[[0,1],[1,0]],'q':2,'repeats':True},TypeError),
             ({'image':[[0,1],[1,0]],'q':2,'exact':1},TypeError)]
    for kwargs,kind in invalid:
        try:
            module.fixed_baci_moments(**kwargs)
        except kind:
            pass
        else:
            raise AssertionError(('invalid input accepted',kwargs))
    return {'status':'PASS','exhaustive_fixed_cases':comparisons,
            'enumerated_uniform_comparison_images':comparison_states,
            'independent_covariance_cases':len(records),'input_rejection_cases':len(invalid),
            'negative_covariance_witness':negative,'covariance_cases':records,
            'reflected_uaci_equalities':5,'checks_per_covariance_case':
            ['exact_mean','exact_variance','pair_occurrence_variance','transpose',
             'global_reflection','coordinate_reflection','repetition_scaling','float_conversion']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--module',type=Path,
        default=Path(__file__).resolve().parents[1]/'fixed_algorithm'/'fixed_image_moments.py')
    parser.add_argument('--output',type=Path,default=Path('algorithm_verification.json'))
    parser.add_argument('--oracles-only',action='store_true')
    args=parser.parse_args()
    started=time.time()
    result={'oracle_agreement':run_oracles_only()}
    if not args.oracles_only:
        result['api_agreement']=run_api(load_module(args.module))
    result['wall_seconds']=round(time.time()-started,6)
    args.output.write_text(json.dumps(pack(result),indent=2)+'\n')
    print(json.dumps({key:{k:v for k,v in value.items() if k not in ('cases','covariance_cases')}
                      if isinstance(value,dict) else value
                      for key,value in pack(result).items()},indent=2))


if __name__=='__main__': main()
