#!/usr/bin/env python3
"""Recompute finite-multinomial entropy moments and reference intervals.

Numerical inputs are transcribed from Wu et al., Information
Sciences 222 (2013), Tables 3--4, DOI 10.1016/j.ins.2012.07.049.
The normal intervals are approximate. Binary rejection probabilities are
computed exactly as rational numbers with denominator 2**k.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from math import comb
from pathlib import Path
import sys

# Use the optional local dependency directory when present.
if (Path(__file__).resolve().parent / '_deps').is_dir():
    sys.path.insert(0, str(Path(__file__).resolve().parent / '_deps'))
import mpmath as mp

BASE = Path(__file__).resolve().parent
ALPHAS = ('0.05', '0.01', '0.001')


def entropy_terms(T):
    return [mp.mpf(0)] + [-mp.mpf(n)/T * mp.log(mp.mpf(n)/T, 2)
                         for n in range(1, T+1)]


def binomial_pmf(n, p):
    if p == 1:
        return [mp.mpf(0)] * n + [mp.mpf(1)]
    values = [(1-p)**n]
    for j in range(1, n+1):
        values.append(values[-1] * (n-j+1)/j * p/(1-p))
    return values


def moments(q, T, method='joint'):
    """Full finite sums, without a count-tail truncation.

    H=sum_l f(N_l), with f(n)=-(n/T)log2(n/T). Marginal and
    two-category multinomial moments give E[H] and E[H**2].
    Two independent organizations of the joint sum are implemented.
    """
    f = entropy_terms(T)
    marginal = binomial_pmf(T, mp.mpf(1)/q)
    mean = q * mp.fsum(f[n]*marginal[n] for n in range(T+1))
    diagonal = q * mp.fsum(f[n]**2*marginal[n] for n in range(T+1))
    if q == 2:
        cross = mp.fsum(f[a]*f[T-a]*marginal[a] for a in range(T+1))
    elif method == 'joint':
        # P(N1=a,N2=0), then recurrence across N2. Exploit symmetry.
        p_a0 = (mp.mpf(q-2)/q)**T
        cross = mp.mpf(0)
        for a in range(1, T):
            p_a0 *= mp.mpf(T-a+1)/(a*(q-2))
            p_ab = p_a0
            for b in range(1, min(a,T-a)+1):
                p_ab *= mp.mpf(T-a-b+1)/(b*(q-2))
                cross += (1 if a == b else 2)*f[a]*f[b]*p_ab
    elif method == 'conditioned_total':
        # S=N1+N2 is Bin(T,2/q), N1|S=s is Bin(s,1/2).
        total = binomial_pmf(T, mp.mpf(2)/q)
        cross = mp.mpf(0)
        for s in range(2, T+1):
            weight = mp.mpf(2)**(-s)
            conditional = mp.mpf(0)
            for a in range(1, s//2+1):
                weight *= mp.mpf(s-a+1)/a
                conditional += (1 if 2*a == s else 2)*f[a]*f[s-a]*weight
            cross += total[s]*conditional
    else:
        raise ValueError(method)
    second = diagonal + q*(q-1)*cross
    variance = second - mean**2
    assert variance >= 0
    return {'mean': mean, 'variance': variance, 'standard_deviation': mp.sqrt(variance),
            'marginal_probability_sum_error': abs(mp.fsum(marginal)-1)}


def number(x, digits=40):
    return mp.nstr(x, digits, strip_zeros=False)


def read_csv(name):
    with (BASE/name).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def binary_rejection(k, lower, upper):
    accepted = [j for j in range(k+1) if lower <= mp.mpf(j)/k <= upper]
    numerator = 2**k-sum(comb(k,j) for j in accepted)
    return {'accepted_count_min': min(accepted) if accepted else None,
            'accepted_count_max': max(accepted) if accepted else None,
            'rejection_numerator': str(numerator), 'rejection_denominator': str(2**k),
            'rejection_probability': number(mp.mpf(numerator)/2**k)}


def exact_binary_bounds(k, alpha):
    """Narrowest symmetric nonrandomized acceptance set with size <= alpha.

    All k values used here are even. Rejection occurs for X < L or X > k-L.
    Tail comparisons use integers, not floating-point arithmetic.
    """
    scale = 10**len(alpha.split('.')[1]); anum = int(alpha.split('.')[1])
    numerator = 0; accepted_lower = 0
    for L in range(1, k//2+1):
        candidate = numerator + 2*comb(k,L-1)
        if candidate*scale > anum*2**k:
            break
        numerator = candidate; accepted_lower = L
    return {'accepted_count_min': accepted_lower, 'accepted_count_max': k-accepted_lower,
            'lower': number(mp.mpf(accepted_lower)/k),
            'upper': number(mp.mpf(k-accepted_lower)/k),
            'rejection_numerator': str(numerator), 'rejection_denominator': str(2**k),
            'rejection_probability': number(mp.mpf(numerator)/2**k),
            'next_narrower_rejection_numerator': str(numerator+2*comb(k,accepted_lower))}


def fixed9(value):
    return f'{float(value):.9f}'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=BASE/'generated')
    parser.add_argument('--dps',type=int,default=60)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    if args.dps < 40: raise ValueError('Use at least 40 decimal digits.')
    mp.mp.dps=args.dps
    source_moments={(int(r['q']),int(r['T'])):r for r in read_csv('source_table3.csv')}
    computed={};comparison=[]
    for q,T in source_moments:
        print(f'Recomputing q={q}, T={T}, dps={args.dps}',flush=True)
        first=moments(q,T,'joint')
        with mp.workdps(args.dps+20):
            second=moments(q,T,'conditioned_total')
        for name in ('mean','variance','standard_deviation'):
            assert abs(first[name]-second[name]) < mp.mpf(10)**(-(args.dps-12)),(q,T,name)
        computed[q,T]=second
        src=source_moments[q,T]
        comparison.append({'q':q,'T':T,'source_mean':src['mean'],
            'source_standard_deviation':src['standard_deviation'],
            **{k:number(v) for k,v in second.items()},
            'source_minus_recomputed_mean':number(mp.mpf(src['mean'])-second['mean']),
            'source_minus_recomputed_standard_deviation':number(mp.mpf(src['standard_deviation'])-second['standard_deviation']),
            'independent_variance_sum_difference':number(abs(first['variance']-second['variance']))})
    write_csv(out/'entropy_moments.csv',comparison)
    rows=[];binary=[];exact=[];errors=[]
    for sr in read_csv('source_table4.csv'):
        q,T,k=map(int,(sr['q'],sr['T'],sr['k']))
        src=source_moments[q,T]; mu0=mp.mpf(src['mean']);sd0=mp.mpf(src['standard_deviation'])
        mean=computed[q,T]['mean'];sd=computed[q,T]['standard_deviation']
        for a in ALPHAS:
            alpha=mp.mpf(a);z=mp.sqrt(2)*mp.erfinv(1-alpha)
            sl=mp.mpf(sr['lower_'+a]);su=mp.mpf(sr['upper_'+a])
            el=abs(sl-(mu0-z*sd0/k));eu=abs(su-(mu0+z*sd0/k));errors.extend((el,eu))
            assert max(el,eu)<=mp.mpf('0.0000000005')
            cl=mean-z*sd/mp.sqrt(k);cu=mean+z*sd/mp.sqrt(k)
            rl=mu0-z*sd0/mp.sqrt(k);ru=mu0+z*sd0/mp.sqrt(k)
            record={'q':q,'T':T,'k':k,'alpha':a,'normal_quantile':number(z),
                'source_lower':sr['lower_'+a],'source_upper':sr['upper_'+a],
                'source_lower_reconstruction_error':number(el),'source_upper_reconstruction_error':number(eu),
                'source_moment_corrected_lower':number(rl),'source_moment_corrected_upper':number(ru),
                'corrected_lower':number(cl),'corrected_upper':number(cu),
                'corrected_lower_9dp':fixed9(cl),'corrected_upper_9dp':fixed9(cu),
                'ideal_normal_source_scale_rejection_probability':number(mp.erfc(z/mp.sqrt(2*k)))}
            rows.append(record)
            if q==2:
                for rule,l,u in [('source_table4',sl,su),('source_moment_gaussian',rl,ru),
                                 ('recomputed_moment_gaussian',cl,cu),
                                 ('recomputed_moment_gaussian_rounded_9dp',mp.mpf(fixed9(cl)),mp.mpf(fixed9(cu)))]:
                    binary.append({'k':k,'alpha':a,'rule':rule,'lower':number(l),'upper':number(u),
                                   **binary_rejection(k,l,u)})
                exact.append({'k':k,'alpha':a,**exact_binary_bounds(k,a)})
    write_csv(out/'entropy_intervals.csv',rows)
    write_csv(out/'entropy_binary_sizes.csv',binary)
    write_csv(out/'entropy_binary_exact_intervals.csv',exact)
    metadata={'source':{'citation':'Y. Wu, Y. Zhou, G. Saveriades, S. Agaian, J. P. Noonan, P. Natarajan, Local Shannon entropy measure with statistical tests for image randomness, Information Sciences 222 (2013) 323–342',
        'doi':'10.1016/j.ins.2012.07.049','numerical_inputs':'Tables 3 and 4, printed page 334',
        'table4_rows':26,'intervals':78,'endpoints':156,
        'transcription_check':'All 26 rows compared with extracted full text and visually inspected on source page 334.',
        'source_table3_sha256':hashlib.sha256((BASE/'source_table3.csv').read_bytes()).hexdigest(),
        'source_table4_sha256':hashlib.sha256((BASE/'source_table4.csv').read_bytes()).hexdigest()},
        'precision':{'primary_dps':args.dps,'independent_dps':args.dps+20,'output_significant_digits':40,
        'table_decimal_places':9,'method':'Complete marginal and bivariate multinomial finite sums; no tail truncation',
        'independent_method':'Condition on the combined count of two symbols, then use its binomial split',
        'normal_quantiles':'mpmath erfinv at primary_dps'},
        'assumptions':['Uniform independent scalar entries','k independent blocks with T scalar entries each',
        'The q=256 RGB interpretation uses 8-bit scalar channel entries, not a 24-bit alphabet or T RGB triples',
        'Binary T=2 block entropy is Bernoulli(1/2)','Endpoints define a closed acceptance interval'],
        'interpretation':{'corrected_intervals':'Gaussian approximations using finite-multinomial moments',
        'binary_sizes':'Exact rational binomial sums, with decimal rendering',
        'binary_exact_intervals':'Narrowest symmetric nonrandomized acceptance interval with size no greater than alpha',
        'binary_exact_interval_implementation':'Use inclusive integer accepted_count_min and accepted_count_max; decimal entropy endpoints are display values',
        'ideal_normal_source_scale_rejection_probability':'Normal-law diagnostic erfc(z/sqrt(2k)); not an exact finite-k size'},
        'checks':{'all_source_endpoints_match_inverse_k_scale_to_nearest_9dp':True,
                  'maximum_source_endpoint_reconstruction_error':number(max(errors)),
                  'independent_full_moment_sums_agree':True},
        'software':{'python':sys.version.split()[0],'mpmath':mp.__version__}}
    (out/'entropy_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata['checks'],indent=2),flush=True)

if __name__ == '__main__':
    main()
