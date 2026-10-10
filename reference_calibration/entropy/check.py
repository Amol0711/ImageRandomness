#!/usr/bin/env python3
"""Independent numerical and exact-discrete checks for generated tables."""
from __future__ import annotations
import csv
from itertools import product
import json
import math
from pathlib import Path
import sys
import generate as g

mp=g.mp
BASE=Path(__file__).resolve().parent


def main():
    mp.mp.dps=60
    small=[]
    for q,Ts in ((2,range(2,9)),(3,range(2,7)),(4,range(2,6))):
        for T in Ts:
            values=[]
            f=[0]+[-n/T*math.log2(n/T) for n in range(1,T+1)]
            for image in product(range(q),repeat=T):
                counts=[image.count(i) for i in range(q)]
                values.append(math.fsum(f[n] for n in counts))
            mean=math.fsum(values)/len(values)
            var=math.fsum((x-mean)**2 for x in values)/len(values)
            joint=g.moments(q,T,'joint');conditional=g.moments(q,T,'conditioned_total')
            assert abs(float(joint['mean'])-mean)<3e-15
            assert abs(float(joint['variance'])-var)<3e-15
            assert abs(joint['variance']-conditional['variance'])<mp.mpf('1e-55')
            small.append({'q':q,'T':T,'enumerated_images':len(values)})
    def read(name):
        return list(csv.DictReader((BASE/'generated'/name).open()))
    intervals=read('entropy_intervals.csv');sizes=read('entropy_binary_sizes.csv')
    exact=read('entropy_binary_exact_intervals.csv')
    assert len(intervals)==78 and len(sizes)==156 and len(exact)==39
    assert len({(r['q'],r['T'],r['k'],r['alpha']) for r in intervals})==78
    for r in sizes:
        k=int(r['k']);L=int(r['accepted_count_min']);U=int(r['accepted_count_max'])
        # Sum rejected counts directly, independently of generator's complement.
        numerator=sum(math.comb(k,j) for j in range(k+1) if j<L or j>U)
        assert numerator==int(r['rejection_numerator'])
        assert int(r['rejection_denominator'])==2**k
    for r in exact:
        k=int(r['k']);L=int(r['accepted_count_min']);U=int(r['accepted_count_max'])
        assert U==k-L
        rejected=sum(math.comb(k,j) for j in range(k+1) if j<L or j>U)
        narrower=sum(math.comb(k,j) for j in range(k+1) if j<L+1 or j>U-1)
        a=r['alpha'];scale=10**len(a.split('.')[1]);a_int=int(a.split('.')[1])
        assert rejected==int(r['rejection_numerator'])
        assert rejected*scale<=a_int*2**k<narrower*scale
    # Binary source input and finite-moment normal intervals must agree exactly.
    for r in intervals:
        assert mp.mpf(r['corrected_lower'])<mp.mpf(r['corrected_upper'])
        if r['q']=='2':
            assert r['source_moment_corrected_lower']==r['corrected_lower']
            assert r['source_moment_corrected_upper']==r['corrected_upper']
    groups={}
    for r in sizes:
        groups.setdefault((r['k'],r['alpha']),{})[r['rule']]=r
    for group in groups.values():
        assert group['recomputed_moment_gaussian']['rejection_numerator']==group['recomputed_moment_gaussian_rounded_9dp']['rejection_numerator']
    # Concrete discreteness check: corrected normal interval can still exceed alpha.
    r=groups['100','0.05']['recomputed_moment_gaussian']
    assert int(r['accepted_count_min'])==41 and int(r['accepted_count_max'])==59
    assert mp.mpf(r['rejection_probability'])>mp.mpf('0.05')
    report={'status':'PASS','small_alphabets_exhaustive':small,
       'total_enumerated_images':sum(r['enumerated_images'] for r in small),
       'source_interval_count':len(intervals),'exact_binary_rule_sizes_checked':len(sizes),
       'exact_discrete_acceptance_sets_checked':len(exact),
       'all_exact_discrete_sets_control_size_and_are_narrowest_symmetric':True,
       'nine_decimal_rounding_preserves_all_binary_gaussian_decisions':True,
       'binary_normal_oversize_at_k100_alpha005_confirmed':True}
    (BASE/'generated'/'entropy_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
