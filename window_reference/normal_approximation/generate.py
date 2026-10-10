"""Evaluate numerical local-dependence bound examples, without simulation."""
import csv
import json
from fractions import Fraction as F
from math import sqrt
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'windows'))
from core import window_moments


def lower_residual_bound(q):
    ell=q//2+1
    return F((ell*ell-1)*(2*ell*ell+7),45*q*q)


def main():
    destination=Path(__file__).resolve().parent/'generated'
    destination.mkdir(exist_ok=True)
    examples=[]
    for q,M,N,w,R in [(256,256,256,2,1),(256,256,256,2,100),
                      (2,256,256,2,1)]:
        moments=window_moments(M,N,w,q)
        K=moments['window_count']
        J=moments['pairs_per_window']
        beta=lower_residual_bound(q)
        score_normalizer=K*J*(q-1)
        variance_sum=moments['variance']*score_normalizer**2
        assert variance_sum>=beta*moments['pair_squares']>=beta*K*J
        factor=75*(10*w-9)**4
        raw=factor*K*(J*(q-1))**3/(sqrt(R)*float(variance_sum)**1.5)
        uniform=factor*(J*(q-1)**2/float(beta))**1.5/sqrt(R*K)
        assert raw<=uniform
        assert min(1.0,raw)==1.0
        examples.append({'q':q,'rows':M,'columns':N,'window_width':w,
                         'independent_image_replications':R,'window_count':K,
                         'pairs_per_window':J,'residual_bound':str(beta),
                         'unnormalized_sum_variance':str(variance_sum),
                         'actual_variance_raw_cdf_bound':raw,
                         'uniform_raw_cdf_bound':uniform,
                         'clipped_cdf_bound':min(1.0,raw)})
    with (destination/'normal_approximation_bound_examples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(examples[0]))
        writer.writeheader();writer.writerows(examples)
    metadata={'status':'PASS','example_count':len(examples),
              'model':'Two independent iid uniform images; complete-pair sliding square windows.',
              'arithmetic':'Exact reference variances and residual bounds; floating-point bound evaluations.',
              'quantity':'Upper bound on the supremum difference between standardized-score and normal CDFs.',
              'interpretation':'All clipped bounds equal 1. These examples provide no nontrivial finite-sample Gaussian accuracy or tail certification.',
              'monte_carlo_trials':0}
    (destination/'normal_approximation_bound_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':
    main()
