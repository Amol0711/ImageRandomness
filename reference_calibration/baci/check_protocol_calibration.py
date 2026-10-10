"""Checks of boundary conventions, exact moments, and aggregate arithmetic."""
import itertools
import json
import math
import argparse
import sys
from pathlib import Path
from fractions import Fraction
from protocol_calibration import (exact_moments, poisson_binomial, count_tail,
    aggregate_cutoff, strict_majority, interval_pass, relative_pass,
    joint_rule_probabilities, generate)


def close(a, b, tol=2e-13):
    assert abs(a-b) <= tol, (a,b)


def main(calibration_root=None):
    checks = 0
    for q in (2,3,4,5,16,256):
        c = exact_moments(q)
        assert c['mu_pair'] == Fraction(4*(q**4-1),15*q**3)
        assert c['A'] == Fraction(9*q**8+25*q**6-18*q**4-16,225*q**6)
        assert c['C'] == Fraction((q*q-4)*(q*q-1)*(3*q**4+20*q*q-8),450*q**6)
        checks += 3
    for probs in ([0,1], [Fraction(1,3),Fraction(2,5),Fraction(4,7)],
                  [Fraction(99,100)]*4):
        expected = [Fraction(0)]*(len(probs)+1)
        for outcomes in itertools.product((0,1),repeat=len(probs)):
            expected[sum(outcomes)] += math.prod(p if s else 1-p for p,s in zip(probs,outcomes))
        actual = poisson_binomial(list(map(float,probs)))
        for a,b in zip(actual,expected): close(a,float(b)); checks+=1
    assert not strict_majority(10,20) and strict_majority(11,20)
    assert not strict_majority(7,15) and strict_majority(8,15)
    assert interval_pass(10,10,1) and not interval_pass(9,10,1) and not interval_pass(11,10,1)
    assert relative_pass(1000,1000) and not relative_pass(999,1000) and not relative_pass(1001,1000)
    checks += 10
    # Exhaust all four elementary categories, with both nesting directions
    # appearing within one heterogeneous vector and endpoint probabilities.
    for ps,ts in [([.2,.8,.4],[.7,.3,.4]), ([0,1,.2],[1,0,.9]),
                  ([.99]*4,[.998]*4), ([.99]*4,[.9]*4)]:
        cats = [[(1,1,min(p,t)),(1,0,max(p-t,0)),(0,1,max(t-p,0)),(0,0,1-max(p,t))]
                for p,t in zip(ps,ts)]
        for threshold in range(len(ps)+2):
            direct = {k:0.0 for k in ('interval_accept','interval_reject','tolerance_accept',
                'tolerance_reject','intersection_accept','intersection_reject','union_accept','union_reject')}
            for outcome in itertools.product(*cats):
                mass = math.prod(x[2] for x in outcome)
                A = sum(x[0] for x in outcome)>=threshold
                B = all(x[1] for x in outcome)
                for name,test in [('interval',A),('tolerance',B),('intersection',A and B),('union',A or B)]:
                    direct[name+('_accept' if test else '_reject')] += mass
            got = joint_rule_probabilities(ps,ts,threshold)
            for key in got: close(got[key],direct[key]); checks+=1
    # Exact rational binomial sums independently check minimality and bounds.
    for m in (1,2,15,20,100):
        for a in (Fraction(0),Fraction(1,100),Fraction(1,20),Fraction(1)):
            for eta in (Fraction(0),Fraction(1,100),Fraction(1,20)):
                c,b = aggregate_cutoff(m,float(a),float(eta))
                tail = lambda k: sum(Fraction(math.comb(m,j))*a**j*(1-a)**(m-j) for j in range(k,m+1))
                assert tail(c) <= eta
                assert c == 1 or tail(c-1) > eta
                close(b,float(tail(c)))
                checks += 3
    api_checks = 0
    if calibration_root is not None:
        sys.path.insert(0,str(Path(calibration_root).resolve()))
        from calibration.core import iid_baci_moments
        c = exact_moments()
        api = iid_baci_moments(256,256,256,exact=True)
        for a,b in [('mean','mean'),('variance','variance'),
                    ('pair_variance','independent_pair_variance'),
                    ('A','A'),('C','C'),('K','K'),('S2','S2'),('W','W')]:
            assert c[a] == getattr(api,b), (a,c[a],getattr(api,b))
            api_checks += 1
    print(json.dumps(dict(status='PASS',checks=checks,api_exact_equalities=api_checks,
                          generated=generate()),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-root',help='optional directory containing calibration/core.py')
    main(parser.parse_args().calibration_root)
