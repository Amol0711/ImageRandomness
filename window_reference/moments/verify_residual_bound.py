"""Independently check the common-support lower bound for edge residuals.

All conditional calculations enumerate underlying uniform random variables.
Python's standard library is sufficient.
"""
import csv
import json
from fractions import Fraction as F
from pathlib import Path

OUT=Path(__file__).resolve().parent/'data'

def uniform_residual(ell):
    # Two-way centered distance matrix. This is an independent finite sum,
    # without using the claimed closed form for either conditional variance.
    matrix=[[abs(x-y) for y in range(ell)] for x in range(ell)]
    rows=[sum(r) for r in matrix]
    cols=[sum(matrix[x][y] for x in range(ell)) for y in range(ell)]
    total=sum(rows)
    numerator=sum((ell*ell*matrix[x][y]-ell*rows[x]-ell*cols[y]+total)**2
                  for x in range(ell) for y in range(ell))
    return F(numerator,ell**6)

def fixed_residual(q,a,b):
    # Enumerate the original independent U and V, not the transformed laws.
    matrix=[[abs(abs(a-u)-abs(b-v)) for v in range(q)] for u in range(q)]
    rows=[sum(r) for r in matrix]
    cols=[sum(matrix[u][v] for u in range(q)) for v in range(q)]
    total=sum(rows)
    numerator=sum((q*q*matrix[u][v]-q*rows[u]-q*cols[v]+total)**2
                  for u in range(q) for v in range(q))
    return F(numerator,q**6)

def triangular_residual(q):
    # Construct all q^2 difference outcomes directly to get the weights.
    weights=[0]*q
    for u in range(q):
        for v in range(q):
            weights[abs(u-v)]+=1
    denom=q*q
    row=[sum(weights[y]*abs(x-y) for y in range(q)) for x in range(q)]
    total=sum(weights[x]*row[x] for x in range(q))
    numerator=sum(weights[x]*weights[y]*(denom*denom*abs(x-y)-denom*row[x]-denom*row[y]+total)**2
                  for x in range(q) for y in range(q))
    return F(numerator,denom**6)

checks=0
rows=[]
minimum_ratio=None
for ell in list(range(2,31))+[32,41,64,128,129,256]:
    exact=uniform_residual(ell)
    formula=F((ell*ell-1)*(2*ell*ell+7),45*ell*ell)
    assert exact==formula,(ell,exact,formula)
    checks+=1
for q in range(2,15):
    ell=q//2+1
    beta=F((ell*ell-1)*(2*ell*ell+7),45*q*q)
    for a in range(q):
        for b in range(q):
            exact=fixed_residual(q,a,b)
            assert exact>=beta,(q,a,b,exact,beta)
            checks+=1
            minimum_ratio=exact/beta if minimum_ratio is None else min(minimum_ratio,exact/beta)
            rows.append({'q':q,'a':a,'b':b,'residual_variance':str(exact),
                         'uniform_common_support_bound':str(beta),'ratio':str(exact/beta)})
    exact=triangular_residual(q)
    assert exact>=beta,(q,exact,beta)
    checks+=1
# Larger alphabets check the random/random law and representative fixed laws.
for q in [32,41,64,128,256]:
    ell=q//2+1
    beta=F((ell*ell-1)*(2*ell*ell+7),45*q*q)
    exact=triangular_residual(q)
    assert exact>=beta,(q,exact,beta)
    checks+=1
    for a,b in [(0,0),(0,q//2),(q//2,q//2),(q//3,q-1)]:
        exact=fixed_residual(q,a,b)
        assert exact>=beta,(q,a,b,exact,beta)
        checks+=1
        rows.append({'q':q,'a':a,'b':b,'residual_variance':str(exact),
                     'uniform_common_support_bound':str(beta),'ratio':str(exact/beta)})
OUT.mkdir(exist_ok=True)
with (OUT/'common_support_residual_bound.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'status':'passed','exact_assertions':checks,'fixed_law_designs':len(rows),
         'uniform_alphabets':35,'triangular_alphabets':18,
         'minimum_ratio_among_exhaustive_small_fixed_laws':str(minimum_ratio),
         'formula':'ell=floor(q/2)+1; beta=(ell^2-1)*(2*ell^2+7)/(45*q^2)',
         'method':'Exact two-way centering of the raw independent-uniform distance matrix.'}
(OUT/'residual_bound_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
