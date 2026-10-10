"""Reproduce exact numerical moment checks by independent finite sums.

Run with Python 3.10+ from any working directory. Outputs are written into data/
next to this script. Only Python's standard library is used.
"""
import csv
import json
from fractions import Fraction as F
from pathlib import Path
from moment_identities import (pair_moments, conditional_pair_mean,
    uaci_coordinate_mean, uaci_pooling_excess, shared_fixed_covariance)

OUT = Path(__file__).resolve().parent/'data'
checks = 0

def check(condition, description):
    global checks
    if not condition:
        raise AssertionError(description)
    checks += 1

moment_rows=[]
for q in list(range(2,31))+[32,41,64,128,256]:
    # Enumerate the triangular law, independently of the polynomial g formula.
    p=[F(1,q)]+[F(2*(q-d),q*q) for d in range(1,q)]
    g=[sum((p[y]*abs(x-y) for y in range(q)),F(0)) for x in range(q)]
    mu=sum((p[x]*g[x] for x in range(q)),F(0))
    e2=sum((p[x]*p[y]*(x-y)**2 for x in range(q) for y in range(q)),F(0))
    eg2=sum((p[x]*g[x]**2 for x in range(q)),F(0))
    ans=pair_moments(q)
    for x in range(q):
        check(g[x]==conditional_pair_mean(x,q),f'g direct q={q}, x={x}')
    check(ans['mean']==mu,f'mu direct q={q}')
    check(ans['variance']==e2-mu*mu,f'A direct q={q}')
    check(ans['shared_covariance']==eg2-mu*mu,f'C direct q={q}')
    check(ans['conditional_second_moment']==eg2,f'Eg2 direct q={q}')
    check(ans['variance']>0,f'A positive q={q}')
    check((ans['shared_covariance']==0)==(q==2),f'C zero iff q=2 q={q}')
    moment_rows.append({'q':q,**{k:str(v) for k,v in ans.items()}})

uaci_rows=[]
for q in range(2,15):
    direct_means=[sum((F(abs(v-u),q*(q-1)) for u in range(q)),F(0)) for v in range(q)]
    for v in range(q):
        check(direct_means[v]==uaci_coordinate_mean(v,q),f'UACI mean q={q},v={v}')
    for a in range(q):
        for b in range(q):
            means=[direct_means[a],direct_means[b]]
            mean=sum(means)/2
            direct_excess=sum((x-mean)**2 for x in means)/4
            formula=uaci_pooling_excess([a,b],q)
            check(formula==direct_excess,f'UACI pair excess q={q},a={a},b={b}')
            check((formula==0)==(a==b or a+b==q-1),f'UACI reflected equality q={q},a={a},b={b}')
            uaci_rows.append({'q':q,'a':a,'b':b,'pooling_excess':str(formula),'equality':formula==0})
    # Multi-coordinate examples ensure that divisor n is not silently replaced by n-1.
    for values in ([0], [0,q-1,0], [0,q//2,q-1],list(range(q))):
        n=len(values)
        means=[direct_means[v] for v in values]
        mean=sum(means)/n
        direct_excess=sum((x-mean)**2 for x in means)/n**2
        check(uaci_pooling_excess(values,q)==direct_excess,f'UACI vector divisor n q={q},values={values}')

# Direct conditional expectations over the underlying uniform variables.
# This path uses neither difference-law counting nor prefix sums.
def direct_covariance(shared,a,b,q):
    left=[sum(abs(d-abs(a-u)) for u in range(q)) for d in range(q)]
    right=[sum(abs(d-abs(b-u)) for u in range(q)) for d in range(q)]
    ds=[abs(shared-u) for u in range(q)]
    ma=sum(left[d] for d in ds)
    mb=sum(right[d] for d in ds)
    prod=sum(left[d]*right[d] for d in ds)
    return F(q*prod-ma*mb,q**4)

cov_rows=[]
for q in range(2,15):
    for shared,a,b in [(0,0,q-1),(q//2,0,q-1),(q//3,0,q//2),(q-1,q//2,q//2)]:
        got=shared_fixed_covariance(shared,a,b,q)
        check(got==direct_covariance(shared,a,b,q),f'fixed covariance q={q},s={shared},a={a},b={b}')
        cov_rows.append({'q':q,'shared':shared,'a':a,'b':b,'covariance':str(got)})
negative=shared_fixed_covariance(54,0,127,256)
check(negative==direct_covariance(54,0,127,256),'negative covariance independent conditional enumeration')
check(negative==F(-151916170599,1073741824),'negative covariance exact specified value')
cov_rows.append({'q':256,'shared':54,'a':0,'b':127,'covariance':str(negative)})
check(uaci_pooling_excess([0,255],256)==0,'0/255 reflected checkerboard has zero pooling excess')
check(uaci_pooling_excess([0,128],256)==F(16129,2080800),'0/128 two-pixel positive pooling excess')

OUT.mkdir(exist_ok=True)
def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write_csv('pair_moments_exact.csv',moment_rows)
write_csv('uaci_two_coordinate_exact.csv',uaci_rows)
write_csv('fixed_covariance_exact.csv',cov_rows)
summary={'status':'passed','exact_assertions':checks,'pair_moment_alphabets':len(moment_rows),
    'uaci_two_coordinate_designs':len(uaci_rows),'fixed_covariance_designs':len(cov_rows),
    'negative_covariance':str(negative),'negative_covariance_decimal':float(negative),
    'checkerboard_0_255_excess':str(uaci_pooling_excess([0,255],256)),
    'two_pixel_0_128_excess':str(uaci_pooling_excess([0,128],256)),
    'method':'Exact rational arithmetic and independent finite-law enumeration.'}
(OUT/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
