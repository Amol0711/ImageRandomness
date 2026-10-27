"""Independent finite-state validation for the calibration and power study.

Oracles enumerate finite product laws and explicitly form graph scores. They do
not reuse the computational projection or covariance decomposition. The rank
checks use every possible held-out rank in a fixed exchangeable pooled sample,
including a tied primary score with distinct independent tie coordinates.
"""
from collections import Counter
from fractions import Fraction as Q
from itertools import product, combinations
from pathlib import Path
import hashlib
import json
import math
import random
import time

import numpy as np
import moment_engine as engine

ROOT = Path(__file__).resolve().parent.parent


def triangular_law(q):
    counts = Counter(abs(u-v) for u in range(q) for v in range(q))
    return tuple(Q(counts[x], q*q) for x in range(q))


def fixed_laws(reference, q):
    return tuple(tuple(Q(sum(abs(v-u)==x for u in range(q)),q)
                       for x in range(q)) for v in reference)


def scalar_score(values, edges):
    return sum(c*abs(int(values[i])-int(values[j])) for i,j,c in edges)


def enumerate_laws(laws, edges):
    supports = [tuple((x,Q(p)) for x,p in enumerate(row) if p) for row in laws]
    mean=second=Q()
    for sample in product(*supports):
        values=[x for x,p in sample]
        probability=math.prod(p for x,p in sample)
        score=scalar_score(values,edges)
        mean+=probability*score
        second+=probability*score*score
    return mean,second-mean*mean


def enumerate_copy(law, edges, n, pairs, theta):
    """Exact Bernoulli activation plus independent base-field enumeration."""
    theta=Q(theta)
    total=second=count_mean=intensity_mean=count_second=intensity_second=Q()
    support=tuple((x,Q(p)) for x,p in enumerate(law) if p)
    for initial in product(support, repeat=n):
        probability=math.prod(p for x,p in initial)
        for flags in product((0,1),repeat=len(pairs)):
            activation_probability=math.prod(theta if flag else 1-theta for flag in flags)
            if not activation_probability:continue
            values=[x for x,p in initial]
            for flag,(donor,recipient) in zip(flags,pairs):
                if flag:values[recipient]=values[donor]
            weight=probability*activation_probability
            score=scalar_score(values,edges)
            total+=weight*score
            second+=weight*score*score
            count=sum(x!=0 for x in values);intensity=sum(values)
            count_mean+=weight*count
            intensity_mean+=weight*intensity
            count_second+=weight*count*count
            intensity_second+=weight*intensity*intensity
    return total,second-total*total,count_mean,intensity_mean,count_second-count_mean**2,intensity_second-intensity_mean**2


def enumerate_identity(law, edges, n, epsilon):
    """Identity independently replaces each difference by zero."""
    epsilon=Q(epsilon)
    changed=[(1-epsilon)*p for p in law]
    changed[0]+=epsilon
    mean,var=enumerate_laws([changed]*n,edges)
    return mean,var,n*(1-changed[0]),n*sum(x*p for x,p in enumerate(changed))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value,Q):return {'numerator':value.numerator,'denominator':value.denominator}
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


# The execution checks below are finalized against the shared module API.
def run():
    import statistical_core as core
    import alternative_analysis as alternative
    started=time.perf_counter()
    rng=random.Random(2026102504)
    counts=Counter()
    details={}

    # These constants are independently summed over all endpoint pairs/triples.
    for q in range(2,9):
        law=triangular_law(q)
        g=[sum((py*abs(x-y) for y,py in enumerate(law)),Q()) for x in range(q)]
        mean=sum((px*g[x] for x,px in enumerate(law)),Q())
        second=sum((px*py*(x-y)**2 for x,px in enumerate(law) for y,py in enumerate(law)),Q())
        covariance=sum((px*g[x]*g[x] for x,px in enumerate(law)),Q())-mean*mean
        actual=core.iid_law_moments(q)
        assert (actual.mean,actual.variance,actual.covariance)==(mean,second-mean*mean,covariance)
        assert tuple(Q(x,actual.denominator) for x in actual.counts)==law
        assert alternative.iid_triangular_constants(q)==(mean,second-mean*mean,covariance)
        counts['iid_law_constant_cases']+=1

    # Graph oracle enumerates the complete independent field, not edge moments.
    for q in (2,3,4):
        for n in range(1,6):
            for repetition in range(4):
                edges=[(i,j,rng.choice([-3,-1,1,2])) for i,j in combinations(range(n),2) if rng.random()<.55]
                design=engine.prepare_graph(n,edges)
                law=triangular_law(q)
                expected=enumerate_laws([law]*n,edges)
                assert alternative.iid_triangular_moments(design,q)==expected
                counts['signed_iid_graph_exact_cases']+=1
                positive_edges=[(i,j,abs(c)) for i,j,c in edges]
                positive_design=engine.prepare_graph(n,positive_edges)
                positive_expected=enumerate_laws([law]*n,positive_edges)
                actual=core.iid_moments(q,positive_design)
                assert (actual.mean,actual.variance)==positive_expected
                counts['iid_graph_exact_cases']+=1
                design=positive_design;edges=positive_edges
                reference=[rng.randrange(q) for i in range(n)]
                expected_fixed=enumerate_laws(fixed_laws(reference,q),edges)
                actual_fixed=core.null_moments('fixed',design,q,reference=reference)
                assert (actual_fixed.mean,actual_fixed.variance)==expected_fixed
                previous=engine.moments_prepared(reference,q,design)
                assert (previous.mean,previous.variance)==expected_fixed
                counts['fixed_graph_exact_cases']+=1

    score_outcomes=0
    for n,q,edges in [(4,3,[(0,1,2),(1,2,-3),(0,3,1)]),
                      (6,2,engine.rectangular_edges(2,3,2,2)),
                      (6,2,engine.rectangular_edges(1,6,1,3)),
                      (3,3,[])]:
        design=engine.prepare_graph(n,edges)
        values=np.asarray(list(product(range(q),repeat=n)),dtype=np.int32)
        expected=np.asarray([scalar_score(row,edges) for row in values],dtype=np.int64)
        score_outcomes+=len(values)
        for chunk in (1,3,8192):
            prepared=core.prepare_score(design,edge_chunk_size=chunk)
            actual=core.score_fields(values,prepared)
            assert np.array_equal(actual,expected)
            counts['score_kernel_batch_checks']+=1
    counts['score_distinct_outcomes']=score_outcomes

    # Nonuniform IID finite probabilities are part of the known-law illustration.
    for law in [(Q(1,2),Q(1,3),Q(1,6)),(Q(1),Q(0)),(Q(0),Q(1)),(Q(1,7),Q(2,7),Q(4,7))]:
        q=len(law);design=engine.prepare_graph(3,[(0,1,2),(0,2,1),(1,2,3)])
        expected=enumerate_laws([law]*3,design.edges)
        actual=core.iid_moments(q,design,law='pmf',pmf=law)
        assert (actual.mean,actual.variance)==expected
        counts['general_iid_law_cases']+=1

    # Independently enumerate both field and copy-activation randomness.
    copying_cases=[(1,4,1,2,2,1),(1,4,1,3,3,1),(2,3,2,2,2,1),
                   (2,4,2,2,2,2),(1,5,1,3,2,2)]
    for M,N,a,b,q,lag in copying_cases:
        design=engine.prepare_image_graph(M,N,a,b)
        pairs=alternative.copying_pairs(M,N,lag)
        expected_pairs=tuple((r*N+start+j,r*N+start+j+lag)
                             for r in range(M) for start in range(0,N-2*lag+1,2*lag)
                             for j in range(lag))
        assert pairs==expected_pairs
        assert len({v for pair in pairs for v in pair})==2*len(pairs)
        law=triangular_law(q)
        null_mean,null_var=enumerate_laws([law]*(M*N),design.edges)
        for theta in (Q(0),Q(1,3),Q(1)):
            mean,var,npcr_mean,uaci_mean,npcr_var,uaci_var=enumerate_copy(law,design.edges,M*N,pairs,theta)
            shift=alternative.exact_mean_shift(design,q,lag,theta,M,N)
            assert null_mean-mean==shift
            assert npcr_mean==M*N*(1-law[0])
            assert uaci_mean==M*N*sum(x*p for x,p in enumerate(law))
            counts['copy_model_exact_cases']+=1
            additive=alternative.additive_copy_moments(q,theta,M,N,lag)
            assert additive['NPCR_count']['mean']==npcr_mean and additive['NPCR_count']['copy_variance']==npcr_var
            assert additive['UACI_sum']['mean']==uaci_mean and additive['UACI_sum']['copy_variance']==uaci_var
            counts['additive_copy_moment_checks']+=2
            if (M,N,a,b,q,lag,theta)==(2,4,2,2,2,2,Q(1)):
                assert (null_mean,null_var,mean,var)==(Q(9),Q(11,2),Q(9),Q(27,2))
                details['zero_mean_shift_counterexample']={'null_mean':null_mean,'null_variance':null_var,'copy_mean':mean,'copy_variance':var}
        for epsilon in (Q(1,10),Q(1,2),Q(1)):
            mean,var,npcr_mean,uaci_mean=enumerate_identity(law,design.edges,M*N,epsilon)
            assert npcr_mean==(1-epsilon)*M*N*(1-law[0])
            assert uaci_mean==(1-epsilon)*M*N*sum(x*p for x,p in enumerate(law))
            predicted=alternative.partial_identity_means(q,epsilon,M*N,design.weight_sum)
            assert (predicted['S'],predicted['NPCR_count'],predicted['UACI_sum'])==(mean,npcr_mean,uaci_mean)
            counts['partial_identity_exact_cases']+=1

    # In an exchangeable continuous lexicographic pooled sample, every held-out
    # observation rank has the same chance. Finite leave-one-out enumeration
    # checks the exact attainable rejection fraction k/(m+1), including ties.
    rank_details=[]
    for size in (2,3,5,10,20):
        for alpha in (.01,.05,.1,.2,.5):
            if math.floor(Q(str(alpha))*size)==0:
                try:core.fit_rank_threshold(np.arange(size-1,dtype=float),alpha,ties=np.zeros(size-1))
                except ValueError:counts['unattainable_rank_levels_rejected']+=1
                else:raise AssertionError('Unattainable rank level should be rejected')
                continue
            for values in [np.zeros(size),np.arange(size,dtype=float),np.arange(size,dtype=float)%3]:
                ties=np.asarray([(i+.25)/size for i in range(size)])
                rejected=0
                for holdout in range(size):
                    mask=np.arange(size)!=holdout
                    threshold=core.fit_rank_threshold(values[mask],alpha,ties=ties[mask])
                    actual=core.reject_rank_threshold(values[holdout:holdout+1],threshold,ties=ties[holdout:holdout+1])
                    rejected+=int(actual[0])
                    assert threshold.k==math.floor(Q(str(alpha))*size)
                    assert threshold.marginal_level==Q(threshold.k,size)
                expected=math.floor(Q(str(alpha))*size)
                assert rejected==expected,(size,alpha,values,rejected,expected)
                counts['rank_exchangeable_configurations']+=1
                counts['rank_heldout_classifications']+=size
            rank_details.append({'pooled_size':size,'alpha':alpha,'rejection_count':expected,'exact_marginal_level':Q(expected,size)})
    details['rank_checks']=rank_details

    # Exact discrepancy construction must preserve adjacent large integers.
    for raw,center in [(np.asarray([2**53,2**53+1,2**53+2],dtype=np.int64),Q(0)),
                       (np.asarray([2**63+4,2**63+5,2**63+7],dtype=object),Q(2**63+5,3)),
                       (np.asarray([-2,-1,0,1,2],dtype=np.int64),Q(1,10**20+39)),
                       (np.asarray([-2**63,2**63-1],dtype=np.int64),Q(-1,3))]:
        keys=core.exact_discrepancy(raw,center)
        expected=[abs(center.denominator*int(x)-center.numerator) for x in raw]
        assert list(keys)==expected
        counts['large_integer_discrepancy_cases']+=1
    for offset in (2**53,2**63,2**100):
        keys=np.asarray([offset+i for i in range(5)],dtype=object)
        threshold=core.fit_rank_threshold(keys[:4],Q(1,5),ties=np.asarray([.1,.2,.3,.4]))
        assert threshold.value==offset+3
        actual=core.reject_rank_threshold(keys,threshold,ties=np.repeat(.5,5))
        assert actual.tolist()==[False,False,False,True,True]
        counts['large_integer_rank_cases']+=1

    import run_calibration
    summary=run_calibration.summarize_scores(np.asarray([-2,-1,0,1,2],dtype=np.int64),
             Q(0),Q(1),Q(4),alphas=[.05],quantile_probabilities=[.5])
    rows={r['method']:r for r in summary['rejections']}
    assert rows['exact_moment_gaussian']['lower_tail']['count']==1
    assert rows['exact_moment_gaussian']['upper_tail']['count']==1
    assert rows['exact_moment_gaussian']['two_sided']['count']==2
    assert rows['pair_occurrence_independent_gaussian']['two_sided']['count']==0
    assert summary['sample_mean']==0 and summary['sample_variance_ddof1']==2.5
    counts['calibration_summary_cases']+=1

    # Wilson intervals are descriptive pointwise binomial intervals. Edge cases
    # and their score-equation form are checked without scipy or the core code.
    z=1.959963984540054
    for n in (1,2,10,100,20000):
        for k in sorted(set((0,1,n//2,max(0,n-1),n))):
            lo,hi=core.wilson_interval(k,n)
            assert 0<=lo<=k/n<=hi<=1
            denominator=1+z*z/n
            center=(k/n+z*z/(2*n))/denominator
            radius=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/denominator
            assert abs(lo-max(0,center-radius))<2e-12
            assert abs(hi-min(1,center+radius))<2e-12
            counts['wilson_interval_cases']+=1

    # Fixed seeds reproduce samples and changing the seed changes a nondegenerate
    # sample. These checks establish deterministic plumbing, not RNG quality.
    for model,q,pmf in [('two_random',3,None),('fixed',3,None),('independent_pmf',3,[Q(1,2),Q(1,3),Q(1,6)])]:
        ref=core.reference_for('palette',2,3,q)
        first=core.generate_null(model,np.random.default_rng(1985),11,2,3,q,reference=ref,pmf=pmf)
        second=core.generate_null(model,np.random.default_rng(1985),11,2,3,q,reference=ref,pmf=pmf)
        other=core.generate_null(model,np.random.default_rng(1986),11,2,3,q,reference=ref,pmf=pmf)
        assert first.shape==(11,6) and np.all((first>=0)&(first<q))
        assert np.array_equal(first,second) and not np.array_equal(first,other)
        counts['seeded_generator_cases']+=1

    import run_power
    # A scalar binomial-tail inversion independently validates the CP bounds.
    def binomial_tail(n,k,p,upper):
        support=range(k,n+1) if upper else range(k+1)
        return sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in support)
    def invert_tail(n,k,target,upper):
        low,high=0.,1.
        for _ in range(80):
            middle=(low+high)/2
            value=binomial_tail(n,k,middle,upper)
            if (value<target)==upper:low=middle
            else:high=middle
        return (low+high)/2
    for n in (2,5,20):
        for k in sorted(set((0,1,n//2,n-1,n))):
            lo,hi=run_power.clopper_pearson(k,n,.975)
            expected_lo=0 if k==0 else invert_tail(n,k,.0125,True)
            expected_hi=1 if k==n else invert_tail(n,k,.0125,False)
            assert abs(lo-expected_lo)<2e-12 and abs(hi-expected_hi)<2e-12
            counts['clopper_pearson_tail_checks']+=1
    first=np.asarray([1,1,1,1,0,0,0,0],dtype=bool)
    second=np.asarray([1,0,0,0,1,0,0,0],dtype=bool)
    paired=run_power.paired_difference(first,second)
    assert paired['first_only']==3 and paired['second_only']==1
    assert paired['difference']==.25
    lo10,hi10=run_power.clopper_pearson(3,8,.975)
    lo01,hi01=run_power.clopper_pearson(1,8,.975)
    assert paired['difference_lower']==lo10-hi01 and paired['difference_upper']==hi10-lo01
    swapped=run_power.paired_difference(second,first)
    assert swapped['difference_lower']==-paired['difference_upper']
    assert swapped['difference_upper']==-paired['difference_lower']
    counts['paired_difference_checks']+=1

    equal=run_power.paired_difference(first,first)
    assert equal['first_only']==equal['second_only']==0 and equal['difference']==0
    assert equal['difference_lower']<0<equal['difference_upper']
    complementary=run_power.paired_difference(np.ones(8,dtype=bool),np.zeros(8,dtype=bool))
    assert complementary['first_only']==8 and complementary['second_only']==0
    assert complementary['difference']==1 and complementary['difference_upper']==1
    assert 0<complementary['difference_lower']<1
    counts['paired_difference_checks']+=2
    tie_values,tie_seed=run_power.ties_for(20,4,2026,0,1)
    reproduced,repeated_seed=run_power.ties_for(20,4,2026,0,1)
    different,_=run_power.ties_for(20,4,2026,0,2)
    assert tie_seed==repeated_seed and np.array_equal(tie_values,reproduced)
    assert not np.array_equal(tie_values,different)
    assert np.all((tie_values>=0)&(tie_values<1))
    counts['tie_seed_reproduction_checks']+=1

    geometry={'rows':2,'columns':4,'q':2,'windows':[[2,2],[1,3]]}
    statistics=run_power.build_statistics(geometry,3)
    values=np.asarray(list(product(range(2),repeat=8)),dtype=np.int32)
    actual=run_power.statistics_from_fields(values,statistics)
    expected=np.asarray([[sum(x!=0 for x in row),sum(row),
                          *[scalar_score(row,stat.design.edges) for stat in statistics[2:]]]
                         for row in values],dtype=np.int64)
    assert np.array_equal(actual,expected)
    counts['paired_statistic_outcomes']+=len(values)
    pairs=alternative.copying_pairs(2,4,2)
    for theta in (0,1):
        changed=run_power.apply_alternative(values.copy(),np.random.default_rng(15),geometry,
                   {'kind':'copying','lag':2,'theta':theta})
        expected_values=values.copy()
        if theta:
            for donor,recipient in pairs:expected_values[:,recipient]=expected_values[:,donor]
        assert np.array_equal(changed,expected_values)
        counts['alternative_endpoint_transforms']+=1
    for epsilon in (0,1):
        changed=run_power.apply_alternative(values.copy(),np.random.default_rng(16),geometry,
                   {'kind':'partial_identity','epsilon':epsilon})
        assert np.array_equal(changed,np.zeros_like(values) if epsilon else values)
        counts['alternative_endpoint_transforms']+=1
    law=triangular_law(2)
    for epsilon in (Q(1,10),Q(1,2),Q(1)):
        alt={'kind':'partial_identity','epsilon':float(epsilon)}
        modified=[(1-epsilon)*p for p in law];modified[0]+=epsilon
        for stat in statistics:
            if stat.name=='NPCR_count':expected=8*(1-modified[0])
            elif stat.name=='UACI_sum':expected=8*sum(x*p for x,p in enumerate(modified))
            else:expected=enumerate_laws([modified]*8,stat.design.edges)[0]
            assert run_power.expected_mean(stat,geometry,alt)==expected
            counts['power_expected_mean_checks']+=1

    # Nested alpha rules share the same bank ties and test ties.
    rng_np=np.random.default_rng(99)
    bank=rng_np.integers(0,6,99);bank_ties=rng_np.random(99)
    test=np.repeat(np.arange(7),100);test_ties=rng_np.random(len(test))
    low=core.fit_rank_threshold(bank,.01,ties=bank_ties)
    high=core.fit_rank_threshold(bank,.05,ties=bank_ties)
    low_reject=core.reject_rank_threshold(test,low,ties=test_ties)
    high_reject=core.reject_rank_threshold(test,high,ties=test_ties)
    assert np.all(~low_reject|high_reject)
    counts['nested_rank_decision_checks']+=1

    files=['moment_engine.py','alphabet_baseline.py','statistical_core.py','alternative_analysis.py','run_calibration.py','run_power.py',Path(__file__).name]
    report={'status':'PASS','counts':dict(sorted(counts.items())),
            'details':details,'elapsed_seconds':time.perf_counter()-started,
            'sha256':{name:sha(ROOT/'code'/name) for name in files},
            'scope':'Finite exact checks of moment/scoring/copying identities and rank construction; no empirical calibration or power claim is inferred.',
            'shared_pool_interpretation':'A fresh exchangeable observation has unconditional rank level k/(m+1). Reusing one calibration pool yields a random conditional rejection probability; independent assessment observations admit a conditional binomial Wilson interval, not a conditional nominal-size guarantee.'}
    destination=ROOT/'results/verification.json';destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(json_safe(report),indent=2)+'\n')
    print(json.dumps({'status':'PASS','counts':report['counts'],'elapsed_seconds':report['elapsed_seconds']},indent=2))


if __name__=='__main__':
    run()
