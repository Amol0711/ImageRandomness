"""Deterministic audit of saved calibration and paired-power experiment outputs.

This audit does not call either experiment driver's decision or summary code.
It reconstructs rejection rules from raw integer statistics, stored null
moments, frozen seeds, and the recorded rank cutoffs. No image is resimulated.
The NPCR conditional-size/power formula is the exact binomial-law formula,
evaluated numerically with SciPy float64 probabilities, including boundary
randomization. It is not an arbitrary-precision probability computation.
"""
from __future__ import annotations
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from export_provenance import recorded_digest, matches_recorded_digest
from statistics import NormalDist
import hashlib
import json
import math
import time
import numpy as np
from scipy.stats import beta, binom

ROOT=Path(__file__).resolve().parent.parent
RESULTS=ROOT/'results'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rational(item):
    return Q(int(item['numerator']),int(item['denominator']))


def canonical_hash(values):
    return hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()


def wilson(k,n,confidence=.95):
    z=NormalDist().inv_cdf((1+confidence)/2)
    denominator=1+z*z/n
    center=(k/n+z*z/(2*n))/denominator
    half=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/denominator
    return (0. if k==0 else max(0.,center-half),1. if k==n else min(1.,center+half))


def cp(k,n,confidence=.975):
    error=(1-confidence)/2
    return (0. if k==0 else float(beta.ppf(error,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-error,k+1,n-k)))


def assert_close(first,second,tolerance=2e-12):
    assert abs(first-second)<=tolerance*max(1,abs(first),abs(second)),(first,second)


def check_sources(mapping, prefix, counts):
    for name,digest in mapping.items():
        assert matches_recorded_digest(prefix/name,digest),(name,'source digest mismatch')
        counts['source_hash_checks']+=1


def exact_keys(raw,center):
    # The audit deliberately uses Python integers, independent of the runtime
    # implementation's guarded int64 fast path.
    return np.asarray([abs(center.denominator*int(x)-center.numerator)
                       for x in raw],dtype=object)


def seeded_ties(group,columns):
    return np.random.default_rng(np.random.SeedSequence(group['tie_seed_entropy'])).random(
        (group['sample_size'],columns))


def json_safe(value):
    if isinstance(value,Q):return {'numerator':str(value.numerator),'denominator':str(value.denominator)}
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def integer_ranges(values):
    values=list(map(int,values))
    if not values:return []
    ranges=[];start=previous=values[0]
    for value in values[1:]:
        if value!=previous+1:ranges.append([start,previous]);start=value
        previous=value
    ranges.append([start,previous]);return ranges


def run():
    started=time.perf_counter();counts=Counter()
    protocol_path=RESULTS/'protocol.json';digest=sha(protocol_path)
    protocol=json.loads(protocol_path.read_text())
    calibration=json.loads((RESULTS/'calibration.json').read_text())
    power=json.loads((RESULTS/'power.json').read_text())
    sensitivity=json.loads((RESULTS/'sensitivity.json').read_text())
    verification=json.loads((RESULTS/'verification.json').read_text())
    for report in (calibration,power,sensitivity):assert report['status']=='completed'
    assert verification['status']=='PASS'
    assert matches_recorded_digest(protocol_path,calibration['metadata']['protocol_sha256'])
    assert matches_recorded_digest(protocol_path,power['metadata']['protocol_sha256'])
    assert matches_recorded_digest(protocol_path,sensitivity['protocol_sha256'])
    check_sources(calibration['metadata']['code_sha256'],ROOT,counts)
    check_sources(power['metadata']['source_sha256'],ROOT/'code',counts)
    check_sources(sensitivity['code_sha256'],ROOT/'code',counts)
    check_sources(verification['sha256'],ROOT/'code',counts)
    normal=NormalDist()
    with np.load(RESULTS/'calibration_scores.npz',allow_pickle=False) as archive:
        assert len(archive.files)==len(calibration['cases'])
        for case_index,record in enumerate(calibration['cases']):
            expected_case=protocol['calibration']['cases'][case_index]
            assert record['case']==expected_case and record['case_index']==case_index
            assert record['seed_entropy']==[protocol['calibration']['master_seed'],case_index]
            scores=archive[record['raw_scores_member']]
            assert canonical_hash(scores)==record['raw_scores_sha256']
            counts['raw_score_hash_checks']+=1
            mean=rational(record['exact_mean']);var=rational(record['exact_variance'])
            naive=rational(record['naive_pair_occurrence_variance'])
            summary=record['summary'];n=len(scores)
            assert n==summary['sample_size']==expected_case.get('sample_size',protocol['calibration']['sample_size'])
            assert len(summary['rejections'])==2*len(protocol['calibration']['alphas'])
            assert_close(float(np.mean(scores)),summary['sample_mean'])
            assert_close(float(np.var(scores,ddof=1)),summary['sample_variance_ddof1'])
            decisions={}
            for row in summary['rejections']:
                variance=var if row['method']=='exact_moment_gaussian' else naive
                radius=normal.inv_cdf(1-row['alpha']/2)*math.sqrt(float(variance))
                centered=scores.astype(np.float64)-float(mean)
                lower=centered < -radius;upper=centered > radius;both=lower|upper
                decisions[row['method'],row['alpha']]=both
                for field,values in [('lower_tail',lower),('upper_tail',upper),('two_sided',both)]:
                    k=int(np.count_nonzero(values));reported=row[field]
                    assert reported['count']==k
                    assert_close(reported['rate'],k/n)
                    lo,hi=wilson(k,n)
                    assert_close(lo,reported['wilson_lower']);assert_close(hi,reported['wilson_upper'])
                    counts['calibration_tail_count_checks']+=1
                counts['calibration_rule_rows']+=1
            for row in summary['rejections']:
                corrected=decisions['exact_moment_gaussian',row['alpha']]
                naive_decision=decisions['pair_occurrence_independent_gaussian',row['alpha']]
                assert row['paired_naive_only_rejections']==int(np.count_nonzero(naive_decision&~corrected))
                assert row['paired_exact_only_rejections']==int(np.count_nonzero(corrected&~naive_decision))
                counts['calibration_paired_count_checks']+=1
            counts['calibration_cases']+=1

    threshold_lookup={(r['geometry_id'],r['statistic'],r['alpha']):r for r in power['thresholds']}
    rate_lookup={(r['geometry_id'],r['model_id'],r['statistic'],r['alpha']):r for r in power['rates']}
    paired_lookup={}
    for row in power['paired_differences']:
        paired_lookup.setdefault((row['geometry_id'],row['model_id']),[]).append(row)
    group_data={}
    # Rebuild each rank threshold from the stored calibration bank and its tie seed.
    for geometry_index,geometry_record in enumerate(power['geometries']):
        geometry=geometry_record['geometry'];gid=geometry['id']
        assert geometry==protocol['power']['geometries'][geometry_index]
        statistics=geometry_record['statistics'];names=[s['name'] for s in statistics]
        assert names==protocol['power']['statistics']
        centers={s['name']:rational(s['null_mean']) for s in statistics}
        group_data[gid]=(geometry,names,centers)
        for group in [g for g in power['groups'] if g['geometry_id']==gid]:
            if group['model_id']=='calibration':phase=0;expected_n=protocol['power']['calibration_size']
            elif group['model_id']=='null':phase=1;expected_n=protocol['power']['null_sample_size']
            else:
                alternative_index=next(i for i,a in enumerate(protocol['power']['alternatives']) if a['id']==group['model_id'])
                phase=2+alternative_index;expected_n=protocol['power']['alternative_sample_size']
                assert group['alternative']==protocol['power']['alternatives'][alternative_index]
            assert group['sample_size']==expected_n
            assert group['field_seed_entropy']==[protocol['power']['master_seed'],geometry_index,phase,0]
            assert group['tie_seed_entropy']==[protocol['power']['master_seed'],geometry_index,phase,1]
            path=RESULTS/group['raw_file']
            assert sha(path)==group['raw_file_sha256'];counts['raw_archive_hash_checks']+=1
            with np.load(path,allow_pickle=False) as archive:
                raw=archive['statistics'];stored_names=list(archive['statistic_names'])
            assert stored_names==names and raw.shape==(group['sample_size'],len(names))
            assert canonical_hash(raw)==group['raw_statistics_sha256'];counts['raw_score_hash_checks']+=1
            ties=seeded_ties(group,len(names))
            decisions={}
            for j,name in enumerate(names):
                keys=exact_keys(raw[:,j],centers[name])
                if group['model_id']=='calibration':
                    order=np.lexsort((ties[:,j],keys))
                    for alpha in protocol['power']['alphas']:
                        row=threshold_lookup[gid,name,alpha]
                        level=Q(str(alpha));m=len(raw);k=(level.numerator*(m+1))//level.denominator
                        rank=m+1-k;index=int(order[rank-1])
                        assert row['m']==m and row['k']==k and row['rank']==rank
                        assert row['value']==int(keys[index]) and row['tie']==float(ties[index,j])
                        assert rational(row['marginal_level'])==Q(k,m+1)
                        assert row['beta_a']==k and row['beta_b']==m+1-k
                        beta_interval=beta.ppf([.025,.975],k,m+1-k)
                        for observed,wanted in zip(row['conditional_size_beta_interval'],beta_interval):assert_close(observed,wanted)
                        counts['rank_threshold_reconstructions']+=1
                    continue
                for alpha in protocol['power']['alphas']:
                    threshold=threshold_lookup[gid,name,alpha]
                    rejected=(keys>threshold['value'])|((keys==threshold['value'])&(ties[:,j]>threshold['tie']))
                    decisions[name,alpha]=rejected
                    row=rate_lookup[gid,group['model_id'],name,alpha]
                    k=int(np.count_nonzero(rejected));n=len(raw)
                    assert row['rejections']==k and row['sample_size']==n
                    assert_close(row['rate'],k/n)
                    lo,hi=wilson(k,n)
                    assert_close(row['wilson_lower'],lo);assert_close(row['wilson_upper'],hi)
                    counts['power_rate_rows']+=1
                lower_alpha,higher_alpha=sorted(protocol['power']['alphas'])
                assert np.all(~decisions[name,lower_alpha]|decisions[name,higher_alpha])
                counts['nested_power_decision_checks']+=1
            if group['model_id']=='calibration':continue
            for row in paired_lookup[gid,group['model_id']]:
                first=decisions[row['first_statistic'],row['alpha']]
                second=decisions[row['second_statistic'],row['alpha']]
                n10=int(np.count_nonzero(first&~second));n01=int(np.count_nonzero(~first&second));n=len(first)
                assert row['first_only']==n10 and row['second_only']==n01
                assert row['sample_size']==n
                assert_close(row['difference'],(n10-n01)/n)
                lo10,hi10=cp(n10,n);lo01,hi01=cp(n01,n)
                assert_close(row['difference_lower'],lo10-hi01)
                assert_close(row['difference_upper'],hi10-lo01)
                counts['power_paired_rows']+=1

    # NPCR is binomial under the iid null and under independent zero replacement.
    # Integrating over a fresh independent tie U retains the equality mass exactly
    # in the formula. The numerical pmf summation uses float64, not rational sums.
    binomial_rows=[]
    for gid,(geometry,names,centers) in group_data.items():
        n=geometry['rows']*geometry['columns'];q=geometry['q'];center=centers['NPCR_count']
        support=np.arange(n+1,dtype=np.int64);keys=exact_keys(support,center)
        for group in [g for g in power['groups'] if g['geometry_id']==gid and g['model_id']!='calibration']:
            alternative=group.get('alternative')
            if alternative is not None and alternative['kind']!='partial_identity':continue
            epsilon=Q(0) if alternative is None else Q(str(alternative['epsilon']))
            success=(1-epsilon)*Q(q-1,q)
            masses=binom.pmf(support,n,float(success))
            mass_sum=float(np.sum(masses));assert abs(mass_sum-1)<2e-12
            for alpha in protocol['power']['alphas']:
                threshold=threshold_lookup[gid,'NPCR_count',alpha]
                strict=keys>threshold['value'];equality=keys==threshold['value']
                strict_mass=float(np.sum(masses[strict]));equality_mass=float(np.sum(masses[equality]))
                conditional=strict_mass+(1-threshold['tie'])*equality_mass
                observed=rate_lookup[gid,group['model_id'],'NPCR_count',alpha]
                binomial_rows.append({'geometry_id':gid,'model_id':group['model_id'],'alpha':alpha,
                    'n':n,'success_probability':success,'threshold_value':threshold['value'],
                    'threshold_tie':threshold['tie'],'strict_rejection_count_ranges':integer_ranges(support[strict]),
                    'boundary_counts':support[equality].tolist(),'strict_probability_float64':strict_mass,
                    'equality_probability_float64':equality_mass,'conditional_probability_float64':conditional,
                    'observed_rejections':observed['rejections'],'observed_rate':observed['rate'],
                    'observed_wilson_lower':observed['wilson_lower'],'observed_wilson_upper':observed['wilson_upper'],
                    'formula_inside_observed_wilson':observed['wilson_lower']<=conditional<=observed['wilson_upper'],
                    'pmf_sum_float64':mass_sum,
                    'interpretation':'Exact conditional binomial-law formula, including uniform tie randomization; numerical probability evaluated in float64, not exact arithmetic. Wilson coverage is not asserted for every realization.'})
                counts['npcr_binomial_formula_rows']+=1

    report={'status':'PASS','protocol_sha256':digest,'counts':dict(sorted(counts.items())),
        'audit_code_sha256':sha(__file__),'elapsed_seconds':time.perf_counter()-started,
        'input_result_sha256':{name:sha(RESULTS/name) for name in
             ['calibration.json','power.json','sensitivity.json','verification.json','calibration_scores.npz']},
        'npcr_conditional_binomial':binomial_rows,
        'scope':'Deterministic reanalysis of saved statistics and frozen tie seeds. No new image simulation, threshold fitting design, or selection of alternatives. All reported raw rejection and discordance counts were independently reproduced.'}
    (RESULTS/'audit.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','counts':report['counts'],'npcr_conditional_binomial':[{k:r[k] for k in ['model_id','alpha','conditional_probability_float64','observed_rate','formula_inside_observed_wilson']} for r in binomial_rows]},indent=2))


if __name__=='__main__':run()
