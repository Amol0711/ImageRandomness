#!/usr/bin/env python3
"""Analyze archived BACI simulations, regenerate data, or verify the engine."""
from __future__ import annotations
import argparse
import csv
from fractions import Fraction
from itertools import product
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from calibration import baci_statistic, fixed_baci_moments
from analysis import analyze, fixed_pattern, read_design, write_csv, write_json
from fixed_moments import check_small_examples, independent_exact_moments
from independent_stream import MT19937_64, independent_trial


def config():
    result = json.loads((HERE/'config.json').read_text())
    ranges = []
    for d in result['designs']:
        if d['q'] != 256 or d['rows'] != 256 or d['columns'] != 256:
            raise ValueError('this campaign uses 256 by 256 eight-bit images')
        lo = d['seed_base']; hi = lo+d['bank_size']+d['evaluation_size']-1
        if lo < 0 or hi >= 2**64 or any(lo <= b and a <= hi for a,b in ranges):
            raise ValueError('overlapping or invalid seed-input range')
        if d['bank_trial_ids'] != [0,d['bank_size']-1] or d['evaluation_trial_ids'] != [d['bank_size'],d['bank_size']+d['evaluation_size']-1]:
            raise ValueError('trial split mismatch')
        ranges.append((lo,hi))
    return result


def new_output(path):
    path = Path(path).resolve()
    # Refuse even an empty existing directory and every shipped-data descendant.
    if path == HERE or HERE in path.parents:
        raise ValueError('output must be outside the expanded component')
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_engine(directory):
    compiler = shutil.which('g++')
    if compiler is None:
        raise RuntimeError('g++ with C++17 and OpenMP support is required')
    executable = directory/'sim_engine'
    subprocess.run([compiler,'-O3','-std=c++17','-fopenmp',str(HERE/'sim_engine.cpp'),
                    '-o',str(executable)],check=True,capture_output=True,text=True)
    return executable


def draw(executable, directory, *, rows, columns, repeats, start, count, seed_base, fixed, threads):
    # Raw files are temporary; neither commands nor local paths are recorded.
    fixed_path = directory/'input.u8'
    if fixed is not None:
        np.asarray(fixed,dtype=np.uint8).tofile(fixed_path)
    output = directory/'raw.u32'
    if output.exists():
        output.unlink()
    result = subprocess.run([str(executable),'simulate',str(rows),str(columns),str(repeats),
        str(start),str(count),str(seed_base),'iid' if fixed is None else 'fixed',
        '-' if fixed is None else str(fixed_path),str(output),str(threads)],
        check=True,capture_output=True,text=True)
    summary = json.loads(result.stdout)
    raw = np.fromfile(output,dtype='<u4')
    if raw.size != count*repeats or int(raw.sum(dtype=np.uint64)) != summary['checksum']:
        raise ValueError('engine shape or integer checksum mismatch')
    scale = 6*(rows-1)*(columns-1)*255
    if np.any(raw > scale):
        raise ValueError('numerator exceeds its bound')
    return raw.reshape(count,repeats)


def compare_rows(actual, expected_path):
    with expected_path.open(newline='') as stream:
        expected = list(csv.DictReader(stream))
    if len(actual) != len(expected):
        raise ValueError('expected result row count mismatch')
    fields = 0
    for index,(a,e) in enumerate(zip(actual,expected)):
        if set(a) != set(e):
            raise ValueError('expected result column mismatch')
        for key,value in a.items():
            target = e[key]
            if isinstance(value,(int,float)):
                if isinstance(value,int):
                    equal = value == int(target)
                else:
                    equal = math.isclose(value,float(target),rel_tol=2e-13,abs_tol=2e-15)
            else:
                equal = value == target
            if not equal:
                raise ValueError(f'expected value mismatch in row {index}, field {key}')
            fields += 1
    return fields


def analyze_data(raw_dir, output=None, verify=False):
    rows,moments,details = analyze(config(),Path(raw_dir))
    result = {'status':'PASS','design_count':len(details),'rejection_comparison_count':len(rows),
              'random_draws':0,'designs':details}
    if verify:
        result['verified_summary_fields'] = compare_rows(rows,HERE/'expected_rejection_rates.csv') + compare_rows(moments,HERE/'expected_moments.csv')
    if output is not None:
        write_csv(output/'rejection_rates.csv',rows)
        write_csv(output/'moments.csv',moments)
        write_json(output/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k != 'designs'},indent=2))
    return result


def simulate(output, threads):
    protocol = config()
    with tempfile.TemporaryDirectory(prefix='baci_sim_') as temp:
        directory = Path(temp)
        executable = build_engine(directory)
        for d in protocol['designs']:
            fixed = None if d['mode'] == 'random' else fixed_pattern(d['mode'],d['rows'],d['columns'])
            raw = draw(executable,directory,rows=d['rows'],columns=d['columns'],repeats=d['repeats'],
                start=0,count=d['bank_size']+d['evaluation_size'],seed_base=d['seed_base'],fixed=fixed,threads=threads)
            scale = 6*(d['rows']-1)*(d['columns']-1)*255
            np.savez_compressed(output/f"{d['id']}_raw.npz",raw_numerators=raw,
                trial_totals=raw.sum(axis=1,dtype=np.uint64),bank_trials=np.uint64(d['bank_size']),
                test_trials=np.uint64(d['evaluation_size']),repeats=np.uint64(d['repeats']),
                per_image_scale=np.uint64(scale))
            print(json.dumps({'design':d['id'],'image_numerators':int(raw.size),'status':'COMPLETE'}),flush=True)
    # Numeric configuration accompanies freshly generated data without machine paths.
    write_json(output/'config.json',protocol)


def check():
    import calibration
    report = {'status':'PASS'}
    # Independent moment enumeration includes all 81 ternary 2x2 fixed patterns.
    report['finite_law_checks'] = check_small_examples(calibration)
    with tempfile.TemporaryDirectory(prefix='baci_check_') as temp:
        directory = Path(temp)
        executable = build_engine(directory)
        images = [np.array(v,dtype=np.uint8).reshape(2,2) for v in product(range(3),repeat=4)]
        for m,n in [(2,3),(3,2),(2,33),(33,2),(3,7),(7,3),(9,13),(256,256)]:
            i,j = np.indices((m,n))
            images.extend([np.zeros((m,n),dtype=np.uint8),np.full((m,n),255,dtype=np.uint8),
                           (255*((i+j)%2)).astype(np.uint8),((73*i+97*j)%256).astype(np.uint8)])
        for image in images:
            path = directory/'stat.u8'; image.tofile(path)
            answer = json.loads(subprocess.check_output([str(executable),'stat',str(image.shape[0]),str(image.shape[1]),str(path)],text=True))
            value = Fraction(answer['raw_numerator'],answer['scale'])
            if value != baci_statistic(image,exact=True,method='literal') or value != baci_statistic(image,exact=True,method='weighted'):
                raise ValueError('C++ statistic differs from independent literal or weighted evaluation')
        report['deterministic_statistic_images'] = len(images)
        generator = MT19937_64(5489)
        vector = [14514284786278117030,4620546740167642908,13109570281517897720,17462938647148434322,355488278567739596]
        if [generator() for _ in vector] != vector:
            raise ValueError('independent MT19937_64 known-vector mismatch')
        stream_trials = 0
        for mode in ['iid','fixed']:
            for m,n in [(2,3),(7,9),(31,43)]:
                i,j = np.indices((m,n))
                fixed = None if mode == 'iid' else ((73*i+19*j)%256).astype(np.uint8)
                params = dict(rows=m,columns=n,repeats=3,start=5,count=4,seed_base=900000000,fixed=fixed)
                one = draw(executable,directory,**params,threads=1)
                many = draw(executable,directory,**params,threads=3)
                expected = np.stack([independent_trial(m,n,3,900000000,t,fixed) for t in range(5,9)])
                np.testing.assert_array_equal(one,expected)
                np.testing.assert_array_equal(one,many)
                part1 = draw(executable,directory,**{**params,'count':2},threads=2)
                part2 = draw(executable,directory,**{**params,'start':7,'count':2},threads=2)
                np.testing.assert_array_equal(one,np.concatenate([part1,part2]))
                stream_trials += 4
        report['independent_python_stream_trials'] = stream_trials
        replays = []
        for d in config()['designs']:
            saved,totals,scale,_ = read_design(d,HERE/'data')
            fixed = None if d['mode'] == 'random' else fixed_pattern(d['mode'],d['rows'],d['columns'])
            for trial in [0,d['bank_size']-1,d['bank_size'],len(totals)-1]:
                for threads in [1,3]:
                    actual = draw(executable,directory,rows=d['rows'],columns=d['columns'],repeats=d['repeats'],
                        start=trial,count=1,seed_base=d['seed_base'],fixed=fixed,threads=threads)
                    np.testing.assert_array_equal(actual[0],saved[trial])
                replays.append({'design':d['id'],'trial_id':trial,'repeats':d['repeats'],'threads':[1,3]})
        report['selected_saved_trial_replays'] = replays
        report['cpp_replayed_image_numerators'] = sum(x['repeats']*2 for x in replays)
        # Explicitly verify widened accumulation beyond the uint32 boundary.
        overflow_control = np.full((3,100),99488250,dtype=np.uint32).sum(axis=1,dtype=np.uint64)
        np.testing.assert_array_equal(overflow_control,np.full(3,9948825000,dtype=np.uint64))
        path = directory/'bad.u8'; path.write_bytes(b'\0\xff')
        for args in [('stat','2','2',str(path)),('stat','1','4',str(path)),('stat','-2','4',str(path)),('stat','100000','100000',str(path))]:
            if subprocess.run([str(executable),*args],capture_output=True,text=True).returncode != 2:
                raise ValueError('engine accepted invalid dimensions or input length')
    print(json.dumps(report,indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    for command in ['analyze','verify']:
        p = sub.add_parser(command)
        p.add_argument('--data',type=Path,default=HERE/'data')
        p.add_argument('--output',type=Path,required=command=='analyze')
    p = sub.add_parser('simulate')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=1)
    p = sub.add_parser('check')
    p.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.action == 'simulate' and not 1 <= args.threads <= 1024:
        parser.error('--threads must be from 1 through 1024')
    output = new_output(args.output) if args.output is not None else None
    if args.action in ['analyze','verify']:
        analyze_data(args.data,output,verify=args.action=='verify')
    elif args.action == 'simulate':
        simulate(output,args.threads)
    else:
        report = check()
        if output is not None:
            write_json(output/'checks.json',report)


if __name__ == '__main__':
    main()
