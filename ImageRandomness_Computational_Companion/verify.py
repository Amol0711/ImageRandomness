"""Run deterministic and saved-record checks in a separate unused directory."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from verify_export import verify
ROOT=Path(__file__).resolve().parent

def snapshot():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists(): parser.error('Output directory must not already exist')
    if out==ROOT or ROOT in out.parents: parser.error('Output directory must be outside the distributed companion')
    before=snapshot()
    exported=verify()
    work=out/'working_copy'
    shutil.copytree(ROOT,work,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    steps=[
      ('moment_implementation','moment_computation/code/verify_algorithms.py'),
      ('statistics_implementation','statistical_validation/code/verify_statistics.py'),
      ('statistics_saved_data','statistical_validation/code/audit_results.py'),
      ('scaling_record','moment_computation/analysis/verify_scaling_record.py'),
      ('vertex_record','moment_computation/analysis/verify_vertex_record.py'),
    ]
    records=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for label,script in steps:
        proc=subprocess.run([sys.executable,'-B',str(work/script)],cwd=work,env=env,
                            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
        (out/(label+'.txt')).write_text(proc.stdout)
        records.append({'check':label,'returncode':proc.returncode})
        if proc.returncode:
            raise RuntimeError('Verification failed: '+label+'; see '+str(out/(label+'.txt')))
    if before!=snapshot(): raise RuntimeError('Distributed input files changed')
    result={'status':'passed','public_export':exported,'checks':records,
            'distributed_files_unchanged':True,'new_image_simulations':0,'new_timing_measurements':0,
            'scope':'Deterministic implementation checks and saved-data audits. No full Monte Carlo or timing regeneration.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
