#!/usr/bin/env python3
"""Evaluate files in bounded isolated workers and retain independent partial credit."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(os.environ.get('AUTOGRADER_ROOT','/autograder'));source=root/'source'
(root/'results').mkdir(parents=True,exist_ok=True)
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'

def limits():
    if sys.platform.startswith('linux'):
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
        resource.setrlimit(resource.RLIMIT_CPU,(120,120))

reference=json.loads((source/'data.json').read_text());tests=[];reports={}
for part,config in reference['parts'].items():
    try:
        run=subprocess.run([sys.executable,str(source/'grader.py'),part,str(root/'submission'/(part+'.json')),str(source/'data.json')],capture_output=True,text=True,timeout=150,check=True,preexec_fn=limits)
        report=json.loads(run.stdout);tests.extend(report['tests']);reports[part]=report
    except Exception:
        for name,points in config['points'].items():
            tests.append(dict(name=part+' — '+name,score=0,max_score=points,output='The saved model exceeded the bounded evaluator limits or could not be evaluated.',visibility='visible'))
rule=reference['comparison'];parts=rule['parts']
ok=all(p in reports and reports[p].get('adequate_base') for p in parts)
for field in ('base_fingerprint','sae_fingerprint'):
    values=[reports.get(p,{}).get(field) for p in parts]
    ok=ok and all(values) and len(set(values))==1
features=[reports.get(p,{}).get('feature') for p in ('zero','amplified','ablated')]
ok=ok and all(x is not None for x in features) and len(set(features))==1
tests.append(dict(name='Unchanged transformer, sparse autoencoder and selected feature',score=rule['points'] if ok else 0,max_score=rule['points'],output='All four files must preserve one adequately trained transformer and sparse autoencoder; the three intervention files must select the same feature.',visibility='visible'))
result=dict(tests=tests,score=sum(t['score'] for t in tests),output='Metrics recomputed from submitted operations and values on canonical data. Cached scores and training histories are ignored.',visibility='visible',stdout_visibility='hidden',output_format='text',test_output_format='text',test_name_format='text')
(root/'results/results.json').write_text(json.dumps(result,allow_nan=False))
