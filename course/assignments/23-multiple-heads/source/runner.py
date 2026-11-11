#!/usr/bin/env python3
"""Isolate each submitted model and preserve partial credit after bad files."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ.get('AUTOGRADER_ROOT','/autograder'));source=root/'source';(root/'results').mkdir(parents=True,exist_ok=True)
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
def limits():
    if sys.platform.startswith('linux'):
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(45,45))
try:
    reference=json.loads((source/'data.json').read_text());tests=[];fingerprints={}
    for part,config in reference['parts'].items():
        try:
            run=subprocess.run([sys.executable,str(source/'grader.py'),part,str(root/'submission'/(part+'.json')),str(source/'data.json')],capture_output=True,text=True,timeout=60,check=True,preexec_fn=limits)
            report=json.loads(run.stdout);tests.extend(report['tests']);fingerprints[part]=report.get('fingerprint')
        except Exception:tests.append(dict(name=part+' — evaluation',score=0,max_score=sum(config['points']),output='Unable to evaluate within the limits. Check the saved graph and tensor sizes.',visibility='visible'))
    if reference.get('sameParameters'):
        rule=reference['sameParameters'];values=[fingerprints.get(part) for part in rule['parts']];ok=all(values) and len(set(values))==1
        tests.append(dict(name='Unchanged parameters across comparison files',score=rule['points'] if ok else 0,max_score=rule['points'],output='Comparison files must use the same learned model parameters.',visibility='visible'))
    result=dict(tests=tests,score=sum(t['score'] for t in tests),output='Predictions recomputed from saved parameters on canonical data; cached losses and predictions ignored.',visibility='visible',stdout_visibility='hidden',output_format='text',test_output_format='text',test_name_format='text')
except Exception:result=dict(score=0,output='Autograder setup failed. Contact the instructor.',visibility='visible')
(root/'results'/'results.json').write_text(json.dumps(result,allow_nan=False))
