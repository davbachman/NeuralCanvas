#!/usr/bin/env python3
"""Run each bounded model check independently, preserving partial credit."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ.get('AUTOGRADER_ROOT','/autograder'));source=root/'source';(root/'results').mkdir(parents=True,exist_ok=True)
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
def limits():
    if sys.platform.startswith('linux'):
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(100,100))
try:
    data=json.loads((source/'data.json').read_text());tests=[]
    for part,spec in data['parts'].items():
        try:
            run=subprocess.run([sys.executable,str(source/'grader.py'),part,str(root/'submission'/(part+'.json')),str(source/'data.json')],capture_output=True,text=True,timeout=120,check=True,preexec_fn=limits)
            tests.extend(json.loads(run.stdout)['tests'])
        except Exception:tests.append(dict(name=part+' — bounded evaluation',score=0,max_score=spec['points'],output='Could not evaluate this saved model within the limits.',visibility='visible'))
    result=dict(tests=tests,score=sum(t['score'] for t in tests),output='Independent canonical-data evaluation; cached reward and training reports ignored.',visibility='visible',stdout_visibility='hidden')
except Exception:result=dict(score=0,output='Autograder setup failed. Contact the instructor.',visibility='visible')
(root/'results/results.json').write_text(json.dumps(result,allow_nan=False))
