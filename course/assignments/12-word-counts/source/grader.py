"""Grade saved text models against canonical data with independent predictions."""
import json,sys
from pathlib import Path
from graph_eval import load,evaluate
PART_POINTS = {'reviews': 60, 'stories': 40}
LIMITS = {'reviews': (0.3, 0.6, 0.7), 'stories': (1.25, 1.45, 0)}
POINTS = {'reviews': (5, 25, 10, 10, 10), 'stories': (5, 20, 10, 5, 0)}

def model_result(path,part,data):
    tests=[];metrics={};points=POINTS[part]
    names=['saved model and original text','model and probability branch','training loss','held-out loss','held-out accuracy']
    def add(i,ok,message):
        tests.append({'name':part+' — '+names[i],'score':points[i] if ok else 0,'max_score':points[i],'output':message,'visibility':'visible'})
    try:g,source=load(path,data,part);add(0,True,'Original text, vocabulary, labels and split verified.')
    except Exception as e:
        for i in range(5):add(i,False,str(e) if i==0 else 'Upload a valid saved model for this part.')
        return {'tests':tests,'metrics':metrics}
    try:metrics=evaluate(g,source,data,part);add(1,True,'Correct assigned architecture; predictions recomputed from saved parameters.')
    except Exception as e:
        for i in range(1,5):add(i,False,str(e) if i==1 else 'Performance credit requires the assigned model form.')
        return {'tests':tests,'metrics':metrics}
    for i,key in [(2,'train'),(3,'validation')]:
        limit=LIMITS[part][i-2];value=metrics[key]
        add(i,value<=limit+1e-8,f'Recomputed loss {value:.6f}; full credit at or below {limit}.')
    accuracy=metrics['accuracy'];limit=LIMITS[part][2]
    add(4,accuracy>=limit-1e-8,f'Recomputed held-out accuracy {accuracy:.1%}; '+(f'full credit at or above {limit:.0%}.' if points[4] else 'informational only; this part is graded on structure and loss.'))
    return {'tests':tests,'metrics':metrics}

if __name__=='__main__':
    part,path,data_path=sys.argv[1:]
    try:result=model_result(Path(path),part,json.loads(Path(data_path).read_text()))
    except Exception:result={'tests':[{'name':part+' — evaluation','score':0,'max_score':PART_POINTS[part],'output':'Unable to evaluate this file.','visibility':'visible'}]}
    print(json.dumps(result,allow_nan=False))
