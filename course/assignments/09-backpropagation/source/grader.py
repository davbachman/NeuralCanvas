"""Verify saved forward models and derivative constructions, independent of cached gradients."""
import json,sys
from pathlib import Path
import numpy as np
from graph_eval import Graph,require
PART_POINTS={'chain':50,'shared':50}

def grade(part,path,data):
    tests=[]
    def credit(name,points,ok,message):tests.append(dict(name=part+' — '+name,score=points if ok else 0,max_score=points,output=message,visibility='visible'))
    try:
        require(part in PART_POINTS,'Unknown part.');m=Graph(path,'line-1d');require(m.loss_kind=='mse','Use Mean squared error.')
        roles={'w':.8,'b1':-.2,'v1':1.1,'c':.1}
        if part=='shared':roles.update(b2=.4,v2=-.7)
        require(len(m.params)==len(roles) and all(v.size==1 and v.ndim==0 for v in m.params.values()),'Use exactly the assigned scalar Params, sharing one w.')
        ids={}
        for role,val in roles.items():
            matches=[k for k,v in m.params.items() if abs(float(v)-val)<1e-12]
            require(len(matches)==1,'Keep the prescribed initial value for '+role+'.');ids[role]=matches[0]
        require(sum(n['type']=='activation' for n in m.nodes.values())==(2 if part=='shared' else 1),'Use the assigned number of hidden activations.')
        x=np.array(data['x']);y=np.array(data['y']);candidates=None;paths=[None,None];error=0
        # Vary parameters and targets during evaluation: copying an inspector value or
        # hardcoding the published initial weights cannot implement a derivative.
        for trial in range(4):
            p={role:val+(trial*.07*(i+1) if trial else 0) for i,(role,val) in enumerate(roles.items())}
            yy=y+trial*.31;xx=x+trial*.023
            h1=np.tanh(p['w']*xx+p['b1']);h2=np.tanh(p['w']*xx+p.get('b2',0))
            want=p['v1']*h1+p['c']+(p['v2']*h2 if part=='shared' else 0)
            pred,_,values,used=m.evaluate(xx,yy,{ids[r]:np.asarray(v) for r,v in p.items()})
            require(used==set(m.params),'Every assigned Param must contribute to the prediction.')
            require(pred.shape==want.shape and np.allclose(pred,want,rtol=1e-9,atol=1e-10),'Build the assigned forward network using its Params.')
            d1=2*(want-yy)*p['v1']*(1-h1*h1)*xx
            d2=2*(want-yy)*p.get('v2',0)*(1-h2*h2)*xx
            total=d1+d2
            def matching(expected):return {k for k,v in values.items() if v.shape==expected.shape and np.allclose(v,expected,rtol=1e-8,atol=1e-9)}
            found=matching(total);candidates=found if candidates is None else candidates&found
            for i,desired in enumerate((d1,d2)):
                found=matching(desired);paths[i]=found if paths[i] is None else paths[i]&found
        credit('forward model',15,True,'Correct fixed forward network, including parameter sharing.')
        if part=='chain':credit('chain-rule derivative',35,bool(candidates),'The grader recomputes dL/dw across examples and changed parameters; a connected derivative branch must agree.')
        else:
            credit('first path contribution',10,bool(paths[0]),'Construct the first contribution to dL/dw.')
            credit('second path contribution',10,bool(paths[1]),'Construct the second contribution to dL/dw.')
            credit('accumulated gradient',15,bool(candidates),'The shared parameter receives the sum of both path contributions.')
    except Exception as e:
        remaining=PART_POINTS.get(part,50)-sum(t['max_score'] for t in tests)
        if remaining:credit('model validation',remaining,False,str(e)[:500])
    return {'tests':tests,'score':sum(t['score'] for t in tests)}
if __name__=='__main__':print(json.dumps(grade(sys.argv[1],sys.argv[2],json.loads(Path(sys.argv[3]).read_text())),allow_nan=False))
