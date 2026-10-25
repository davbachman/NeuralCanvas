"""Recompute polynomial predictions and L2 objectives from saved model graphs."""
import json,sys
from pathlib import Path
import numpy as np
from graph_eval import graph_from_file,compile_graph,require,finite,tensor,ZERO
VERSION='regularization-02-v1'
PART_POINTS={'baseline':30,'polynomial':30,'regularized':40}

def features(data,degree):
 x=np.array([r['weight'] for r in data['rows']]);y=np.array([r['mpg'] for r in data['rows']]);train=np.array([r['split']=='train' for r in data['rows']])
 z=(x-x[train].mean())/x[train].std()
 powers=np.column_stack([z**k for k in range(1,degree+1)])
 powers=(powers-powers[train].mean(0))/powers[train].std(0)
 return x,y,train,np.column_stack([np.ones(len(x)),powers])

def evaluate(poly,x):
 return sum((c*x**p[0] for p,c in poly.terms.items()), np.zeros_like(x))

def model_result(path,part,data):
 tests=[];metrics={};degree=1 if part=='baseline' else 5
 weights={'saved model and dataset':5,'model form':15,'training fit':5,'validation MSE':5}
 if part=='regularized':weights['L2 configuration']=10
 def add(name,ok,message):tests.append({'name':part+' — '+name,'score':weights[name] if ok else 0,'max_score':weights[name],'output':message,'visibility':'visible'})
 try:
  graph=graph_from_file(path,data);add('saved model and dataset',True,'Original dataset and split verified.')
 except Exception as e:
  for name in weights:add(name,False,str(e) if name=='saved model and dataset' else 'Upload a readable saved model to evaluate this part.')
  return {'tests':tests,'metrics':metrics}
 try:
  poly,params,used,scaled=compile_graph(graph)
  require(used=={0} and used<=scaled,'Use weight only, standardized before learned coefficients; leave model_year unused.')
  require(set(poly.terms)<={(k,0,0) for k in range(degree+1)},'Use exactly the assigned polynomial family.')
  require(len(params)==degree+1,'Use one scalar Param for each coefficient, including the intercept.')
  x,y,train,A=features(data,degree);coeff=np.zeros(degree+1);matched={};nodes={n['id']:n for n in graph['nodes']}
  for key,index in params:
   value=tensor(nodes[key]['params'].get('value',.5));require(value.size==1 and index==0,'Use separate scalar Param blocks for this assignment.')
   original=value.flat[0].terms.get(ZERO,0)
   plus=compile_graph(graph,(key,index,original+.25))[0];minus=compile_graph(graph,(key,index,original-.25))[0]
   for changed in [plus,minus]:require(set(changed.terms)<={(k,0,0) for k in range(degree+1)},'Unassigned terms depend on a parameter.')
   direction=evaluate(plus-poly,x)/.25
   require(np.allclose(direction,evaluate(poly-minus,x)/.25,atol=2e-5,rtol=2e-5),'Parameters must be linear coefficients, not multiplied together.')
   matches=[k for k in range(degree+1) if np.allclose(direction,A[:,k],atol=2e-5,rtol=2e-5)]
   require(len(matches)==1 and matches[0] not in matched,'Each coefficient must multiply its own standardized power (or 1 for the intercept). Fit every feature on the 24 training rows.')
   k=matches[0];matched[k]=key;coeff[k]=original
  require(np.allclose(evaluate(poly,x),A@coeff,atol=1e-4,rtol=1e-5),'Do not add fixed fitted coefficients or other prediction terms.')
  add('model form',True,'Correct independently trainable polynomial coefficients and training-fitted feature scaling.')
 except Exception as e:
  for name in weights:
   if name!='saved model and dataset':add(name,False,str(e) if name=='model form' else 'Requires the assigned model form.')
  return {'tests':tests,'metrics':metrics}
 loss=next(n for n in graph['nodes'] if n['type']=='loss')['params'];lam=loss.get('regularizationStrength',0);kind=loss.get('regularization','none')
 config=True
 if part=='regularized':
  ids=loss.get('regularizationParameterIds',[n['id'] for n in graph['nodes'] if n['type']=='weight'])
  config=kind=='l2' and finite(lam) and 0<lam<=10 and isinstance(ids,list) and all(isinstance(i,str) for i in ids) and len(ids)==len(set(ids)) and set(ids)=={matched[k] for k in range(1,degree+1)}
  add('L2 configuration',config,'Penalize exactly the five feature coefficients, excluding the intercept; use 0 < lambda <= 10.' if not config else f'L2 lambda={lam:g}, feature coefficients only.')
 else:
  config=kind=='none' or (finite(lam) and lam==0)
  lam=0
 if not config:
  add('training fit',False,'Use the required penalty configuration before grading fit.')
  add('validation MSE',False,'Requires the required penalty configuration.')
  return {'tests':tests,'metrics':metrics}
 pred=evaluate(poly,x);tr=float(np.mean((pred[train]-y[train])**2));val=float(np.mean((pred[~train]-y[~train])**2));penalty=float(lam/2*np.sum(coeff[1:]**2));objective=tr+penalty
 T=A[train];optimal=np.linalg.solve(2*T.T@T/train.sum()+np.diag([0]+[lam]*degree),2*T.T@y[train]/train.sum());best=float(np.mean((T@optimal-y[train])**2)+lam/2*np.sum(optimal[1:]**2))
 require(all(finite(v) for v in [tr,val,objective,best]),'Nonfinite predictions.')
 metrics={'train':tr,'validation':val,'penalty':penalty,'objective':objective,'optimalObjective':best,'lambda':lam}
 add('training fit',objective<=best+.2+1e-8,f'Recomputed training objective {objective:.5f}; full credit within 0.20 of the independently calculated minimum {best:.5f}.')
 limit={'baseline':14,'polynomial':60,'regularized':16}[part]
 add('validation MSE',val<=limit+1e-8,f'Recomputed validation MSE {val:.5f} mpg²; full credit at or below {limit}.')
 return {'tests':tests,'metrics':metrics}

if __name__=='__main__':
 part,path,data_path=sys.argv[1:]
 try:r=model_result(Path(path),part,json.loads(Path(data_path).read_text()))
 except Exception as e:r={'tests':[{'name':part+' — evaluation','score':0,'max_score':PART_POINTS[part],'output':'Unable to evaluate: '+str(e),'visibility':'visible'}]}
 print(json.dumps(r,allow_nan=False))
