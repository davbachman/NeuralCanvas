"""Bounded semantic grading of editable scalar regression graphs.

No cached outputs, gradients, training history, or block labels are trusted.
"""
import ast
import json
import math
from pathlib import Path
import sys
import numpy as np

DATA = json.loads((Path(__file__).resolve().parent/'data.json').read_text()) if (Path(__file__).resolve().parent/'data.json').exists() else json.loads((Path(__file__).resolve().parents[1]/'reference/data.json').read_text())
PART_POINTS = {k:v['points'] for k,v in DATA['parts'].items()}

def require(ok,message):
 if not ok: raise ValueError(message)

def scalar(v):
 if isinstance(v,dict):
  require(v.get('shape') in ([],[1]) and len(v.get('data',[]))==1,'Use scalar Params and constants.')
  v=v['data'][0]
 require(type(v) in (int,float) and math.isfinite(v) and abs(v)<=1e6,'Invalid or excessive scalar.')
 return float(v)

def expression(text,args):
 require(isinstance(text,str) and len(text)<=120,'Use a short arithmetic expression.')
 tree=ast.parse(text.replace('^','**'),mode='eval')
 require(sum(1 for _ in ast.walk(tree))<=50,'Expression too complex.')
 used=set()
 def walk(n):
  if isinstance(n,ast.Constant):return scalar(n.value)
  if isinstance(n,ast.Name) and n.id.startswith('x') and n.id[1:].isdigit():
   i=int(n.id[1:])-1;require(0<=i<len(args),'Missing arithmetic input.');used.add(i);return args[i]
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):return -walk(n.operand) if isinstance(n.op,ast.USub) else walk(n.operand)
  if isinstance(n,ast.BinOp):
   a,b=walk(n.left),walk(n.right)
   if isinstance(n.op,ast.Add):return a+b
   if isinstance(n.op,ast.Sub):return a-b
   if isinstance(n.op,ast.Mult):return a*b
  raise ValueError('Use only numeric constants, x1/x2/…, parentheses, +, -, and *.')
 out=walk(tree.body);require(used==set(range(len(args))),'Connect exactly the inputs used.');return out

def graph_from_file(path):
 require(path.is_file() and not path.is_symlink() and path.stat().st_size<=1_000_000,'Upload a saved JSON at most 1 MB.')
 obj=json.loads(path.read_text(),parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON.')))
 require(obj.get('kind') in ('neural-canvas-state','backprop-builder-state') and obj.get('version')==1,'Use File → Save in Neural Canvas.')
 graph=obj.get('state',{}).get('graph',{});nodes=graph.get('nodes',[]);edges=graph.get('edges',[])
 require(isinstance(nodes,list) and 1<=len(nodes)<=30 and isinstance(edges,list) and len(edges)<=50,'Use at most 30 blocks and 50 wires.')
 allowed={'dataset','input','target','weight','bias','arithmetic','add','multiply','loss'}
 require(all(isinstance(n,dict) and isinstance(n.get('id'),str) and n.get('type') in allowed and isinstance(n.get('params'),dict) for n in nodes),'Use scalar arithmetic, Param, Input, Target, Dataset and Loss blocks.')
 ids=[n['id'] for n in nodes];require(len(set(ids))==len(ids),'Duplicate block IDs.')
 datasets=[n for n in nodes if n['type']=='dataset'];require(len(datasets)==1,'Use one Dataset block.')
 p=datasets[0]['params'];require(p.get('dataset')=='line-1d' and p.get('trainPercent') is None and not p.get('customCsv'),'Use unchanged Line regression (1D) with the default split.')
 for e in edges:
  require(isinstance(e,dict) and e.get('source') in ids and e.get('target') in ids,'Invalid wire.')
  require(type(e.get('inputSlot',0)) is int and 0<=e.get('inputSlot',0)<10 and type(e.get('sourceSlot',0)) is int and 0<=e.get('sourceSlot',0)<=1,'Invalid wire ports.')
 return graph

def evaluate(graph,x,y,override=None):
 nodes={n['id']:n for n in graph['nodes']};incoming={k:[] for k in nodes}
 for e in graph['edges']:incoming[e['target']].append(e)
 for es in incoming.values():
  es.sort(key=lambda e:e.get('inputSlot',0));require([e.get('inputSlot',0) for e in es]==list(range(len(es))),'Connect each input once without gaps.')
 losses=[n for n in nodes.values() if n['type']=='loss'];require(len(losses)==1,'Use one MSE Loss block.')
 loss=losses[0];p=loss['params'];require(p.get('loss')=='mse' and p.get('regularization','none')=='none' and not p.get('activationPenaltyNodeId') and not p.get('lossMaskNodeId') and p.get('targetMode','dataset')!='internal','Use MSE with no penalties.')
 memo={};visiting=set();used_params=set();feature_slots=set()
 def value(key,slot=0,prediction=True):
  n=nodes[key];kind=n['type'];p=n['params']
  if kind=='dataset':
   require(not incoming[key],'Dataset has no inputs.')
   if prediction:feature_slots.add(slot)
   return x if slot==0 else y
  require(slot==0,'Only Dataset has multiple output ports.')
  require(key not in visiting,'The graph contains a cycle.')
  cache=(key,prediction)
  if cache in memo:return memo[cache]
  visiting.add(key);args=[value(e['source'],e.get('sourceSlot',0),prediction) for e in incoming[key]]
  if kind in ('weight','bias'):
   require(not args and p.get('trainable',True) is not False,'Use unfrozen scalar Params.');used_params.add(key);out=scalar(override[key] if override and key in override else p.get('value',.5))
  elif kind in ('input','target'):
   require(len(args)<=1,'Input/Target accepts one wire.');out=args[0] if args else scalar(p.get('value',1 if kind=='input' else 0))
  elif kind=='arithmetic':out=expression(p.get('expression','x1 * x2'),args)
  elif kind in ('add','multiply'):
   require(len(args)==2,'Connect two operands.');out=args[0]+args[1] if kind=='add' else args[0]*args[1]
  else:raise ValueError('Unexpected operation upstream of Loss.')
  require(np.all(np.isfinite(out)) and np.max(np.abs(out))<1e12,'Invalid numerical computation.')
  visiting.remove(key);memo[cache]=out;return out
 require(len(incoming[loss['id']])==2,'Connect prediction and target to Loss.')
 prededge,targedge=incoming[loss['id']]
 pred=value(prededge['source'],prededge.get('sourceSlot',0),True)
 require(feature_slots=={0},'Prediction must use x and must not use y, even through cancellation.')
 target=value(targedge['source'],targedge.get('sourceSlot',0),False)
 require(np.allclose(target,y,atol=0,rtol=0),'Connect the unmodified y target to Loss.')
 require(len(used_params)==2,'Use exactly two independent scalar Params, slope and intercept.')
 return np.broadcast_to(pred,x.shape),used_params

def check_linear_structure(graph):
 """Symbolically bound degree before numeric checks, including zero coefficients."""
 nodes={n['id']:n for n in graph['nodes']};incoming={k:[] for k in nodes}
 for e in graph['edges']:incoming[e['target']].append(e)
 for es in incoming.values():es.sort(key=lambda e:e.get('inputSlot',0))
 visiting=set();memo={}
 def degree(key,slot=0):
  n=nodes[key];kind=n['type']
  if kind=='dataset':
   require(slot==0,'Never route the target into prediction.');return 1
  require(key not in visiting,'Cycle in prediction.');require(slot==0,'Invalid source port.')
  if key in memo:return memo[key]
  visiting.add(key);args=[degree(e['source'],e.get('sourceSlot',0)) for e in incoming[key]]
  if kind in ('weight','bias'):out=0
  elif kind in ('input','target'):out=args[0] if args else 0
  elif kind=='add':out=max(args)
  elif kind=='multiply':out=sum(args)
  elif kind=='arithmetic':
   text=n['params'].get('expression','x1 * x2');require(isinstance(text,str) and len(text)<=120,'Use a short arithmetic expression.');tree=ast.parse(text.replace('^','**'),mode='eval')
   def walk(a):
    if isinstance(a,ast.Constant):return 0
    if isinstance(a,ast.Name) and a.id.startswith('x') and a.id[1:].isdigit():return args[int(a.id[1:])-1]
    if isinstance(a,ast.UnaryOp) and isinstance(a.op,(ast.UAdd,ast.USub)):return walk(a.operand)
    if isinstance(a,ast.BinOp):
     l,r=walk(a.left),walk(a.right)
     if isinstance(a.op,(ast.Add,ast.Sub)):return max(l,r)
     if isinstance(a.op,ast.Mult):return l+r
    raise ValueError('Use linear arithmetic operations.')
   out=walk(tree.body)
  else:raise ValueError('Unsupported prediction block.')
  require(out<=1,'Do not build powers or products of input features, even if their current coefficients are zero.')
  visiting.remove(key);memo[key]=out;return out
 loss=next(n for n in graph['nodes'] if n['type']=='loss')
 e=next(e for e in incoming[loss['id']] if e.get('inputSlot',0)==0)
 degree(e['source'],e.get('sourceSlot',0))

def coefficients(graph):
 check_linear_structure(graph)
 # Probe target separately so labels cannot masquerade as features.
 x=np.array([-3.,-1.,0.,.5,2.,4.]);y=np.array([7.,2.,-4.,8.,1.,9.])
 pred,params=evaluate(graph,x,y);b=pred[2];w=pred[3]*2-b*2
 require(np.allclose(pred,b+w*x,atol=1e-9,rtol=1e-9),'Build a linear model b + w*x.')
 directions=[]
 for key in params:
  n=next(n for n in graph['nodes'] if n['id']==key);v=scalar(n['params'].get('value',.5))
  pp,_=evaluate(graph,x,y,{key:v+.125});pm,_=evaluate(graph,x,y,{key:v-.125})
  d=(pp-pred)/.125
  require(np.allclose(d,(pred-pm)/.125,atol=1e-8) and (np.allclose(d,x,atol=1e-8) or np.allclose(d,np.ones_like(x),atol=1e-8)),'One Param must be the slope and the other the intercept, each used once.')
  directions.append(d)
 require(np.linalg.matrix_rank(np.array(directions))==2,'Slope and intercept must vary independently.')
 return float(w),float(b)

def model_result(path,part,data):
 cfg=data['parts'][part];points=cfg['points'];tests=[];metrics={}
 def add(name,p,ok,msg):tests.append({'name':part+' — '+name,'max_score':p,'score':p if ok else 0,'output':msg,'visibility':'visible'})
 try:
  graph=graph_from_file(path);w,b=coefficients(graph);add('editable linear model',points*.4,True,'Verified original dataset, two coefficients, no target leakage, and MSE wiring.')
 except Exception as e:
  add('editable linear model',points*.4,False,str(e));add('parameter result',points*.6,False,'Requires a valid linear model.');return {'tests':tests,'metrics':metrics}
 rows=data['rows'];x=np.array([r['x'] for r in rows]);y=np.array([r['y'] for r in rows]);train=np.array([r['split']=='train' for r in rows]);mse=float(np.mean((b+w*x[train]-y[train])**2));val=float(np.mean((b+w*x[~train]-y[~train])**2));metrics={'slope':w,'intercept':b,'train':mse,'validation':val}
 if cfg['kind']=='zero':ok=abs(w)<=1e-8 and abs(b)<=1e-8;msg='Both coefficients must start at zero.'
 elif cfg['kind']=='update':
  rate=cfg['rate'];expected_w=rate*2*np.mean(x[train]*y[train]);expected_b=rate*2*np.mean(y[train]);ok=abs(w-expected_w)<=1e-6 and abs(b-expected_b)<=1e-6;msg=f'Checked one full-batch MSE gradient update from zero at learning rate {rate:g}; train MSE {mse:.6f}.'
 else:ok=mse<=cfg['trainMax'] and val<=cfg['validationMax'];msg=f'Recomputed training MSE {mse:.6f} (≤{cfg["trainMax"]}) and validation MSE {val:.6f} (≤{cfg["validationMax"]}).'
 add('parameter result',points*.6,ok,msg);return {'tests':tests,'metrics':metrics}

if __name__=='__main__':
 part,path,data_path=sys.argv[1:]
 try:report=model_result(Path(path),part,json.loads(Path(data_path).read_text()))
 except Exception as e:report={'tests':[{'name':part+' — evaluation','score':0,'max_score':PART_POINTS.get(part,0),'output':str(e),'visibility':'visible'}]}
 print(json.dumps(report,allow_nan=False))
