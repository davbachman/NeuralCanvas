"""Bounded, independent evaluation of the prescribed rings classification graphs."""
import ast
import json
import sys
from pathlib import Path
import numpy as np

PART_POINTS = {'linear': 40, 'mlp': 60}
LIMITS = {'linear': (0.56, 0.57, 0), 'mlp': (0.08, 0.10, 0.97)}

def require(condition, message):
    if not condition: raise ValueError(message)

def load(path, data):
    require(path.is_file() and not path.is_symlink(), 'Required saved model is missing.')
    require(path.stat().st_size <= 32 * 1024**2, 'Keep each saved model under 32 MiB.')
    f = json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON.')))
    require(f.get('kind') in ('neural-canvas-state', 'backprop-builder-state') and f.get('version') == 1, 'Use File → Save in Neural Canvas.')
    g = f['state']['graph']; nodes = g['nodes']; edges = g['edges']
    require(isinstance(nodes, list) and 1 <= len(nodes) <= 100 and isinstance(edges, list) and len(edges) <= 200, 'Limit models to 100 blocks and 200 wires.')
    require(all(isinstance(n, dict) and isinstance(n.get('id'), str) and len(n['id']) <= 128 and isinstance(n.get('params'), dict) for n in nodes), 'Invalid blocks.')
    ids = [n['id'] for n in nodes]; require(len(set(ids)) == len(ids), 'Block IDs must be unique.')
    datasets = [n for n in nodes if n.get('type') == 'dataset']; require(len(datasets) == 1, 'Use one Dataset block.')
    source = datasets[0]; p = source['params']
    csv=p.get('customCsv', {})
    require(p.get('dataset')=='custom-csv' and csv.get('task') == 'classification' and csv.get('hasHeader') is True and csv.get('targetColumn')==2, 'In the dataset browser choose Classification → Concentric rings, with class as the target.')
    rows=csv.get('rows',[]); expected=data['rows']
    require(len(rows)==len(expected)+1 and rows[0]==data['headers'] and csv.get('splits')==[r['split'] for r in expected], 'Keep the issued columns, rows, and split unchanged.')
    for actual, original in zip(rows[1:],expected):
        require(isinstance(actual,list) and len(actual)==3 and all(isinstance(v,str) and len(v)<=64 for v in actual), 'Invalid CSV cells.')
        require(np.allclose(np.array(actual,float), [*original['values'],original['label']], atol=1e-12,rtol=0), 'Dataset values differ from rings.csv.')
    for e in edges:
        require(isinstance(e, dict) and e.get('source') in ids and e.get('target') in ids, 'A wire references an unknown block.')
        require(all(type(e.get(k, 0)) is int and 0 <= e.get(k, 0) < 64 for k in ['inputSlot', 'sourceSlot']), 'Invalid wire ports.')
    return g, source['id']

def evaluate(g, source, data, part):
    nodes = {n['id']: n for n in g['nodes']}; incoming = {i: [] for i in nodes}
    for e in g['edges']: incoming[e['target']].append(e)
    for edges in incoming.values():
        edges.sort(key=lambda e:e.get('inputSlot',0))
        require([e.get('inputSlot',0) for e in edges] == list(range(len(edges))), 'Connect input ports once each without gaps.')
    require(not incoming[source], 'Dataset must not have incoming wires.')
    losses = [n for n in nodes.values() if n['type'] in ('loss','cross-entropy')]
    require(len(losses) == 1, 'Use one cross-entropy Loss block.')
    loss = losses[0]; lp = loss['params']
    require((loss['type']=='cross-entropy' or lp.get('loss','mse') == 'cross-entropy'), 'Use cross-entropy on logits.')
    strength = lp.get('regularizationStrength',0)
    require(isinstance(strength,(int,float)) and np.isfinite(strength) and strength >= 0 and lp.get('regularization','none') in ('none','l1','l2') and (lp.get('regularization','none') == 'none' or strength == 0), 'Use no loss penalty.')
    raw = np.array([r['values'] for r in data['rows']], dtype=float)
    labels = np.array([r['label'] for r in data['rows']])
    train = np.array([r['split']=='train' for r in data['rows']])
    cache={}; active=set(); parameters={}
    def edge_value(e):
        slot=e.get('sourceSlot',0)
        if e['source']==source:
            require(slot in (0,1,2), 'Invalid Dataset output.')
            return (raw[:,slot], ('column',slot)) if slot<2 else (labels, ('target',))
        require(slot==0, 'Use the first output of operation blocks.')
        return visit(e['source'])
    def visit(key):
        if key in cache:return cache[key]
        require(key not in active, 'Graph contains a cycle.');active.add(key)
        n=nodes[key];p=n['params'];kind=n['type'];args=[edge_value(e) for e in incoming[key]]
        if kind=='tensor-transform':kind=p.get('transform','reshape')
        if kind in ('input','target'):
            require(len(args)==1, 'Aliases must pass through a connected value.');out,sig=args[0]
        elif kind in ('weight','bias'):
            require(not args, 'Params must not have inputs.')
            value=p.get('value'); require(isinstance(value,dict), 'Initialize every Param tensor.')
            shape=value.get('shape');values=value.get('data')
            require(isinstance(shape,list) and len(shape)<=2 and all(type(d) is int and 1<=d<=128 for d in shape), 'Invalid Param shape.')
            require(isinstance(values,list) and 1<=len(values)<=4096 and all(type(x) in (int,float) and np.isfinite(x) for x in values), 'Use finite Param tensors of at most 4096 values.')
            require(int(np.prod(shape))==len(values), 'Param shape and value count differ.')
            out=np.array(values).reshape(shape);parameters[key]=out;sig=('param',key)
        elif kind=='matmul':
            require(len(args)==2 and all(v.ndim==2 for v,_ in args), 'Matrix product requires two matrices.')
            out=args[0][0]@args[1][0];sig=('matmul',args[0][1],args[1][1])
        elif kind in ('arithmetic','add'):
            require(len(args)==2, 'Use two-input addition for biases.')
            if kind=='arithmetic':
                tree=ast.parse(p.get('expression','x1 * x2'),mode='eval').body
                require(isinstance(tree,ast.BinOp) and isinstance(tree.op,ast.Add) and isinstance(tree.left,ast.Name) and isinstance(tree.right,ast.Name) and {tree.left.id,tree.right.id}=={'x1','x2'}, 'Use x1 + x2 for bias addition.')
            out=args[0][0]+args[1][0];sig=('add',args[0][1],args[1][1])
        elif kind=='activation':
            require(len(args)==1 and p.get('activation','identity')=='relu', 'Use ReLU in the hidden layer.')
            out=np.maximum(args[0][0],0);sig=('relu',args[0][1])
        elif kind=='concat':
            require(len(args)==2 and p.get('axis',1)==1 and [sig for _,sig in args]==[('column',0),('column',1)], 'Concatenate raw x1 then x2 on axis 1; do not add engineered features.')
            out=np.column_stack([v for v,_ in args]);sig=('features',)
        else:raise ValueError('Use the assigned concatenation, matrix, addition, and ReLU operations.')
        require(out.size<=len(raw)*128 and np.isfinite(out).all(), 'Invalid or oversized intermediate tensor.')
        active.remove(key);cache[key]=(out,sig);return cache[key]
    edges=incoming[loss['id']];require(len(edges)==2, 'Connect logits and class labels to Loss.')
    pred,sig=edge_value(edges[0]);target,ts=edge_value(edges[1])
    require(ts==('target',) and pred.shape==(len(raw),3) and target.shape==(len(raw),), 'Use three logits and the original class labels.')
    used=[]
    def dense(signature, inputs, outputs, bias_shape):
        require(signature[0]=='add', 'Add a separate learned bias to each matrix product.')
        a,b=signature[1:]
        if b[0]=='matmul':a,b=b,a
        require(a[0]=='matmul' and a[2][0]=='param' and b[0]=='param', 'Each layer must be X @ W + b with independent Params.')
        w,bi=a[2][1],b[1]
        require(parameters[w].shape==(inputs,outputs) and parameters[bi].shape==bias_shape, 'Check the assigned matrix and bias shapes.')
        used.extend([w,bi]);return a[1]
    if part=='linear':
        first=dense(sig,2,3,(3,))
    else:
        hidden=dense(sig,16,3,(3,));require(hidden[0]=='relu', 'Place ReLU between the two learned layers.')
        first=dense(hidden[1],2,16,(16,))
    require(first==('features',) and len(used)==len(set(used)) and set(used)==set(parameters), 'Use the specified features and separate Params for each layer.')
    shifted=pred-pred.max(axis=1,keepdims=True)
    ce=np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(raw)),labels]
    return float(ce[train].mean()), float(ce[~train].mean()), float((pred[~train].argmax(axis=1)==labels[~train]).mean())

def model_result(path,part,data):
    tests=[];metrics={};points=(5,20,10,5,0) if part=='linear' else (5,25,10,10,10)
    def add(i,ok,msg):tests.append({'name':part+' — '+['saved model and dataset','model form','training cross-entropy','validation cross-entropy','validation accuracy'][i],'score':points[i] if ok else 0,'max_score':points[i],'output':msg,'visibility':'visible'})
    try:g,source=load(path,data);add(0,True,'Original rings dataset and split verified.')
    except Exception as e:
        for i in range(5):add(i,False,str(e) if i==0 else 'Upload a valid model to evaluate this part.')
        return {'tests':tests,'metrics':metrics}
    try:tr,va,acc=evaluate(g,source,data,part);add(1,True,'Correct raw feature matrix, model, activation, and label connection.')
    except Exception as e:
        for i in range(1,5):add(i,False,str(e) if i==1 else 'Performance credit requires the assigned model form.')
        return {'tests':tests,'metrics':metrics}
    metrics={'train':tr,'validation':va,'accuracy':acc}
    for i,v,limit in [(2,tr,LIMITS[part][0]),(3,va,LIMITS[part][1])]:add(i,v<=limit+1e-8,f'Recomputed cross-entropy {v:.6f}; full credit at or below {limit}.')
    add(4,acc>=LIMITS[part][2]-1e-8, f'Recomputed held-out accuracy {acc:.1%}; informational only for the linear baseline.' if part=='linear' else f'Recomputed held-out accuracy {acc:.1%}; full credit at or above {LIMITS[part][2]:.0%}.')
    return {'tests':tests,'metrics':metrics}

if __name__=='__main__':
    part,path,data_path=sys.argv[1:]
    try:result=model_result(Path(path),part,json.loads(Path(data_path).read_text()))
    except Exception:result={'tests':[{'name':part+' — evaluation','score':0,'max_score':PART_POINTS[part],'output':'Unable to evaluate this file.','visibility':'visible'}]}
    print(json.dumps(result,allow_nan=False))
