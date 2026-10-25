"""Bounded, independent evaluation of the prescribed concrete regression graphs."""
import ast
import json
import sys
from pathlib import Path
import numpy as np

PART_POINTS = {'linear': 40, 'mlp': 60}
LIMITS = {'linear': (120, 150), 'mlp': (45, 65)}

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
    source = datasets[0]; p = source['params']; csv = p.get('customCsv', {})
    require(p.get('dataset') == 'custom-csv' and csv.get('task') == 'regression' and csv.get('hasHeader') is True and csv.get('targetColumn') == 8, 'In the dataset browser choose Regression → Concrete strength, with strength as the target.')
    rows = csv.get('rows', []); expected = data['rows']
    require(len(rows) == len(expected)+1 and rows[0] == data['headers'] and csv.get('splits') == [r['split'] for r in expected], 'Keep the issued columns, rows, and split unchanged.')
    for actual, original in zip(rows[1:], expected):
        require(isinstance(actual, list) and len(actual) == 9 and all(isinstance(v, str) and len(v) <= 64 for v in actual), 'Invalid CSV cells.')
        require(np.allclose(np.array(actual, float), original['values'], atol=1e-9, rtol=0), 'Dataset values differ from concrete.csv.')
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
    require(len(losses) == 1 and losses[0]['type'] == 'loss', 'Use one MSE Loss block.')
    loss = losses[0]; lp = loss['params']
    require(lp.get('loss','mse') == 'mse', 'Use mean squared error.')
    strength = lp.get('regularizationStrength',0)
    require(isinstance(strength,(int,float)) and np.isfinite(strength) and strength >= 0 and lp.get('regularization','none') in ('none','l1','l2') and (lp.get('regularization','none') == 'none' or strength == 0), 'Use no loss penalty.')
    raw = np.array([r['values'] for r in data['rows']]); train = np.array([r['split']=='train' for r in data['rows']])
    means = raw[train,:8].mean(0); scales = raw[train,:8].std(0); scales[scales==0]=1
    cache={}; active=set(); parameters={}
    def edge_value(e):
        slot=e.get('sourceSlot',0)
        if e['source']==source:
            require(slot<=8, 'Invalid Dataset column.')
            return raw[:,slot], ('column',slot)
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
            require(isinstance(values,list) and 1<=len(values)<=512 and all(type(x) in (int,float) and np.isfinite(x) for x in values), 'Use finite Param tensors of at most 512 values.')
            require(int(np.prod(shape))==len(values), 'Param shape and value count differ.')
            out=np.array(values).reshape(shape);parameters[key]=out;sig=('param',key)
        elif kind=='concat':
            require(len(args)==8 and p.get('axis',1)==1 and all(s==('column',i) for i,(_,s) in enumerate(args)), 'Concatenate the eight raw features in CSV order on axis 1.')
            out=np.column_stack([v for v,_ in args]);sig=('raw_features',)
        elif kind=='standardize':
            require(len(args)==1 and args[0][1]==('raw_features',), 'Standardize the concatenated feature matrix.')
            st=p.get('standardization',{});m=np.asarray(st.get('mean',[]));s=np.asarray(st.get('scale',[]))
            require(m.shape==s.shape==(8,) and np.allclose(m,means,atol=1e-7,rtol=1e-7) and np.allclose(s,scales,atol=1e-7,rtol=1e-7), 'Fit all eight feature columns on training rows only.')
            out=(args[0][0]-m)/s;sig=('features',)
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
        elif kind=='reshape':
            require(len(args)==1 and args[0][1]==('column',8), 'Only reshape the strength target.')
            shape=p.get('shape',[-999]);require(shape in ([-1,1],[len(raw),1]), 'Reshape strength to -1, 1.')
            out=args[0][0].reshape(-1,1);sig=('target',)
        else:raise ValueError('Use the assigned matrix, addition, standardization, and ReLU operations.')
        require(out.size<=len(raw)*128 and np.isfinite(out).all(), 'Invalid or oversized intermediate tensor.')
        active.remove(key);cache[key]=(out,sig);return cache[key]
    edges=incoming[loss['id']];require(len(edges)==2, 'Connect predictions and strength to Loss.')
    pred,sig=edge_value(edges[0]);target,ts=edge_value(edges[1])
    require(ts==('target',) and pred.shape==target.shape==(len(raw),1), 'Predictions and strength targets must both be batch-by-1 columns.')
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
        first=dense(sig,8,1,())
    else:
        hidden=dense(sig,16,1,());require(hidden[0]=='relu', 'Place ReLU between the two learned layers.')
        first=dense(hidden[1],8,16,(16,))
    require(first==('features',) and len(used)==len(set(used)) and set(used)==set(parameters), 'Use the specified features and separate Params for each layer.')
    return float(np.mean((pred[train]-target[train])**2)),float(np.mean((pred[~train]-target[~train])**2))

def model_result(path,part,data):
    tests=[];metrics={};points=(5,15,10,10) if part=='linear' else (5,25,15,15)
    def add(i,ok,msg):tests.append({'name':part+' — '+['saved model and dataset','model form','training MSE','validation MSE'][i],'score':points[i] if ok else 0,'max_score':points[i],'output':msg,'visibility':'visible'})
    try:g,source=load(path,data);add(0,True,'Original concrete dataset and split verified.')
    except Exception as e:
        for i in range(4):add(i,False,str(e) if i==0 else 'Upload a valid model to evaluate this part.')
        return {'tests':tests,'metrics':metrics}
    try:tr,va=evaluate(g,source,data,part);add(1,True,'Correct matrix model, fitted scaling, activation, and target shape.')
    except Exception as e:
        for i in range(1,4):add(i,False,str(e) if i==1 else 'Performance credit requires the assigned model form.')
        return {'tests':tests,'metrics':metrics}
    metrics={'train':tr,'validation':va}
    for i,v,limit in [(2,tr,LIMITS[part][0]),(3,va,LIMITS[part][1])]:add(i,v<=limit+1e-8,f'Recomputed MSE {v:.6f} MPa²; full credit at or below {limit}.')
    return {'tests':tests,'metrics':metrics}

if __name__=='__main__':
    part,path,data_path=sys.argv[1:]
    try:result=model_result(Path(path),part,json.loads(Path(data_path).read_text()))
    except Exception:result={'tests':[{'name':part+' — evaluation','score':0,'max_score':PART_POINTS[part],'output':'Unable to evaluate this file.','visibility':'visible'}]}
    print(json.dumps(result,allow_nan=False))
