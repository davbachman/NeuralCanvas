"""Independent, bounded NumPy evaluator for the prescribed text models.

The companion data.json is instructor-owned. Submitted vocabularies, documents,
and splits must match it; cached features, predictions and losses are never read.
"""
import ast
import json
from pathlib import Path
import numpy as np

def require(ok, message):
    if not ok: raise ValueError(message)

def load(path, data, part):
    data=data['parts'][part]
    counts=part in ('reviews','stories')
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Upload the required saved model JSON.')
    require(path.stat().st_size <= 32*1024**2, 'Keep saved files under 32 MiB.')
    f = json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON.')))
    require(f.get('kind') in ('neural-canvas-state','backprop-builder-state') and f.get('version') == 1, 'Use File → Save in Neural Canvas.')
    g = f['state']['graph']; ns = g['nodes']; es = g['edges']
    require(isinstance(ns,list) and 1 <= len(ns) <= 100 and isinstance(es,list) and len(es) <= 200, 'Use at most 100 blocks and 200 wires.')
    require(all(isinstance(n,dict) and isinstance(n.get('id'),str) and len(n['id'])<=128 and isinstance(n.get('params'),dict) for n in ns), 'Invalid blocks.')
    ids = [n['id'] for n in ns]
    require(len(set(ids)) == len(ids), 'Block IDs must be unique.')
    sources = [n for n in ns if n.get('type') == 'dataset']
    require(len(sources) == 1, 'Use exactly one Dataset block.')
    source = sources[0]; p = source['params']; actual = p.get('textData',{}); expected = data['textData']
    require(p.get('dataset') == 'custom-text', 'Import the prepared dataset JSON using Text / sentiment… .')
    for field in ['version','task','tokenizer','lowercase','vocabulary','maxLength','stride','documents','fixedLength','classLabels','factWords','maxFacts','targetMode']:
        # UI saves false for non-fixed-length inputs; older exports may omit it.
        default = False if field == 'fixedLength' else None
        require(actual.get(field,default) == expected.get(field,default), 'Keep the issued data, split, vocabulary and import settings unchanged (check '+field+').')
    require(actual.get('representation') == ('counts' if counts else 'tokens'), 'Use the prepared '+('word-count' if counts else 'token-ID')+' dataset for this assignment.')
    require(p.get('trainPercent') is None, 'Keep the supplied train/held-out split.')
    for e in es:
        require(isinstance(e,dict) and e.get('source') in ids and e.get('target') in ids, 'A wire references an unknown block.')
        require(all(type(e.get(k,0)) is int and 0<=e.get(k,0)<64 for k in ['sourceSlot','inputSlot']), 'Invalid ports.')
    return g, source['id']

def evaluate(g, source, data, part):
    data=data['parts'][part]
    counts=part in ('reviews','stories')
    length=data['textData']['maxLength']
    embedding_width=data['embeddingWidth']
    hidden_width=data['hiddenWidth']
    nodes = {n['id']:n for n in g['nodes']}; incoming = {i:[] for i in nodes}
    for e in g['edges']: incoming[e['target']].append(e)
    for edges in incoming.values():
        edges.sort(key=lambda e:e.get('inputSlot',0))
        require([e.get('inputSlot',0) for e in edges] == list(range(len(edges))), 'Connect each input once, without gaps.')
    require(not incoming[source], 'Dataset must have no inputs.')
    losses = [n for n in nodes.values() if n['type'] in ('loss','cross-entropy')]
    require(len(losses)==1, 'Use exactly one Loss block.')
    loss = losses[0]; p=loss['params']; sentiment=data['textData']['task']=='sentiment'
    kind = 'cross-entropy' if loss['type']=='cross-entropy' else p.get('loss','mse')
    require(kind==('binary-cross-entropy-with-logits' if sentiment else 'cross-entropy'), 'Connect logits directly to the assigned logits loss.')
    strength=p.get('regularizationStrength',0)
    require(type(strength) in (int,float) and np.isfinite(strength) and strength>=0 and p.get('regularization','none') in ('none','l1','l2') and (p.get('regularization','none')=='none' or strength==0), 'Use no loss penalty.')
    docs=data['textData']['documents']; count=len(docs); width=len(data['textData']['vocabulary'])
    labels=np.array(data['labels'],dtype=int); train=np.array([d['split']=='train' for d in docs])
    if counts:
        features=np.stack([np.bincount(ids,minlength=width) for ids in data['ids']])[:,None,:]
        feature_sig=('counts',)
    else:
        require(len({len(ids) for ids in data['ids']})==1, 'Canonical story lengths differ.')
        features=np.array(data['ids'],dtype=int); feature_sig=('ids',)
    cache={}; active=set(); params={}
    # Each signal carries a batch flag so axis 0 in the app means the first
    # PER-EXAMPLE axis, never the leading evaluation-batch axis here.
    def edge_value(e):
        slot=e.get('sourceSlot',0)
        if e['source']==source:
            require(slot in (0,2), 'Use text features and labels, not position IDs.')
            return (features,feature_sig,True) if slot==0 else (labels,('target',),True)
        require(slot==0, 'Use the first output of operation blocks.')
        return visit(e['source'])
    def visit(key):
        if key in cache:return cache[key]
        require(key not in active,'Graph contains a cycle.');active.add(key)
        n=nodes[key];p=n['params'];kind=n['type'];args=[edge_value(e) for e in incoming[key]]
        if kind=='tensor-transform':kind=p.get('transform','reshape')
        batch=any(a[2] for a in args)
        if kind in ('weight','bias'):
            require(not args,'Params must not have inputs.')
            value=p.get('value');require(isinstance(value,dict),'Initialize Param tensors.')
            shape=value.get('shape');values=value.get('data')
            require(isinstance(shape,list) and 1<=len(shape)<=2 and all(type(d) is int and 1<=d<=1000 for d in shape), 'Invalid Param shape.')
            require(isinstance(values,list) and 1<=len(values)<=16000 and all(type(x) in (int,float) and np.isfinite(x) for x in values), 'Use finite Params with at most 16,000 values.')
            require(int(np.prod(shape))==len(values),'Param shape and value count differ.')
            out=np.array(values).reshape(shape);sig=('param',key);params[key]=out
        elif kind in ('input','target'):
            require(len(args)==1,'Aliases must have one input.');out,sig,batch=args[0]
        elif kind=='matmul':
            require(len(args)==2 and args[0][2] and not args[1][2] and args[0][0].ndim==3 and args[1][0].ndim==2,'Multiply the feature matrix by a Param matrix.')
            out=args[0][0]@args[1][0];sig=('matmul',args[0][1],args[1][1])
        elif kind in ('arithmetic','add'):
            require(len(args)==2,'Use two-input addition for a bias.')
            if kind=='arithmetic':
                tree=ast.parse(p.get('expression','x1 * x2'),mode='eval').body
                require(isinstance(tree,ast.BinOp) and isinstance(tree.op,ast.Add) and isinstance(tree.left,ast.Name) and isinstance(tree.right,ast.Name) and {tree.left.id,tree.right.id}=={'x1','x2'},'Use x1 + x2 for bias addition.')
            require(sum(a[2] for a in args)==1,'Add an independent bias Param to model output.')
            out=args[0][0]+args[1][0];sig=('add',args[0][1],args[1][1])
        elif kind=='embedding':
            require(len(args)==2 and args[0][1][0]=='param' and args[1][1]==('ids',),'Embedding takes the table first and token IDs second.')
            require(args[0][0].shape==(width,embedding_width),'Use an embedding table of shape vocabulary size × '+str(embedding_width)+'.')
            out=args[0][0][args[1][0]];sig=('embedding',args[0][1])
        elif kind=='one-hot':
            require(len(args)==1 and args[0][1]==('ids',) and p.get('numClasses',2)==width,'One-hot width must equal the vocabulary size.')
            out=np.eye(width)[args[0][0]];sig=('one-hot',)
        elif kind=='reshape':
            features_per_token=width if part=='onehot' else embedding_width
            flat_width=length*features_per_token
            require(len(args)==1 and args[0][2] and args[0][0].shape==(count,length,features_per_token) and p.get('shape') in ([1,flat_width],[1,-1]),'Preserve all '+str(length)+' token vectors in order as a single row.')
            out=args[0][0].reshape(count,1,flat_width);sig=('ordered',args[0][1])
        elif kind=='activation' and p.get('activation','identity')=='relu':
            require(len(args)==1,'ReLU needs one input.')
            out=np.maximum(args[0][0],0);sig=('relu',args[0][1])
        elif kind=='softmax' or kind=='activation':
            require(len(args)==1,'Probability branch needs one input.')
            if kind=='activation':
                require(sentiment and p.get('activation','identity')=='sigmoid','Use Sigmoid on the probability branch.')
                out=np.exp(-np.logaddexp(0,-args[0][0]));sig=('sigmoid',args[0][1])
            else:
                require(not sentiment and p.get('axis',-1) in (-1,1),'Use Softmax over the six class scores.')
                z=args[0][0]-args[0][0].max(axis=-1,keepdims=True);out=np.exp(z);out/=out.sum(axis=-1,keepdims=True);sig=('softmax',args[0][1])
        else:raise ValueError('Use the assigned matrix, bias, one-hot/embedding, reshape and ReLU operations.')
        require(out.size<=count*2000 and np.isfinite(out).all(),'Invalid or oversized intermediate result.')
        active.remove(key);cache[key]=(out,sig,batch);return cache[key]
    edges=incoming[loss['id']];require(len(edges)==2,'Connect logits and the original target to Loss.')
    pred,sig,_=edge_value(edges[0]);_,target_sig,_=edge_value(edges[1])
    outputs=1 if sentiment else len(data['textData']['classLabels'])
    require(target_sig==('target',) and pred.shape==(count,1,outputs),'Check the output scores and target connection.')
    used=[]
    def dense(signature,inputs,outputs):
        require(signature[0]=='add','Each learned layer must be X @ W + b.')
        a,b=signature[1:]
        if b[0]=='matmul':a,b=b,a
        require(a[0]=='matmul' and a[2][0]=='param' and b[0]=='param','Use separate matrix and bias Params.')
        w,bi=a[2][1],b[1]
        require(params[w].shape==(inputs,outputs) and params[bi].shape==(outputs,),'Check matrix and bias-vector shapes.')
        used.extend([w,bi]);return a[1]
    first=dense(sig,hidden_width,outputs)
    require(first[0]=='relu','Use the assigned ReLU hidden layer before the six logits.')
    flattened=dense(first[1],length*(width if part=='onehot' else embedding_width),hidden_width)
    require(flattened[0]=='ordered','Flatten all token vectors in their original order.')
    representation=flattened[1]
    if part=='onehot':
        require(representation==('one-hot',),'Part A requires flattened one-hot vectors without an embedding table.')
    else:
        if representation[0]=='embedding':table=representation[1]
        else:
            require(representation[0]=='matmul' and representation[1]==('one-hot',),'Use Embedding lookup or one-hot @ embedding table before flattening.')
            table=representation[2]
        require(table[0]=='param' and params[table[1]].shape==(width,embedding_width),'Use a vocabulary size × '+str(embedding_width)+' embedding table.')
        used.append(table[1])
    all_params={n['id'] for n in nodes.values() if n['type'] in ('weight','bias')}
    require(len(used)==len(set(used)) and set(used)==set(params)==all_params,'Use only separate Params for the assigned layers; remove unused Params.')
    probability_kind='sigmoid' if sentiment else 'softmax'
    valid_branch=False
    for node in nodes.values():
        if (sentiment and node['type']=='activation' and node['params'].get('activation')=='sigmoid') or (not sentiment and node['type']=='softmax'):
            try: valid_branch |= visit(node['id'])[1]==(probability_kind,sig)
            except (ValueError,KeyError,IndexError): pass
    require(valid_branch,'Keep a separate '+('Sigmoid' if sentiment else 'Softmax')+' branch connected to the logits.')
    pred=pred[:,0,:]
    if sentiment:
        z=pred[:,0];losses=np.logaddexp(0,z)-labels*z;predicted=(z>=0).astype(int)
    else:
        shifted=pred-pred.max(axis=1,keepdims=True)
        losses=np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(count),labels]
        predicted=pred.argmax(axis=1)
    metrics={'train':float(losses[train].mean()),'validation':float(losses[~train].mean()),'accuracy':float((predicted[~train]==labels[~train]).mean())}
    metrics['parameters']=sum(v.size for v in params.values())
    for variant in (0,1):
        mask=(~train)&(np.arange(count)%2==variant)
        metrics['dropBeforeMovingAccuracy' if variant==0 else 'moveBeforeDroppingAccuracy']=float((predicted[mask]==labels[mask]).mean())
    return metrics
