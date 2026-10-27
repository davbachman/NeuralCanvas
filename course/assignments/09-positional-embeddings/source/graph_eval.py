"""Bounded NumPy execution of editable story/language graphs; no submitted code runs."""
import ast, hashlib, json, math, re, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np


def require(ok, message):
    if not ok: raise ValueError(message)


def tokens(text, data):
    text=unicodedata.normalize('NFC',re.sub(r'<br\s*/?\s*>',' ',text,flags=re.I))
    if data['lowercase']:text=text.lower()
    return list(text) if data['tokenizer']=='character' else re.findall(r"[^\W_]+(?:['’][^\W_]+)*|[^\s\w]|_",text)


def encode_story(facts, question, data):
    if data['representation']=='story-question':
        lengths=[data['maxLength'],data.get('questionLength')]
        require(all(type(n) is int and 1<=n<=256 for n in lengths),'Invalid story/question lengths.')
        require(data['task']=='classification' and data['vocabulary'][:2]==['<unk>','<pad>'],'Invalid story/question vocabulary or task.')
        index={t:i for i,t in enumerate(data['vocabulary'])}
        require(isinstance(facts,str) and isinstance(question,str),'Supply a story and separate question.')
        rows=[[index.get(t,0) for t in tokens(s,data)] for s in (facts,question)]
        require(all(0<len(row)<=length for row,length in zip(rows,lengths)),'Story or question exceeds its token limit.')
        story,question=[np.array(row+[1]*(length-len(row))) for row,length in zip(rows,lengths)]
        return [story,question,np.arange(lengths[0]),(story!=1)[:,None],(question!=1)[:,None]]
    n,w=data['maxFacts'],data['factWords'];index={t:i for i,t in enumerate(data['vocabulary'])}
    require(0<len(facts)<=n and question.strip(),'Invalid story.')
    rows=[[index.get(t,0) for t in tokens(s,data)] for s in [*facts,question]]
    require(all(0<len(r)<=w for r in rows),'Story exceeds word limit.')
    padded=[r+[1]*(w-len(r)) for r in rows]
    f=np.array(padded[:-1]+[[1]*w]*(n-len(facts)));q=np.array([padded[-1]])
    return [f,q,np.arange(n),np.array([[1]*len(facts)+[0]*(n-len(facts))]),(f!=1)[...,None],(q!=1)[...,None]]


def examples(data):
    index={t:i for i,t in enumerate(data['vocabulary'])}
    for doc in data['documents']:
        if data['representation']=='story-question':
            yield encode_story(doc['text'],doc['question'],data),np.array([data['classLabels'].index(doc['label'])]),doc['split']
        elif data['representation']=='facts':
            yield encode_story(doc['facts'],doc['question'],data),np.array([data['classLabels'].index(doc['label'])]),doc['split']
        else:
            ids=[index.get(t,0) for t in tokens(doc['text'],data)]
            for start in range(0,len(ids)-data['maxLength'],data['stride']):
                n=data['maxLength'];x=np.array(ids[start:start+n]);target=ids[start+n:start+n+1] if data.get('targetMode')=='last' else ids[start+1:start+n+1]
                yield [x,np.arange(n)],np.array(target),doc['split']


def expression(source,args,signature=False):
    require(isinstance(source,str) and len(source)<=256,'Arithmetic expression too long.')
    tree=ast.parse(source,mode='eval').body
    def visit(t,depth=0):
        require(depth<20,'Arithmetic expression too deep.')
        if isinstance(t,ast.Constant) and type(t.value) in (int,float):
            require(math.isfinite(t.value) and abs(t.value)<=1e10,'Invalid arithmetic constant.')
            return ('number',round(t.value,5)) if signature else t.value
        if isinstance(t,ast.Name) and re.fullmatch('x[1-9][0-9]*',t.id):
            i=int(t.id[1:])-1;require(i<len(args),'Missing arithmetic input.');return args[i]
        if isinstance(t,ast.UnaryOp) and isinstance(t.op,ast.USub):
            x=visit(t.operand,depth+1);return ('neg',x) if signature else -x
        require(isinstance(t,ast.BinOp) and isinstance(t.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)),'Use only addition, subtraction, multiplication, and division.')
        a,b=visit(t.left,depth+1),visit(t.right,depth+1);op=type(t.op).__name__
        if signature:
            if op in ('Add','Mult'):
                terms=[]
                for item in (a,b):terms.extend(item[1:] if isinstance(item,tuple) and item[0]==op else [item])
                return (op,*sorted(terms,key=repr))
            return (op,a,b)
        return {'Add':np.add,'Sub':np.subtract,'Mult':np.multiply,'Div':np.divide}[op](a,b)
    return visit(tree)


class Graph:
    def __init__(self,g,data):
        ns,es=g['nodes'],g['edges'];require(1<=len(ns)<=200 and len(es)<=400,'Graph exceeds block/wire limits.')
        self.nodes={n['id']:n for n in ns};require(len(self.nodes)==len(ns),'Duplicate block IDs.')
        self.inputs={i:[] for i in self.nodes};self.data=data
        for e in es:
            require(e['source'] in self.nodes and e['target'] in self.nodes,'Invalid wire.')
            require(all(type(e.get(k,0)) is int and 0<=e.get(k,0)<64 for k in ('sourceSlot','inputSlot')),'Invalid port.')
            self.inputs[e['target']].append(e)
        for edges in self.inputs.values():
            edges.sort(key=lambda e:e.get('inputSlot',0));require([e.get('inputSlot',0) for e in edges]==list(range(len(edges))),'Connect each input exactly once.')
        sources=[n for n in ns if n['type']=='dataset'];losses=[n for n in ns if n['type'] in ('loss','cross-entropy')]
        require(len(sources)==len(losses)==1,'Use one Dataset and one Loss.')
        self.source=sources[0];self.loss=losses[0];p=self.loss['params']
        require(self.loss['type']=='cross-entropy' or p.get('loss')=='cross-entropy','Use cross-entropy on logits.')
        require(p.get('regularization','none')=='none' or p.get('regularizationStrength',0)==0,'Use no loss penalty.')
        require(len(self.inputs[self.loss['id']])==2,'Connect logits and targets to Loss.')
        require(not self.inputs[self.source['id']],'Dataset has no inputs.')
        self.params={}
        for n in ns:
            if n['type'] in ('weight','bias'):
                v=n['params'].get('value');require(isinstance(v,dict),'Initialize each Param tensor.')
                shape=v['shape'];values=v['data']
                require(1<=len(shape)<=3 and all(type(d) is int and 1<=d<=8192 for d in shape) and math.prod(shape)<=262144,'Param shape exceeds limits.')
                require(len(values)==math.prod(shape) and all(type(x) in (int,float) and math.isfinite(x) for x in values),'Invalid Param values.')
                self.params[n['id']]=np.array(values,dtype=float).reshape(shape)
        require(sum(x.size for x in self.params.values())<=1000000,'Too many parameters.')
        self._structural_cache={}

    def execute(self,features,target,with_values=False):
        cache={};active=set();used=set()
        def edge(e):
            if e['source']==self.source['id']:
                slot=e.get('sourceSlot',0);require(slot<=len(features),'Invalid dataset output.')
                return (target,('target',)) if slot==len(features) else (features[slot],('feature',slot))
            require(e.get('sourceSlot',0)==0,'Invalid operation output.');return visit(e['source'])
        def visit(key):
            if key in cache:return cache[key]
            require(key not in active,'Cycle in graph.');active.add(key)
            n=self.nodes[key];p=n['params'];kind=p.get('transform','reshape') if n['type']=='tensor-transform' else n['type'];args=[edge(e) for e in self.inputs[key]];xs=[a[0] for a in args];ss=[a[1] for a in args]
            require(all(s!=('target',) for s in ss) or kind in ('target','input','reshape'),'Target must not enter predictions.')
            if kind in ('weight','bias'):
                require(not args,'Params have no inputs.');out=self.params[key];used.add(key);sig=('param',out.shape) + ((hashlib.sha256(out.tobytes()).hexdigest(),) if with_values else ())
            elif kind in ('input','target'):
                require(len(args)==1,'Connect alias blocks.');out,sig=args[0]
            elif kind=='matmul':
                out=xs[0]@xs[1];sig=('matmul',*ss)
                if ss[0][0]=='one-hot':sig=('embedding',ss[1],ss[0][2])
            elif kind=='one-hot':
                width=p.get('numClasses',2);require(type(width) is int and 2<=width<=8192 and xs[0].size*width<=262144,'Invalid one-hot width.')
                require(np.equal(xs[0],np.floor(xs[0])).all() and (xs[0]>=0).all() and (xs[0]<width).all(),'Invalid one-hot IDs.')
                out=np.zeros((*xs[0].shape,width));np.put_along_axis(out,xs[0].astype(int)[...,None],1,axis=-1);sig=('one-hot',width,ss[0])
            elif kind in ('arithmetic','add','multiply'):
                expr=p.get('expression','x1 * x2') if kind=='arithmetic' else (' + ' if kind=='add' else ' * ').join('x'+str(i+1) for i in range(len(xs)))
                out=expression(expr,xs);sig=self._structural_cache.get(key) if not with_values else None
                if sig is None:sig=expression(expr,ss,True)
            elif kind=='embedding':
                require(np.equal(xs[1],np.floor(xs[1])).all() and (xs[1]>=0).all() and (xs[1]<xs[0].shape[0]).all(),'Invalid token ID.')
                out=xs[0][xs[1].astype(int)];sig=('embedding',*ss)
            elif kind=='reshape':
                shape=p.get('shape',list(xs[0].shape));require(shape.count(-1)<=1 and all(type(d) is int and (d>0 or d==-1) for d in shape),'Invalid reshape.')
                out=xs[0].reshape(shape);sig=('reshape',out.shape,ss[0])
            elif kind=='transpose':
                axes=p.get('axes',list(reversed(range(xs[0].ndim))));out=xs[0].transpose(axes);sig=('transpose',tuple(axes),ss[0])
            elif kind=='slice':
                axis=p.get('axis',0);sl=[slice(None)]*xs[0].ndim;start=p.get('start',0);end=p.get('end',xs[0].shape[axis]);sl[axis]=slice(start,end);out=xs[0][tuple(sl)];sig=('slice',axis,start,end,ss[0])
            elif kind=='concat':
                axis=p.get('axis',1);out=np.concatenate(xs,axis=axis);sig=('concat',axis,*ss)
            elif kind=='activation':
                require(p.get('activation')=='relu','Use the assigned ReLU activation.');out=np.maximum(xs[0],0);sig=('relu',ss[0])
            elif kind=='softmax':
                require(p.get('axis',-1) in (-1,xs[0].ndim-1),'Softmax must operate across each row.');z=xs[0]-xs[0].max(axis=-1,keepdims=True);out=np.exp(z);out/=out.sum(axis=-1,keepdims=True);sig=('softmax',ss[0])
            elif kind=='causal-mask':
                out=np.where(np.triu(np.ones(xs[0].shape,dtype=bool),1),-1e9,xs[0]);sig=('causal-mask',ss[0])
            elif kind=='layer-norm':
                eps=p.get('epsilon',1e-5);require(type(eps) in (int,float) and 0<eps<=.1,'Invalid normalization epsilon.')
                out=(xs[0]-xs[0].mean(axis=-1,keepdims=True))/np.sqrt(xs[0].var(axis=-1,keepdims=True)+eps)*xs[1]+xs[2];sig=('layer-norm',eps,*ss)
            else:raise ValueError('Unsupported block in assigned model: '+kind)
            out=np.asarray(out);require(out.size<=262144 and np.isfinite(out).all(),'Invalid or oversized intermediate tensor.')
            if not with_values:self._structural_cache[key]=sig
            active.remove(key);cache[key]=(out,sig);return cache[key]
        pred,sig=edge(self.inputs[self.loss['id']][0]);_,target_sig=edge(self.inputs[self.loss['id']][1])
        require(target_sig==('target',),'Use the original dataset target.')
        require(pred.ndim==2 and pred.shape[:-1]==target.shape,'Logits and target shapes do not match.')
        seen=set()
        def contains_target(s):
            if s==('target',):return True
            if not isinstance(s,tuple) or id(s) in seen:return False
            seen.add(id(s));return any(contains_target(x) for x in s)
        require(not contains_target(sig),'Target leakage is not allowed.')
        prediction_params=set(used)
        branches=[visit(n['id'])[1] for n in self.nodes.values() if n['type']=='softmax']
        require(('softmax',sig) in branches,'Keep a separate Softmax branch on the output logits.')
        return pred,sig,Counter(self.params[k].shape for k in prediction_params),cache


def load(path,data):
    path=Path(path);require(path.is_file() and not path.is_symlink() and path.stat().st_size<=32*1024**2,'Upload a saved model under 32 MiB.')
    f=json.loads(path.read_text(),parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON.')))
    require(f.get('kind') in ('neural-canvas-state','backprop-builder-state') and f.get('version')==1,'Use File → Save in Neural Canvas.')
    model=Graph(f['state']['graph'],data);p=model.source['params'];actual=p.get('textData',{})
    for k in ['version','task','tokenizer','representation','lowercase','vocabulary','maxLength','questionLength','stride','documents','fixedLength','classLabels','factWords','maxFacts','targetMode']:
        require(actual.get(k)==data.get(k),'Keep the supplied dataset and preprocessing unchanged: '+k)
    require(p.get('trainPercent') is None,'Keep the supplied split.')
    return model


def evaluate(model, template, data):
    rows=list(examples(data));xs,y,_=rows[0];_,expected,expected_params,_=template.execute(xs,y)
    losses={'train':[],'test':[]};final={'train':[],'test':[]};accuracy=[]
    for xs,y,split in rows:
        pred,sig,params,_=model.execute(xs,y)
        require(sig==expected and params==expected_params,'Use the assigned architecture, shapes, masks, and independent Params.')
        z=pred-pred.max(axis=-1,keepdims=True);ls=np.log(np.exp(z).sum(axis=-1))-z[np.arange(len(y)),y]
        losses[split].append(float(ls.mean()));final[split].append(float(ls[-1]))
        if split=='test':accuracy.append(float((pred.argmax(axis=-1)==y).mean()))
    return {'train':float(np.mean(losses['train'])),'validation':float(np.mean(losses['test'])),'accuracy':float(np.mean(accuracy)),
            'final_train':float(np.mean(final['train'])),'final_validation':float(np.mean(final['test']))}


def parameter_fingerprint(model,data):
    xs,y,_=next(examples(data));_,sig,_,_=model.execute(xs,y,with_values=True)
    if sig[0]=='slice' and sig[1:4]==(0,data['maxLength']-1,data['maxLength']):sig=sig[4]
    return hashlib.sha256(repr(sig).encode()).hexdigest()
