"""Bounded NumPy evaluation of submitted graphs. Never executes submitted code."""
import ast,json,math,re
from pathlib import Path
import numpy as np

def require(ok,message):
    if not ok: raise ValueError(message)

def arithmetic(source,args):
    require(isinstance(source,str) and len(source)<=256,'Invalid Arithmetic expression.')
    tree=ast.parse(source.replace('^','**'),mode='eval')
    require(len(list(ast.walk(tree)))<=100,'Arithmetic expression too large.')
    def visit(t):
        if isinstance(t,ast.Constant) and type(t.value) in (int,float):
            require(math.isfinite(t.value) and abs(t.value)<=1e6,'Invalid constant.');return t.value
        if isinstance(t,ast.Name) and re.fullmatch(r'x[1-9][0-9]*',t.id):
            i=int(t.id[1:])-1;require(i<len(args),'Missing input.');return args[i]
        if isinstance(t,ast.UnaryOp) and isinstance(t.op,(ast.USub,ast.UAdd)):
            return (-1 if isinstance(t.op,ast.USub) else 1)*visit(t.operand)
        require(isinstance(t,ast.BinOp),'Use numeric arithmetic only.')
        a,b=visit(t.left),visit(t.right)
        if isinstance(t.op,ast.Pow):
            require(isinstance(t.right,ast.Constant) and type(b) in (float,int) and abs(b)<=4,'Use a small constant exponent.');return a**b
        require(type(t.op) in (ast.Add,ast.Sub,ast.Mult,ast.Div),'Unsupported arithmetic operator.')
        return {ast.Add:np.add,ast.Sub:np.subtract,ast.Mult:np.multiply,ast.Div:np.divide}[type(t.op)](a,b)
    return visit(tree.body)

class Graph:
    def __init__(self,path,dataset):
        p=Path(path);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=8*1024**2,'Upload a saved project under 8 MiB.')
        f=json.loads(p.read_text(),parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON.')))
        require(f.get('kind') in ('neural-canvas-state','backprop-builder-state') and f.get('version')==1,'Use File → Save.')
        g=f['state']['graph'];ns,es=g['nodes'],g['edges'];require(1<=len(ns)<=100 and len(es)<=200,'Use at most 100 blocks and 200 wires.')
        require(all(isinstance(n.get('id'),str) and len(n['id'])<=128 and isinstance(n.get('params'),dict) for n in ns),'Invalid block.')
        self.nodes={n['id']:n for n in ns};require(len(self.nodes)==len(ns),'Duplicate block IDs.')
        self.incoming={key:[] for key in self.nodes}
        for e in es:
            require(e['source'] in self.nodes and e['target'] in self.nodes,'Invalid wire.')
            require(all(type(e.get(k,0)) is int and 0<=e.get(k,0)<16 for k in ('sourceSlot','inputSlot')),'Invalid port.')
            self.incoming[e['target']].append(e)
        for edges in self.incoming.values():
            edges.sort(key=lambda e:e.get('inputSlot',0));require([e.get('inputSlot',0) for e in edges]==list(range(len(edges))),'Connect each port exactly once.')
        allowed={'dataset','input','target','weight','bias','arithmetic','multiply','add','activation','matmul','reshape','tensor-transform','conv2d','avgpool2d','loss','cross-entropy','softmax'}
        require(all(n['type'] in allowed for n in ns),'Unsupported operation.')
        sources=[n for n in ns if n['type']=='dataset'];losses=[n for n in ns if n['type'] in ('loss','cross-entropy')]
        require(len(sources)==len(losses)==1,'Use one Dataset and one Loss.')
        self.source=sources[0]['id'];self.loss=losses[0]['id'];sp=sources[0]['params'];lp=losses[0]['params']
        require(sp.get('dataset')==dataset and sp.get('trainPercent') is None and not sp.get('datasetValues') and not sp.get('customCsv'),'Keep the built-in dataset and default split unchanged.')
        require(not self.incoming[self.source] and len(self.incoming[self.loss])==2,'Connect prediction and target to Loss.')
        require(lp.get('regularization','none')=='none' and not lp.get('activationPenaltyStrength') and not lp.get('targetMode'),'Use the unpenalized dataset loss.')
        self.loss_kind=lp.get('loss','cross-entropy' if losses[0]['type']=='cross-entropy' else 'mse')
        self.params={}
        arity={'dataset':0,'weight':0,'bias':0,'activation':1,'matmul':2,'reshape':1,'tensor-transform':1,'conv2d':3,'avgpool2d':1,'loss':2,'cross-entropy':2,'softmax':1}
        for n in ns:
            k=n['id'];kind=n['type'];p=n['params'];count=len(self.incoming[k])
            if kind in arity:require(count==arity[kind],'Invalid input count for '+kind)
            elif kind in ('input','target'):require(count==1,'Use connected aliases; no fixed prediction/gradient Inputs.')
            elif kind in ('add','multiply'):require(count==p.get('inputCount',2) and 2<=count<=8,'Invalid arithmetic inputs.')
            else:require(1<=count<=8,'Invalid arithmetic inputs.')
            if kind=='tensor-transform':require(p.get('transform','reshape')=='reshape','Use Reshape only in these constructions.')
            if kind in ('weight','bias'):
                v=p.get('value');require(isinstance(v,dict),'Initialize every Param.');shape=v.get('shape');nums=v.get('data')
                require(isinstance(shape,list) and len(shape)<=4 and all(type(x) is int and 1<=x<=128 for x in shape) and math.prod(shape)<=4096,'Invalid Param shape.')
                require(isinstance(nums,list) and len(nums)==math.prod(shape) and all(type(x) in (int,float) and math.isfinite(x) and abs(x)<=1e6 for x in nums),'Invalid Param values.')
                require(p.get('trainable',True) is True and not p.get('constraint'),'Use trainable unconstrained Params.')
                self.params[k]=np.array(nums).reshape(shape)
        require(sum(v.size for v in self.params.values())<=4096,'Too many parameters.')
        self.ancestors(self.loss)

    def ancestors(self,key):
        found=set();active=set()
        def visit(k):
            require(k not in active,'Cycle in graph.')
            if k in found:return
            active.add(k)
            for e in self.incoming[k]:visit(e['source'])
            active.remove(k);found.add(k)
        visit(key);return found

    def evaluate(self,x,y,overrides=None):
        cache={};active=set();overrides=overrides or {};signatures={};tainted={};used=set()
        batch=len(y)
        def edge(e):
            if e['source']==self.source:
                slot=e.get('sourceSlot',0);require(slot in (0,1),'Invalid dataset output.')
                return (x,('features',),False) if slot==0 else (y,('target',),True)
            require(e.get('sourceSlot',0)==0,'Operation output must be port zero.');visit(e['source']);return cache[e['source']],signatures[e['source']],tainted[e['source']]
        def visit(k):
            if k in cache:return cache[k]
            require(k not in active,'Cycle in graph.');active.add(k)
            n=self.nodes[k];p=n['params'];kind=n['type'];args=[edge(e) for e in self.incoming[k]];xs=[a[0] for a in args];ss=[a[1] for a in args];t=any(a[2] for a in args)
            if kind in ('input','target'):out,sig,t=args[0]
            elif kind in ('weight','bias'):out=overrides.get(k,self.params[k]);sig=('param',tuple(self.params[k].shape));used.add(k)
            elif kind in ('arithmetic','add','multiply'):
                expr=p.get('expression','x1*x2') if kind=='arithmetic' else ('+' if kind=='add' else '*').join('x'+str(i+1) for i in range(len(xs)))
                out=arithmetic(expr,xs)
                # Bias addition is commutative; other expressions retain their text signature.
                compact=re.sub(r'\s','',expr)
                if compact in ('x1+x2','x2+x1'):sig=('add',*sorted(ss,key=repr))
                else:sig=('arithmetic',compact,*ss)
            elif kind=='activation':
                require(p.get('activation') in ('relu','tanh'),'Use the assigned activation.');act=p['activation'];out=np.maximum(xs[0],0) if act=='relu' else np.tanh(xs[0]);sig=(act,ss[0])
            elif kind=='matmul':out=xs[0]@xs[1];sig=('matmul',*ss)
            elif kind in ('reshape','tensor-transform'):
                shape=p.get('shape');require(isinstance(shape,list) and 1<=len(shape)<=3 and all(type(a) is int and (1<=a<=4096 or a==-1) for a in shape) and shape.count(-1)<=1,'Invalid reshape.')
                require(xs[0].shape[0]==batch,'Reshape must preserve examples.');out=xs[0].reshape((batch,*shape));sig=('reshape',tuple(shape),ss[0])
            elif kind=='conv2d':
                image,w,b=xs;require(image.ndim==4 and w.shape==(4,3,3,1) and b.shape==(4,),'Use four 3×3 one-channel filters.')
                win=np.lib.stride_tricks.sliding_window_view(image,(3,3),axis=(1,2));out=np.einsum('nhwcij,fijc->nhwf',win,w,optimize=True)+b;sig=('conv2d',*ss)
            elif kind=='avgpool2d':
                a=xs[0];require(a.ndim==4 and a.shape[1:3]==(6,6),'Pool a 6×6 feature map.');out=(a[:,0::2,0::2,:]+a[:,1::2,0::2,:]+a[:,0::2,1::2,:]+a[:,1::2,1::2,:])/4;sig=('avgpool2d',ss[0])
            elif kind=='softmax':
                z=xs[0]-xs[0].max(axis=-1,keepdims=True);out=np.exp(z);out/=out.sum(axis=-1,keepdims=True);sig=('softmax',ss[0])
            else:raise ValueError('Unexpected endpoint operation.')
            out=np.asarray(out);require(out.size<=2000000 and np.isfinite(out).all(),'Invalid or oversized intermediate tensor.')
            active.remove(k);cache[k]=out;signatures[k]=sig;tainted[k]=t;return out
        pred,signature,leak=edge(self.incoming[self.loss][0]);require(not leak,'Targets must never enter predictions.')
        target,tsig,_=edge(self.incoming[self.loss][1]);require(tsig==('target',),'Wire the canonical Dataset target directly or through Target.')
        prediction_params=set(used)
        for n in self.nodes.values():
            if n['type'] not in ('dataset','loss','cross-entropy'):visit(n['id'])
        return pred,signature,cache,prediction_params
