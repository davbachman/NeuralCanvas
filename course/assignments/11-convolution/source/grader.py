"""Independently recompute digit predictions and verify MLP/CNN architecture."""
import json,sys
from pathlib import Path
import numpy as np
from graph_eval import Graph,require
PART_POINTS={'mlp':50,'cnn':50}

def signature(part):
    p=lambda *s:('param',s)
    add=lambda a,b:('add',*sorted((a,b),key=repr))
    if part=='mlp':
        h=('relu',add(('matmul',('reshape',(1,64),('features',)),p(64,16)),p(16)))
        return add(('matmul',h,p(16,10)),p(10))
    h=('conv2d',('features',),p(4,3,3,1),p(4))
    h=('reshape',(1,36),('avgpool2d',('relu',h)))
    return add(('matmul',h,p(36,10)),p(10))

def grade(part,path,data):
    tests=[]
    def credit(name,points,ok,message):tests.append(dict(name=part+' — '+name,score=points if ok else 0,max_score=points,output=message,visibility='visible'))
    try:
        require(part in PART_POINTS,'Unknown part.');m=Graph(path,'digits-8x8');require(m.loss_kind=='cross-entropy','Use Cross entropy (logits).')
        rows=data['digits'];x=np.asarray([r['pixels'] for r in rows],float).reshape(-1,8,8,1)/16;y=np.asarray([r['label'] for r in rows]);train=np.asarray([r['split']=='train' for r in rows])
        pred,sig,_,used=m.evaluate(x,y)
        require(sig==signature(part),'Use the assigned shapes, ReLU, and linear logits; do not normalize the already normalized pixels again.')
        require(used==set(m.params) and len(used)==4,'Use exactly four independent Params in the prediction.')
        pred=pred.reshape(len(y),10);z=pred-pred.max(axis=1,keepdims=True);loss=np.log(np.exp(z).sum(axis=1))-z[np.arange(len(y)),y]
        accuracy=(pred.argmax(axis=1)==y);tr=float(loss[train].mean());va=float(loss[~train].mean());acc=float(accuracy[~train].mean())
        limits=data['limits'][part]
        credit('architecture',15,True,f'Correct model with {sum(v.size for v in m.params.values())} parameters.')
        credit('training fit',15,tr<=limits['train'],f'Training cross-entropy {tr:.4f}; required ≤ {limits["train"]}.')
        credit('validation predictions',20,acc>=limits['accuracy'] and va<=limits['validation'],f'Validation accuracy {acc:.1%}, cross-entropy {va:.4f}; required ≥ {limits["accuracy"]:.0%} and ≤ {limits["validation"]}.')
    except Exception as e:
        remaining=PART_POINTS.get(part,50)-sum(t['max_score'] for t in tests)
        if remaining:credit('model validation',remaining,False,str(e)[:500])
    return {'tests':tests,'score':sum(t['score'] for t in tests)}
if __name__=='__main__':print(json.dumps(grade(sys.argv[1],sys.argv[2],json.loads(Path(sys.argv[3]).read_text())),allow_nan=False))
