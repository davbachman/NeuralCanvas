"""Independent saved-graph assessment of frozen residual autoencoders.

No submitted Python is imported. Values are recomputed on canonical examples;
IDs, labels, cached outputs, scores and training histories are not evidence.
"""
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from graph_eval import Graph, examples, load, require


def fingerprint(signature):
    return hashlib.sha256(repr(signature).encode()).hexdigest()


def add_signature(*terms):
    flat=[]
    for term in terms:
        flat.extend(term[1:] if term[0]=='Add' else [term])
    return ('Add',*sorted(flat,key=repr))


def canonical(reference):
    data=reference['dataset']
    model=Graph(reference['template'],data)
    xs,y,_=next(examples(data))
    _,logits,_,_=model.execute(xs,y)
    _,residual,_,_=model.execute(xs,y,endpoint=reference['residualNode'])
    return logits,residual


def discover(model,reference,autoencoder=False):
    data=reference['dataset'];xs,y,_=next(examples(data))
    expected,expected_residual=canonical(reference)
    _,_,_,cache=model.execute(xs,y,all_nodes=True)
    endpoints=[key for key,(_,sig) in cache.items() if sig==expected]
    require(endpoints,'Preserve the trained two-block, two-head generator and vocabulary output.')
    endpoint=endpoints[0] if autoencoder else model.loss_sources()[0]
    require(endpoint in endpoints,'Connect the preserved generator logits to baseline Loss.')
    base_keys=model.ancestors(endpoint);base_params=base_keys & model.params.keys()
    require(sum(model.params[key].size for key in base_params)==5689,'Keep 5,689 independent transformer values.')
    require(any(sig==('softmax',expected) for _,sig in cache.values()),'Keep a separate vocabulary Softmax branch.')
    residuals=[key for key,(_,sig) in cache.items() if sig==expected_residual and key in base_keys]
    require(residuals,'Use the final second-block MLP residual, before output Layer norm.')
    _,base_sig,_,value_cache=model.execute(xs,y,with_values=True,endpoint=endpoint,all_nodes=True)
    residual_value_sig=value_cache[residuals[0]][1]
    require(any(sig==('softmax',base_sig) for _,sig in value_cache.values()),'Keep a Softmax branch on the preserved vocabulary logits.')
    result=dict(endpoint=endpoint,residual=residuals[0],base_params=base_params,base_fingerprint=fingerprint(base_sig))
    if not autoencoder:
        require(set(model.params)==base_params,'Baseline must contain only transformer Params.')
        p=model.loss['params'];require(p.get('loss')=='cross-entropy' and p.get('targetMode','dataset')!='internal','Baseline uses dataset next-token cross entropy.')
        edges=model.inputs[model.loss['id']]
        require(cache[edges[0]['source']][1]==expected,'Connect vocabulary logits to baseline Loss.')
        target=edges[1]
        require(target['source']==model.source['id'] and target.get('sourceSlot',0)==2,'Use the original Dataset targets.')
        return result
    model.frozen(base_keys)
    p=model.loss['params'];require(p.get('loss','mse')=='mse' and p.get('targetMode')=='internal','Use MSE with frozen internal targets.')
    predicted,target=model.loss_sources()
    require(value_cache[target][1]==residual_value_sig,'Loss target must be the original final MLP residual, using the same frozen values as the vocabulary path.')
    require(p.get('lossMaskNodeId') in (None,''),'The canonical full narrative windows contain no padding; use all 32 token positions.')
    extra=set(model.params)-base_params
    require(len(extra)==4,'Add exactly encoder/decoder matrices and their two biases.')
    require(all(model.nodes[k]['params'].get('trainable',True) is True for k in extra),'Leave the four autoencoder Params trainable.')
    width=reference['hiddenWidth'];d=16
    def shape(key):return model.params[key].shape
    match=None
    for we,be,wd,bd in itertools.permutations(extra):
        if shape(we)!=(d,width) or shape(wd)!=(width,d) or shape(be) not in [(width,),(1,width)] or shape(bd) not in [(d,),(1,d)]:continue
        hidden=('relu',add_signature(('matmul',residual_value_sig,value_cache[we][1]),value_cache[be][1]))
        reconstructed=add_signature(('matmul',hidden,value_cache[wd][1]),value_cache[bd][1])
        if value_cache[predicted][1]==reconstructed:
            hkeys=[k for k,(_,sig) in value_cache.items() if sig==hidden]
            if hkeys:match=(we,be,wd,bd,hkeys[0]);break
    require(match is not None,'Build ReLU(r We + be) Wd + bd with the assigned independent feature width.')
    we,be,wd,bd,hidden=match
    require(model.nodes[wd]['params'].get('constraint')=='unit-row-norm','Constrain decoder directions to unit row norm.')
    norms=np.linalg.norm(model.params[wd],axis=1)
    require(np.allclose(norms,1,rtol=0,atol=5e-5),'Every saved decoder direction must have norm one; zero and rescaled directions are invalid.')
    require(set(model.params)==base_params|set(match[:4]),'Remove unused Params.')
    _,sae_sig,_,_=model.execute(xs,y,with_values=True,endpoint=predicted)
    result.update(predicted=predicted,hidden=hidden,decoder=wd,encoder=we,encoder_bias=be,decoder_bias=bd,sae_fingerprint=fingerprint(sae_sig),norm_min=float(norms.min()),norm_max=float(norms.max()))
    return result


def contribution(model,info,reference):
    """Find a genuine same-feature rank-one reconstruction, independent of names."""
    xs,y,_=next(examples(reference['dataset']))
    _,_,_,cache=model.execute(xs,y,all_nodes=True,with_values=True)
    hsig=cache[info['hidden']][1];dsig=cache[info['decoder']][1]
    for j in range(reference['hiddenWidth']):
        expected=('matmul',('slice',1,j,j+1,hsig),('slice',0,j,j+1,dsig))
        for key,(value,sig) in cache.items():
            if sig==expected:
                wanted=cache[info['hidden']][0][:,j:j+1] @ model.params[info['decoder']][j:j+1]
                require(value.shape==wanted.shape and np.allclose(value,wanted,rtol=1e-10,atol=1e-12),'Feature contribution values do not match h_j v_j.')
                return j,key
    raise ValueError('Build a separate rank-one feature contribution: Slice one ReLU feature column, Slice the same decoder row, then Matrix product.')


def metrics(model,info,reference):
    data=reference['dataset'];accum={s:dict(loss=0.,n=0,mse=0.,l1=0.,active=0.,tokens=0,ever=np.zeros(reference['hiddenWidth'],bool)) for s in ('train','test')}
    for xs,y,split in examples(data):
        logits,_,_,_=model.execute(xs,y,endpoint=info['endpoint'])
        z=logits-logits.max(axis=-1,keepdims=True)
        ls=np.log(np.exp(z).sum(axis=-1))-z[np.arange(len(y)),y]
        a=accum[split];a['loss']+=float(ls.sum());a['n']+=len(y)
        if 'hidden' in info:
            r,_,_,_=model.execute(xs,y,endpoint=info['residual'])
            reconstructed,_,_,cache=model.execute(xs,y,endpoint=info['predicted'])
            h=cache[info['hidden']][0]
            valid=xs[0]!=data['vocabulary'].index('<pad>')
            # Exact ReLU zeros; no small-magnitude threshold silently substitutes for zero.
            a['mse']+=float(np.square(reconstructed[valid]-r[valid]).sum()/r.shape[-1]);a['l1']+=float(np.abs(h[valid]).sum())
            a['active']+=int((h[valid]>0).sum());a['tokens']+=int(valid.sum());a['ever']|=(h[valid]>0).any(axis=0)
    out={}
    for split,a in accum.items():
        label='validation' if split=='test' else 'train';out[label+'_loss']=a['loss']/a['n']
        if a['tokens']:
            for metric in ('mse','l1','active'):out[label+'_'+metric]=a[metric]/a['tokens']
            out[label+'_inactive']=int((~a['ever']).sum())
    if 'hidden' in info:
        out.update(norm_min=info['norm_min'],norm_max=info['norm_max'])
        strength=float(model.loss['params'].get('activationPenaltyStrength',0))
        for split in ('train','validation'):out[split+'_penalty']=strength*out[split+'_l1'];out[split+'_objective']=out[split+'_mse']+out[split+'_penalty']
    return out


def grade_model(path,part,reference):
    rules=reference['parts'][part];tests=[];out={};info=None
    def add(name,ok,message):
        points=rules['points'][name];tests.append(dict(name=f'{part} — {name}',score=points if ok else 0,max_score=points,output=message,visibility='visible'))
    try:model=load(path,reference['dataset']);add('data',True,'Canonical narratives, vocabulary and split verified.')
    except Exception as error:
        for name in rules['points']:add(name,False,str(error) if name=='data' else 'Upload this saved model to earn these independent points.')
        return dict(tests=tests,metrics={})
    try:
        info=discover(model,reference,part!='baseline')
        if part!='baseline':
            p=model.loss['params'];strength=p.get('activationPenaltyStrength',0)
            require(type(strength) in (int,float) and np.isfinite(strength),'Invalid activation penalty strength.')
            if part=='reconstruction':require(strength==0,'Use zero activation penalty in the reconstruction baseline.')
            else:
                require(reference['lambdaBounds'][0]<=strength<=reference['lambdaBounds'][1],'Use the stated bounded activation L1 strength.')
                node=p.get('activationPenaltyNodeId')
                require(node in model.nodes,'Choose the encoder ReLU for the activation penalty.')
                xs,y,_=next(examples(reference['dataset']))
                selected=model.execute(xs,y,endpoint=node,with_values=True)[1]
                hidden=model.execute(xs,y,endpoint=info['hidden'],with_values=True)[1]
                require(selected==hidden,'Penalize the encoder ReLU activations.')
        add('form',True,'Operations, feature width, frozen target, decoder directions and objective verified.')
    except Exception as error:
        for name in rules['points']:
            if name!='data':add(name,False,str(error) if name=='form' else 'Quality credit requires a valid construction.')
        return dict(tests=tests,metrics={})
    if 'contribution' in rules['points']:
        try:
            j,_=contribution(model,info,reference)
            add('contribution',True,f'Independently verified rank-one contribution of student-selected feature {j}.')
        except Exception as error:add('contribution',False,str(error))
    try:
        out=metrics(model,info,reference)
        adequate=out['train_loss']<=reference['baseLimits'][0] and out['validation_loss']<=reference['baseLimits'][1]
        quality_ok=adequate
        if part=='baseline':add('quality',adequate,f"Recomputed generator CE train {out['train_loss']:.6f}; validation {out['validation_loss']:.6f}. Limits {reference['baseLimits']}.")
        else:
            good=adequate and all(out[split+'_mse']<=rules['mseLimit'] for split in ('train','validation'))
            quality_ok=bool(good)
            add('quality',good,f"Recomputed MSE train {out['train_mse']:.6f}; validation {out['validation_mse']:.6f}; limit {rules['mseLimit']}. Generator CE {out['train_loss']:.6f}/{out['validation_loss']:.6f}; limits {reference['baseLimits']}.")
            if 'sparsity' in rules['points']:
                good=adequate and out['validation_active']<=rules['activeLimit'] and out['validation_inactive']<=rules['inactiveLimit']
                add('sparsity',good,f"Validation active features/token {out['validation_active']:.4f} (limit {rules['activeLimit']}); inactive {out['validation_inactive']} (limit {rules['inactiveLimit']}). Active means strictly greater than zero. L1 mean/token {out['validation_l1']:.6f}; penalty {out['validation_penalty']:.6f}; total objective {out['validation_objective']:.6f}.")
        return dict(tests=tests,metrics=out,base_fingerprint=info['base_fingerprint'],sae_fingerprint=info.get('sae_fingerprint'),adequate_base=bool(adequate),quality_ok=quality_ok)
    except Exception as error:
        done={test['name'].split(' — ')[1] for test in tests}
        for name in rules['points']:
            if name not in done:add(name,False,'Evaluation failed: '+str(error))
        return dict(tests=tests,metrics={})



def polynomial(signature,h,v):
    """Reduce only elementary scalar/row products; no submitted expression runs."""
    if signature==h:return {('h',):1.}
    if signature==v:return {('v',):1.}
    if signature[0]=='number':return {():float(signature[1])}
    op=signature[0]
    if op=='neg':return {k:-value for k,value in polynomial(signature[1],h,v).items()}
    require(op in ('Add','Sub','Mult','matmul','Div'),'Use visible feature selection, scalar arithmetic and Matrix product for the intervention.')
    parts=[polynomial(s,h,v) for s in signature[1:]]
    if op in ('Add','Sub'):
        result={}
        for i,p in enumerate(parts):
            for key,value in p.items():result[key]=result.get(key,0)+(value if op=='Add' or i==0 else -value)
    elif op=='Div':
        require(set(parts[1])=={()} and parts[1][()]!=0,'Divide intervention only by nonzero constants.')
        result={k:x/parts[1][()] for k,x in parts[0].items()}
    else:
        result={():1.}
        for p in parts:
            nxt={}
            for k,a in result.items():
                for q,b in p.items():
                    monomial=tuple(sorted(k+q));require(len(monomial)<=2,'Intervention product degree exceeds the assigned construction.')
                    nxt[monomial]=nxt.get(monomial,0)+a*b
            result=nxt
    return {k:v for k,v in result.items() if v!=0}


def intervention_model(model,part,reference):
    data=reference['dataset'];xs,y,_=next(examples(data));expected,rsig=canonical(reference)
    _,actual_sig,_,cache=model.execute(xs,y,all_nodes=True)
    require(model.loss['params'].get('loss')=='cross-entropy' and model.loss['params'].get('targetMode','dataset')!='internal','Restore the single Loss to dataset next-token cross entropy.')
    require(model.loss['params'].get('activationPenaltyStrength',0)==0,'Disable the reconstruction activation penalty while generating.')
    e=model.inputs[model.loss['id']][1];require(e['source']==model.source['id'] and e.get('sourceSlot',0)==2,'Use the canonical Dataset targets.')
    model.frozen(set(model.params))
    prediction=model.loss_sources()[0];ancestors=model.ancestors(prediction)
    chosen=None
    for key in ancestors:
        node=model.nodes[key]
        if node['type'] not in ('arithmetic','add'):continue
        edges=model.inputs[key]
        if len(edges)!=2:continue
        residuals=[e['source'] for e in edges if e['source'] in cache and cache[e['source']][1]==rsig]
        if not residuals:continue
        residual=residuals[0]
        g={'nodes':copy.deepcopy(list(model.nodes.values())),'edges':copy.deepcopy([e for edges in model.inputs.values() for e in edges])}
        for wire in g['edges']:
            if wire['source']==key:wire['source']=residual;wire['sourceSlot']=0
        try:
            view=Graph(g,data)
            _,sig,_,v_cache=view.execute(xs,y,all_nodes=True)
            if sig!=expected:continue
            chosen=(key,residual,g,view,v_cache);break
        except (ValueError,KeyError,TypeError):continue
    require(chosen is not None,'Insert a real residual intervention after the second MLP and before final Layer norm; vocabulary bias or disconnected branches do not count.')
    key,residual,g,view,v_cache=chosen
    base_params=view.ancestors(prediction)&view.params.keys();extra=set(view.params)-base_params
    require(len(extra)==4,'Keep the original sparse autoencoder and no extra Params.')
    for n in g['nodes']:
        if n['id'] in extra:n['params']['trainable']=True
    width=reference['hiddenWidth'];reconstruction=None;hidden=None
    for we,be,wd,bd in itertools.permutations(extra):
        if view.params[we].shape!=(16,width) or view.params[wd].shape!=(width,16) or view.params[be].shape not in [(width,),(1,width)] or view.params[bd].shape not in [(16,),(1,16)]:continue
        hs=('relu',add_signature(('matmul',rsig,v_cache[we][1]),v_cache[be][1]))
        recon=add_signature(('matmul',hs,v_cache[wd][1]),v_cache[bd][1])
        matches=[k for k,(_,sig) in v_cache.items() if sig==recon]
        hmatches=[k for k,(_,sig) in v_cache.items() if sig==hs]
        if matches and hmatches:reconstruction=matches[0];hidden=hmatches[0];break
    require(reconstruction is not None,'Preserve the trained sparse encoder and decoder branches.')
    for n in g['nodes']:
        if n['id']==model.loss['id']:n['params'].update(loss='mse',targetMode='internal',activationPenaltyNodeId=hidden,activationPenaltyStrength=reference['sparseLambda'])
    for wire in g['edges']:
        if wire['target']==model.loss['id']:
            wire['source']=reconstruction if wire.get('inputSlot',0)==0 else residual;wire['sourceSlot']=0
    view=Graph(g,data);info=discover(view,reference,True)
    require(key not in model.ancestors(info['hidden']),'Keep the original residual feeding the encoder; the intervened residual must not enter it.')
    _,_,_,values=model.execute(xs,y,all_nodes=True,with_values=True)
    hsig=values[info['hidden']][1];dsig=values[info['decoder']][1];rval_sig=values[residual][1]
    edges=model.inputs[key];expr=model.nodes[key]['params'].get('expression','x1 + x2')
    from graph_eval import expression
    effect_signature=expression(expr,[('r',) if e['source']==residual else values[e['source']][1] for e in edges],True)
    # The outer operation must add the original residual exactly once.
    require(effect_signature[0] in ('Add','Sub'),'Add the feature change to the original residual.')
    terms=list(effect_signature[1:]);require(terms.count(('r',))==1,'Preserve the original residual with coefficient one.')
    sign=1
    if effect_signature[0]=='Sub':require(terms[0]==('r',),'Subtract the feature contribution from the original residual.');sign=-1
    other=[s for s in terms if s!=('r',)];require(len(other)==1,'Use one feature intervention branch.')
    selected=None
    for j in range(width):
        hj=('slice',1,j,j+1,hsig);vj=('slice',0,j,j+1,dsig)
        try:poly={k:sign*v for k,v in polynomial(other[0],hj,vj).items()}
        except ValueError:continue
        if part=='ablated' and poly=={('h','v'):-1.}:selected=(j,None);break
        if part in ('zero','amplified') and set(poly)<= {('v',)}:
            alpha=poly.get(('v',),0.)
            if (part=='zero' and alpha==0) or (part=='amplified' and 0<alpha<=reference['maxAlpha']):selected=(j,alpha);break
    require(selected is not None,'Use the same selected decoder row: r + alpha v_j (bounded alpha), or r - h_j v_j for ablation.')
    j,alpha=selected
    # A zero scalar must still select a real decoder row: reject literal zero/disconnected branches.
    if part=='zero':
        def has(sig,wanted):return sig==wanted or (isinstance(sig,tuple) and any(has(s,wanted) for s in sig if isinstance(s,tuple)))
        matching=[k for k in range(width) if has(other[0],('slice',0,k,k+1,dsig))]
        require(len(matching)==1,'Keep the selected decoder row connected even at zero strength.');j=matching[0]
    info.update(intervention=key,feature=j,alpha=alpha,actual_endpoint=prediction)
    return view,info


def intervention_metrics(model,view,info,part,reference):
    baseline=metrics(view,info,reference)
    acts=[];tvs=[];loss_change=[];max_activation=0.;zero_error=0.
    j=info['feature'];direction=view.params[info['decoder']][j]
    for xs,y,split in examples(reference['dataset']):
        if split!='test':continue
        original,_,_,cache=view.execute(xs,y,all_nodes=True,endpoint=info['endpoint'])
        changed,_,_,actual=model.execute(xs,y,all_nodes=True,endpoint=info['actual_endpoint'])
        r=cache[info['residual']][0];h=cache[info['hidden']][0][:,j]
        delta=-h[:,None]*direction if part=='ablated' else info['alpha']*direction
        require(np.allclose(actual[info['intervention']][0],r+delta,rtol=1e-9,atol=1e-10),'Recomputed graph does not implement the declared residual intervention at every position.')
        reset=model.execute(xs,y,endpoint=info['actual_endpoint'],overrides={info['intervention']:r})[0]
        require(np.allclose(reset,original,rtol=1e-9,atol=1e-10),'Zero intervention does not restore baseline logits.')
        def softmax(z):z=z-z.max();p=np.exp(z);return p/p.sum()
        p=softmax(original[-1]);q=softmax(changed[-1]);tvs.append(float(np.abs(p-q).sum()/2));acts.append(float(h[-1]))
        loss_change.append(float(-np.log(max(q[y[-1]],1e-300))+np.log(max(p[y[-1]],1e-300))))
        max_activation=max(max_activation,float(h.max()));zero_error=max(zero_error,float(np.max(np.abs(q-p))) if part=='zero' else 0.)
    order=np.argsort(-np.array(acts),kind='stable');low=np.argsort(np.array(acts),kind='stable');n=reference['contextCount'];tvs=np.array(tvs)
    baseline.update(feature=j,alpha=info['alpha'],mean_tv=float(tvs.mean()),high_tv=float(tvs[order[:n]].mean()),low_tv=float(tvs[low[:n]].mean()),max_activation=max_activation,ce_increase=float(np.mean(loss_change)),zero_error=zero_error)
    return baseline


def grade_intervention(path,part,reference):
    if part=='sparse':
        original=copy.deepcopy(reference);original['parts']['sparse']=reference['saeRule']
        result=grade_model(path,'sparse',original)
        # The previous construction is a 20-point prerequisite, retaining partial data/form credit.
        return result
    rules=reference['parts'][part];tests=[]
    def add(name,ok,message):
        points=rules['points'][name];tests.append(dict(name=f'{part} — {name}',score=points if ok else 0,max_score=points,output=message,visibility='visible'))
    try:model=load(path,reference['dataset']);view,info=intervention_model(model,part,reference);add('form',True,'Residual path, original encoder input, same-feature direction, frozen values and bounded strength verified.')
    except Exception as error:
        for name in rules['points']:add(name,False,str(error) if name=='form' else 'Submit a valid independent intervention for behavioral credit.')
        return dict(tests=tests,metrics={})
    try:
        m=intervention_metrics(model,view,info,part,reference)
        adequate=m['train_loss']<=reference['baseLimits'][0] and m['validation_loss']<=reference['baseLimits'][1] and max(m['train_mse'],m['validation_mse'])<=reference['saeRule']['mseLimit'] and m['validation_active']<=reference['saeRule']['activeLimit'] and m['validation_inactive']<=reference['saeRule']['inactiveLimit']
        if part=='zero':good=adequate and m['zero_error']<=reference['zeroTolerance']
        elif part=='amplified':good=adequate and reference['minAmplificationTV']<=m['mean_tv']<=reference['maxMeanTV'] and m['ce_increase']<=reference['maxCEIncrease']
        else:good=adequate and m['high_tv']>=reference['minHighTV'] and m['low_tv']<=reference['maxLowTV'] and m['max_activation']<=reference['maxActivation'] and m['mean_tv']<=reference['maxMeanTV'] and m['ce_increase']<=reference['maxCEIncrease']
        add('effect',bool(good),f"Feature {m['feature']}; mean TV {m['mean_tv']:.6f}; high-{reference['contextCount']} TV {m['high_tv']:.6f}; low-{reference['contextCount']} TV {m['low_tv']:.6f}; maximum h across validation tokens {m['max_activation']:.6f}; final-token CE change {m['ce_increase']:.6f}; zero probability error {m['zero_error']:.3g}.")
        return dict(tests=tests,metrics=m,base_fingerprint=info['base_fingerprint'],sae_fingerprint=info['sae_fingerprint'],adequate_base=bool(adequate),quality_ok=bool(good),feature=info['feature'])
    except Exception as error:
        add('effect',False,'Evaluation failed: '+str(error));return dict(tests=tests,metrics={})


if __name__=='__main__':
    part,path,reference_path=sys.argv[1:];reference=json.loads(Path(reference_path).read_text())
    print(json.dumps(grade_intervention(path,part,reference),allow_nan=False))
