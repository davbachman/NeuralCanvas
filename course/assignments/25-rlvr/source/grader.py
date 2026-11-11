"""Independently evaluate bounded finite-answer policies and learned rewards."""
import copy
import json
import math
import sys
from pathlib import Path
import numpy as np
from graph_eval import Graph, load, examples, require, parameter_fingerprint


def softmax(values):
    values=np.asarray(values,dtype=float); shifted=values-values.max(axis=-1,keepdims=True)
    result=np.exp(shifted);return result/result.sum(axis=-1,keepdims=True)


def javascript_json(value):
    """JSON.stringify-compatible serialization for the finite JSON graph schema."""
    if value is None:return 'null'
    if value is True:return 'true'
    if value is False:return 'false'
    if isinstance(value,str):return json.dumps(value,ensure_ascii=False,separators=(',',':'))
    if isinstance(value,(int,float)):
        require(math.isfinite(value),'Nonfinite signature value.')
        if value==0:return '0'
        if isinstance(value,int):return str(value)
        if value.is_integer() and abs(value)<1e21:return str(int(value))
        text=repr(value)
        if 'e' in text:
            mantissa,exponent=text.split('e');power=int(exponent)
            if 1e-6<=abs(value)<1e21:
                from decimal import Decimal
                return format(Decimal(text),'f')
            return mantissa.removesuffix('.0')+'e'+('+' if power>=0 else '-')+str(abs(power))
        return text
    if isinstance(value,list):return '['+','.join(javascript_json(v)for v in value)+']'
    if isinstance(value,dict):return '{'+','.join(javascript_json(k)+':'+javascript_json(v)for k,v in value.items())+'}'
    raise ValueError('Invalid graph JSON.')


def structure_signature(graph):
    omitted={'outputName','initializer','inputMode','inputSeed','datasetIndex','datasetSplit','datasetValues','storyExperiment'}
    nodes=[]
    for n in graph['nodes']:
        params={k:v for k,v in n['params'].items()if k not in omitted and not(n['type']in('weight','bias')and k in ('value','trainable','constraint'))}
        nodes.append({'id':n['id'],'type':n['type'],'params':params})
    edges=[{k:e[k]for k in ('source','target','sourceSlot','inputSlot')if k in e}for e in graph['edges']]
    text=javascript_json({'nodes':nodes,'edges':edges});units=text.encode('utf-16-le');h=2166136261
    for i in range(0,len(units),2):h=((h^(units[i]+256*units[i+1]))*16777619)&0xffffffff
    return f'{len(units)//2}:{h:x}'


def validate_reward_schema(graph,state):
    finite=lambda n:type(n)in(int,float)and math.isfinite(n)
    require(state.get('structureSignature')==structure_signature(graph),'Reward reference is stale or incompatible with this graph. Restore the captured graph.')
    prefs=state.get('preferences');require(isinstance(prefs,list)and len(prefs)<=5000,'Invalid preference records.')
    actions=state['actionTokenIds']
    for p in prefs:
        require(isinstance(p,dict)and type(p.get('row'))is int and 0<=p['row']<=50000 and p.get('preferred')in actions and p.get('rejected')in actions and p['preferred']!=p['rejected'],'Invalid saved preference record.')
    reward=state.get('rewardModel')
    if reward is not None:
        require(isinstance(reward,dict)and type(reward.get('steps'))is int and 1<=reward['steps']<=10000 and finite(reward.get('preferenceLoss')),'Reward model is not a valid app checkpoint.')
        k=len(actions)
        require(isinstance(reward.get('weights'),list)and len(reward['weights'])==k*k and isinstance(reward.get('bias'),list)and len(reward['bias'])==k and all(finite(n)for n in reward['weights']+reward['bias']),'Invalid saved reward weights.')
        if 'preferenceAccuracy'in reward:require(finite(reward['preferenceAccuracy'])and 0<=reward['preferenceAccuracy']<=1,'Invalid cached preference agreement.')
    report=state.get('lastReport')
    if report is not None:
        require(isinstance(report,dict)and type(report.get('epoch'))is int and report['epoch']>=0,'Invalid reward report schema.')
        for key in ('train','validation'):
            m=report.get(key);require(isinstance(m,dict)and all(finite(m.get(k))for k in ('expectedReward','exactReward','accuracy','kl','objective','examples')),'Invalid reward report schema.')


def policy_state(path,reference):
    model=load(path,reference['dataset'])
    raw=json.loads(Path(path).read_text());graph=raw['state']['graph'];state=graph.get('rewardLearning')
    require(isinstance(state,dict) and state.get('version')==1,'Capture a reward reference in Train → Reward learning.')
    require(state.get('actionTokenIds')==reference['actionTokenIds'],'Keep all canonical candidate answers in vocabulary order.')
    require(state.get('mode') in ('verifiable','preferences'),'Choose the required reward source.')
    validate_reward_schema(graph,state)
    beta=state.get('klStrength');require(type(beta) in (int,float) and math.isfinite(beta) and abs(beta-.1)<1e-12,'Use Reference KL strength 0.1.')
    weights=state.get('referenceParameters');require(isinstance(weights,dict) and set(weights)==set(model.params),'Reference must contain each transformer parameter exactly once.')
    frozen=copy.deepcopy(graph)
    for n in frozen['nodes']:
        if n['id'] in weights:
            value=weights[n['id']];require(isinstance(value,dict) and value.get('shape')==list(model.params[n['id']].shape),'Reference parameter shapes must match.')
            n['params']['value']=value
    origin=Graph(frozen,reference['dataset'])
    require(sum(p.size for p in model.params.values())==reference['parameterCount'],'Keep the complete two-block two-head transformer.')
    rows=list(examples(reference['dataset']));template=Graph(reference['template'],reference['dataset']);expected=template.execute(rows[0][0],rows[0][1])[1:3]
    for candidate in (model,origin):
        require(candidate.execute(rows[0][0],rows[0][1])[1:3]==expected,'Keep the assigned architecture and causal masks.')
    return model,origin,state,rows


def preference_metrics(origin,state,rows,reference):
    prefs=state.get('preferences');require(isinstance(prefs,list) and reference['minimumPreferences']<=len(prefs)<=5000,'Record at least 24 distinct human comparisons.')
    require(len({p.get('row') for p in prefs if isinstance(p,dict)})>=reference['minimumPreferenceRows'],'Compare answers on at least 12 different training examples.')
    actions=state['actionTokenIds']; k=len(actions); index={a:i for i,a in enumerate(actions)}; pairs=set();targets=set();features=[]; chosen=[];rejected=[]
    for p in prefs:
        require(isinstance(p,dict) and type(p.get('row')) is int and 0<=p['row']<len(rows),'Invalid preference example.')
        row=p['row'];x,y,split=rows[row]; a=p.get('preferred');b=p.get('rejected')
        require(split=='train','Human comparison fitting may use training examples only.')
        require(a in index and b in index and a!=b,'Choose two distinct candidate answers.')
        require((row,min(a,b),max(a,b)) not in pairs,'Duplicate preference pairs do not count.')
        pairs.add((row,min(a,b),max(a,b)));targets.add(a)
        require(a==int(y[-1]) and b!=int(y[-1]),'For this task, prefer the answer supported by the story over a false answer.')
        features.append(softmax(origin.execute(x,y)[0][-1,actions]));chosen.append(index[a]);rejected.append(index[b])
    require(len(targets)>=3,'Include at least three distinct preferred answers.')
    reward=state.get('rewardModel');require(isinstance(reward,dict),'Fit a reward model after recording comparisons.')
    w=np.asarray(reward.get('weights'),dtype=float);b=np.asarray(reward.get('bias'),dtype=float)
    require(w.shape==(k*k,) and b.shape==(k,) and np.isfinite(w).all() and np.isfinite(b).all() and max(np.abs(w).max(),np.abs(b).max())<=1000,'Invalid bounded reward model.')
    w=w.reshape(k,k);scores=np.asarray(features)@w+b
    margins=scores[np.arange(len(prefs)),chosen]-scores[np.arange(len(prefs)),rejected]
    return {'preferenceLoss':float(np.logaddexp(0,-margins).mean()),'preferenceAccuracy':float((margins>0).mean()),'comparisons':len(prefs),'rows':len({p['row'] for p in prefs})},w,b


def evaluate_policy(path,reference):
    model,origin,state,rows=policy_state(path,reference);actions=state['actionTokenIds'];beta=state['klStrength'];prefs={};w=b=None
    if state['mode']=='preferences':prefs,w,b=preference_metrics(origin,state,rows,reference)
    metrics={}
    for split in ('train','test'):
        available=[(i,row) for i,row in enumerate(rows) if row[2]==split];limit=reference['evaluationRows'][split]
        selected=[available[i] for i in np.linspace(0,len(available)-1,min(limit,len(available)),dtype=int)]
        values=[]
        for _,(x,y,_) in selected:
            logits=model.execute(x,y)[0][-1];ref_logits=origin.execute(x,y)[0][-1]
            p=softmax(logits[actions]);q=softmax(ref_logits[actions]);target=actions.index(int(y[-1]));true=np.eye(len(actions))[target]
            rewards=true if w is None else 1/(1+np.exp(-np.clip(q@w+b,-60,60)))
            kl=float(np.sum(p*(np.log(np.maximum(p,1e-30))-np.log(np.maximum(q,1e-30)))))
            expected=float(p@rewards);baseline=float(q@rewards)
            values.append([expected,baseline,float(p@true),float(q@true),kl,expected-beta*kl-baseline,float(logits.argmax()==int(y[-1])),float(ref_logits.argmax()==int(y[-1]))])
        means=np.mean(values,axis=0);metrics['validation' if split=='test' else 'train']=dict(zip(['reward','referenceReward','exactReward','referenceExactReward','kl','objectiveGain','accuracy','referenceAccuracy'],map(float,means)))
    return model,origin,state,dict(metrics,**prefs)


def model_result(path,part,reference):
    points=reference['parts'][part]['points'];tests=[];metrics={}
    def add(name,ok,score,message):tests.append({'name':part+' — '+name,'score':score if ok else 0,'max_score':score,'output':message,'visibility':'visible'})
    try:
        model,origin,state,metrics=evaluate_policy(path,reference)
        add('saved transformer and canonical data',True,points*.2,'Architecture, data, reference tensors and predictions recomputed independently.')
        expected='preferences' if part in ('preferences','rlhf') else 'verifiable';require(state['mode']==expected,'Choose the required reward source.')
        require(metrics['validation']['referenceAccuracy']>=reference['minimumAccuracy'],'The captured predecessor must remain an adequately trained QA model.')
        add('reward source and predecessor quality',True,points*.2,'Validated the reward source and independently evaluated the frozen predecessor.')
        unchanged=all(np.array_equal(model.params[key],origin.params[key]) for key in model.params)
        if reference['parts'][part].get('role')=='handoff':
            require(not unchanged and metrics['train']['objectiveGain']>=reference.get('minimumObjectiveGain',.00001),'Supply the completed verifiable-reward policy from the preceding lesson.')
            require(metrics['validation']['accuracy']>=reference['minimumAccuracy'],'Keep the predecessor QA accuracy at least 65%.')
            add('completed predecessor',True,points*.6,'Verified the preceding reward policy independently.')
        elif part=='before':
            require(unchanged,'Save before.json immediately after capturing the instruction-tuned reference, before reward updates.')
            add('unchanged baseline',True,points*.6,'Saved policy exactly matches its captured reference.')
        elif part=='preferences':
            predecessor,_,previous,_=evaluate_policy(Path(path).with_name('rlvr.json'),reference)
            require(previous['mode']=='verifiable' and all(np.array_equal(origin.params[key],predecessor.params[key]) for key in origin.params),'Capture the exact submitted rlvr.json transformer as the preference reference.')
            require(unchanged,'Keep policy weights unchanged while fitting the preference reward model.')
            require(metrics['preferenceLoss']<=.68 and metrics['preferenceAccuracy']>=.6,'Fit the reward model until comparison loss is at most 0.68 and agreement at least 60%.')
            add('human comparisons and fitted reward',True,points*.6,f"Recomputed pairwise loss {metrics['preferenceLoss']:.4f}, agreement {metrics['preferenceAccuracy']:.1%}; no cached fit score is trusted.")
        else:
            parent=Path(path).with_name('before.json' if part=='rlvr' else 'preferences.json')
            pm,po,ps,parent_metrics=evaluate_policy(parent,reference)
            require(all(np.array_equal(pm.params[key],po.params[key])for key in pm.params),'Preserve the paired pre-update policy exactly.')
            require(ps['mode']==expected,'The paired checkpoint must use the same reward source.')
            if part=='rlhf':require(parent_metrics['preferenceLoss']<=.68 and parent_metrics['preferenceAccuracy']>=.6,'The paired preference model must meet the required fit targets.')
            require(all(np.array_equal(origin.params[key],po.params[key]) for key in origin.params),'Use the same captured predecessor in both submitted files.')
            if part=='rlhf':require(state['preferences']==ps['preferences'] and state['rewardModel']==ps['rewardModel'],'Keep the fitted reward model and comparisons fixed during policy optimization.')
            require(not unchanged,'Update the transformer policy using reward feedback.')
            add('paired checkpoint and actual parameter update',True,points*.2,'Paired files share the same reference; transformer parameters changed.')
            require(metrics['train']['objectiveGain']>=reference.get('minimumObjectiveGain',.00001),'Increase the independently recomputed training reward objective relative to the captured reference.')
            require(metrics['validation']['accuracy']>=reference['minimumAccuracy'],'Keep full-vocabulary QA accuracy at least 65% on validation examples.')
            add('reward optimization and QA retention',True,points*.4,'The reward objective improved on the fixed training benchmark and validation QA quality meets its floor.')
    except Exception as exc:
        remaining=points-sum(t['max_score'] for t in tests)
        if remaining:add('remaining requirements',False,remaining,str(exc))
    return {'tests':tests,'metrics':metrics}

if __name__=='__main__':
    part,path,reference=sys.argv[1:4]
    print(json.dumps(model_result(path,part,json.loads(Path(reference).read_text())),allow_nan=False))
