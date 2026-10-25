"""Grade a model using canonical data and independent bounded NumPy evaluation."""
import json,sys
from pathlib import Path
from graph_eval import Graph,load,evaluate,require,examples,parameter_fingerprint,encode_story,tokens
import numpy as np

def experiment_tokens(value,data):
    if value is None:return None
    encode_story(value['facts'],value['question'],data)
    return [tokens(f,data) for f in value['facts']],tokens(value['question'],data)

def validate_experiment(value,config,data):
    rule=config.get('interventionRule')
    if not rule:
        require(experiment_tokens(value,data)==experiment_tokens(config.get('experiment'),data),'Restore the original story, or apply the experiment required for this file.')
        return config.get('answer')
    require(isinstance(value,dict) and isinstance(value.get('facts'),list) and len(value['facts'])==2 and all(isinstance(f,str) for f in value['facts']) and isinstance(value.get('question'),str),'Apply exactly two facts and the specified question.')
    facts,question=experiment_tokens(value,data)
    require(question==tokens('Where is Mary?',data),'Keep the question Where is Mary? for this intervention.')
    original=[tokens('Mary moved to the kitchen.',data),tokens('John moved to the garden.',data)]
    changed=1 if rule=='irrelevant' else 0
    require(facts[1-changed]==original[1-changed],'Change only the designated fact; keep the other reference fact unchanged.')
    row=facts[changed];expected=original[changed]
    require(len(row)==len(expected) and row[:4]==expected[:4] and row[5:]==expected[5:],'Change only the location, preserving the person, wording and fact order.')
    require(row[4] in data['classLabels'] and row[4]!=expected[4],'Choose a supplied location different from the original location.')
    return 'kitchen' if rule=='irrelevant' else row[4]


def scalar_multiple(signature,original):
    """Recognize equivalent constant scaling without executing submitted code."""
    if signature==original:return 1.0
    if not isinstance(signature,tuple):return None
    if signature[0]=='Div' and signature[2][0]=='number' and signature[2][1]!=0:
        a=scalar_multiple(signature[1],original)
        return None if a is None else a/signature[2][1]
    if signature[0]=='Mult':
        constants=[s[1] for s in signature[1:] if s[0]=='number'];signals=[s for s in signature[1:] if s[0]!='number']
        if len(signals)!=1:return None
        a=scalar_multiple(signals[0],original)
        return None if a is None else a*float(np.prod(constants))
    return None


def verify_temperatures(model,data,temperatures):
    xs,y,_=next(examples(data));logits,sig,_,cache=model.execute(xs,y,with_values=True)
    for temperature in temperatures:
        matching=[values for values,s in cache.values() if s[0]=='softmax' and (factor:=scalar_multiple(s[1],sig)) is not None and abs(factor-1/temperature)<1e-10]
        require(bool(matching),f'Add a separate Softmax branch at temperature {temperature}, scaling the original logits.')
        shifted=logits/temperature;shifted-=shifted.max(axis=-1,keepdims=True);expected=np.exp(shifted);expected/=expected.sum(axis=-1,keepdims=True)
        require(any(v.shape==expected.shape and np.allclose(v,expected,rtol=1e-10,atol=1e-12) for v in matching),'Temperature branch probabilities do not match the required transformation.')

def narrative_diagnostics(model, data, probes):
    """Generate from canonical held-out openings, without feeding future words.

    These observations describe the submitted model, not a claimed UI action.
    The caller has already verified its architecture and canonical dataset.
    """
    vocabulary=data['vocabulary'];index={w:i for i,w in enumerate(vocabulary)}
    grammatical=consistent=0;observations=[]
    for probe in probes:
        ids=[index[w] for w in probe['prompt'].split()];generated=[]
        require(len(ids)>=data['maxLength'],'Canonical generation prompt is too short.')
        for _ in range(12):
            context=np.array(ids[-data['maxLength']:]);target=np.zeros(1 if data.get('targetMode')=='last' else len(context),dtype=int)
            logits,_,_,_=model.execute([context,np.arange(len(context))],target)
            scores=logits[-1].copy()
            if '<pad>' in index:scores[index['<pad>']]=-np.inf
            word=int(scores.argmax());ids.append(word);generated.append(vocabulary[word])
            if vocabulary[word]=='end':break
        people=('liam','nora');rooms=('garden','kitchen','bedroom','office','hall','bathroom')
        grammar=(len(generated)==7 and generated[0] in people and
                 generated[1:4]==['is','in','the'] and generated[4] in rooms and generated[5:]==['.','end']) or (
                 len(generated)==8 and generated[:5]==['the','football','is','in','the'] and
                 generated[5] in rooms and generated[6:]==['.','end']) or (
                 len(generated)==6 and generated[0] in (*people,'nobody') and
                 generated[1:]==['has','the','football','.','end'])
        faithful=grammar and generated in probe['validContinuations']
        grammatical+=grammar;consistent+=faithful
        if len(observations)<3:observations.append({'prompt':probe['prompt'],'continuation':' '.join(generated),'consistent':bool(faithful)})
    return {'generation_count':len(probes),'grammatical_fraction':grammatical/len(probes),
            'consistent_fraction':consistent/len(probes),'generated_examples':observations}

def model_result(path,part,reference):
    config=reference['parts'][part];data=reference['datasets'][config['dataset']];tests=[];metrics={};points=config['points']
    def add(i,ok,message):
        name=['saved model and data','model form','training loss','held-out loss','held-out accuracy'][i]
        tests.append(dict(name=part+' — '+name,score=points[i] if ok else 0,max_score=points[i],output=message,visibility='visible'))
    try:
        model=load(path,data)
        answer=validate_experiment(model.source['params'].get('storyExperiment'),config,data)
        add(0,True,'Original dataset, vocabulary, labels, and splits verified.')
    except Exception as e:
        for i in range(5):add(i,False,str(e) if i==0 else 'Upload the required saved model.')
        return dict(tests=tests,metrics=metrics)
    try:
        metrics=evaluate(model,Graph(config['template'],data),data)
        if config.get('parameterCount') is not None:
            require(sum(value.size for value in model.params.values())==config['parameterCount'],
                    'Keep only the required trainable parameters.')
        if config.get('temperatures'):verify_temperatures(model,data,config['temperatures'])
        add(1,True,'Assigned architecture verified; predictions recomputed independently.')
    except Exception as e:
        for i in range(1,5):add(i,False,str(e) if i==1 else 'Performance credit requires the assigned model form.')
        return dict(tests=tests,metrics=metrics)
    for i,key in [(2,'train'),(3,'validation'),(4,'accuracy')]:
        limit=config['limits'][i-2];value=metrics[key];ok=limit is None or (value>=limit-1e-8 if key=='accuracy' else value<=limit+1e-8)
        add(i,ok,f'Recomputed {key}: {value:.6f}. '+('Informational only.' if points[i]==0 else f'Full-credit threshold: {limit}.'))
    if config.get('experiment'):
        experiment=model.source['params']['storyExperiment'];xs=encode_story(experiment['facts'],experiment['question'],data)
        logits,_,_,_=model.execute(xs,np.array([0]));prediction=data['classLabels'][int(logits.argmax(axis=-1)[0])]
        metrics['experiment_prediction']=prediction
        metrics['experiment_answer']=answer
        tests.append(dict(name=part+' — intervention observation',score=0,max_score=0,output=f"Recomputed prediction: {prediction}. Expected answer from your edited story: {answer}. A failed intervention is evidence to investigate, not an extra grading penalty.",visibility='visible'))
    tests.append(dict(name=part+' — common final-position metric',score=0,max_score=0,output=f"Final-position held-out cross-entropy: {metrics['final_validation']:.6f}.",visibility='visible')) if data['task']=='language' else None
    if reference.get('narrativeProbes'):
        metrics.update(narrative_diagnostics(model,data,reference['narrativeProbes']))
        metrics['parameters']=sum(value.size for value in model.params.values())
        tests.append(dict(name=part+' — narrative continuation',score=0,max_score=0,visibility='visible',
                         output=f"Parameters: {metrics['parameters']}. On {metrics['generation_count']} fixed held-out football-story openings, greedy continuations had the expected consequence grammar in {metrics['grammatical_fraction']:.1%} and stated a fact consistent with the events in {metrics['consistent_fraction']:.1%}. These are informational diagnostics, not extra thresholds.\n"+'\n'.join('Generated: '+row['continuation'] for row in metrics['generated_examples'])))
    return dict(tests=tests,metrics=metrics,fingerprint=parameter_fingerprint(model,data))

if __name__=='__main__':
    part,path,reference_path=sys.argv[1:];reference=json.loads(Path(reference_path).read_text())
    try:result=model_result(path,part,reference)
    except Exception:result={'tests':[dict(name=part+' — evaluation',score=0,max_score=sum(reference['parts'][part]['points']),output='Unable to evaluate this saved model.',visibility='visible')]}
    print(json.dumps(result,allow_nan=False))
