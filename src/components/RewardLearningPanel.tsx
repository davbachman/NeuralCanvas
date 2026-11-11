import {useState} from 'react'
import {datasetExamplesForNode} from '../domain/datasets'
import {captureRewardReference,recordPreference,rewardActionLabels,type RewardLearning} from '../domain/rewardLearning'
import type {GraphModel} from '../domain/types'

export function RewardLearningPanel({graph,filename,disabled,onChange,onRun}:{graph:GraphModel;filename?:string;disabled:boolean;onChange:(graph:GraphModel)=>void;onRun:(kind:'fit'|'policy'|'evaluate')=>void}){
  const identity=`${filename??'untitled'}:${graph.rewardLearning?.mode??'none'}`
  const [sourceSelection,setSourceSelection]=useState<{identity:string;mode:RewardLearning['mode']}>({identity,mode:graph.rewardLearning?.mode??'verifiable'})
  const mode=sourceSelection.identity===identity?sourceSelection.mode:graph.rewardLearning?.mode??'verifiable'
  const [rowIndex,setRowIndex]=useState(0),[preferred,setPreferred]=useState(0),[rejected,setRejected]=useState(1),[error,setError]=useState('')
  const state=graph.rewardLearning,source=graph.nodes.find(n=>n.type==='dataset'),examples=source?datasetExamplesForNode(source):[]
  const trainingRows=examples.flatMap((row,i)=>row.split==='train'?[i]:[]),row=trainingRows[Math.min(rowIndex,Math.max(0,trainingRows.length-1))]
  const labels=rewardActionLabels(graph),report=state?.lastReport
  const context=source?.params.textData?.representation==='tokens'&&examples[row]
    ?examples[row].features[0].data.map(id=>source.params.textData!.vocabulary[id]??'<unk>').filter(token=>token!=='<pad>').join(' ')
    :examples[row]?.features.map((value,i)=>`Feature ${i+1}: ${value.data.join(', ')}`).join('; ')
  const change=(action:()=>GraphModel)=>{try{onChange(action());setError('')}catch(e){setError(e instanceof Error?e.message:String(e))}}
  return <details className="training-walkthrough">
    <summary>Reward learning</summary>
    <p className="run-intro">Adapt a categorical answer model with a one-step reward. The policy considers every candidate answer; the reference model stays fixed.</p>
    <label className="run-field">Reward source for new reference<select aria-label="Reward source" disabled={disabled} value={mode} onChange={e=>setSourceSelection({identity,mode:e.target.value as RewardLearning['mode']})}><option value="verifiable">Verifiable answer correctness</option><option value="preferences">Human preference comparisons</option></select></label>
    <button type="button" disabled={disabled} onClick={()=>change(()=>captureRewardReference(graph,mode))}>Capture current model as reward reference</button>
    <p className="run-intro">Capturing again replaces the reference and clears recorded preferences, reward weights, and reward reports. Save the current project first if you need that state.</p>
    {state&&<>
      <p className="run-intro">Active reward: {state.mode==='verifiable'?'verifiable correctness':'learned human preferences'}. Candidate answers: {labels.join(', ')}.</p>
      <button type="button" disabled={disabled} onClick={()=>onChange({...graph,rewardLearning:undefined})}>Clear reward setup</button>
      <p className="run-intro">Clear the setup after saving the reward checkpoint when moving to a different task. Current model parameters are preserved.</p>
      <label className="run-field">Reference KL strength<input aria-label="Reference KL strength" type="number" min="0" max="100" step="0.01" disabled={disabled} value={state.klStrength} onChange={e=>{const value=Number(e.target.value);if(Number.isFinite(value)&&value>=0&&value<=100)onChange({...graph,rewardLearning:{...state,klStrength:value,lastReport:undefined}})}}/></label>
      {state.mode==='preferences'&&<>
        <p className="run-intro">Read the context and record your own preferred answer. Comparisons use training contexts only. The small reward model learns from these judgments using the frozen reference model’s answer probabilities.</p>
        <label className="run-field">Training context<input aria-label="Preference training context" type="number" min="1" max={trainingRows.length} value={Math.min(rowIndex+1,trainingRows.length)} disabled={disabled} onChange={e=>setRowIndex(Math.max(0,Math.min(trainingRows.length-1,Number(e.target.value)-1)))}/></label>
        <p className="run-intro" aria-label="Preference context">Dataset row {row}: {context}</p>
        <label className="run-field">Preferred answer<select aria-label="Preferred answer" value={preferred} disabled={disabled} onChange={e=>setPreferred(Number(e.target.value))}>{labels.map((label,i)=><option key={i} value={i}>{label}</option>)}</select></label>
        <label className="run-field">Rejected answer<select aria-label="Rejected answer" value={rejected} disabled={disabled} onChange={e=>setRejected(Number(e.target.value))}>{labels.map((label,i)=><option key={i} value={i}>{label}</option>)}</select></label>
        <button type="button" disabled={disabled||preferred===rejected||row===undefined} onClick={()=>change(()=>recordPreference(graph,{row,preferred:state.actionTokenIds[preferred],rejected:state.actionTokenIds[rejected]}))}>Record human preference</button>
        <p className="run-intro">{state.preferences.length} comparisons across {new Set(state.preferences.map(p=>p.row)).size} training contexts.</p>
        <button type="button" disabled={disabled||state.preferences.length<2} onClick={()=>onRun('fit')}>Fit preference reward model</button>
        {state.rewardModel&&<p className="run-intro">Pairwise training loss: {state.rewardModel.preferenceLoss.toFixed(4)}.{state.rewardModel.preferenceAccuracy!==undefined&&` Comparison agreement: ${(100*state.rewardModel.preferenceAccuracy).toFixed(1)}%.`} Reward weights stay fixed during policy optimization.</p>}
      </>}
      <button type="button" disabled={disabled||(state.mode==='preferences'&&!state.rewardModel)} onClick={()=>onRun('evaluate')}>Evaluate reward policy</button>
      <button type="button" disabled={disabled||(state.mode==='preferences'&&!state.rewardModel)} onClick={()=>onRun('policy')}>Optimize policy with rewards</button>
      <p className="run-intro">Policy optimization uses Epochs per run, Examples per update, learning rate, optimizer, and gradient norm limit above. It keeps the final parameters; early stopping is not used. Ordinary Run epochs still trains the canvas Loss.</p>
      {report&&<div className="run-intro" aria-label="Reward report"><strong>Reward report · epoch {report.epoch}</strong><table><thead><tr><th>Measure</th><th>Training</th><th>Validation</th></tr></thead><tbody>{(['expectedReward','exactReward','accuracy','kl','objective'] as const).map(key=><tr key={key}><td>{{expectedReward:'Expected reward',exactReward:'Expected correctness',accuracy:'Candidate accuracy',kl:'Reference KL',objective:'Reward objective'}[key]}</td><td>{report.train[key].toFixed(4)}</td><td>{report.validation[key].toFixed(4)}</td></tr>)}</tbody></table><p>Candidate probabilities are normalized over the listed answers. Ordinary Test reports the full output vocabulary.</p></div>}
    </>}
    {error&&<p role="alert">{error}</p>}
  </details>
}
