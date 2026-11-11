import {datasetExamplesForNode, datasetForNode} from './datasets'
import {toTensor} from './tensor'
import type {GraphModel, TensorValue} from './types'

export interface Preference {row:number; preferred:number; rejected:number}
export interface RewardModel {
  /** Row-major [reference-answer feature, candidate action]. */
  weights:number[]
  bias:number[]
  steps:number
  preferenceLoss:number
  preferenceAccuracy?:number
}
export interface RewardLearning {
  version:1
  mode:'verifiable'|'preferences'
  actionTokenIds:number[]
  referenceParameters:Record<string,TensorValue>
  structureSignature:string
  klStrength:number
  preferences:Preference[]
  rewardModel?:RewardModel
  lastReport?:RewardReport
}
export interface RewardMetrics {expectedReward:number; exactReward:number; accuracy:number; kl:number; objective:number; examples:number}
export interface RewardReport {epoch:number; train:RewardMetrics; validation:RewardMetrics}

/** Detect changed data/operations while allowing renamed blocks and learned weights. */
export function rewardStructureSignature(graph:GraphModel):string {
  const text=JSON.stringify({nodes:graph.nodes.map(n=>({id:n.id,type:n.type,params:Object.fromEntries(Object.entries(n.params).filter(([key])=>
    !['outputName','initializer','inputMode','inputSeed','datasetIndex','datasetSplit','datasetValues','storyExperiment'].includes(key) &&
    !(['weight','bias'].includes(n.type)&&['value','trainable','constraint'].includes(key))))})),edges:graph.edges.map(({source,target,sourceSlot,inputSlot})=>({source,target,sourceSlot,inputSlot}))})
  let hash=2166136261
  for(let i=0;i<text.length;i++){hash^=text.charCodeAt(i);hash=Math.imul(hash,16777619)}
  return `${text.length}:${(hash>>>0).toString(16)}`
}

export function captureRewardReference(graph:GraphModel,mode:RewardLearning['mode']='verifiable'):GraphModel {
  const datasets=graph.nodes.filter(n=>n.type==='dataset'),losses=graph.nodes.filter(n=>n.type==='loss'||n.type==='cross-entropy')
  if(datasets.length!==1||losses.length!==1||!['cross-entropy'].includes(losses[0].params.loss??(losses[0].type==='cross-entropy'?'cross-entropy':''))) throw Error('Reward learning requires one dataset and one cross-entropy Loss on answer logits.')
  const rows=datasetExamplesForNode(datasets[0]),train=rows.filter(r=>r.split==='train')
  if(!train.length||rows.some(r=>r.target.data.length!==1)) throw Error('Choose a task with one categorical answer per example.')
  const actionTokenIds=[...new Set(train.map(r=>r.target.data[0]))].sort((a,b)=>a-b)
  if(actionTokenIds.length<2||actionTokenIds.length>64||actionTokenIds.some(a=>!Number.isInteger(a)||a<0)) throw Error('Choose a task with 2–64 categorical answers in its training split.')
  const referenceParameters=Object.fromEntries(graph.nodes.filter(n=>n.type==='weight'||n.type==='bias').map(n=>[n.id,structuredClone(toTensor(n.params.value))]))
  if(!Object.keys(referenceParameters).length) throw Error('Build a model with parameters before capturing a reference.')
  return {...graph,rewardLearning:{version:1,mode,actionTokenIds,referenceParameters,structureSignature:rewardStructureSignature(graph),klStrength:.1,preferences:[]}}
}

export function validateRewardReference(graph:GraphModel):RewardLearning {
  const state=graph.rewardLearning
  if(!state||!isRewardLearning(state)) throw Error('Capture a reward reference first.')
  if(state.structureSignature!==rewardStructureSignature(graph)) throw Error('The dataset or graph changed. Capture a new reward reference for this task.')
  const parameters=graph.nodes.filter(n=>n.type==='weight'||n.type==='bias')
  if(parameters.length!==Object.keys(state.referenceParameters).length||parameters.some(n=>{
    const saved=state.referenceParameters[n.id],value=toTensor(n.params.value)
    return !saved||JSON.stringify(saved.shape)!==JSON.stringify(value.shape)
  })) throw Error('Reference parameters do not match this graph.')
  const source=graph.nodes.find(n=>n.type==='dataset')!,rows=datasetExamplesForNode(source)
  if(state.preferences.some(p=>rows[p.row]?.split!=='train')) throw Error('Preference comparisons must use training examples only.')
  return state
}

export function referenceGraph(graph:GraphModel):GraphModel {
  const state=validateRewardReference(graph)
  return {...graph,rewardLearning:undefined,nodes:graph.nodes.map(n=>({...n,params:state.referenceParameters[n.id]?{...n.params,value:structuredClone(state.referenceParameters[n.id]),trainable:false}:n.type==='dropout'?{...n.params,inferenceOnly:true}:n.params}))}
}

export function recordPreference(graph:GraphModel,preference:Preference):GraphModel {
  const state=validateRewardReference(graph)
  const next={...state,preferences:[...state.preferences.filter(p=>!(p.row===preference.row&&
    ((p.preferred===preference.preferred&&p.rejected===preference.rejected)||(p.rejected===preference.preferred&&p.preferred===preference.rejected)))),preference],rewardModel:undefined,lastReport:undefined}
  if(!isRewardLearning(next)) throw Error('Choose two distinct candidate answers and a valid training row.')
  const result={...graph,rewardLearning:next};validateRewardReference(result);return result
}

export function rewardActionLabels(graph:GraphModel):string[] {
  const source=graph.nodes.find(n=>n.type==='dataset')
  if(!source||!graph.rewardLearning)return []
  const data=datasetForNode(source),labels=source.params.textData?.task==='language'?data.vocabulary:data.classLabels
  return graph.rewardLearning.actionTokenIds.map(id=>labels?.[id]??String(id))
}

export function isRewardLearning(value:unknown):value is RewardLearning {
  if(!value||typeof value!=='object')return false
  const v=value as RewardLearning,k=v.actionTokenIds?.length
  const finite=(n:unknown):n is number=>typeof n==='number'&&Number.isFinite(n)
  if(v.version!==1||!['verifiable','preferences'].includes(v.mode)||!Array.isArray(v.actionTokenIds)||k<2||k>64||new Set(v.actionTokenIds).size!==k||v.actionTokenIds.some(n=>!Number.isInteger(n)||n<0)||!finite(v.klStrength)||v.klStrength<0||v.klStrength>100||typeof v.structureSignature!=='string'||v.structureSignature.length>100||!v.referenceParameters||typeof v.referenceParameters!=='object')return false
  let size=0
  for(const tensor of Object.values(v.referenceParameters)){
    if(!tensor||!Array.isArray(tensor.shape)||!Array.isArray(tensor.data)||tensor.shape.some(n=>!Number.isInteger(n)||n<1)||tensor.shape.reduce((a,b)=>a*b,1)!==tensor.data.length||tensor.data.some(n=>!finite(n)))return false
    size+=tensor.data.length
  }
  if(size===0||size>2_000_000||!Array.isArray(v.preferences)||v.preferences.length>5000||v.preferences.some(p=>!p||!Number.isInteger(p.row)||p.row<0||p.row>50000||p.preferred===p.rejected||!v.actionTokenIds.includes(p.preferred)||!v.actionTokenIds.includes(p.rejected)))return false
  if(v.rewardModel){const m=v.rewardModel;if(!Array.isArray(m.weights)||m.weights.length!==k*k||!Array.isArray(m.bias)||m.bias.length!==k||[...m.weights,...m.bias,m.preferenceLoss].some(n=>!finite(n))||!Number.isInteger(m.steps)||m.steps<1||m.steps>10000||(m.preferenceAccuracy!==undefined&&(!finite(m.preferenceAccuracy)||m.preferenceAccuracy<0||m.preferenceAccuracy>1)))return false}
  if(v.lastReport){const r=v.lastReport;if(!Number.isInteger(r.epoch)||r.epoch<0)return false;for(const m of [r.train,r.validation])if(!m||[m.expectedReward,m.exactReward,m.accuracy,m.kl,m.objective,m.examples].some(n=>!finite(n)))return false}
  return true
}
