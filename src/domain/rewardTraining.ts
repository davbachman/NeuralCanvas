import {datasetExamplesForNode,type DatasetExample} from './datasets'
import {trainingBatches} from './datasetTraining'
import {TensorGraph,TensorOptimizer,selectTensorBackend,tf} from './tensorTraining'
import {DEFAULT_TRAINING} from './trainingSettings'
import {referenceGraph,validateRewardReference,type RewardMetrics,type RewardReport,type RewardModel} from './rewardLearning'
import type {GraphModel} from './types'

export function logProbabilities(values:number[]):number[]{
  const max=Math.max(...values),normalizer=max+Math.log(values.reduce((s,n)=>s+Math.exp(n-max),0))
  return values.map(n=>n-normalizer)
}
export function learnedRewards(features:number[],model:RewardModel):number[]{
  const k=features.length
  return model.bias.map((bias,a)=>1/(1+Math.exp(-Math.max(-30,Math.min(30,bias+features.reduce((sum,x,i)=>sum+x*model.weights[i*k+a],0))))))
}
export async function candidateLogProbabilities(model:TensorGraph,rows:DatasetExample[],actions:number[],batchSize=32,signal?:AbortSignal):Promise<number[][]>{
  const result:number[][]=[]
  for(let start=0;start<rows.length;start+=batchSize){
    signal?.throwIfAborted()
    const batch=rows.slice(start,start+batchSize),prediction=await model.inspect(batch),width=prediction.shape.at(-1)!
    if(prediction.prediction.length!==batch.length*width||actions.some(a=>a>=width))throw Error('The model must produce one answer-logit vector for each example.')
    for(let i=0;i<batch.length;i++)result.push(logProbabilities(actions.map(a=>prediction.prediction[i*width+a])))
    await new Promise(resolve=>setTimeout(resolve,0))
  }
  return result
}

/** Pairwise logistic reward model on reference-policy probabilities. Its input
 * representation remains fixed when the policy is subsequently optimized. */
export async function fitRewardModel(graph:GraphModel,options:{steps?:number;rate?:number;signal?:AbortSignal}={}):Promise<GraphModel>{
  const state=validateRewardReference(graph),steps=options.steps??200,rate=options.rate??.5,k=state.actionTokenIds.length
  if(state.preferences.length<2)throw Error('Record at least two human preference comparisons first.')
  if(!Number.isInteger(steps)||steps<1||steps>10000||!Number.isFinite(rate)||rate<=0||rate>10)throw Error('Choose valid reward-model steps and learning rate.')
  await selectTensorBackend(graph,(graph.training??DEFAULT_TRAINING).backend)
  const model=new TensorGraph(referenceGraph(graph))
  try{
    const ids=[...new Set(state.preferences.map(p=>p.row))],all=model.examples,rows=ids.map(i=>all[i])
    const logps=await candidateLogProbabilities(model,rows,state.actionTokenIds,32,options.signal)
    const features=new Map(ids.map((id,i)=>[id,logps[i].map(Math.exp)]))
    const weights=Array(k*k).fill(0) as number[],bias=Array(k).fill(0) as number[]
    const loss=()=>state.preferences.reduce((sum,p)=>{
      const x=features.get(p.row)!,a=state.actionTokenIds.indexOf(p.preferred),b=state.actionTokenIds.indexOf(p.rejected)
      const margin=bias[a]-bias[b]+x.reduce((s,n,i)=>s+n*(weights[i*k+a]-weights[i*k+b]),0)
      return sum+Math.max(0,-margin)+Math.log1p(Math.exp(-Math.abs(margin)))
    },0)/state.preferences.length
    for(let step=0;step<steps;step++){
      options.signal?.throwIfAborted()
      const dw=weights.map(w=>.001*w),db=bias.map(()=>0)
      for(const p of state.preferences){
        const x=features.get(p.row)!,a=state.actionTokenIds.indexOf(p.preferred),b=state.actionTokenIds.indexOf(p.rejected)
        const margin=bias[a]-bias[b]+x.reduce((s,n,i)=>s+n*(weights[i*k+a]-weights[i*k+b]),0)
        const gradient=-1/(1+Math.exp(Math.max(-30,Math.min(30,margin))))/state.preferences.length
        db[a]+=gradient;db[b]-=gradient
        x.forEach((n,i)=>{dw[i*k+a]+=gradient*n;dw[i*k+b]-=gradient*n})
      }
      weights.forEach((w,i)=>{weights[i]=w-rate*dw[i]});bias.forEach((b,i)=>{bias[i]=b-rate*db[i]})
      if(step%20===0)await new Promise(resolve=>setTimeout(resolve,0))
    }
    const preferenceAccuracy=state.preferences.reduce((sum,p)=>{
      const x=features.get(p.row)!,a=state.actionTokenIds.indexOf(p.preferred),b=state.actionTokenIds.indexOf(p.rejected)
      const margin=bias[a]-bias[b]+x.reduce((s,n,i)=>s+n*(weights[i*k+a]-weights[i*k+b]),0)
      return sum+(margin>0?1:margin===0?.5:0)
    },0)/state.preferences.length
    return {...graph,rewardLearning:{...state,rewardModel:{weights,bias,steps,preferenceLoss:loss(),preferenceAccuracy},lastReport:undefined}}
  }finally{model.dispose()}
}

function metrics(logps:number[][],reference:number[][],rewards:number[][],rows:DatasetExample[],actions:number[],klStrength:number):RewardMetrics{
  let expectedReward=0,exactReward=0,accuracy=0,kl=0
  logps.forEach((lp,i)=>{
    const probabilities=lp.map(Math.exp),correct=actions.indexOf(rows[i].target.data[0])
    expectedReward+=probabilities.reduce((s,p,a)=>s+p*rewards[i][a],0)
    exactReward+=correct<0?0:probabilities[correct]
    accuracy+=Number(actions[lp.indexOf(Math.max(...lp))]===rows[i].target.data[0])
    kl+=probabilities.reduce((s,p,a)=>s+p*(lp[a]-reference[i][a]),0)
  })
  const n=rows.length||1
  return {expectedReward:expectedReward/n,exactReward:exactReward/n,accuracy:accuracy/n,kl:kl/n,objective:(-expectedReward+klStrength*kl)/n,examples:rows.length}
}
export interface RewardTrainOptions {epochs:number;batchSize:number;reportEvery?:number;shuffle?:boolean;signal?:AbortSignal;onReport?:(report:RewardReport)=>void;onProgress?:(done:number,total:number)=>void}
export async function trainRewardPolicy(graph:GraphModel,options:RewardTrainOptions):Promise<GraphModel>{
  const state=validateRewardReference(graph),settings=graph.training??DEFAULT_TRAINING
  if(!Number.isInteger(options.epochs)||options.epochs<0||options.epochs>10000||!Number.isInteger(options.batchSize)||options.batchSize<1||options.batchSize>4096)throw Error('Choose valid reward epochs and batch size.')
  if(state.mode==='preferences'&&!state.rewardModel)throw Error('Fit the preference reward model before optimizing the policy.')
  await selectTensorBackend(graph,settings.backend,settings)
  const policy=new TensorGraph(graph),reference=new TensorGraph(referenceGraph(graph)),optimizer=new TensorOptimizer({...settings,patience:0},graph.learningRate)
  try{
    const train=policy.examples.filter(r=>r.split==='train'),validation=policy.examples.filter(r=>r.split==='test')
    if(!train.length||!validation.length||options.batchSize>train.length)throw Error('Reward learning requires training and validation examples and a valid batch size.')
    const trainRef=await candidateLogProbabilities(reference,train,state.actionTokenIds,options.batchSize,options.signal)
    const valRef=await candidateLogProbabilities(reference,validation,state.actionTokenIds,options.batchSize,options.signal)
    const rewards=(rows:DatasetExample[],ref:number[][])=>rows.map((row,i)=>state.mode==='verifiable'?state.actionTokenIds.map(a=>Number(a===row.target.data[0])):learnedRewards(ref[i].map(Math.exp),state.rewardModel!))
    const trainRewards=rewards(train,trainRef),valRewards=rewards(validation,valRef)
    let report:RewardReport|undefined
    const evaluate=async(epoch:number)=>{
      const trainPolicy=await candidateLogProbabilities(policy,train,state.actionTokenIds,options.batchSize,options.signal)
      const valPolicy=await candidateLogProbabilities(policy,validation,state.actionTokenIds,options.batchSize,options.signal)
      report={epoch,train:metrics(trainPolicy,trainRef,trainRewards,train,state.actionTokenIds,state.klStrength),validation:metrics(valPolicy,valRef,valRewards,validation,state.actionTokenIds,state.klStrength)}
      options.onReport?.(report)
    }
    await evaluate(0)
    for(let epoch=1;epoch<=options.epochs;epoch++){
      const batches=trainingBatches(train.map((_,i)=>i),options.batchSize,epoch-1,options.shuffle??true)
      let done=0
      for(const ids of batches){
        options.signal?.throwIfAborted()
        const result=policy.rewardGradients(ids.map(i=>train[i]),state.actionTokenIds,ids.map(i=>trainRewards[i]),ids.map(i=>trainRef[i]),state.klStrength)
        try{
          if(!Number.isFinite((await result.loss.data())[0]))throw Error('Reward objective diverged. Reduce the learning rate.')
          optimizer.update(policy.variables,result.grads)
        }finally{tf.dispose([result.loss,...Object.values(result.grads)])}
        done+=ids.length;options.onProgress?.((epoch-1)*train.length+done,options.epochs*train.length)
        await new Promise(resolve=>setTimeout(resolve,0))
      }
      if(epoch%(options.reportEvery??1)===0||epoch===options.epochs)await evaluate(epoch)
    }
    const updated=options.epochs===0?graph:await policy.snapshot()
    return {...updated,rewardLearning:{...state,lastReport:report}}
  }finally{policy.dispose();reference.dispose();optimizer.dispose()}
}

export async function evaluateRewardPolicy(graph:GraphModel,batchSize=32,signal?:AbortSignal):Promise<RewardReport>{
  const result=await trainRewardPolicy(graph,{epochs:0,batchSize,signal})
  return result.rewardLearning!.lastReport!
}

export function preferenceTrainingRows(graph:GraphModel):number[]{
  const source=graph.nodes.find(n=>n.type==='dataset')
  return source?datasetExamplesForNode(source).flatMap((row,i)=>row.split==='train'?[i]:[]):[]
}
