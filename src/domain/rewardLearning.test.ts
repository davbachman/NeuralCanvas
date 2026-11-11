import {afterEach,beforeAll,expect,it} from 'vitest'
import {builder} from '../test/curriculumModels'
import {parseCustomCsv} from './customCsv'
import {TensorGraph,tf} from './tensorTraining'
import {DEFAULT_TRAINING} from './trainingSettings'
import {captureRewardReference,recordPreference,referenceGraph,validateRewardReference} from './rewardLearning'
import {evaluateRewardPolicy,fitRewardModel,trainRewardPolicy} from './rewardTraining'
import {createProjectStateFile,parseProjectStateFile} from './session'
import {recoveryWorkspace} from '../test/recoveryWorkspace'

beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready()})
afterEach(()=>expect(tf.memory().numTensors).toBe(0))
function fixture(){
  const {graph,add}=builder()
  const csv=parseCustomCsv('x,target,split\n-2,a,train\n-1,a,train\n1,b,train\n2,b,train\n-1.5,a,test\n1.5,b,test\n','reward.csv');csv.task='classification'
  add('data','dataset',{dataset:'custom-csv',customCsv:csv,datasetMode:'sample'})
  add('input','reshape',{shape:[1,1]},[['data',0]])
  add('weight','weight',{value:{shape:[1,2],data:[-.1,.1]}})
  add('logits','matmul',{},['input','weight'])
  add('loss','loss',{loss:'cross-entropy'},['logits',['data',1]])
  graph.training={...DEFAULT_TRAINING,backend:'cpu',optimizer:'sgd',clipNorm:1};graph.learningRate=.2
  return graph
}
it('computes the analytic expected-reward gradient and keeps the reference constant',async()=>{
  const graph=captureRewardReference(fixture()),model=new TensorGraph(graph),row=model.examples[2]
  try{
    const p=1/(1+Math.exp(.2)),lp=[Math.log(p),Math.log(1-p)]
    const result=model.rewardGradients([row],[0,1],[[0,1]],[lp],.1)
    try{
      const grad=Array.from(await result.grads[model.variables.get('weight')!.name].data())
      expect(grad[0]).toBeCloseTo(p*(1-p),6);expect(grad[1]).toBeCloseTo(-p*(1-p),6)
      expect((await result.loss.data())[0]).toBeCloseTo(-(1-p),6)
    }finally{tf.dispose([result.loss,...Object.values(result.grads)])}
    expect(referenceGraph(graph).nodes.find(n=>n.id==='weight')!.params.trainable).toBe(false)
  }finally{model.dispose()}
})
it('optimizes the same model and saves independently reevaluable reward reports',async()=>{
  const graph=captureRewardReference(fixture()),original=JSON.stringify(graph)
  const evaluation=await trainRewardPolicy(graph,{epochs:0,batchSize:2})
  expect(evaluation.nodes).toEqual(graph.nodes)
  expect(evaluation.rewardLearning!.referenceParameters).toEqual(graph.rewardLearning!.referenceParameters)
  const before=await evaluateRewardPolicy(graph,2)
  const result=await trainRewardPolicy(graph,{epochs:4,batchSize:2})
  expect(JSON.stringify(graph)).toBe(original)
  expect(result.rewardLearning!.lastReport!.validation.exactReward).toBeGreaterThan(before.validation.exactReward)
  expect(result.rewardLearning!.referenceParameters).toEqual(graph.rewardLearning!.referenceParameters)
  const parsed=parseProjectStateFile(JSON.stringify(createProjectStateFile({...recoveryWorkspace().state,graph:result,visualizationGraph:result})))
  expect(parsed.ok).toBe(true)
  if(!parsed.ok)throw Error(parsed.error)
  const evaluated=await evaluateRewardPolicy(parsed.file.state.graph,2)
  expect(evaluated.validation.expectedReward).toBeCloseTo(result.rewardLearning!.lastReport!.validation.expectedReward,6)
})
it('fits a preference model, freezes it during optimization, and rejects held-out comparisons',async()=>{
  let graph=captureRewardReference(fixture(),'preferences')
  expect(()=>recordPreference(graph,{row:4,preferred:0,rejected:1})).toThrow(/training/)
  expect(()=>recordPreference(graph,{row:0,preferred:0,rejected:0})).toThrow(/distinct/)
  for(let row=0;row<4;row++)graph=recordPreference(graph,{row,preferred:row<2?0:1,rejected:row<2?1:0})
  const fitted=await fitRewardModel(graph)
  expect(fitted.rewardLearning!.rewardModel!.preferenceLoss).toBeLessThan(Math.log(2))
  const result=await trainRewardPolicy(fitted,{epochs:3,batchSize:2})
  expect(result.rewardLearning!.rewardModel).toEqual(fitted.rewardLearning!.rewardModel)
  expect(result.rewardLearning!.lastReport!.validation.kl).toBeGreaterThan(0)
  const changed=structuredClone(result);changed.nodes.find(n=>n.id==='data')!.params.customCsv!.rows[1][0]='99'
  expect(()=>validateRewardReference(changed)).toThrow(/changed/)
})
it('rejects nonfinite and malformed reward state on import',()=>{
  const graph=captureRewardReference(fixture())
  const file=createProjectStateFile({...recoveryWorkspace().state,graph,visualizationGraph:graph})
  file.state.graph.rewardLearning!.referenceParameters.weight.data=[0]
  expect(parseProjectStateFile(JSON.stringify(file)).ok).toBe(false)
})
