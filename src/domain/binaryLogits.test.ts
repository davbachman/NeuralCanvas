import { afterEach, beforeAll, expect, it } from 'vitest'
import { builder } from '../test/curriculumModels'
import { parseCustomCsv } from './customCsv'
import { backwardPass, forwardPass, lossOptionsForNode } from './engine'
import { evaluateDataset, withDatasetExample } from './datasetTraining'
import { TensorGraph, tf } from './tensorTraining'
import { generatePyTorchExport } from './pytorchExport'
import { createProjectStateFile, parseProjectStateFile } from './session'
import { recoveryWorkspace } from '../test/recoveryWorkspace'

beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready()})
afterEach(()=>expect(tf.memory().numTensors).toBe(0))
function fixture() {
  const {graph,add}=builder()
  const csv=parseCustomCsv('x,target,split\n-1000,1,train\n1000,0,train\n0,1,train\n0.1,1,test\n-0.1,0,test\n','binary.csv');csv.task='binary-classification'
  add('data','dataset',{dataset:'custom-csv',customCsv:csv,datasetMode:'sample'})
  add('weight','weight',{value:1})
  add('logits','multiply',{},[['data',0],'weight'])
  add('loss','loss',{loss:'binary-cross-entropy-with-logits'},['logits',['data',1]])
  return graph
}
it('computes finite losses and correct gradients for extreme and zero logits',async()=>{
  const graph=fixture(),model=new TensorGraph(graph)
  try {
    for(const [index,expected] of [[0,1000],[1,1000],[2,Math.log(2)]] as const) {
      const traced=backwardPass(forwardPass(withDatasetExample(graph,'data',index)).graph)
      expect(traced.loss).toBeCloseTo(expected,6)
      const result=model.gradients([model.examples[index]])
      try {
        expect((await result.loss.data())[0]).toBeCloseTo(expected,5)
        expect((await result.grads[model.variables.get('weight')!.name].data())[0]).toBeCloseTo(traced.graph.nodes.find(n=>n.id==='weight')!.grad!.data[0],5)
      } finally {tf.dispose([result.loss,...Object.values(result.grads)])}
    }
    const batch=model.gradients(model.examples.slice(0,3))
    try {
      expect((await batch.loss.data())[0]).toBeCloseTo((2000+Math.log(2))/3,3)
      expect((await batch.grads[model.variables.get('weight')!.name].data())[0]).toBeCloseTo(2000/3,3)
    } finally {tf.dispose([batch.loss,...Object.values(batch.grads)])}
    tf.tidy(()=>model.variables.get('weight')!.assign(tf.scalar(0)))
    const atZero=model.gradients([model.examples[3]])
    try {expect((await atZero.grads[model.variables.get('weight')!.name].data())[0]).toBeCloseTo(-.05,6)}
    finally {tf.dispose([atZero.loss,...Object.values(atZero.grads)])}
    const zero=backwardPass(forwardPass(withDatasetExample(graph,'data',2)).graph)
    expect(zero.graph.nodes.find(n=>n.id==='logits')!.grad!.data[0]).toBe(-.5)
  } finally {model.dispose()}
})
it('uses the zero-logit threshold in trace and tensor inference and evaluation',async()=>{
  const graph=fixture(),model=new TensorGraph(graph)
  try {
    const traced=evaluateDataset(graph,'data','test')
    expect(traced.rows.map(row=>row.predicted)).toEqual(['1','0'])
    const result=await model.inference(model.examples.filter(row=>row.split==='test'),2)
    expect(result.rows).toEqual(traced.rows)
    expect(result.accuracy).toBe(1)
    expect((await model.evaluate(model.examples.filter(row=>row.split==='test'),2)).accuracy).toBe(1)
  } finally {model.dispose()}
})
it('offers logits for new losses while keeping explicit legacy probability models readable',()=>{
  const graph=fixture(),loss=graph.nodes.find(n=>n.id==='loss')!
  expect(lossOptionsForNode(loss,graph).map(x=>x.kind)).toContain('binary-cross-entropy-with-logits')
  expect(lossOptionsForNode(loss,graph).map(x=>x.kind)).not.toContain('binary-cross-entropy')
  loss.params.loss='binary-cross-entropy'
  expect(lossOptionsForNode(loss,graph).map(x=>x.kind)).toContain('binary-cross-entropy')
  const workspace=recoveryWorkspace()
  for(const kind of ['binary-cross-entropy','binary-cross-entropy-with-logits'] as const) {
    loss.params.loss=kind
    const parsed=parseProjectStateFile(JSON.stringify(createProjectStateFile({...workspace.state,graph})))
    expect(parsed.ok).toBe(true)
    if(parsed.ok) expect(parsed.file.state.graph.nodes.find(n=>n.id==='loss')!.params.loss).toBe(kind)
  }
})
it('exports the stable PyTorch logits loss and matching prediction threshold',()=>{
  const script=generatePyTorchExport({...fixture(),training:undefined}).script
  expect(script).toContain('F.binary_cross_entropy_with_logits(*torch.broadcast_tensors(')
  expect(script).toContain('scores[:, 0] >= 0.0')
})
