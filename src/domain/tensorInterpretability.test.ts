import {afterEach,beforeAll,describe,expect,it} from 'vitest'
import {backwardPass,forwardPass} from './engine'
import {evaluateDataset,withDatasetExample} from './datasetTraining'
import {DEFAULT_TRAINING} from './trainingSettings'
import {TensorGraph,TensorOptimizer,tf} from './tensorTraining'
import {prepareTextDocuments} from './textData'
import {buildTextModel} from '../test/textModels'
import {parameterPenalty} from './regularization'
import type {GraphModel,GraphNode,NodeType} from './types'

beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready()})
afterEach(()=>expect(tf.memory().numTensors).toBe(0))
function autoencoder():GraphModel {
  const graph:GraphModel={nodes:[],edges:[],learningRate:.01}
  const add=(id:string,type:NodeType,params:GraphNode['params'],inputs:Array<string|[string,number]>=[])=>{
    graph.nodes.push({id,type,label:id,position:{x:0,y:0},params})
    inputs.forEach((source,inputSlot)=>graph.edges.push({id:`${id}-${inputSlot}`,source:typeof source==='string'?source:source[0],sourceSlot:typeof source==='string'?0:source[1],target:id,inputSlot}))
  }
  add('data','dataset',{dataset:'attention-sequence',datasetMode:'sample'})
  add('base','weight',{value:{shape:[2,2],data:[1.123456789123456,0,0,1]},trainable:false})
  add('residual','matmul',{},[['data',0],'base'])
  add('encoder','weight',{value:{shape:[2,3],data:[.8,-.6,-.4,.2,.9,-.7]}})
  add('hidden-input','matmul',{},['residual','encoder'])
  add('hidden','activation',{activation:'relu'},['hidden-input'])
  add('decoder','weight',{value:{shape:[3,2],data:[1,0,0,1,.6,.8]},constraint:'unit-row-norm'})
  add('reconstruction','matmul',{},['hidden','decoder'])
  add('valid','input',{value:{shape:[3,1],data:[1,1,0]}})
  add('loss','loss',{loss:'mse',targetMode:'internal',lossMaskNodeId:'valid',activationPenaltyNodeId:'hidden',activationPenaltyStrength:.1},['reconstruction','residual'])
  return graph
}

describe('tensor sparse reconstruction',()=>{
 it('matches Trace masked objectives and gradients and independently reports sparsity',async()=>{
  const graph=autoencoder(),model=new TensorGraph(graph)
  try {
    const traced=backwardPass(forwardPass(withDatasetExample(graph,'data',0)).graph)
    const result=model.gradients([model.examples[0]])
    try {
      expect((await result.loss.data())[0]).toBeCloseTo(traced.loss!,6)
      for(const id of ['encoder','decoder']) {
        const actual=await result.grads[model.variables.get(id)!.name].data()
        const expected=traced.graph.nodes.find(n=>n.id===id)!.grad!.data
        actual.forEach((n,i)=>expect(n).toBeCloseTo(expected[i],5))
      }
      expect(result.grads[model.variables.get('base')!.name]).toBeUndefined()
    } finally {tf.dispose([result.loss,...Object.values(result.grads)])}
    const report=await model.evaluate(model.examples,4)
    expect(report.objective! - report.loss).toBeCloseTo(report.activationPenalty!,6)
    expect(report.activationPenalty).toBeCloseTo(.1*(1.123456789123456*.8+.2+.9)/2,6)
    expect(report.activePerToken).toBe(1.5)
    expect(report.inactiveFeatures).toBe(1)
    expect(report.sparsityTolerance).toBe(0)
    const inference=await model.inference(model.examples.slice(0,1),1)
    expect(inference.accuracy).toBeUndefined()
    expect(report.accuracy).toBeUndefined()
    expect(inference.objective).toBeCloseTo(report.objective!,6)
    expect(inference.activePerToken).toBe(1.5)
    expect(inference.rows).toHaveLength(4)
    expect(inference.rows[0].actual).toBe('1.1235')
  } finally {model.dispose()}
 })
 it('agrees with finite differences and uses zero subgradients at exactly zero',async()=>{
  const graph=autoencoder(),model=new TensorGraph(graph)
  try {
    const result=model.gradients([model.examples[0]])
    try {
      const actual=await result.grads[model.variables.get('encoder')!.name].data()
      for(const index of [0,1,2,3,4,5]) {
        const objectives:number[]=[]
        for(const direction of [-1,1]) {
          const changed=structuredClone(graph),node=changed.nodes.find(n=>n.id==='encoder')!
          ;(node.params.value as {data:number[]}).data[index]+=direction*.001
          const probe=new TensorGraph(changed)
          try {objectives.push((await probe.inspect([probe.examples[0]])).loss)} finally {probe.dispose()}
        }
        expect(actual[index]).toBeCloseTo((objectives[1]-objectives[0])/.002,4)
      }
      expect(actual[2]).toBe(0)
      expect(actual[5]).toBe(0)
    } finally {tf.dispose([result.loss,...Object.values(result.grads)])}
  } finally {model.dispose()}
 })
 it.each(['sgd','adam','adamw'] as const)('freezes values exactly under %s and projects decoder rows',async optimizer=>{
  const graph=autoencoder(),model=new TensorGraph(graph)
  const opt=new TensorOptimizer({...DEFAULT_TRAINING,optimizer,weightDecay:.4,clipNorm:1},.03)
  try {
    for(let i=0;i<4;i++) {
      const result=model.gradients(model.examples.slice(0,2))
      try {opt.update(model.variables,result.grads)} finally {tf.dispose([result.loss,...Object.values(result.grads)])}
    }
    const snapshot=await model.snapshot()
    expect(snapshot.nodes.find(n=>n.id==='base')!.params.value).toEqual(graph.nodes.find(n=>n.id==='base')!.params.value)
    const values=await model.variables.get('decoder')!.data()
    for(let i=0;i<values.length;i+=2) expect(Math.hypot(values[i],values[i+1])).toBeCloseTo(1,6)
    expect(snapshot.nodes.find(n=>n.id==='encoder')!.params.value).not.toEqual(graph.nodes.find(n=>n.id==='encoder')!.params.value)
  } finally {opt.dispose();model.dispose()}
 })
 it('projects zero decoder directions safely and preserves gradient flow through frozen multiplications',async()=>{
  const graph=autoencoder()
  graph.nodes.find(n=>n.id==='decoder')!.params.value={shape:[3,2],data:[0,0,1e30,1e30,1e-30,1e-30]}
  const model=new TensorGraph(graph),opt=new TensorOptimizer({...DEFAULT_TRAINING,optimizer:'sgd',clipNorm:0},.01)
  try {
    expect(Array.from(await model.variables.get('decoder')!.data()).slice(0,2)).toEqual([0,0])
    opt.update(model.variables,{})
    const values=Array.from(await model.variables.get('decoder')!.data())
    expect(values.slice(0,2)).toEqual([1,0])
    for(let i=0;i<values.length;i+=2) expect(Math.hypot(values[i],values[i+1])).toBeCloseTo(1,6)
  } finally {opt.dispose();model.dispose()}
  graph.nodes.find(n=>n.id==='base')!.params.trainable=true
  graph.nodes.find(n=>n.id==='encoder')!.params.trainable=false
  graph.nodes.find(n=>n.id==='decoder')!.params.trainable=false
  graph.nodes.find(n=>n.id==='loss')!.params={loss:'mse'}
  graph.edges.find(e=>e.target==='loss'&&e.inputSlot===1)!.source='data'
  graph.edges.find(e=>e.target==='loss'&&e.inputSlot===1)!.sourceSlot=3
  const frozen=new TensorGraph(graph)
  try {
    const result=frozen.gradients([frozen.examples[0]])
    try {expect(Array.from(await result.grads[frozen.variables.get('base')!.name].data()).some(n=>Math.abs(n)>0)).toBe(true)}
    finally {tf.dispose([result.loss,...Object.values(result.grads)])}
  } finally {frozen.dispose()}
 })
 it('rejects trainable target ancestors and invalid masks before optimization',()=>{
  const graph=autoencoder()
  graph.nodes.find(n=>n.id==='base')!.params.trainable=true
  expect(()=>new TensorGraph(graph)).toThrow(/Freeze/)
  graph.nodes.find(n=>n.id==='base')!.params.trainable=false
  graph.nodes.find(n=>n.id==='valid')!.params.value={shape:[3,1],data:[1,.5,0]}
  const model=new TensorGraph(graph)
  try {expect(()=>model.gradients([model.examples[0]])).toThrow(/zero or one/)} finally {model.dispose()}
 })
 it('requires stable target dropout and applies inference-only behavior while fitting',async()=>{
  const graph=autoencoder()
  graph.nodes.push({id:'stable',type:'dropout',label:'stable',position:{x:0,y:0},params:{dropoutRate:.8}})
  graph.edges.push({id:'stable-in',source:'residual',target:'stable',inputSlot:0})
  graph.edges.find(e=>e.target==='hidden-input'&&e.inputSlot===0)!.source='stable'
  graph.edges.find(e=>e.target==='loss'&&e.inputSlot===1)!.source='stable'
  expect(()=>new TensorGraph(graph)).toThrow(/Inference only/)
  graph.nodes.find(n=>n.id==='stable')!.params.inferenceOnly=true
  const model=new TensorGraph(graph)
  try {
    const a=model.gradients([model.examples[0]]),b=model.gradients([model.examples[0]])
    try {
      expect(Array.from(await a.loss.data())).toEqual(Array.from(await b.loss.data()))
      expect(Array.from(await a.grads[model.variables.get('encoder')!.name].data())).toEqual(Array.from(await b.grads[model.variables.get('encoder')!.name].data()))
    } finally {tf.dispose([a.loss,b.loss,...Object.values(a.grads),...Object.values(b.grads)])}
  } finally {model.dispose()}
 })
 it('has a finite no-op update for entirely frozen graphs',async()=>{
  const graph=autoencoder()
  graph.nodes.filter(n=>n.type==='weight').forEach(n=>{n.params.trainable=false})
  const model=new TensorGraph(graph),opt=new TensorOptimizer({...DEFAULT_TRAINING,optimizer:'adamw',weightDecay:1},.03)
  try {
    const result=model.gradients(model.examples.slice(0,2))
    try {expect(Object.keys(result.grads)).toHaveLength(0);opt.update(model.variables,result.grads)} finally {tf.dispose([result.loss,...Object.values(result.grads)])}
    expect((await model.snapshot()).nodes.filter(n=>n.type==='weight').map(n=>n.params.value)).toEqual(graph.nodes.filter(n=>n.type==='weight').map(n=>n.params.value))
  } finally {opt.dispose();model.dispose()}
 })
 it('reports a token-weighted activation penalty independently of classification loss and minibatch size',async()=>{
  const data=prepareTextDocuments([{text:'one two one two',label:'positive',split:'train'},{text:'two',label:'negative',split:'train'},{text:'one two',label:'positive',split:'test'}],'tiny.json',{task:'sentiment',representation:'tokens',maxLength:4})
  const graph=buildTextModel(data,'mean',2)
  const loss=graph.nodes.find(n=>n.id==='loss')!
  Object.assign(loss.params,{activationPenaltyNodeId:'embedded-tokens',activationPenaltyStrength:.1,regularization:'l2',regularizationStrength:.002,regularizationParameterIds:['word-embeddings']})
  const model=new TensorGraph(graph)
  try {
    const rows=model.examples.filter(row=>row.split==='train'),single=await model.evaluate(rows,1),batch=await model.evaluate(rows,2)
    const trace=evaluateDataset(graph,'text-data','train',{includePredictions:false})
    expect(single.loss).toBeCloseTo(batch.loss,6)
    expect(single.activationPenalty).toBeCloseTo(batch.activationPenalty!,6)
    expect(single.objective).toBeCloseTo(batch.objective!,6)
    expect(batch.objective).toBeCloseTo(batch.loss+batch.activationPenalty!+parameterPenalty(graph),6)
    expect(batch.objective).toBeCloseTo(trace.objective!,6)
    expect(batch.activePerToken).toBe(trace.activePerToken)
  } finally {model.dispose()}
  loss.params.activationPenaltyNodeId='mean-review'
  const invalid=new TensorGraph(graph)
  try {expect(()=>invalid.gradients(invalid.examples)).toThrow(/preserve token rows/)} finally {invalid.dispose()}
 })
 it('rejects an empty individual example mask even when another batch example has valid tokens',()=>{
  const graph=autoencoder()
  graph.nodes.push({id:'data-mask',type:'slice',label:'mask',position:{x:0,y:0},params:{axis:1,start:0,end:1}})
  graph.edges.push({id:'mask-in',source:'data',sourceSlot:0,target:'data-mask',inputSlot:0})
  graph.nodes.find(n=>n.id==='loss')!.params.lossMaskNodeId='data-mask'
  const model=new TensorGraph(graph)
  try {
    const empty=structuredClone(model.examples[0]);empty.features[0].data=empty.features[0].data.map(()=>0)
    expect(()=>model.gradients([model.examples[0],empty])).toThrow(/each example/)
  } finally {model.dispose()}
 })
 it('excludes fixed token padding from reconstruction and penalty, including across evaluation batches',async()=>{
  const data=prepareTextDocuments([{text:'one two',label:'positive',split:'train'},{text:'two',label:'negative',split:'train'},{text:'one',label:'positive',split:'test'}],'tiny.json',{task:'sentiment',representation:'tokens',maxLength:4,fixedLength:true})
  const graph=autoencoder()
  const source=graph.nodes.find(n=>n.id==='data')!
  source.params={dataset:'custom-text',textData:data,datasetMode:'sample'}
  const base=graph.nodes.find(n=>n.id==='base')!
  base.params.value={shape:[data.vocabulary.length,2],data:data.vocabulary.flatMap((_,i)=>i===1?[1e4,1e4]:[1,0])}
  graph.nodes.find(n=>n.id==='residual')!.type='embedding'
  graph.edges.find(e=>e.target==='residual'&&e.inputSlot===0)!.source='base'
  graph.edges.find(e=>e.target==='residual'&&e.inputSlot===1)!.source='data'
  graph.nodes.find(n=>n.id==='loss')!.params.lossMaskNodeId=undefined
  const model=new TensorGraph(graph)
  try {
    const all=await model.evaluate(model.examples,16),single=await model.evaluate(model.examples,1)
    expect(all.loss).toBeCloseTo(.02,5)
    expect(all.activationPenalty).toBeCloseTo(.08,5)
    expect(all.loss).toBeCloseTo(single.loss,6)
    expect(all.objective).toBeCloseTo(single.objective!,6)
    const trace=evaluateDataset(graph,'data','train',{includePredictions:false})
    const tensor=await model.evaluate(model.examples.filter(row=>row.split==='train'),2)
    expect(tensor.loss).toBeCloseTo(trace.loss,6)
    expect(tensor.objective).toBeCloseTo(trace.objective!,6)
    expect(tensor.activePerToken).toBe(trace.activePerToken)
    // A shape-preserving Reshape erases Tensor axis provenance; padding must
    // still be excluded because this objective reconstructs token rows.
    graph.nodes.push({id:'residual-view',type:'reshape',label:'view',position:{x:0,y:0},params:{shape:[4,2]}})
    graph.edges.push({id:'view-in',source:'residual',target:'residual-view',inputSlot:0})
    graph.edges.find(e=>e.target==='hidden-input'&&e.inputSlot===0)!.source='residual-view'
    graph.edges.find(e=>e.target==='loss'&&e.inputSlot===1)!.source='residual-view'
    const reshaped=new TensorGraph(graph)
    try {
      const result=await reshaped.evaluate(reshaped.examples,16)
      expect(result.loss).toBeCloseTo(all.loss,6)
      expect(result.activationPenalty).toBeCloseTo(all.activationPenalty!,6)
    } finally {reshaped.dispose()}
  } finally {model.dispose()}
 })
})
