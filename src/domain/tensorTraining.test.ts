import {afterEach,beforeAll,describe,expect,it} from 'vitest'
import {tf,TensorGraph,TensorOptimizer,trainTensorGraph} from './tensorTraining'
import {prepareTextDocuments} from './textData'
import {buildTextModel,type TextModelKind} from '../test/textModels'
import {backwardPass,forwardPass,parameterValues} from './engine'
import {withDatasetExample} from './datasetTraining'
import {DEFAULT_TRAINING} from './trainingSettings'
import {createNode} from './examples'

beforeAll(async()=>{await tf.setBackend('cpu');await tf.ready()})
afterEach(()=>expect(tf.memory().numTensors).toBe(0))
const data=(language=false)=>prepareTextDocuments([{text:'good funny good film',label:'positive',split:'train'},{text:'bad film',label:'negative',split:'train'},{text:'good film again',label:'positive',split:'test'},{text:'bad dull movie',label:'negative',split:'test'}],'reviews.csv',{task:language?'language':'sentiment',maxLength:4})
const kinds:TextModelKind[]=['counts-linear','counts-mlp','mean','position-mean','attention','position-attention','transformer','alice-baseline','alice-transformer']
describe('batched tensor compiler',()=>{
 it('evaluates indexed sampling branches without changing the training objective or gradients',async()=>{
  const graph=buildTextModel(data(true),'alice-baseline',4)
  const loss=graph.nodes.find(n=>n.type==='loss'||n.type==='cross-entropy')!
  const logits=graph.edges.find(e=>e.target===loss.id&&e.inputSlot===0)!.source
  const baseline=new TensorGraph(graph)
  const ids={...createNode('input',100),id:'excluded-id',params:{value:{shape:[1],data:[1]}}}
  const mask={...createNode('index-mask',101),id:'selected',params:{indexMaskMode:'exclude' as const}}
  const rank={...createNode('argsort',102),id:'ranked'}
  const prob={...createNode('softmax',103),id:'selected-prob'}
  graph.nodes.push(ids,mask,rank,prob)
  graph.edges.push({id:'l-m',source:logits,target:mask.id,inputSlot:0},{id:'i-m',source:ids.id,target:mask.id,inputSlot:1},{id:'m-r',source:mask.id,target:rank.id,inputSlot:0},{id:'m-p',source:mask.id,target:prob.id,inputSlot:0})
  const modified=new TensorGraph(graph)
  try {
   const rows=modified.examples.slice(0,2)
   const signal=await modified.inspectSignal(rows,prob.id)
   const masked=await modified.inspectSignal(rows,mask.id)
   expect(masked!.value.data.every(Number.isFinite)).toBe(true)
   expect(masked!.value.excluded?.some(Boolean)).toBe(true)
   const width=signal!.value.shape.at(-1)!
   for(let i=0;i<signal!.value.data.length;i+=width) {
    expect(signal!.value.data[i+1]).toBe(0)
    expect(signal!.value.data.slice(i,i+width).reduce((a,b)=>a+b,0)).toBeCloseTo(1,6)
   }
   const ranks=await modified.inspectSignal(rows,rank.id)
   for(let i=0;i<ranks!.value.data.length;i+=width) expect(ranks!.value.data[i+width-1]).toBe(1)
   const a=baseline.gradients(rows),b=modified.gradients(rows)
   try {
    expect(Array.from(await a.loss.data())).toEqual(Array.from(await b.loss.data()))
    for(const [id,v] of baseline.variables) expect(Array.from(await a.grads[v.name].data())).toEqual(Array.from(await b.grads[modified.variables.get(id)!.name].data()))
   } finally {tf.dispose([a.loss,b.loss,...Object.values(a.grads),...Object.values(b.grads)])}
  } finally {baseline.dispose();modified.dispose()}
 })
 it.each(kinds)('matches per-example loss and mean parameter gradients for %s, including unequal lengths',async kind=>{
  const graph=buildTextModel(data(kind.startsWith('alice')),kind,4), model=new TensorGraph(graph)
  try {
   const rows=model.examples.slice(0,2), expected=rows.map((_,i)=>forwardPass(withDatasetExample(graph,'text-data',i),false))
   const gradients=expected.map(result=>backwardPass(result.graph).graph)
   const actual=model.gradients(rows)
   try {
    expect((await actual.loss.data())[0]).toBeCloseTo(expected.reduce((sum,r)=>sum+r.loss!,0)/2,5)
    for(const [id,v] of model.variables) {
      const values=await actual.grads[v.name].data()
      const a=gradients[0].nodes.find(n=>n.id===id)!.grad!.data,b=gradients[1].nodes.find(n=>n.id===id)!.grad!.data
      values.forEach((value,i)=>expect(value,id+' '+i).toBeCloseTo((a[i]+b[i])/2,4))
    }
   }finally{tf.dispose([actual.loss,...Object.values(actual.grads)])}
   const single=await model.inspect([rows[1]]),batch=await model.inspect(rows)
   if(!kind.startsWith('alice')) expect(batch.prediction[1]).toBeCloseTo(single.prediction[0],6)
  }finally{model.dispose()}
 })
 it('does not let another review or padding change a prediction',async()=>{
  const graph=buildTextModel(data(),'transformer',4),model=new TensorGraph(graph)
  try {
   const a=await model.inspect([model.examples[1]]),b=await model.inspect([model.examples[0],model.examples[1]])
   expect(a.prediction[0]).toBeCloseTo(b.prediction[1],6)
  }finally{model.dispose()}
 })
 it('keeps AdamW moments across updates and applies decoupled decay',async()=>{
  const v=tf.tidy(()=>tf.variable(tf.scalar(2))),g=tf.scalar(.5)
  const opt=new TensorOptimizer({...DEFAULT_TRAINING,optimizer:'adamw',weightDecay:.1,clipNorm:0},.01)
  try {
   opt.update(new Map([['p',v]]),{[v.name]:g})
   expect((await v.data())[0]).toBeCloseTo(2*.999-.01,6)
   opt.update(new Map([['p',v]]),{[v.name]:g})
   expect((await v.data())[0]).toBeCloseTo((2*.999-.01)*.999-.01,6)
  }finally{opt.dispose();v.dispose();g.dispose()}
 })
 it('trains, restores the best validation checkpoint, and never mutates the input',async()=>{
  const graph=buildTextModel(data(),'counts-linear',4);graph.learningRate=.02
  const before=JSON.stringify(parameterValues(graph))
  const result=await trainTensorGraph(graph,{epochs:12,batchSize:2,settings:{...DEFAULT_TRAINING,engine:'tensor',backend:'cpu',optimizer:'adam',minDelta:100,patience:2}})
  expect(result.completed).toBe(2);expect(result.bestEpoch).toBe(0)
  const original=parameterValues(graph),best=parameterValues(result.graph)
  for(const id of Object.keys(original)) original[id].data.forEach((n,i)=>expect(best[id].data[i]).toBeCloseTo(n,6))
  expect(JSON.stringify(parameterValues(graph))).toBe(before)
 })
 it.each([0,2])('honors reporting intervals without changing early stopping, patience %i',async patience=>{
  const graph=buildTextModel(data(),'counts-linear',4)
  const published:number[]=[]
  const result=await trainTensorGraph(graph,{epochs:5,reportEvery:3,epochOffset:10,batchSize:2,settings:{...DEFAULT_TRAINING,engine:'tensor',backend:'cpu',patience,minDelta:100},onReport:report=>published.push(report.epoch)})
  expect(result.completed).toBe(patience?2:5)
  expect(published).toEqual(patience?[10,12]:[10,13,15])
  expect(result.reports.map(report=>report.epoch)).toEqual(published)
  if(patience) expect(result.bestEpoch).toBe(0)
 })
 it.each([0,-1,1.5,NaN,100001])('rejects invalid reporting interval %s',async reportEvery=>{
  await expect(trainTensorGraph(buildTextModel(data(),'counts-linear',4),{epochs:1,batchSize:2,reportEvery})).rejects.toThrow('reporting interval')
 })
 it('publishes matching parameters and epochs for durable recovery checkpoints',async()=>{
  const graph=buildTextModel(data(),'counts-linear',4)
  const checkpoints:Array<{graph:typeof graph;epoch:number;loss:number}>=[]
  const result=await trainTensorGraph(graph,{epochs:3,reportEvery:2,batchSize:2,settings:{...DEFAULT_TRAINING,engine:'tensor',backend:'cpu',patience:0},onCheckpoint:async(graph,report)=>{
   checkpoints.push({graph,epoch:report.epoch,loss:report.train.loss})
  }})
  expect(checkpoints.map(point=>point.epoch)).toEqual([0,2,3])
  expect(parameterValues(checkpoints.at(-1)!.graph)).toEqual(parameterValues(result.graph))
  for(const checkpoint of checkpoints) {
   const model=new TensorGraph(checkpoint.graph)
   try {expect((await model.evaluate(model.examples.filter(row=>row.split==='train'),2)).loss).toBeCloseTo(checkpoint.loss,5)}
   finally{model.dispose()}
  }
 })
 it('rejects unsupported nodes before allocating variables',()=>{
  const graph=buildTextModel(data(),'mean',4)
  graph.nodes.find(n=>n.id==='mean-review')!.params.axis=1
  expect(()=>new TensorGraph(graph)).toThrow()
 })
 it('stops during evaluation and returns a finite checkpoint without leaking tensors',async()=>{
  const graph=buildTextModel(data(),'mean',4),controller=new AbortController()
  const result=await trainTensorGraph(graph,{epochs:3,batchSize:2,signal:controller.signal,settings:{...DEFAULT_TRAINING,engine:'tensor',backend:'cpu'},onReport:()=>controller.abort()})
  expect(result.stopped).toBe(true)
  expect(result.bestEpoch).toBe(0)
  expect(Object.values(parameterValues(result.graph)).every(v=>v.data.every(Number.isFinite))).toBe(true)
 })
 it('handles an already cancelled run and invalid learning rates without leaks',async()=>{
  const graph=buildTextModel(data(),'mean',4),controller=new AbortController();controller.abort()
  const result=await trainTensorGraph(graph,{epochs:2,batchSize:2,signal:controller.signal,settings:{...DEFAULT_TRAINING,backend:'cpu'}})
  expect(result.stopped).toBe(true)
  await expect(trainTensorGraph({...graph,learningRate:NaN},{epochs:2,batchSize:2,settings:{...DEFAULT_TRAINING,backend:'cpu'}})).rejects.toThrow('learning rate')
 })
 it('does not cross-broadcast labels when pooling drops the token dimension',async()=>{
  const graph=buildTextModel(data(),'mean',4)
  graph.nodes.push({id:'squeeze-probability',type:'tensor-transform',label:'Mean',position:{x:0,y:0},params:{transform:'mean',axis:0,keepDims:false}})
  const lossEdge=graph.edges.find(e=>e.target==='loss'&&e.inputSlot===0)!
  graph.edges.push({id:'squeeze-input',source:lossEdge.source,target:'squeeze-probability',inputSlot:0})
  lossEdge.source='squeeze-probability'
  const model=new TensorGraph(graph)
  try {
   const rows=model.examples.slice(0,2),single=await Promise.all(rows.map(r=>model.inspect([r]))),batch=await model.inspect(rows)
   expect(batch.loss).toBeCloseTo((single[0].loss+single[1].loss)/2,6)
  }finally{model.dispose()}
 })

})

describe('tensor dropout',()=>{
 it('is disabled in evaluation, active only in training, and advances masks across updates',async()=>{
  const {addTransformerDropout}=await import('../test/textModels')
  const baseline=buildTextModel(data(),'transformer',4),graph=addTransformerDropout(baseline,.5),model=new TensorGraph(graph)
  try {
   const rows=model.examples.slice(0,2),before=await model.inspect(rows)
   const expected=rows.map((_,i)=>forwardPass(withDatasetExample(baseline,'text-data',i),false).loss!)
   expect(before.loss).toBeCloseTo((expected[0]+expected[1])/2,5)
   const first=model.gradients(rows),second=model.gradients(rows)
   try {
    const a=await first.loss.data(),b=await second.loss.data()
    expect(Number.isFinite(a[0])&&Number.isFinite(b[0])).toBe(true)
    expect(a[0]).not.toBe(b[0])
   }finally{tf.dispose([first.loss,second.loss,...Object.values(first.grads),...Object.values(second.grads)])}
   expect(await model.inspect(rows)).toEqual(before)
   expect((await model.evaluate(rows,2)).loss).toBeCloseTo(before.loss,6)
  }finally{model.dispose()}
 })
 it('rate zero leaves training losses and gradients unchanged',async()=>{
  const {addTransformerDropout}=await import('../test/textModels')
  const baseline=buildTextModel(data(),'transformer',4),a=new TensorGraph(baseline),b=new TensorGraph(addTransformerDropout(baseline,0))
  const ga=a.gradients(a.examples.slice(0,2)),gb=b.gradients(b.examples.slice(0,2))
  try {
   expect(Array.from(await ga.loss.data())).toEqual(Array.from(await gb.loss.data()))
   for(const [id,v] of a.variables)expect(Array.from(await ga.grads[v.name].data())).toEqual(Array.from(await gb.grads[b.variables.get(id)!.name].data()))
  }finally{tf.dispose([ga.loss,gb.loss,...Object.values(ga.grads),...Object.values(gb.grads)]);a.dispose();b.dispose()}
 })
})
