import { describe, expect, it } from 'vitest'
import { backwardPass, createForwardEvaluator, forwardPass, runTrainingStep, runTrainingStepFast, topologicalSort, validateGraph } from './engine'
import { activationObjective, normalizeParameterValue } from './interpretability'
import { copyGraphSelection, pasteGraphClipboard } from './clipboard'
import type { GraphModel, GraphNode, NodeParams, NodeType } from './types'

function reconstruction(): GraphModel {
  const nodes:GraphNode[] = [], edges:GraphModel['edges'] = []
  const add = (id:string,type:NodeType,params:NodeParams,sources:string[] = []) => {
    nodes.push({id,type,label:id,params,position:{x:nodes.length*100,y:0}})
    sources.forEach((source,inputSlot)=>edges.push({id:`${id}-${inputSlot}`,source,target:id,inputSlot}))
  }
  add('r','weight',{trainable:false,value:{shape:[3,2],data:[1,2,-1,3,900,-800]}})
  add('e','weight',{value:{shape:[2,3],data:[.2,-.3,.1,.4,.1,-.2]}})
  add('he','matmul',{},['r','e'])
  add('h','activation',{activation:'relu'},['he'])
  add('d','weight',{constraint:'unit-row-norm',value:{shape:[3,2],data:[1,0,0,1,.6,.8]}})
  add('pred','matmul',{},['h','d'])
  add('loss','loss',{loss:'mse',targetMode:'internal',activationPenaltyNodeId:'h',activationPenaltyStrength:.07,lossMaskNodeId:'mask'},['pred','r'])
  // Intentionally after Loss in authoring order: the saved reference is a dependency.
  add('mask','input',{value:{shape:[3],data:[1,1,0]}})
  return {nodes,edges,learningRate:.01}
}

const node = (g:GraphModel,id:string) => g.nodes.find(n=>n.id===id)!

describe('internal reconstruction and activation objective',()=>{
  it('uses valid-token coordinate MSE and per-token feature-summed L1, independently of padded values',()=>{
    const graph = reconstruction(), result = forwardPass(graph)
    const h = node(result.graph,'h').value!, pred = node(result.graph,'pred').value!, r = node(result.graph,'r').value!
    const mse = pred.data.slice(0,4).reduce((sum,n,i)=>sum+(n-r.data[i])**2,0)/4
    const l1 = .07*h.data.slice(0,6).reduce((sum,n)=>sum+Math.abs(n),0)/2
    expect(result.loss).toBeCloseTo(mse+l1,12)
    expect(createForwardEvaluator(graph)().loss).toBeCloseTo(result.loss!,12)
    expect(topologicalSort(graph).indexOf('mask')).toBeLessThan(topologicalSort(graph).indexOf('loss'))
    const changed = structuredClone(graph)
    node(changed,'r').params.value = {shape:[3,2],data:[1,2,-1,3,-1e5,2e5]}
    expect(forwardPass(changed).loss).toBeCloseTo(result.loss!,12)
    const diagnostics=activationObjective(result.graph)!
    expect(diagnostics.activePerToken).toBe(1.5)
    expect(diagnostics.inactiveFeatures).toBe(1)
    expect(diagnostics.sparsityTolerance).toBe(0)
  })

  it('matches central differences for encoder and decoder, with detached target and exact zero subgradients',()=>{
    const graph=reconstruction(), result=backwardPass(forwardPass(graph).graph)
    for(const id of ['e','d']) {
      const value=node(graph,id).params.value as {shape:number[];data:number[]}
      value.data.forEach((_,i)=>{
        const plus=structuredClone(graph),minus=structuredClone(graph),step=1e-6
        ;(node(plus,id).params.value as typeof value).data[i]+=step
        ;(node(minus,id).params.value as typeof value).data[i]-=step
        expect(node(result.graph,id).grad!.data[i]).toBeCloseTo((forwardPass(plus).loss!-forwardPass(minus).loss!)/(2*step),6)
      })
    }
    expect(result.graph.edges.find(e=>e.target==='loss'&&e.inputSlot===1)!.grad!.data).toEqual(Array(6).fill(0))
    expect(activationObjective(result.graph)!.gradient.data[2]).toBe(0)
  })

  it('freezes values but preserves prediction-side gradients through frozen operations, in both SGD paths',()=>{
    const graph=reconstruction(), back=backwardPass(forwardPass(graph).graph)
    expect(node(back.graph,'r').grad!.data.some(n=>n!==0)).toBe(true)
    for(const trained of [runTrainingStep(graph).graph,runTrainingStepFast(graph)]) {
      expect(node(trained,'r').params.value).toEqual(node(graph,'r').params.value)
      expect(node(trained,'e').params.value).not.toEqual(node(graph,'e').params.value)
      const decoder=(node(trained,'d').params.value as {data:number[]}).data
      for(let row=0;row<3;row++) expect(Math.hypot(...decoder.slice(row*2,row*2+2))).toBeCloseTo(1,12)
    }
  })

  it('rejects unfrozen or stochastic internal targets and malformed mask rows',()=>{
    const graph=reconstruction();node(graph,'r').params.trainable=true
    expect(validateGraph(graph).some(i=>/Freeze every/.test(i.message))).toBe(true)
    node(graph,'r').params.trainable=false
    graph.nodes.push({id:'drop',type:'dropout',label:'drop',position:{x:0,y:0},params:{dropoutRate:.5}})
    graph.edges.push({id:'rd',source:'r',target:'drop',inputSlot:0})
    graph.edges.find(e=>e.target==='loss'&&e.inputSlot===1)!.source='drop'
    expect(validateGraph(graph).some(i=>/Inference only/.test(i.message))).toBe(true)
    node(graph,'drop').params.inferenceOnly=true
    expect(forwardPass(graph,false,{training:true,random:()=>0}).loss).toBeCloseTo(forwardPass(graph).loss!,12)
    node(graph,'mask').params.value={shape:[3],data:[1,.2,0]}
    expect(()=>forwardPass(graph)).toThrow(/only 0 and 1/)
    node(graph,'mask').params.value={shape:[3],data:[0,0,0]}
    expect(()=>forwardPass(graph)).toThrow(/at least one valid/)
  })

  it('normalizes zero and extreme rows safely and never projects frozen parameters',()=>{
    const decoder=node(reconstruction(),'d'),value={shape:[3,2],data:[0,0,1e300,1e300,1e-300,0]}
    const normalized=normalizeParameterValue(decoder,value)
    normalized.data.forEach((n,i)=>expect(n).toBeCloseTo([1,0,Math.SQRT1_2,Math.SQRT1_2,1,0][i],15))
    decoder.params.trainable=false
    expect(normalizeParameterValue(decoder,value)).toBe(value)
  })

  it('remaps objective signal references when copying a construction',()=>{
    const graph=reconstruction(), fragment=copyGraphSelection(graph,{nodeIds:graph.nodes.map(n=>n.id)})!
    const pasted=pasteGraphClipboard(graph,fragment,{x:50,y:50})
    const newLoss=pasted.graph.nodes.find(n=>n.id!=='loss'&&n.type==='loss')!
    const newHidden=pasted.graph.nodes.find(n=>n.id!=='h'&&n.type==='activation')!
    const newMask=pasted.graph.nodes.find(n=>n.id!=='mask'&&n.type==='input')!
    expect(newLoss.params.activationPenaltyNodeId).toBe(newHidden.id)
    expect(newLoss.params.lossMaskNodeId).toBe(newMask.id)
  })
})
