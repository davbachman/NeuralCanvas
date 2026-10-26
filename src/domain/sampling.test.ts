import {describe,expect,it} from 'vitest'
import * as math from '../learning/math'
import {textSamplingDistribution} from './textGeneration'
import {forwardPass,validateGraph} from './engine'
import {createProjectStateFile,parseProjectStateFile} from './session'
import type {GraphModel,GraphNode} from './types'

function samplingGraph():GraphModel {
  const nodes:GraphNode[]=( [
    {id:'logits',type:'input',params:{value:{shape:[1,5],data:[2,100,1,2,-3]}}},
    {id:'pad',type:'input',params:{value:{shape:[1],data:[1]}}},
    {id:'eligible',type:'index-mask',params:{indexMaskMode:'exclude'}},
    {id:'rank',type:'argsort',params:{descending:true}},
    {id:'slice',type:'tensor-transform',params:{transform:'slice',axis:1,start:0,end:2}},
    {id:'keep',type:'index-mask',params:{indexMaskMode:'keep'}},
    {id:'prob',type:'softmax',params:{}},
  ] satisfies Array<Pick<GraphNode,'id'|'type'|'params'>>).map(n=>({...n,label:n.id,position:{x:0,y:0}}))
  const wires:[string,string,number][]=[['logits','eligible',0],['pad','eligible',1],['eligible','rank',0],['rank','slice',0],['logits','keep',0],['slice','keep',1],['keep','prob',0]]
  return {nodes,edges:wires.map(([source,target,inputSlot],i)=>({id:String(i),source,target,inputSlot})),learningRate:.01}
}

describe('sorting and indexed probability selection',()=>{
  it('matches the existing generator, retaining token positions and stable ties',()=>{
    const graph=samplingGraph()
    expect(validateGraph(graph)).toEqual([])
    const result=forwardPass(graph).graph
    expect(result.nodes.find(n=>n.id==='rank')!.value!.data).toEqual([0,3,2,4,1])
    const probabilities=result.nodes.find(n=>n.id==='prob')!.value!.data
    expect(probabilities).toEqual(textSamplingDistribution([2,100,1,2,-3],['a','<pad>','b','c','d'],1,2))
    expect(probabilities).toEqual([.5,0,0,.5,0])
  })

  it('sorts arbitrary rows ascending or descending and places exclusions last',()=>{
    const x=math.tensor([2,4],[2,2,1,5,0,-2,-2,1],true)
    x.excluded=[false,false,false,true,false,false,false,false]
    expect(math.argsort(x).data).toEqual([0,1,2,3,3,0,1,2])
    const order=math.argsort(x,false)
    expect(order.data).toEqual([2,0,1,3,1,2,0,3])
    order.backward(Array(8).fill(1))
    expect(x.grad).toEqual(Array(8).fill(0))
  })

  it('preserves exact exclusions when temperature is applied after selection',()=>{
    const graph=samplingGraph()
    graph.nodes.find(n=>n.id==='logits')!.params.value={shape:[1,5],data:[-2e9,100,-3e9,-2.1e9,-4e9]}
    graph.nodes.push({id:'scale',type:'arithmetic',label:'Temperature',params:{expression:'x1 / 2'},position:{x:0,y:0}})
    graph.edges.find(e=>e.target==='prob')!.source='scale'
    graph.edges.push({id:'scale-in',source:'keep',target:'scale',inputSlot:0})
    const result=forwardPass(graph).graph
    expect(result.nodes.find(n=>n.id==='prob')!.value!.data).toEqual([1,0,0,0,0])
  })

  it('uses exact masks even when selected logits are below the display sentinel',()=>{
    const x=math.tensor([2,3],[-2e9,10,-3e9,5,8,6],true), ids=math.tensor([2,1],[0,2],true)
    const kept=math.indexMask(x,ids), p=math.softmax(kept)
    expect(p.data).toEqual([1,0,0,0,0,1])
    kept.backward([2,3,4,5,6,7])
    expect(x.grad).toEqual([2,0,0,0,0,7])
    expect(ids.grad).toEqual([0,0])
  })

  it('broadcasts a vector mask, composes exclusions, and rejects invalid IDs/shapes',()=>{
    const x=math.tensor([2,3],[1,2,3,4,5,6])
    const eligible=math.indexMask(x,math.tensor([1],[1]),'exclude')
    expect(math.softmax(math.indexMask(eligible,math.tensor([3],[0,1,2]))).data[1]).toBe(0)
    for(const ids of [math.tensor([1],[.5]),math.tensor([1],[3]),math.tensor([3,1],[0,0,0])]) expect(()=>math.indexMask(x,ids)).toThrow()
    expect(()=>math.softmax(math.indexMask(x,math.tensor([3],[0,1,2]),'exclude'))).toThrow('at least one allowed')
  })

  it('round-trips settings and exact masks through File Save/Import',()=>{
    const graph=forwardPass(samplingGraph()).graph
    const file=createProjectStateFile({graph,visualizationGraph:graph,initialParameterValues:{},selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:0,currentLoss:null,display:{showMath:true,showGradient:false,showCode:false,showVisualization:true}})
    const parsed=parseProjectStateFile(JSON.stringify(file))
    expect(parsed.ok).toBe(true)
    if(parsed.ok) {
      const result=forwardPass(parsed.file.state.graph).graph
      expect(result.nodes.find(n=>n.id==='prob')!.value!.data).toEqual([.5,0,0,.5,0])
      expect(result.nodes.find(n=>n.id==='keep')!.value!.excluded).toEqual([false,true,true,false,true])
    }
    file.state.graph.nodes.find(n=>n.id==='rank')!.params.descending='invalid' as unknown as boolean
    expect(parseProjectStateFile(JSON.stringify(file)).ok).toBe(false)
  })
})
