import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { scratchModel } from '../test/scratchModels'
import { createNode } from '../domain/examples'
import { blockPalette } from '../domain/blockPalette'
import { formulaForNode, forwardPass } from '../domain/engine'
import { datasetOutputValueForSlot } from '../domain/datasets'
import { tensorValue } from '../domain/tensor'
import { initializeTensor } from '../domain/authoring'
import { predictionNode } from '../domain/datasetTraining'
import type { GraphModel, GraphNode } from '../domain/types'
import { ModelInspector } from './ModelInspector'
import { DecoderControls } from './DecoderControls'
import { ConvolutionInspector } from './ConvolutionInspector'
import { parseCustomCsv } from '../domain/customCsv'
import { DatasetWorkbench } from './DatasetWorkbench'

const callbacks = {onParams:vi.fn(),onValue:vi.fn(),onDataset:vi.fn(),onOpen:vi.fn(),onInspectNeuron:vi.fn(),onGroup:vi.fn(),selectionCount:1}
function InspectorHarness({initial}: {initial:GraphNode}) {
  const [node,setNode] = useState(initial)
  return <ModelInspector {...callbacks} onRename={vi.fn()} graph={{nodes:[node],edges:[],learningRate:.01}} node={node} onParams={(_,params)=>setNode({...node,params:{...node.params,...params}})} onValue={(_,value)=>setNode({...node,params:{...node.params,value}})}/>
}

describe('scratch model authoring controls',()=>{
  it('defaults inputs to manual entry and applies matrix dimensions from brackets', () => {
    const initial = createNode('input', 1), onValue = vi.fn()
    render(<ModelInspector {...callbacks} graph={{ nodes: [initial], edges: [], learningRate: .01 }} node={initial} onValue={onValue}/>)
    expect(screen.getByLabelText('Input value mode')).toHaveValue('manual')
    expect(screen.getByLabelText('Tensor values')).toBeVisible()
    expect(screen.queryByLabelText('Tensor shape')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Tensor initializer')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Initializer seed')).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Tensor values'), { target: { value: '[[1, 2], [3, 4]]' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply values' }))
    expect(onValue).toHaveBeenLastCalledWith(initial.id, tensorValue([2, 2], [1, 2, 3, 4]))
    fireEvent.change(screen.getByLabelText('Tensor values'), { target: { value: '[[1, 2], [3]]' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply values' }))
    expect(screen.getByRole('alert')).toHaveTextContent('rectangular')
    expect(onValue).toHaveBeenCalledTimes(1)
  })

  it('generates seeded inputs only on request and preserves values when switching modes', () => {
    const initial = createNode('input', 1)
    initial.params.value = tensorValue([2, 2], [1, 2, 3, 4])
    render(<InspectorHarness initial={initial}/>)
    const manual = (screen.getByLabelText('Tensor values') as HTMLTextAreaElement).value
    fireEvent.change(screen.getByLabelText('Input value mode'), { target: { value: 'random' } })
    expect(screen.queryByLabelText('Tensor values')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Apply values' })).not.toBeInTheDocument()
    expect(screen.getByLabelText('Tensor shape')).toHaveValue('2, 2')
    fireEvent.change(screen.getByLabelText('Tensor initializer'), { target: { value: 'uniform' } })
    fireEvent.change(screen.getByLabelText('Initializer seed'), { target: { value: '137' } })
    fireEvent.change(screen.getByLabelText('Input value mode'), { target: { value: 'manual' } })
    expect(screen.getByLabelText('Tensor values')).toHaveValue(manual)
    fireEvent.change(screen.getByLabelText('Input value mode'), { target: { value: 'random' } })
    expect(screen.getByLabelText('Initializer seed')).toHaveValue(137)
    fireEvent.click(screen.getByRole('button', { name: 'Generate inputs' }))
    fireEvent.change(screen.getByLabelText('Input value mode'), { target: { value: 'manual' } })
    const first = (screen.getByLabelText('Tensor values') as HTMLTextAreaElement).value
    expect(JSON.parse(first)).toEqual([
      initializeTensor([2, 2], 'uniform', 137).data.slice(0, 2),
      initializeTensor([2, 2], 'uniform', 137).data.slice(2),
    ])
    fireEvent.change(screen.getByLabelText('Input value mode'), { target: { value: 'random' } })
    fireEvent.click(screen.getByRole('button', { name: 'Generate inputs' }))
    fireEvent.change(screen.getByLabelText('Input value mode'), { target: { value: 'manual' } })
    expect(screen.getByLabelText('Tensor values')).toHaveValue(first)
  })

  it('exposes sort order and index selection mode in Details',()=>{
    const first=render(<InspectorHarness initial={createNode('argsort',1)}/>)
    expect(screen.getByLabelText('Argsort order')).toHaveValue('descending')
    fireEvent.change(screen.getByLabelText('Argsort order'),{target:{value:'ascending'}})
    expect(screen.getByLabelText('Argsort order')).toHaveValue('ascending')
    first.unmount()
    render(<InspectorHarness initial={createNode('index-mask',2)}/>)
    fireEvent.change(screen.getByLabelText('Index mask mode'),{target:{value:'exclude'}})
    expect(screen.getByLabelText('Index mask mode')).toHaveValue('exclude')
  })
  it('builds both scratch architectures using only blocks in the current palette',()=>{
    const available = new Set(blockPalette.map(item => item.type))
    for (const kind of ['cnn', 'transformer'] as const) {
      const { graph } = scratchModel(kind)
      expect(graph.nodes.every(node => available.has(node.type))).toBe(true)
      expect(forwardPass(graph).loss).toBeGreaterThan(0)
    }
  })
  it('creates an entire filter tensor and retains the selected initializer',()=>{
    render(<InspectorHarness initial={createNode('weight',1)}/>)
    expect(screen.queryByLabelText('Input value mode')).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Tensor shape'),{target:{value:'4,3,3,1'}})
    fireEvent.change(screen.getByLabelText('Tensor initializer'),{target:{value:'he'}})
    fireEvent.click(screen.getByRole('button',{name:'Initialize tensor'}))
    expect(screen.getByLabelText('Tensor shape')).toHaveValue('4, 3, 3, 1')
    expect(screen.getByLabelText('Tensor initializer')).toHaveValue('he')
    const values = (screen.getByLabelText('Tensor values') as HTMLTextAreaElement).value.split(',').map(Number)
    expect(values).toHaveLength(36)
    expect(values.every(Number.isFinite)).toBe(true)
    fireEvent.change(screen.getByLabelText('Tensor shape'),{target:{value:'-3,0'}})
    fireEvent.click(screen.getByRole('button',{name:'Initialize tensor'}))
    expect(screen.getByRole('alert')).toHaveTextContent('positive dimensions')
  })

  it('accepts parenthesized parameter shapes and initializes every entry on Enter',()=>{
    render(<InspectorHarness initial={createNode('weight',1)}/>)
    const shape = screen.getByLabelText('Tensor shape')
    fireEvent.change(shape,{target:{value:'(3,1)'}})
    fireEvent.keyDown(shape,{key:'Enter'})
    expect(shape).toHaveValue('3, 1')
    expect((screen.getByLabelText('Tensor values') as HTMLTextAreaElement).value.split(',')).toHaveLength(3)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('supports default transpose axes and a whole-tensor mean with keepDims',()=>{
    const node = createNode('transpose',1), onParams=vi.fn()
    const props={...callbacks,onParams,graph:{nodes:[node],edges:[],learningRate:.01}}
    const view=render(<ModelInspector {...props} node={node}/>)
    expect(screen.getByLabelText('Operation axes')).toHaveValue('')
    fireEvent.click(screen.getByRole('button',{name:'Apply operation'}))
    expect(onParams).toHaveBeenLastCalledWith(node.id,{axes:undefined})
    view.rerender(<ModelInspector {...props} node={createNode('mean',1)}/>)
    fireEvent.click(screen.getByLabelText('Keep dimensions'))
    fireEvent.click(screen.getByRole('button',{name:'Apply operation'}))
    expect(onParams).toHaveBeenLastCalledWith('mean-1',{axis:undefined,keepDims:true})
  })

  it('edits tensor transform parameters from Details at any zoom level',()=>{
    const node=createNode('tensor-transform',1), onParams=vi.fn()
    const props={...callbacks,onParams,graph:{nodes:[node],edges:[],learningRate:.01}}
    const view=render(<ModelInspector {...props} node={node}/>)
    expect(screen.getByLabelText('Tensor transform operation')).toHaveValue('reshape')
    fireEvent.change(screen.getByLabelText('Tensor transform operation'),{target:{value:'mean'}})
    expect(onParams).toHaveBeenCalledWith(node.id,{transform:'mean'})
    const changed={...node,params:{...node.params,transform:'mean' as const}}
    view.rerender(<ModelInspector {...props} graph={{...props.graph,nodes:[changed]}} node={changed}/>)
    expect(screen.getByLabelText('Operation axis')).toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('Keep dimensions'))
    fireEvent.click(screen.getByRole('button',{name:'Apply operation'}))
    expect(onParams).toHaveBeenLastCalledWith(node.id,{axis:undefined,keepDims:true})
  })

  it('keeps loss editing visible and exact numbers available on demand',()=>{
    const prediction={...createNode('input',1),params:{value:tensorValue([2],[.2,.8])}}
    const target={...createNode('target',1),params:{value:tensorValue([2],[0,1])}}
    const loss=createNode('loss',1)
    const graph:GraphModel={nodes:[prediction,target,loss],edges:[
      {id:'prediction-loss',source:prediction.id,target:loss.id,inputSlot:0},
      {id:'target-loss',source:target.id,target:loss.id,inputSlot:1},
    ],learningRate:.01}
    const evaluated=forwardPass(graph).graph
    render(<ModelInspector {...callbacks} graph={evaluated} node={evaluated.nodes.find(node=>node.id===loss.id)}/>)
    expect(screen.getByLabelText('Loss function')).toHaveValue('mse')
    expect(screen.getByLabelText('Loss formula')).toHaveTextContent('L = (1/n) * Σ_i (x1_i - y1_i)^2')
    const disclosure=screen.getByText('Exact values and gradients').closest('details')!
    expect(disclosure).not.toHaveAttribute('open')
    fireEvent.click(screen.getByText('Exact values and gradients'))
    expect(disclosure).toHaveAttribute('open')
    expect(disclosure).toHaveTextContent('Value')
  })

  it('updates the formula when choosing a loss or changing regularization', () => {
    render(<InspectorHarness initial={createNode('loss', 1)}/>)
    expect(screen.getByLabelText('Loss formula')).toHaveTextContent('L = 0.5 * (prediction - target)^2')
    for (const [kind, formula] of [
      ['mse', 'L = (prediction - target)^2'],
      ['mae', 'L = |prediction - target|'],
      ['binary-cross-entropy-with-logits', 'L = mean(softplus(prediction) - target * prediction)'],
      ['cross-entropy', 'L = mean(-log softmax(prediction)[target])'],
    ]) {
      fireEvent.change(screen.getByLabelText('Loss function'), { target: { value: kind } })
      expect(screen.getByLabelText('Loss formula')).toHaveTextContent(formula)
    }
    fireEvent.change(screen.getByLabelText('Regularization penalty'), { target: { value: 'l1' } })
    fireEvent.change(screen.getByLabelText('Strength λ'), { target: { value: '0.2' } })
    expect(screen.getByLabelText('Loss formula')).toHaveTextContent('+ 0.2 Σ |w|')
  })

  it.each(['loss', 'cross-entropy'] as const)('shows the formula for imported legacy %s blocks', type => {
    const node = createNode(type, 1)
    node.params.loss = 'binary-cross-entropy'
    render(<InspectorHarness initial={node}/>)
    expect(screen.getByLabelText('Loss formula')).toHaveTextContent(type === 'loss'
      ? 'L = -(target * log(prediction) + (1 - target) * log(1 - prediction))'
      : 'L = mean(−log softmax(prediction)[target])')
  })

  it.each(['loss', 'cross-entropy'] as const)('explains the target index in %s Details without renaming the connected answer', type => {
    const prediction = createNode('input', 1), target = createNode('target', 1), loss = createNode(type, 1)
    prediction.label = 'z7'
    target.label = 'answer'
    loss.params.loss = 'cross-entropy'
    const graph: GraphModel = { nodes: [prediction, target, loss], learningRate: .1, edges: [
      { id: 'prediction', source: prediction.id, target: loss.id, inputSlot: 0 },
      { id: 'target', source: target.id, target: loss.id, inputSlot: 1 },
    ] }
    render(<ModelInspector {...callbacks} graph={graph} node={loss}/>)
    expect(screen.getByLabelText('Loss formula')).toHaveTextContent('softmax(z7)[target]')
    expect(screen.getByText(/target is the correct-class index/)).toBeInTheDocument()
    expect(target.label).toBe('answer')
    expect(formulaForNode(loss, graph)).toContain('softmax(z7)[answer]')
  })

  it('shows inputs inferred from the loss path and updates after disconnecting it', () => {
    const data = createNode('dataset', 0), operation = createNode('standardize', 0), loss = createNode('loss', 0)
    data.params = { dataset: 'custom-csv', customCsv: parseCustomCsv('radius,unused,y\n1,9,0\n2,8,1\n', 'features.csv') }
    const graph: GraphModel = { nodes: [data, operation, loss], learningRate: .1, edges: [
      { id: 'feature', source: data.id, target: operation.id, inputSlot: 0 },
      { id: 'prediction', source: operation.id, target: loss.id, inputSlot: 0 },
      { id: 'target', source: data.id, sourceSlot: 2, target: loss.id, inputSlot: 1 },
    ] }
    const { rerender } = render(<DatasetWorkbench {...callbacks} graph={graph} node={data}/>)
    expect(screen.getByLabelText('Detected dataset inputs')).toHaveTextContent('Connected inputs: radius.')
    expect(screen.getByLabelText('Detected dataset inputs')).not.toHaveTextContent('unused')
    rerender(<DatasetWorkbench {...callbacks} graph={{ ...graph, edges: graph.edges.filter(edge => edge.id !== 'prediction') }} node={data}/>)
    expect(screen.getByLabelText('Detected dataset inputs')).not.toHaveTextContent('Connected inputs:')
  })

  it('sets and clears a computed output variable independently of its title', () => {
    const node = createNode('standardize', 1), onParams = vi.fn()
    const props = { ...callbacks, graph: { nodes: [node], edges: [], learningRate: .01 }, node, onParams }
    const { rerender } = render(<ModelInspector {...props}/>)
    fireEvent.change(screen.getByLabelText('Output variable'), { target: { value: 'u' } })
    expect(onParams).toHaveBeenLastCalledWith(node.id, { outputName: 'u' })
    node.params.outputName = 'u'
    rerender(<ModelInspector {...props}/>)
    fireEvent.change(screen.getByLabelText('Output variable'), { target: { value: '' } })
    expect(onParams).toHaveBeenLastCalledWith(node.id, { outputName: undefined })
    expect(screen.getByRole('heading', { name: 'Standardize features' })).toBeInTheDocument()
  })

  it('renames a selected block from its title, without the old selection heading',()=>{
    const node=createNode('input',1), onRename=vi.fn()
    render(<ModelInspector {...callbacks} graph={{nodes:[node],edges:[],learningRate:.01}} node={node} onRename={onRename}/>)
    expect(screen.queryByText('Selected block')).not.toBeInTheDocument()
    expect(screen.queryByText('Rename block')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button',{name:'Rename x1'}))
    const name=screen.getByRole('textbox',{name:'Block name'})
    fireEvent.change(name,{target:{value:'Kernel'}})
    fireEvent.keyDown(name,{key:'Enter'})
    expect(onRename).toHaveBeenCalledWith(node.id,'Kernel')
  })

  it.each(['weight', 'bias'] as const)('edits the %s output variable while keeping the Param heading', type => {
    const node = createNode(type, 1)
    render(<InspectorHarness initial={node}/>)
    const field = screen.getByLabelText('Output variable')
    expect(field).toHaveAttribute('placeholder', 'Automatic (p1, p2, …)')
    expect(field).toHaveValue('')
    fireEvent.change(field, { target: { value: 'kernel' } })
    expect(field).toHaveValue('kernel')
    expect(screen.getByRole('heading', { name: 'Param' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^Rename/ })).not.toBeInTheDocument()
    fireEvent.change(field, { target: { value: 'not a variable' } })
    expect(field).toHaveValue('kernel')
    fireEvent.change(field, { target: { value: '' } })
    expect(field).toHaveValue('')
    expect(screen.getByRole('heading', { name: 'Param' })).toBeInTheDocument()
  })

  it('renames a group from its title while keeping its kind selector',()=>{
    const group={id:'group-1',label:'Group 1',kind:'module',nodeIds:[],position:{x:0,y:0},dimensions:{width:200,height:100}}
    const onGroupChange=vi.fn()
    render(<ModelInspector {...callbacks} graph={{nodes:[],edges:[],groups:[group],learningRate:.01}} group={group} onGroupChange={onGroupChange}/>)
    fireEvent.click(screen.getByRole('button',{name:'Rename Group 1'}))
    const name=screen.getByRole('textbox',{name:'Block name'})
    fireEvent.change(name,{target:{value:'Attention head'}})
    fireEvent.blur(name)
    expect(onGroupChange).toHaveBeenCalledWith(group.id,{label:'Attention head'})
    fireEvent.click(screen.getByText('Group settings'))
    fireEvent.change(screen.getByRole('combobox',{name:'Block kind'}),{target:{value:'head'}})
    expect(onGroupChange).toHaveBeenCalledWith(group.id,{kind:'head'})
  })

  it('generates on a transformer whose nodes have only palette IDs',()=>{
    const {graph,datasetId}=scratchModel('transformer'), changed=vi.fn()
    render(<DecoderControls graph={graph} onGraphChange={changed}/>)
    fireEvent.change(screen.getByLabelText('Prompt'),{target:{value:'<bos> red green blue'}})
    fireEvent.click(screen.getByRole('button',{name:'Apply prompt'}))
    const next:GraphModel=changed.mock.calls[0][0]
    expect(datasetOutputValueForSlot(next.nodes.find(node=>node.id === datasetId)!,0).data).toEqual([0,1,2,3])
    expect(predictionNode(next)?.value?.shape).toEqual([4,5])
    fireEvent.click(screen.getByRole('button',{name:'Generate next token'}))
    expect(changed).toHaveBeenCalledTimes(2)
  })

  it('inspects arbitrary CNN filter counts and edits the actual shared kernel',()=>{
    const {graph}=scratchModel('cnn'), evaluated=forwardPass(graph).graph
    const node=evaluated.nodes.find(node=>node.type === 'conv2d')!, onValue=vi.fn()
    render(<ConvolutionInspector graph={evaluated} node={node} onValue={onValue}/>)
    expect((screen.getByLabelText('Filter to inspect') as HTMLSelectElement).options).toHaveLength(2)
    fireEvent.change(screen.getByLabelText('Filter to inspect'),{target:{value:'1'}})
    fireEvent.change(screen.getByLabelText('Selected matrix coordinate'),{target:{value:'0.75'}})
    expect(onValue.mock.calls[0][1].data[9]).toBe(.75)
    expect(screen.getByText('Feature map · before activation')).toBeVisible()
  })

  it('keeps dataset details for configuration, including a selectable train/test split',()=>{
    const {graph,datasetId}=scratchModel('transformer'), onParams=vi.fn()
    const node=graph.nodes.find(candidate=>candidate.id === datasetId)!
    render(<DatasetWorkbench graph={graph} node={node} onParams={onParams} onDataset={vi.fn()}/>)
    fireEvent.change(screen.getByLabelText('Train/test split'),{target:{value:'80'}})
    expect(onParams).toHaveBeenCalledWith(datasetId,expect.objectContaining({trainPercent:80,datasetSplit:'train'}))
    expect(screen.queryByRole('button',{name:/Train 1 epoch|Run inference|Evaluate training/i})).not.toBeInTheDocument()
  })
})

it('offers an editable dropout probability and explains evaluation behavior',()=>{
 const node=createNode('dropout',1),onParams=vi.fn()
 render(<ModelInspector {...callbacks} node={node} graph={{nodes:[node],edges:[],learningRate:.01}} onParams={onParams}/>)
 expect(blockPalette.some(item=>item.type==='dropout')).toBe(true)
 expect(screen.getByLabelText('Dropout probability')).toHaveValue(.1)
 fireEvent.change(screen.getByLabelText('Dropout probability'),{target:{value:'.2'}})
 expect(onParams).toHaveBeenCalledWith(node.id,{dropoutRate:.2})
 expect(screen.getByText(/Evaluation passes them unchanged/)).toBeInTheDocument()
})
