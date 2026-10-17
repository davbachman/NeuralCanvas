import { expect, it } from 'vitest'
import { tensorExampleRows } from './tensorExampleRows'
import { exampleRowsModel } from '../test/exampleRowsModel'
import { backwardPass, cloneGraph, forwardPass, parameterValues } from './engine'
import { trainingBatches, withDatasetExample, withDatasetIndices } from './datasetTraining'
import { createProjectStateFile, parseProjectStateFile } from './session'
import { tensorValue } from './tensor'
import { createNode } from './examples'

it('preserves original row indices through column concatenation, scaling, matmul, bias, and softmax', () => {
  const {graph} = exampleRowsModel()
  const evaluated = backwardPass(forwardPass(graph).graph).graph
  for (const id of ['data', 'columns', 'scaled', 'product', 'biased', 'probabilities']) {
    expect(tensorExampleRows(evaluated, id)).toEqual([1, 3, 4])
  }
  expect(tensorExampleRows(evaluated, 'data', 2)).toEqual([1, 3, 4])
  for (const id of ['weights', 'bias', 'loss']) expect(tensorExampleRows(evaluated, id)).toBeUndefined()
  graph.nodes[0].params.datasetSplit = 'test'
  expect(tensorExampleRows(forwardPass(graph).graph, 'probabilities')).toEqual([0, 2, 5])
  graph.nodes[0].params.datasetSplit = 'all'
  expect(tensorExampleRows(forwardPass(graph).graph, 'probabilities')).toEqual([0, 1, 2, 3, 4, 5])
})

it('retains shuffled, repeated, and single-row batch identities through clones and saved projects', () => {
  const {graph} = exampleRowsModel()
  const batches = [...trainingBatches([1, 3, 4], 2, 3), [4, 1, 4], [3]]
  for (const indices of batches) {
    const evaluated = forwardPass(withDatasetIndices(graph, 'data', indices)).graph
    expect(tensorExampleRows(cloneGraph(evaluated), 'probabilities')).toEqual(indices)
    expect(evaluated.nodes.find(node => node.id === 'columns')!.value!.data).toEqual(indices.flatMap(index => [index + 1, (index + 1) * 10]))
    const file = createProjectStateFile({graph:evaluated,visualizationGraph:evaluated,initialParameterValues:parameterValues(evaluated),selectedNodeIds:[],phase:'edit',traceSteps:[],traceIndex:0,epoch:0,currentLoss:null,display:{showMath:true,showGradient:true,showCode:false,showVisualization:false}})
    const restored = parseProjectStateFile(JSON.stringify(file))
    expect(restored.ok).toBe(true)
    if (restored.ok) expect(tensorExampleRows(restored.file.state.graph, 'probabilities')).toEqual(indices)
  }
})

it('preserves row identity only when reshape, transpose, slice, and reduction keep the example axis', () => {
  const {graph, add} = exampleRowsModel()
  add('flat', 'reshape', {shape:[6]}, ['columns'])
  add('reshaped', 'reshape', {shape:[3,1,2]}, ['columns'])
  add('transposed', 'transpose', {}, ['columns'])
  add('back', 'transpose', {}, ['transposed'])
  add('slice', 'slice', {axis:0,start:1,end:3}, ['columns'])
  add('feature-mean', 'mean', {axis:1}, ['columns'])
  add('row-mean', 'mean', {axis:0,keepDims:true}, ['columns'])
  add('all-mean', 'mean', {}, ['columns'])
  const evaluated = forwardPass(graph).graph
  for (const id of ['flat','transposed','row-mean','all-mean']) expect(tensorExampleRows(evaluated, id)).toBeUndefined()
  for (const id of ['reshaped','back','feature-mean']) expect(tensorExampleRows(evaluated, id)).toEqual([1,3,4])
  expect(tensorExampleRows(evaluated, 'slice')).toEqual([3,4])
})

it('does not infer example identity from matching shapes or custom values without provenance', () => {
  const {graph, add} = exampleRowsModel()
  add('parameter-rows', 'weight', {value:tensorValue([3,2],[1,2,3,4,5,6])})
  add('parameter-product', 'matmul', {}, ['parameter-rows','weights'])
  let evaluated = forwardPass(graph).graph
  expect(tensorExampleRows(evaluated,'parameter-product')).toBeUndefined()
  graph.nodes[0].params.datasetValues = [tensorValue([3],[8,9,10]),tensorValue([3],[11,12,13]),tensorValue([3],[0,1,0])]
  evaluated = forwardPass(graph).graph
  expect(tensorExampleRows(evaluated, 'data')).toBeUndefined()
  expect(tensorExampleRows(evaluated, 'probabilities')).toBeUndefined()
})

it('rejects conflicting row orders and combines explicit row concatenations', () => {
  const {graph, add} = exampleRowsModel()
  add('first', 'slice', {axis:0,start:0,end:2}, ['columns'])
  add('second', 'slice', {axis:0,start:1,end:3}, ['columns'])
  add('conflict', 'add', {}, ['first','second'])
  add('bad-columns', 'concat', {axis:1}, ['first','second'])
  add('stacked', 'concat', {axis:0}, ['second','first'])
  const evaluated = forwardPass(graph).graph
  expect(tensorExampleRows(evaluated,'conflict')).toBeUndefined()
  expect(tensorExampleRows(evaluated,'bad-columns')).toBeUndefined()
  expect(tensorExampleRows(evaluated,'stacked')).toEqual([3,4,1,3])
})

it('recognizes a scalar example reshaped into one row without labelling image rows as examples', () => {
  const {graph, add} = exampleRowsModel()
  graph.nodes = graph.nodes.filter(node => node.type === 'dataset')
  graph.edges = []
  add('single-row', 'reshape', {shape:[1,1]}, [['data',0]])
  expect(tensorExampleRows(forwardPass(withDatasetExample(graph,'data',4)).graph,'single-row')).toEqual([4])
  const image = createNode('dataset', 0)
  image.params = {dataset:'digits-8x8',datasetIndex:4}
  expect(tensorExampleRows({nodes:[image],edges:[],learningRate:.1},image.id)).toBeUndefined()
})

it('tracks modified single-example origin separately from example axes and shared parameters', async () => {
  const { singleExampleTextModel } = await import('../test/inferenceInspectionModel')
  const { tensorDatasetProvenance } = await import('./tensorExampleRows')
  const { datasetExamplesForNode } = await import('./datasets')
  const graph = withDatasetExample(singleExampleTextModel(), 'text-data', 2)
  const source = graph.nodes.find(node => node.type === 'dataset')!
  const example = datasetExamplesForNode(source)[2]
  source.params.datasetValues = [example.features[0], example.features[1], example.target]
  graph.nodes.push({ ...createNode('mean', 99), id: 'reduced', params: { axis: 0 } })
  graph.edges.push({ id: 'reduce', source: 'review-counts', target: 'reduced', inputSlot: 0 })
  const evaluated = forwardPass(graph).graph
  for (const id of ['text-data', 'review-counts', 'sentiment-product', 'sentiment-bias-add', 'positive-probability', 'reduced']) {
    expect(tensorDatasetProvenance(evaluated, id).sources).toEqual([{ datasetId: 'text-data', rows: [2], modified: true }])
    expect(tensorExampleRows(evaluated, id)).toEqual(id === 'reduced' ? undefined : [2])
  }
  expect(tensorExampleRows(evaluated, 'text-data', 1)).toBeUndefined() // Position IDs are not example IDs.
  for (const id of ['sentiment-weights', 'sentiment-bias']) expect(tensorDatasetProvenance(evaluated, id)).toEqual({ exampleRows: undefined, sources: [] })
  const restored = forwardPass(withDatasetExample(graph, 'text-data', 3)).graph
  expect(tensorDatasetProvenance(restored, 'positive-probability')).toEqual({ exampleRows: [3], sources: [{ datasetId: 'text-data', rows: [3], modified: false }] })
})

it('retains single-example context through token embeddings and attention without relabeling token axes', async () => {
  const { singleExampleTextModel } = await import('../test/inferenceInspectionModel')
  const { buildTextModel } = await import('../test/textModels')
  const { encodeText } = await import('./textData')
  const { tensorDatasetProvenance } = await import('./tensorExampleRows')
  const { datasetExamplesForNode } = await import('./datasets')
  const data = singleExampleTextModel().nodes.find(node => node.type === 'dataset')!.params.textData!
  const graph = withDatasetExample(buildTextModel(data, 'attention', 4), 'text-data', 2)
  const source = graph.nodes.find(node => node.type === 'dataset')!
  const ids = encodeText('good good film', data)
  source.params.datasetValues = [{ shape: [3], data: ids }, { shape: [3], data: [0, 1, 2] }, datasetExamplesForNode(source)[2].target]
  const evaluated = forwardPass(graph).graph
  for (const node of evaluated.nodes.filter(node => ['embedding', 'softmax'].includes(node.type))) {
    expect(tensorDatasetProvenance(evaluated, node.id).sources).toEqual([{ datasetId: 'text-data', rows: [2], modified: true }])
    expect(tensorExampleRows(evaluated, node.id)).toBeUndefined()
  }
})
