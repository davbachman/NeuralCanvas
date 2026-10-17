import { sentimentModel } from '../test/inferenceInspectionModel'
import { afterEach, beforeAll, expect, it } from 'vitest'
import { InferenceInspection, inferenceSignature, type InspectionValue } from './inferenceInspection'
import { prepareTextDocuments } from './textData'
import { buildTextModel } from '../test/textModels'
import { buildQaModel } from '../test/curriculumModels'
import { exampleRowsModel } from '../test/exampleRowsModel'
import { backwardPass, forwardPass, parameterValues } from './engine'
import { evaluateDataset, withDatasetExample, withDatasetIndices } from './datasetTraining'
import { TensorGraph, tf } from './tensorTraining'
import type { GraphModel } from './types'

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready() })
afterEach(() => expect(tf.memory().numTensors).toBe(0))
const available = (result: InspectionValue) => { expect(result.status).toBe('available'); if (result.status !== 'available') throw Error(result.message); return result }
async function report(graph: GraphModel, backend?: string) {
  if (!backend) return evaluateDataset(graph, graph.nodes.find(node => node.type === 'dataset')!.id, 'test')
  const model = new TensorGraph(graph)
  try { return await model.inference(model.examples.filter(row => row.split === 'test'), 64) }
  finally { model.dispose() }
}

it.each([undefined, 'cpu'])('inspects every word-count sigmoid output, including incorrect predictions (%s)', async backend => {
  const graph = backwardPass(forwardPass(sentimentModel()).graph).graph
  const before = structuredClone(graph), metrics = await report(graph, backend)
  const inspection = new InferenceInspection(graph, 'text-data', 'test', backend)
  expect(inspection.indices).toEqual([1, 3])
  expect(metrics.rows.every(row => row.correct === false)).toBe(true)
  for (const row of inspection.indices) {
    const result = available(await inspection.read('positive-probability', 0, row))
    const expected = forwardPass(withDatasetExample(graph, 'text-data', row), false).graph.nodes.find(node => node.id === 'positive-probability')!.value!.data[0]
    expect(result.value.data[0]).toBeCloseTo(expected, 6)
    expect(result.simple).toBe(true)
    expect(metrics.rows.find(item => item.example === `Dataset row ${row}`)!.predicted).toBe(result.value.data[0] >= .5 ? 'positive' : 'negative')
  }
  expect(graph).toEqual(before)
  expect(inspection.graph.nodes.every(node => !node.grad && !node.cache)).toBe(true)
  expect(inspection.graph.edges.every(edge => !edge.grad)).toBe(true)
})

it.each([undefined, 'cpu'])('matches multiclass reports and retains original noncontiguous batch row IDs (%s)', async backend => {
  const { graph: base } = exampleRowsModel()
  const graph = withDatasetIndices(base, 'data', [4, 1, 3])
  const before = structuredClone(graph), metrics = await report({ ...graph, nodes: graph.nodes.map(node => node.id === 'data' ? { ...node, params: { ...node.params, datasetSplit: 'test' } } : node) }, backend)
  const inspection = new InferenceInspection(graph, 'data', 'test', backend)
  expect(inspection.indices).toEqual([0, 2, 5])
  for (const row of inspection.indices) {
    const result = available(await inspection.read('probabilities', 0, row))
    const expected = forwardPass(withDatasetIndices(graph, 'data', [row])).graph.nodes.find(node => node.id === 'probabilities')!.value!.data
    result.value.data.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 6))
    expect(metrics.rows.find(item => item.example === `Dataset row ${row}`)!.predicted).toBe(String(result.value.data.indexOf(Math.max(...result.value.data))))
  }
  const weights = available(await inspection.read('weights', 0, 5))
  expect(weights.kind).toBe('shared')
  expect(weights.value).toEqual(graph.nodes.find(node => node.id === 'weights')!.params.value)
  const reduced = available(await inspection.read('loss', 0, 5))
  expect(reduced.kind).toBe('aggregate')
  expect(reduced.value.shape).toEqual([])
  expect(graph).toEqual(before) // Includes fitted preprocessing and canvas batch settings.
  const training = new InferenceInspection(graph, 'data', 'train', backend)
  expect(training.indices).toEqual([1, 3, 4])
})

it('replays attention using the original batch padding and masks, including the final partial batch', async () => {
  const data = prepareTextDocuments(Array.from({ length: 68 }, (_, i) => ({ text: i % 2 ? 'good' : 'good bad good', label: i % 2 ? 'positive' : 'negative', split: i < 2 ? 'train' as const : 'test' as const })), 'attention.csv', { task: 'sentiment', representation: 'tokens', maxLength: 4 })
  const graph = buildTextModel(data, 'attention', 4)
  const attentionId = graph.nodes.find(node => node.type === 'softmax')!.id
  const inspection = new InferenceInspection(graph, 'text-data', 'test', 'cpu')
  const model = new TensorGraph(graph)
  try {
    for (const row of [3, 67]) {
      const indices = row < 66 ? inspection.indices.slice(0, 64) : inspection.indices.slice(64)
      const expected = (await model.inspectSignal(indices.map(index => model.examples[index]), attentionId))!
      const result = available(await inspection.read(attentionId, 0, row))
      const width = expected.value.data.length / indices.length, index = indices.indexOf(row)
      expect(result.value.data).toEqual(expected.value.data.slice(index * width, (index + 1) * width))
      expect(result.value.shape).toEqual([3, 3])
      expect(result.axes).toEqual(['token', 'token'])
      expect(result.value.excluded).toEqual([false, true, true, true, true, true, true, true, true])
      expect(result.simple).toBe(false)
    }
  } finally { model.dispose() }
})

it('supports multidimensional structured-story tensors and multiclass probabilities', async () => {
  const documents = ['train', 'train', 'test', 'test'].map((split, i) => ({ text: 'Mary kitchen garden where', facts: ['Mary kitchen', 'Mary garden'], question: 'where Mary', label: i % 2 ? 'garden' : 'kitchen', split: split as 'train' | 'test' }))
  const data = prepareTextDocuments(documents, 'stories.csv', { task: 'classification', representation: 'facts', maxFacts: 2, factWords: 3 })
  const graph = buildQaModel(data, 'memory-1', 137, 4)
  const inspection = new InferenceInspection(graph, 'text-data', 'test', 'cpu')
  const result = available(await inspection.read('fact_lookup', 0, 3))
  expect(result.value.shape).toEqual([2, 3, 4])
  expect(result.simple).toBe(false)
})

it('freezes parameters and dataset settings, and bounds the signal cache', async () => {
  const graph = sentimentModel(40)
  const inspection = new InferenceInspection(graph, 'text-data', 'test')
  const parameters = parameterValues(inspection.graph)
  const expected = available(await inspection.read('positive-probability', 0, 1)).value
  const weight = graph.nodes.find(node => node.type === 'weight')!
  weight.params.value = { shape: [1], data: [900] }
  graph.nodes.find(node => node.type === 'dataset')!.params.textData!.documents[1].text = 'changed'
  for (const row of inspection.indices) await inspection.read('positive-probability', 0, row)
  expect(inspection.cacheSize.entries).toBeLessThanOrEqual(InferenceInspection.maxEntries)
  expect(inspection.cacheSize.elements).toBeLessThanOrEqual(InferenceInspection.maxElements)
  expect(available(await inspection.read('positive-probability', 0, 1)).value).toEqual(expected)
  expect(parameterValues(inspection.graph)).toEqual(parameters)
  expect(inferenceSignature(graph)).not.toBe(inspection.signature)
})

it('invalidates numerical and topology changes but tolerates layout and canvas navigation', () => {
  const { graph } = exampleRowsModel(), signature = inferenceSignature(graph)
  const navigation = structuredClone(graph)
  navigation.nodes[0].params.datasetIndex = 5
  navigation.nodes[0].params.datasetSplit = 'test'
  navigation.nodes[0].position.x += 100
  expect(inferenceSignature(navigation)).toBe(signature)
  for (const edit of [
    (g: GraphModel) => { g.nodes.find(n => n.id === 'scaled')!.params.standardization!.scale[0] = 5 },
    (g: GraphModel) => { g.nodes[0].params.customCsv!.rows[0][0] = '99' },
    (g: GraphModel) => { g.edges.pop() },
  ]) {
    const changed = structuredClone(graph); edit(changed)
    expect(inferenceSignature(changed)).not.toBe(signature)
  }
})

it('distinguishes missing run outputs from disconnected blocks', async () => {
  const { graph, add } = exampleRowsModel()
  add('unwired', 'activation', { activation: 'sigmoid' })
  const inspection = new InferenceInspection(graph, 'data', 'test')
  expect((await inspection.read('new-block', 0, 0)).status).toBe('not-evaluated')
  expect((await inspection.read('unwired', 0, 0)).status).toBe('disconnected')
  expect((await inspection.read('probabilities', 0, 1)).status).toBe('not-evaluated')
})

it('does not retain an individual tensor larger than the cache budget', async () => {
  const { graph, add } = exampleRowsModel()
  const value = { shape: [InferenceInspection.maxElements + 1], data: Array(InferenceInspection.maxElements + 1).fill(.5) }
  add('large-shared', 'weight', { value })
  const inspection = new InferenceInspection(graph, 'data', 'test')
  const result = available(await inspection.read('large-shared', 0, 0))
  expect(result.kind).toBe('shared')
  expect(result.value).toEqual(value)
  expect(inspection.cacheSize).toEqual({ entries: 0, elements: 0 })
})
