import { builder } from './curriculumModels'
import { backwardPass, forwardPass } from '../domain/engine'

/** Small graph with large populated tensors, for canvas navigation benchmarks. */
export function canvasPerformanceModel(embedding = false) {
  const { graph, add, weight } = builder()
  add('ids', 'input', { value: { shape: [16], data: Array.from({ length: 16 }, (_, i) => i % 27) } })
  weight('table', [27, 8])
  if (embedding) add('vectors', 'embedding', {}, ['table', 'ids'])
  else {
    add('encoding', 'one-hot', { numClasses: 27 }, ['ids'])
    add('vectors', 'matmul', {}, ['encoding', 'table'])
  }
  add('flatten', 'tensor-transform', { transform: 'reshape', shape: [1, 128] }, ['vectors'])
  weight('w1', [128, 32])
  add('b1', 'bias', { value: { shape: [32], data: Array(32).fill(.1) } })
  add('hidden', 'matmul', {}, ['flatten', 'w1'])
  add('biased', 'arithmetic', { expression: 'x1 + x2' }, ['hidden', 'b1'])
  add('relu', 'activation', { activation: 'relu' }, ['biased'])
  weight('w2', [32, 6])
  add('b2', 'bias', { value: { shape: [6], data: Array(6).fill(.1) } })
  add('output', 'matmul', {}, ['relu', 'w2'])
  add('logits', 'arithmetic', { expression: 'x1 + x2' }, ['output', 'b2'])
  add('probabilities', 'softmax', { axis: 1 }, ['logits'])
  add('target', 'target', { value: { shape: [1], data: [2] } })
  add('loss', 'loss', { loss: 'cross-entropy' }, ['logits', 'target'])
  graph.view = { expandedGroupIds: [], ...graph.view, viewport: { x: 40, y: 30, zoom: .55 } }
  return backwardPass(forwardPass(graph, false).graph).graph
}
