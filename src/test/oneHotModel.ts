import { builder } from './curriculumModels'

/** Repeated IDs exercise equivalence of encoding + matmul and table lookup. */
export function oneHotModel(embedding = false, trainableIds = false) {
  const { graph, add } = builder()
  graph.training!.engine = 'trace'
  add('ids', trainableIds ? 'weight' : 'input', { value: { shape: [3], data: [2, 0, 2] } })
  add('table', 'weight', { value: { shape: [3, 2], data: [.2, .3, .4, .5, .6, .7] } })
  if (embedding) add('vectors', 'embedding', {}, ['table', 'ids'])
  else {
    add('encoding', 'one-hot', { numClasses: 3 }, ['ids'])
    add('vectors', 'matmul', {}, ['encoding', 'table'])
  }
  add('pooled', 'mean', { axis: 0, keepDims: true }, ['vectors'])
  add('head', 'weight', { value: { shape: [2, 1], data: [.5, -.2] } })
  add('logits', 'matmul', {}, ['pooled', 'head'])
  add('probability', 'activation', { activation: 'sigmoid' }, ['logits'])
  add('target', 'target', { value: { shape: [1, 1], data: [1] } })
  add('loss', 'loss', { loss: 'binary-cross-entropy-with-logits' }, ['logits', 'target'])
  return graph
}
