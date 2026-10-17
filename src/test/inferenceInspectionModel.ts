import { prepareTextDocuments } from '../domain/textData'
import { buildTextModel } from './textModels'

export function sentimentModel(count = 4) {
  const data = prepareTextDocuments(Array.from({ length: count }, (_, i) => ({ text: i % 2 ? 'bad' : 'good good', label: i % 2 ? 'negative' : 'positive', split: i === 0 || i === 2 ? 'train' as const : 'test' as const })), 'sentiment.csv', { task: 'sentiment', representation: 'counts', maxLength: 4 })
  const graph = buildTextModel(data, 'counts-linear', 4)
  graph.groups = []
  const probabilityEdge = graph.edges.find(edge => edge.target === 'positive-probability')!
  graph.edges.find(edge => edge.target === 'loss' && edge.inputSlot === 0)!.source = probabilityEdge.source
  graph.nodes.find(node => node.id === 'loss')!.params.loss = 'binary-cross-entropy-with-logits'
  // Deliberately wrong on held-out negative examples; inspection must include them.
  for (const node of graph.nodes.filter(node => node.type === 'weight' || node.type === 'bias')) {
    const value = node.params.value as { shape: number[]; data: number[] }
    node.params.value = { ...value, data: value.data.map(() => .25) }
  }
  return graph
}
