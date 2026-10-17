import { prepareTextDocuments } from '../domain/textData'
import { createNode } from '../domain/examples'
import type { GraphModel } from '../domain/types'

/** A deterministic next-token lookup: pad wins after 'e', and reserved tokens
 * predict comma while their final printed character predicts 'l'. */
export function generationConsistencyModel(): GraphModel {
  const text = 'alice was in the <unk> <pad>, alice was in the'
  const data = prepareTextDocuments([{text, split: 'train'}, {text, split: 'test'}], 'generation.txt', {task: 'language', tokenizer: 'character', maxLength: 8, fixedLength: true})
  const size = data.vocabulary.length, id = (token: string) => data.vocabulary.indexOf(token)
  const values = Array(size * size).fill(-10)
  for (let i = 0; i < size; i++) values[i * size + id('l')] = 10
  const transition = (from: string, to: string, score = 20) => { values[id(from) * size + id(to)] = score }
  transition('e', '<pad>', 100); transition('e', '<unk>', 90)
  transition('<pad>', ','); transition('<unk>', ',')
  transition('<', 'u'); transition('u', 'n'); transition('n', 'k'); transition('k', '>')
  const source = {...createNode('dataset', 1), id: 'data', params: {dataset: 'custom-text' as const, textData: data, datasetMode: 'sample' as const, datasetIndex: 0}}
  const table = {...createNode('weight', 1), id: 'weights', params: {value: {shape: [size,size], data: values}}}
  const logits = {...createNode('embedding', 1), id: 'logits'}
  const loss = {...createNode('loss', 1), id: 'loss', params: {loss: 'cross-entropy' as const}}
  return {learningRate: .01, nodes: [source, table, logits, loss], edges: [
    {id: 'table', source: 'weights', target: 'logits', inputSlot: 0},
    {id: 'ids', source: 'data', sourceSlot: 0, target: 'logits', inputSlot: 1},
    {id: 'prediction', source: 'logits', target: 'loss', inputSlot: 0},
    {id: 'target', source: 'data', sourceSlot: 2, target: 'loss', inputSlot: 1},
  ]}
}
