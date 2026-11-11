import { expect, it } from 'vitest'
import { decodeTokens, encodeText, prepareTextDocuments, tokenize } from './textData'
import { generateText, predictText, textGenerationStates, textSamplingDistribution } from './textGeneration'
import { generationConsistencyModel } from '../test/generationConsistencyModel'
import { createNode } from './examples'
import type { GraphModel, NodeParams, NodeType } from './types'

it.each(['character', 'word'] as const)('round-trips reserved IDs alongside ordinary %s tokens', tokenizer => {
  const text = 'alice was in the garden, alice was in the garden.'
  const data = prepareTextDocuments([{text, split:'train'}, {text, split:'test'}], 'roundtrip.txt', {task:'language', tokenizer, maxLength: 4, fixedLength:true})
  const ids = [0,1,...encodeText('alice',data),1,0]
  expect(encodeText(decodeTokens(ids,data),data)).toEqual(ids)
  expect(encodeText('<pad><unk><pad>',data)).toEqual([1,0,1])
  const unpadded = {...data, fixedLength:false, vocabulary: data.vocabulary.filter(token=>token!=='<pad>')}
  expect(encodeText('<pad>',unpadded)).toEqual(tokenize('<pad>',tokenizer).map(token=>Math.max(0,unpadded.vocabulary.indexOf(token))))
})

it('excludes padding before top-k and renormalizes both inspection and sampling probabilities', () => {
  const graph = generationConsistencyModel(), result = predictText(graph, 'alice was in the')
  expect(result.logits.indexOf(Math.max(...result.logits))).toBe(1)
  expect(textSamplingDistribution([1,100,2],['<unk>','<pad>','a'],1,1)).toEqual([0,0,1])
  const distribution = textSamplingDistribution(result.logits,result.vocabulary)
  expect(distribution[1]).toBe(0)
  expect(distribution.reduce((a,b)=>a+b,0)).toBeCloseTo(1,12)
  for (const random of [0,.5,.999999999]) expect(generateText(graph,'alice was in the',2,{sample:true,topK:1,random:()=>random})).toBe('alice was in the<unk>,')
})

it('makes continuous and consecutive greedy runs identical, including pasted legacy padding and full context windows', () => {
  const graph = generationConsistencyModel(), before = JSON.stringify(graph)
  for (const prompt of ['alice was in the', 'alice was in the<pad>', 'e']) {
    const together = generateText(graph,prompt,2)
    const separate = generateText(graph,generateText(graph,prompt,1),1)
    expect(separate).toBe(together)
  }
  expect(generateText(graph,'alice was in the',2)).toBe('alice was in the<unk>,')
  expect(predictText(graph,'alice was in the<pad>').ids.at(-1)).toBe(1)
  expect(JSON.stringify(graph)).toBe(before)
})

it('retains actual IDs when generated characters happen to spell a reserved marker', () => {
  const graph = generationConsistencyModel(), before = JSON.stringify(graph)
  const continuous = [...textGenerationStates(graph,'<',5)].at(-1)!
  let consecutive = [...textGenerationStates(graph,'<',1)][0]
  for (let i=1;i<5;i++) consecutive=[...textGenerationStates(graph,consecutive,1)][0]
  expect(consecutive).toEqual(continuous)
  expect(consecutive.text).toBe('<unk>l')
  const marker = [...textGenerationStates(graph,'<',4)].at(-1)!
  expect(marker.tokenIds).not.toContain(0)
  expect(predictText(graph,marker).logits).not.toEqual(predictText(graph,marker.text).logits)
  expect(JSON.stringify(graph)).toBe(before)
})

it('reevaluates a decoder-direction residual intervention on every generated token and restores baseline at zero strength', () => {
  const data = prepareTextDocuments([{ text: 'a b c a b c', split: 'train' }, { text: 'a b c a b c', split: 'test' }], 'intervention.txt', { task: 'language', tokenizer: 'word', maxLength: 4, fixedLength: true })
  const graph: GraphModel = { nodes: [], edges: [], learningRate: .01 }
  const add = (id: string, type: NodeType, params: NodeParams, inputs: Array<string | [string, number]> = []) => {
    graph.nodes.push({ ...createNode(type, graph.nodes.length), id, label: id, params })
    inputs.forEach((source, inputSlot) => graph.edges.push({ id: `${id}-${inputSlot}`, source: typeof source === 'string' ? source : source[0], sourceSlot: typeof source === 'string' ? 0 : source[1], target: id, inputSlot }))
  }
  add('data', 'dataset', { dataset: 'custom-text', textData: data, datasetMode: 'sample' })
  add('embedding', 'weight', { trainable: false, value: { shape: [data.vocabulary.length, 2], data: data.vocabulary.flatMap(word => word === 'a' ? [1, 0] : word === 'b' ? [0, 4] : word === 'c' ? [4, 0] : [0, 0]) } })
  add('residual', 'embedding', {}, ['embedding', ['data', 0]])
  add('decoder', 'weight', { trainable: false, constraint: 'unit-row-norm', value: { shape: [2, 2], data: [1, 0, 0, 1] } })
  add('direction', 'slice', { axis: 0, start: 1, end: 2 }, ['decoder'])
  add('strength', 'input', { value: 0 })
  add('delta', 'arithmetic', { expression: 'x1 * x2', inputCount: 2 }, ['strength', 'direction'])
  add('intervened', 'add', { inputCount: 2 }, ['residual', 'delta'])
  add('head', 'weight', { trainable: false, value: { shape: [2, data.vocabulary.length], data: [...data.vocabulary.map(word => Number(word === 'b')), ...data.vocabulary.map(word => Number(word === 'c'))] } })
  add('logits', 'matmul', {}, ['intervened', 'head'])
  add('loss', 'loss', { loss: 'cross-entropy' }, ['logits', ['data', 2]])
  const baseline = structuredClone(graph)
  baseline.edges.find(edge => edge.target === 'logits' && edge.inputSlot === 0)!.source = 'residual'
  for (const prompt of ['a', 'b', 'c', 'a b']) expect(predictText(graph, prompt).logits).toEqual(predictText(baseline, prompt).logits)
  expect(generateText(graph, 'a', 3)).toBe('a b c b')
  graph.nodes.find(node => node.id === 'strength')!.params.value = 2
  const before = JSON.stringify(graph)
  expect(generateText(graph, 'a', 3)).toBe('a c b c')
  const first = [...textGenerationStates(graph, 'a', 1)][0]
  const second = [...textGenerationStates(graph, first, 1)][0]
  expect(predictText(graph, first).logits).not.toEqual(predictText(graph, 'a').logits)
  expect([...textGenerationStates(graph, second, 1)][0].text).toBe('a c b c')
  expect(JSON.stringify(graph)).toBe(before)
})
