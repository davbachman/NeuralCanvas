import { expect, it } from 'vitest'
import { decodeTokens, encodeText, prepareTextDocuments, tokenize } from './textData'
import { generateText, predictText, textGenerationStates, textSamplingDistribution } from './textGeneration'
import { generationConsistencyModel } from '../test/generationConsistencyModel'

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
