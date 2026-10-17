import { forwardPass, isLossNode } from './engine'
import { predictionNode } from './datasetTraining'
import { encodeText, decodeTokens } from './textData'
import { samplingDistribution } from '../learning/decoder'
import type { GraphModel } from './types'

export interface TextGenerationState { text: string; tokenIds: number[] }
export interface TextGenerationOptions { temperature?: number; topK?: number; sample?: boolean; random?: () => number }

/** Padding is input scaffolding, never generated text. Filter before top-k so
 * even a highest-scoring pad cannot displace a real candidate. */
export function textSamplingDistribution(logits: number[], vocabulary: string[], temperature = 1, topK = logits.length): number[] {
  if (logits.length !== vocabulary.length || logits.some(n => !Number.isFinite(n))) throw new Error('Generation needs finite vocabulary logits.')
  const eligible = logits.map((value, id) => ({value, id})).filter(({id}) => vocabulary[id] !== '<pad>')
  const probabilities = samplingDistribution(eligible.map(({value}) => value), temperature, topK)
  const result = logits.map(() => 0)
  eligible.forEach(({id}, i) => { result[id] = probabilities[i] })
  return result
}

/** Prompts are inference-only: evaluate a copy without Loss, never invent targets. */
export function predictText(graph: GraphModel, prompt: string | TextGenerationState) {
  return predictTokenIds(graph, typeof prompt === 'string' ? prompt : prompt.tokenIds)
}

function predictTokenIds(graph: GraphModel, prompt: string | number[]) {
  const source = graph.nodes.find(node => node.type === 'dataset' && node.params.textData?.task === 'language')
  const data = source?.params.textData
  if (!source || !data) throw new Error('Import a next-token text dataset first.')
  const output = predictionNode(graph)
  if (!output) throw new Error('Connect vocabulary logits to a Loss block first.')
  const visited = new Set<string>()
  const checkInputs = (id: string) => {
    if (visited.has(id)) return
    visited.add(id)
    for (const edge of graph.edges.filter(edge => edge.target === id)) {
      if (edge.source === source.id && edge.sourceSlot === 2) throw new Error('Generation output must not depend on dataset targets.')
      checkInputs(edge.source)
    }
  }
  checkInputs(output.id)
  const allIds = typeof prompt === 'string' ? encodeText(prompt, data) : prompt
  if (!allIds.length) throw new Error('Enter a nonempty prompt.')
  const contextIds = allIds.slice(-data.maxLength)
  let ids = contextIds
  if (data.fixedLength && ids.length<data.maxLength) ids=[...Array(data.maxLength-ids.length).fill(1),...ids]
  const predictionGraph: GraphModel = { ...graph, nodes: graph.nodes.filter(node => !isLossNode(node)).map(node => node.id === source.id ? {...node, params: {...node.params, datasetValues: [{shape: [ids.length], data: ids}, {shape: [ids.length], data: ids.map((_, i) => i)}]}} : node), edges: graph.edges.filter(edge => !graph.nodes.some(node => isLossNode(node) && (node.id === edge.target || node.id === edge.source))) }
  const evaluated = forwardPass(predictionGraph, false).graph
  const value = evaluated.nodes.find(node => node.id === output.id)?.value
  if (!value || value.shape.length !== 2 || value.shape[0] !== (data.targetMode==='last'?1:ids.length) || value.shape[1] !== data.vocabulary.length || value.data.some(n => !Number.isFinite(n))) throw new Error('Generation needs logits shaped [context tokens, vocabulary size].')
  return { logits: value.data.slice(-data.vocabulary.length), ids, vocabulary: data.vocabulary, data, contextIds }
}

/** Retain the actual IDs across UI clicks; rendered characters can themselves
 * spell a reserved marker and therefore are not a lossless token container. */
export function* textGenerationStates(graph: GraphModel, prompt: string | TextGenerationState, count: number, options: TextGenerationOptions = {}): Generator<TextGenerationState> {
  if (!Number.isInteger(count) || count < 1 || count > 256) throw new Error('Generate 1–256 tokens at a time.')
  let text = typeof prompt === 'string' ? prompt : prompt.text
  let context: string | number[] = typeof prompt === 'string' ? prompt : prompt.tokenIds
  for (let i = 0; i < count; i++) {
    const {logits, data, contextIds} = predictTokenIds(graph, context)
    const probabilities = textSamplingDistribution(logits, data.vocabulary, options.sample ? options.temperature ?? 1 : 1, options.sample ? options.topK ?? logits.length : logits.length)
    let id = probabilities.indexOf(Math.max(...probabilities))
    if (options.sample) {
      let remaining = (options.random ?? Math.random)()
      // The fallback must also be an eligible token (rounding can leave a residue).
      id = probabilities.findLastIndex(probability => probability > 0)
      for (let j = 0; j < probabilities.length; j++) { remaining -= probabilities[j]; if (remaining < 0) {id = j; break} }
    }
    text += (data.tokenizer === 'word' ? ' ' : '') + decodeTokens([id], data)
    context = [...contextIds, id]
    yield {text, tokenIds: context}
  }
}

export function* textGenerationSteps(graph: GraphModel, prompt: string, count: number, options: TextGenerationOptions = {}): Generator<string> {
  for (const state of textGenerationStates(graph, prompt, count, options)) yield state.text
}

export function generateText(graph: GraphModel, prompt: string, count: number, options: TextGenerationOptions = {}): string {
  let text = prompt
  for (const next of textGenerationSteps(graph, prompt, count, options)) text = next
  return text
}
