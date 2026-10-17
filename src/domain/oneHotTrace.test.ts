import { expect, it } from 'vitest'
import { backwardPass, forwardPass, parameterValues } from './engine'
import { oneHotModel } from '../test/oneHotModel'

it('explains One-hot encoding without changing predictions or discrete-ID gradients', () => {
  const graph = oneHotModel(), before = structuredClone(graph)
  const untraced = forwardPass(graph, false), traced = forwardPass(graph, true)
  expect(traced.graph).toEqual(untraced.graph)
  expect(traced.loss).toBe(untraced.loss)
  const step = traced.steps.find(step => step.nodeId === 'encoding')!
  expect(step.calculation).toContain('3-class one-hot rows')
  expect(step.calculation).toContain('1 at its token ID and 0 elsewhere')
  expect(step.calculation).toContain('discrete constants')
  expect(traced.graph.nodes.find(node => node.id === 'encoding')!.value).toEqual({ shape: [3, 3], data: [0, 0, 1, 1, 0, 0, 0, 0, 1] })
  const backward = backwardPass(traced.graph)
  expect(backward.graph.nodes.find(node => node.id === 'ids')!.grad!.data).toEqual([0, 0, 0])
  expect(backward.steps.find(step => step.nodeId === 'encoding')!.formula).toContain('no gradient')
  expect(graph).toEqual(before)
})

it('matches Embedding predictions, loss, and repeated-token table gradients with tracing enabled', () => {
  const encoded = forwardPass(oneHotModel(), true), lookup = forwardPass(oneHotModel(true), true)
  for (const id of ['vectors', 'pooled', 'logits', 'probability']) {
    expect(encoded.graph.nodes.find(node => node.id === id)!.value).toEqual(lookup.graph.nodes.find(node => node.id === id)!.value)
  }
  expect(encoded.loss).toBe(lookup.loss)
  const a = backwardPass(encoded.graph).graph, b = backwardPass(lookup.graph).graph
  for (const id of ['table', 'head', 'ids']) expect(a.nodes.find(node => node.id === id)!.grad).toEqual(b.nodes.find(node => node.id === id)!.grad)
  expect(a.nodes.find(node => node.id === 'table')!.grad!.data.some(value => value !== 0)).toBe(true)
  expect(parameterValues(a)).toEqual(parameterValues(b))
})
