import { describe, expect, it } from 'vitest'
import { backwardPass, forwardPass, runTrainingStep } from './engine'
import { createNode, createStarterGraph } from './examples'
import { scalarValue } from './tensor'
import type { GraphModel } from './types'

const partialForward = (graph: GraphModel) => forwardPass(graph, true, { allowPartial: true })

describe('partial forward passes', () => {
  it.each([false, true])('preserves complete forward results with dataset sources: %s', datasetBacked => {
    const graph = createStarterGraph(datasetBacked)
    expect(partialForward(graph)).toEqual(forwardPass(graph))
  })

  it('computes predictions with an unconnected loss and no loss value', () => {
    const graph = createStarterGraph(true)
    const expected = forwardPass(graph)
    graph.edges = graph.edges.filter(edge => edge.target !== 'loss')
    const snapshot = structuredClone(graph)

    const result = partialForward(graph)

    expect(result.graph.nodes.find(node => node.id === 'pred')?.value).toEqual(expected.graph.nodes.find(node => node.id === 'pred')?.value)
    expect(result.graph.nodes.find(node => node.id === 'loss')?.value).toBeUndefined()
    expect(result.loss).toBeUndefined()
    expect(result.steps.map(step => step.nodeId)).toEqual(['mul', 'add', 'pred'])
    expect(graph).toEqual(snapshot)
    expect(() => forwardPass(graph)).toThrow(/loss/)
    expect(() => runTrainingStep(graph)).toThrow(/loss/)
  })

  it('ignores an extra unconnected loss while reporting the computed loss', () => {
    const graph = createStarterGraph(true)
    const expected = forwardPass(graph)
    graph.nodes.unshift(createNode('loss', 99))

    const result = partialForward(graph)

    expect(result.loss).toBe(expected.loss)
    expect(result.graph.nodes[0].value).toBeUndefined()
    expect(result.steps).toEqual(expected.steps)
    expect(() => backwardPass(result.graph)).toThrow(/one loss/)
  })

  it('stops at missing inputs, clears stale downstream state, and continues independent branches', () => {
    const graph = forwardPass(createStarterGraph(true)).graph
    graph.edges = graph.edges.filter(edge => edge.id !== 'w-mul')
    graph.nodes.push(
      { id: 'waiting', type: 'input', label: 'Waiting input', position: { x: 1500, y: 0 }, params: { value: 999 }, value: scalarValue(999) },
      { ...createNode('activation', 99), id: 'independent' },
    )
    graph.edges.push(
      { id: 'mul-waiting', source: 'mul', target: 'waiting', inputSlot: 0, value: scalarValue(999) },
      { id: 'independent-input', source: 'starter-data', target: 'independent', sourceSlot: 0, inputSlot: 0 },
    )
    const snapshot = structuredClone(graph)

    const result = partialForward(graph)

    for (const id of ['mul', 'add', 'pred', 'loss', 'waiting']) {
      const node = result.graph.nodes.find(node => node.id === id)!
      expect(node.value, id).toBeUndefined()
      expect(node.grad, id).toBeUndefined()
      expect(node.cache, id).toBeUndefined()
      expect(node.localDerivative, id).toBeUndefined()
    }
    for (const id of ['mul-add', 'add-act', 'act-loss', 'mul-waiting']) {
      const edge = result.graph.edges.find(edge => edge.id === id)!
      expect(edge.value, id).toBeUndefined()
      expect(edge.grad, id).toBeUndefined()
    }
    expect(result.graph.edges.find(edge => edge.id === 'x-mul')?.value).toEqual(result.graph.nodes.find(node => node.id === 'x')?.value)
    expect(result.graph.edges.find(edge => edge.id === 'target-loss')?.value).toBeDefined()
    expect(result.graph.nodes.find(node => node.id === 'independent')?.value).toBeDefined()
    expect(result.steps.map(step => step.nodeId)).toEqual(['independent'])
    expect(result.loss).toBeUndefined()
    expect(graph).toEqual(snapshot)
  })

  it.each(['invalid-output', 'unknown-source', 'shape-mismatch', 'invalid-parameter', 'cycle'] as const)(
    'keeps dataset values flowing on valid branches alongside a branch with %s', problem => {
      const graph = createStarterGraph(true)
      const expected = forwardPass(graph)
      const blocked = { ...createNode('activation', 99), id: 'blocked' }
      graph.nodes.push(blocked)
      graph.edges.push({ id: 'blocked-input', source: 'starter-data', target: 'blocked', inputSlot: 0 })
      if (problem === 'invalid-output') graph.edges.at(-1)!.sourceSlot = 999
      if (problem === 'unknown-source') graph.edges.at(-1)!.source = 'missing'
      if (problem === 'shape-mismatch') {
        blocked.type = 'reshape'
        blocked.params = { shape: [1000] }
      }
      if (problem === 'invalid-parameter') {
        blocked.type = 'dropout'
        blocked.params = { dropoutRate: 1 }
      }
      if (problem === 'cycle') graph.edges.at(-1)!.source = 'blocked'

      const result = partialForward(graph)

      expect(result.loss).toBe(expected.loss)
      expect(result.graph.nodes.find(node => node.id === 'pred')?.value).toEqual(expected.graph.nodes.find(node => node.id === 'pred')?.value)
      expect(result.graph.nodes.find(node => node.id === 'blocked')?.value).toBeUndefined()
      expect(result.steps.map(step => [step.nodeId, step.phase, step.calculation])).toEqual(expected.steps.map(step => [step.nodeId, step.phase, step.calculation]))
    },
  )
})
