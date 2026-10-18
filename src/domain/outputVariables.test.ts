import { expect, it } from 'vitest'
import { createStarterGraph } from './examples'
import { computedOutputLabels, formulaForNode, forwardPass, outputLabelForFormula } from './engine'

it('shares parameter variables with formulas and keeps numbering through moving, grouping, and viewing', () => {
  const graph = createStarterGraph()
  const before = structuredClone(graph)
  const names = computedOutputLabels(graph)
  expect(names.get('w')).toBe('p1')
  expect(names.get('b')).toBe('p2')
  expect(formulaForNode(graph.nodes.find(node => node.id === 'mul')!, graph)).toBe('z1 = x * p1')
  expect(formulaForNode(graph.nodes.find(node => node.id === 'add')!, graph)).toBe('z2 = z1 + p2')
  expect(formulaForNode(graph.nodes.find(node => node.id === 'w')!, graph)).toBe('p1 = 0.500')
  expect(outputLabelForFormula(graph.nodes.find(node => node.id === 'target'), graph)).toBe('y')
  expect(outputLabelForFormula(graph.nodes.find(node => node.id === 'loss'), graph)).toBe('L')
  expect(graph).toEqual(before)
  const moved = { ...graph, nodes: graph.nodes.map(node => ({ ...node, position: { x: -node.position.x, y: -node.position.y } })), view: { expandedGroupIds: ['group'], viewport: { x: 100, y: 200, zoom: .5 } } }
  expect(computedOutputLabels(moved)).toEqual(names)
})

it('avoids collisions with source and custom output names', () => {
  const graph = createStarterGraph()
  graph.nodes.find(node => node.id === 'x')!.label = 'p1'
  graph.nodes.find(node => node.id === 'mul')!.params.outputName = 'p2'
  const names = computedOutputLabels(graph)
  expect(names.get('mul')).toBe('p2')
  expect(names.get('w')).toBe('p3')
  expect(names.get('b')).toBe('p4')
  expect(new Set(names.values()).size).toBe(names.size)
})

it('sets and clears custom parameter names without changing values or execution', () => {
  const graph = createStarterGraph()
  const weight = graph.nodes.find(node => node.id === 'w')!
  const bias = graph.nodes.find(node => node.id === 'b')!
  const values = graph.nodes.map(node => node.params.value)
  const predictions = forwardPass(graph, false).graph.nodes.map(node => node.value)
  weight.params.outputName = 'kernel'
  expect(computedOutputLabels(graph).get('w')).toBe('kernel')
  expect(computedOutputLabels(graph).get('b')).toBe('p2')
  expect(formulaForNode(graph.nodes.find(node => node.id === 'mul')!, graph)).toBe('z1 = x * kernel')
  expect(outputLabelForFormula(weight)).toBe('kernel')
  bias.params.outputName = 'kernel'
  expect(computedOutputLabels(graph).get('b')).toBe('kernel_2')
  expect(forwardPass(graph, false).graph.nodes.map(node => node.value)).toEqual(predictions)
  expect(graph.nodes.map(node => node.params.value)).toEqual(values)
  weight.params.outputName = undefined
  bias.params.outputName = undefined
  expect(computedOutputLabels(graph).get('w')).toBe('p1')
  expect(computedOutputLabels(graph).get('b')).toBe('p2')
  weight.params.outputName = 'p2'
  expect(computedOutputLabels(graph).get('w')).toBe('p2')
  expect(computedOutputLabels(graph).get('b')).toBe('p3')
})
