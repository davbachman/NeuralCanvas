import { expect, it } from 'vitest'
import { createStarterGraph } from './examples'
import { computedOutputLabels, formulaForNode, outputLabelForFormula } from './engine'

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
