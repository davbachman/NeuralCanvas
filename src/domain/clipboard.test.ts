import type { GraphModel } from './types'
import { describe, expect, it } from 'vitest'
import { clipboardPasteOffset, copyGraphSelection, pasteGraphClipboard } from './clipboard'
import { compactVisualHierarchy, layoutContinuousScene, sceneGroupId } from './continuousScene'
import { mergePreservingLayout } from './mergeLayout'
import { createEmptyGraph, createNode, createStarterGraph } from './examples'
import { mergeNodesIntoVisualGroup, visualGroupInterface } from './grouping'

describe('graph clipboard', () => {
  it('copies selected nodes and pastes them with remapped ids and internal edges only', () => {
    const graph = createStarterGraph()

    const fragment = copyGraphSelection(graph, { nodeIds: ['x', 'w', 'mul'] })
    const result = pasteGraphClipboard(graph, fragment!, { x: 40, y: 50 })

    expect(fragment).toBeDefined()
    expect(result.selection.groupId).toBeUndefined()
    expect(result.selection.nodeIds).toHaveLength(3)
    expect(result.graph.nodes).toHaveLength(graph.nodes.length + 3)
    expect(result.graph.edges).toHaveLength(graph.edges.length + 2)

    const pastedNodes = result.selection.nodeIds.map((nodeId) => result.graph.nodes.find((node) => node.id === nodeId))
    expect(pastedNodes.map((node) => node?.label)).toEqual(['x1', 'w1', 'x * w'])
    expect(pastedNodes.map((node) => node?.position)).toEqual([
      { x: 80, y: 110 },
      { x: 80, y: 310 },
      { x: 360, y: 210 },
    ])

    const pastedNodeIds = new Set(result.selection.nodeIds)
    const pastedEdges = result.graph.edges.filter((edge) => pastedNodeIds.has(edge.source) && pastedNodeIds.has(edge.target))
    expect(pastedEdges).toHaveLength(2)
    expect(pastedEdges.map((edge) => edge.inputSlot)).toEqual([0, 1])
    expect(result.graph.edges.some((edge) => pastedNodeIds.has(edge.source) && edge.target === 'add')).toBe(false)
  })

  it('copies a selected visual group and pastes it with the next group label and boundary handles', () => {
    const groupedGraph = mergeNodesIntoVisualGroup(createStarterGraph(), ['mul', 'add']).graph

    const fragment = copyGraphSelection(groupedGraph, { nodeIds: [], groupId: 'group-1' })
    const result = pasteGraphClipboard(groupedGraph, fragment!, { x: 25, y: 35 })

    expect(fragment).toBeDefined()
    expect(result.selection.nodeIds).toEqual([])
    expect(result.selection.groupId).toBe('group-2')
    expect(result.graph.nodes).toHaveLength(groupedGraph.nodes.length + 2)
    expect(result.graph.edges).toHaveLength(groupedGraph.edges.length + 4)
    expect(result.graph.groups).toHaveLength(2)

    const pastedGroup = result.graph.groups?.find((group) => group.id === result.selection.groupId)
    expect(pastedGroup).toBeDefined()
    expect(pastedGroup?.label).toBe('Group 2')
    expect(pastedGroup?.nodeIds).toHaveLength(2)
    expect(pastedGroup?.position).toEqual({
      x: groupedGraph.groups![0].position.x + 25,
      y: groupedGraph.groups![0].position.y + 35,
    })

    const pastedGroupNodeIds = new Set(pastedGroup?.nodeIds)
    const pastedInternalEdges = result.graph.edges.filter(
      (edge) => pastedGroupNodeIds.has(edge.source) && pastedGroupNodeIds.has(edge.target),
    )
    expect(pastedInternalEdges).toHaveLength(1)

    const pastedInterface = visualGroupInterface(result.graph, pastedGroup!)
    expect(pastedInterface.inputs).toHaveLength(3)
    expect(pastedInterface.outputs).toHaveLength(1)
    expect(result.graph.edges.filter(edge => edge.target === 'pred')).toHaveLength(1)
  })

  it('pastes copied source nodes with the next generated label for that node type', () => {
    const input = createNode('input', 1)
    const graph = { ...createEmptyGraph(), nodes: [input] }

    const fragment = copyGraphSelection(graph, { nodeIds: [input.id] })
    const result = pasteGraphClipboard(graph, fragment!, { x: 20, y: 20 })

    expect(result.selection.nodeIds).toEqual(['input-2'])
    expect(result.graph.nodes.find((node) => node.id === 'input-2')).toEqual(
      expect.objectContaining({
        label: 'x2',
        position: { x: input.position.x + 20, y: input.position.y + 20 },
      }),
    )
  })
})

const sceneFor = (graph: GraphModel) => layoutContinuousScene(compactVisualHierarchy(graph))
const expectRect = (actual: { x: number; y: number; width: number; height: number }, expected: typeof actual) => {
  for (const key of ['x', 'y', 'width', 'height'] as const) expect(actual[key]).toBeCloseTo(expected[key], 8)
}

it.each([false, true])('pastes a moved nested selection at its visible position, copying the whole group: %s', wholeGroup => {
  let graph = mergePreservingLayout(createStarterGraph(), ['x', 'w', 'mul', 'b', 'add', 'pred']).graph
  const merged = mergePreservingLayout(graph, ['mul', 'add'])
  graph = merged.graph
  graph.view!.layoutOffsets = { [sceneGroupId('group-1')]: { x: 2200, y: -900 }, mul: { x: 3, y: 2 } }
  const original = structuredClone(graph), before = sceneFor(graph)
  const fragment = copyGraphSelection(graph, wholeGroup ? { nodeIds: [], groupId: merged.group!.id } : { nodeIds: ['mul', 'add'] })!
  const offset = { x: 30, y: 25 }
  const pasted = pasteGraphClipboard(graph, fragment, offset)
  const after = sceneFor(pasted.graph)
  for (const [id, rect] of before.nodes) expectRect(after.nodes.get(id)!, rect)
  for (const [id, rect] of before.groups) expectRect(after.groups.get(id)!, rect)
  for (const [index, source] of fragment.nodes.entries()) {
    const copy = pasted.graph.nodes[graph.nodes.length + index]
    const rect = before.nodes.get(source.id)!
    expectRect(after.nodes.get(copy.id)!, { ...rect, x: rect.x + offset.x, y: rect.y + offset.y })
    expect(after.scales.get(copy.id)).toBeCloseTo(before.scales.get(source.id)!, 8)
    expect(copy.params).toEqual(source.params)
  }
  if (wholeGroup) {
    const rect = before.groups.get(merged.group!.id)!
    expectRect(after.groups.get(pasted.selection.groupId!)!, { ...rect, x: rect.x + offset.x, y: rect.y + offset.y })
  }
  expect(graph).toEqual(original)
  const reloaded = sceneFor(JSON.parse(JSON.stringify(pasted.graph)))
  expect(reloaded.nodes).toEqual(after.nodes)
  expect(reloaded.groups).toEqual(after.groups)
})

it('keeps a pasted module and existing flat blocks in place when entering a grouped canvas', () => {
  const source = mergePreservingLayout(createStarterGraph(), ['mul', 'add'])
  const fragment = copyGraphSelection(source.graph, { nodeIds: [], groupId: source.group!.id })!
  const target = createStarterGraph()
  target.nodes[0].position = { x: -900, y: 1400 }
  const pasted = pasteGraphClipboard(target, fragment, { x: 100, y: 50 })
  const scene = sceneFor(pasted.graph), before = sceneFor(source.graph)
  for (const node of target.nodes) expect(scene.nodes.get(node.id)).toMatchObject(node.position)
  const rect = before.groups.get(source.group!.id)!
  expectRect(scene.groups.get(pasted.selection.groupId!)!, { ...rect, x: rect.x + 100, y: rect.y + 50 })
})

it('preserves automatic layouts and remaps every level of a copied nested module', () => {
  let graph = mergeNodesIntoVisualGroup(createStarterGraph(), ['mul', 'add']).graph
  graph = mergeNodesIntoVisualGroup(graph, ['x', 'w', 'mul', 'b', 'add', 'pred']).graph
  const outer = graph.groups!.find(group => !group.parentId)!
  const before = sceneFor(graph)
  const fragment = copyGraphSelection(graph, { nodeIds: [], groupId: outer.id })!
  const pasted = pasteGraphClipboard(graph, fragment, { x: 400, y: 30 })
  const after = sceneFor(pasted.graph)
  for (const [id, rect] of before.nodes) expectRect(after.nodes.get(id)!, rect)
  for (const [index, source] of fragment.nodes.entries()) {
    const copy = pasted.graph.nodes[graph.nodes.length + index], rect = before.nodes.get(source.id)!
    expectRect(after.nodes.get(copy.id)!, { ...rect, x: rect.x + 400, y: rect.y + 30 })
  }
  const copiedGroups = pasted.graph.groups!.slice(graph.groups!.length)
  expect(copiedGroups).toHaveLength(2)
  expect(copiedGroups[0].parentId).toBeUndefined()
  expect(copiedGroups[1].parentId).toBe(copiedGroups[0].id)
})

it.each([.2, 1, 20])('keeps repeated pastes inside the visible viewport at zoom %s', zoom => {
  const graph = createStarterGraph()
  const fragment = copyGraphSelection(graph, { nodeIds: ['w'] })!
  const viewport = { x: 5000, y: -3000, width: 1000 / zoom, height: 800 / zoom, zoom }
  for (let count = 0; count < 50; count++) {
    const offset = clipboardPasteOffset(graph, fragment, count, viewport)
    const x = fragment.nodes[0].position.x + offset.x, y = fragment.nodes[0].position.y + offset.y
    expect(x).toBeGreaterThanOrEqual(viewport.x)
    expect(x).toBeLessThan(viewport.x + viewport.width)
    expect(y).toBeGreaterThanOrEqual(viewport.y)
    expect(y).toBeLessThan(viewport.y + viewport.height)
    expect(clipboardPasteOffset(graph, fragment, count + 5, viewport)).toEqual(offset)
  }
})

it('uses screen-sized offsets while keeping a multi-selection arrangement intact', () => {
  const graph = createStarterGraph()
  const fragment = copyGraphSelection(graph, { nodeIds: ['x', 'w', 'mul'] })!
  const offset = clipboardPasteOffset(graph, fragment, 0, { x: 0, y: 0, width: 1000, height: 700, zoom: 2 })
  expect(offset).toEqual({ x: 18, y: 18 })
  const pasted = pasteGraphClipboard(graph, fragment, offset)
  for (const [index, node] of fragment.nodes.entries()) {
    expect(pasted.graph.nodes[graph.nodes.length + index].position).toEqual({ x: node.position.x + 18, y: node.position.y + 18 })
  }
})


it('does not reconnect a missing external input to an unrelated newly generated node ID', () => {
  const original: GraphModel = {
    learningRate: .01,
    nodes: [
      { id: 'weight-1', label: 'External parameter', type: 'weight', params: { value: 3 }, position: { x: 0, y: 0 } },
      { id: 'weight-2', label: 'Internal parameter', type: 'weight', params: { value: 4 }, position: { x: 200, y: 0 } },
      { id: 'add', label: 'Sum', type: 'add', params: {}, position: { x: 400, y: 0 } },
    ],
    edges: [{ id: 'external', source: 'weight-1', target: 'add', inputSlot: 0 }, { id: 'internal', source: 'weight-2', target: 'add', inputSlot: 1 }],
    groups: [{ id: 'g', label: 'Building block', nodeIds: ['weight-2', 'add'], position: { x: 200, y: 0 }, dimensions: {} }],
  }
  const fragment = copyGraphSelection(original, { nodeIds: [], groupId: 'g' })!
  const pasted = pasteGraphClipboard({ nodes: [], edges: [], learningRate: .01 }, fragment, { x: 0, y: 0 }).graph
  expect(pasted.nodes.some(node => node.id === 'weight-1')).toBe(true)
  expect(pasted.edges).toHaveLength(1)
  expect(pasted.edges[0].inputSlot).toBe(1)
})
