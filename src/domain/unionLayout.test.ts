import { expect, it } from 'vitest'
import { CONTAINMENT_SCALE, compactVisualHierarchy, layoutContinuousScene } from './continuousScene'
import { forwardPass, parameterValues } from './engine'
import { createStarterGraph } from './examples'
import { mergePreservingLayout, unionPreservingLayout } from './mergeLayout'
import { groupAncestors, visualGroupInterface } from './grouping'

function fixture(nested: boolean, mixed: boolean) {
  let graph = createStarterGraph(true)
  if (nested) graph = mergePreservingLayout(graph, ['x', 'w', 'mul', 'b', 'add', 'pred']).graph
  const first = mergePreservingLayout(graph, ['x', 'w', 'mul'])
  const child = mergePreservingLayout(first.graph, ['x', 'w'])
  graph = child.graph
  let secondId: string | undefined
  if (!mixed) {
    const second = mergePreservingLayout(graph, ['b', 'add', 'pred'])
    graph = second.graph
    secondId = second.group!.id
  }
  return { graph, child: child.group!.id, first: first.group!.id,
    groups: secondId ? [first.group!.id, secondId] : [first.group!.id], nodes: mixed ? ['b'] : [] }
}

it.each([{ nested: false, mixed: false }, { nested: true, mixed: false }, { nested: false, mixed: true }, { nested: true, mixed: true }])(
  'distinguishes an outer merge from a union, preserving nested blocks and model values: %j', ({ nested, mixed }) => {
    const { graph, child, first, groups, nodes } = fixture(nested, mixed)
    const original = structuredClone(graph), before = layoutContinuousScene(compactVisualHierarchy(graph))
    const merge = mergePreservingLayout(graph, nodes, graph, groups)
    const union = unionPreservingLayout(graph, nodes, groups)
    expect(merge.group).toBeDefined()
    expect(union.group).toBeDefined()
    const parent = graph.groups!.find(group => group.id === first)!.parentId
    expect(merge.group!.parentId).toBe(parent)
    expect(union.group!.parentId).toBe(parent)
    for (const id of groups) {
      expect(merge.graph.groups!.find(group => group.id === id)!.parentId).toBe(merge.group!.id)
      expect(union.graph.groups!.some(group => group.id === id)).toBe(false)
      expect(union.graph.view!.preservedLayouts![id]).toBeUndefined()
      expect(union.graph.view!.preservedLayoutBounds![id]).toBeUndefined()
      expect(union.graph.view!.preservedGroupTransforms![id]).toBeUndefined()
    }
    expect(merge.graph.groups!.find(group => group.id === child)!.parentId).toBe(first)
    expect(union.graph.groups!.find(group => group.id === child)!.parentId).toBe(union.group!.id)
    const mergedScene = layoutContinuousScene(compactVisualHierarchy(merge.graph))
    const unionScene = layoutContinuousScene(compactVisualHierarchy(union.graph))
    const selected = new Set(merge.group!.nodeIds)
    for (const node of graph.nodes) {
      const rect = before.nodes.get(node.id)!, combined = unionScene.nodes.get(node.id)!
      if (selected.has(node.id)) {
        expect(mergedScene.scales.get(node.id)).toBe(before.scales.get(node.id)! * CONTAINMENT_SCALE)
        expect(unionScene.scales.get(node.id)).toBe(before.scales.get(node.id)! * (nodes.includes(node.id) ? CONTAINMENT_SCALE : 1))
        // Existing contents keep their positions when their wrappers are united.
        expect(combined.x).toBeCloseTo(rect.x, 8)
        expect(combined.y).toBeCloseTo(rect.y, 8)
      } else expect(combined).toEqual(rect)
    }
    for (const result of [merge, union]) {
      expect(result.graph.nodes).toBe(graph.nodes)
      expect(result.graph.edges).toBe(graph.edges)
      expect(parameterValues(result.graph)).toEqual(parameterValues(graph))
      expect(forwardPass(result.graph).loss).toBe(forwardPass(graph).loss)
      const ids = new Set<string>()
      for (const group of result.graph.groups!) {
        expect(ids.has(group.id)).toBe(false)
        ids.add(group.id)
        expect(groupAncestors(result.graph, group.id).at(-1)!.id).toBe(group.id)
        const parentGroup = result.graph.groups!.find(candidate => candidate.id === group.parentId)
        if (parentGroup) expect(group.nodeIds.every(id => parentGroup.nodeIds.includes(id))).toBe(true)
      }
      expect(visualGroupInterface(result.graph, result.group!).inputs.length).toBeGreaterThan(0)
      const reloaded = layoutContinuousScene(compactVisualHierarchy(JSON.parse(JSON.stringify(result.graph))))
      const expected = result === merge ? mergedScene : unionScene
      expect(reloaded.nodes).toEqual(expected.nodes)
      expect(reloaded.parents).toEqual(expected.parents)
    }
    expect(graph).toEqual(original)
  },
)

it('does not combine a selected container with its own child or a selection crossing container boundaries', () => {
  const { graph, first, child } = fixture(true, false)
  expect(unionPreservingLayout(graph, ['w'], [first]).graph).toBe(graph)
  expect(unionPreservingLayout(graph, ['b'], [child]).graph).toBe(graph)
})
