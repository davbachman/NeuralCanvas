import { expect, it } from 'vitest'
import { CONTAINMENT_SCALE, compactVisualHierarchy, layoutContinuousScene, routeContinuousScene, sceneAtPositions, sceneContentBounds, sceneGroupId } from './continuousScene'
import { captureCanvasLayout } from './deletionLayout'
import { cloneGraph, forwardPass, parameterValues } from './engine'
import { createStarterGraph } from './examples'
import { mergePreservingLayout, ungroupPreservingLayout } from './mergeLayout'
import { createModelPreset } from './modelPresets'
import { LESSONS } from '../learning/presets'
import type { SemanticRect } from './semanticLayout'
import type { GraphModel } from './types'

const rectEqual = (actual: SemanticRect, expected: SemanticRect) => {
  for (const key of ['x', 'y', 'width', 'height'] as const) expect(actual[key]).toBeCloseTo(expected[key], 8)
}

it.each(LESSONS)('uses only visible containment depth to determine every block scale in $id', ({ id }) => {
  const graph = compactVisualHierarchy(createModelPreset(id)), scene = layoutContinuousScene(graph)
  for (const [key, scale] of scene.scales) {
    let parent = scene.parents.get(key), depth = 0
    while (parent) { depth++; parent = scene.parents.get(sceneGroupId(parent)) }
    expect(scale).toBe(CONTAINMENT_SCALE ** depth)
  }
})

it.each([{ x: 450, y: 300 }, { x: -450, y: -300 }])('grows nested frames around moved contents without rescaling or moving other blocks: %j', delta => {
  const outer = mergePreservingLayout(createStarterGraph(), ['x', 'w', 'mul', 'b', 'add', 'pred'])
  const inner = mergePreservingLayout(outer.graph, ['x', 'w', 'mul'])
  const graph = inner.graph, before = layoutContinuousScene(graph), original = structuredClone(graph)
  const source = before.nodes.get('w')!, destination = { x: source.x + delta.x, y: source.y + delta.y }
  const live = sceneAtPositions(before, new Map([['w', destination]]))
  const edited: GraphModel = { ...graph, view: { ...graph.view!, layoutOffsets: { w: delta } } }
  const after = layoutContinuousScene(edited)
  rectEqual(after.nodes.get('w')!, { ...source, ...destination })
  for (const [id, rect] of before.nodes) if (id !== 'w') rectEqual(after.nodes.get(id)!, rect)
  for (const id of [inner.group!.id, outer.group!.id]) {
    const frame = after.groups.get(id)!, previous = before.groups.get(id)!
    expect(frame.width).toBeGreaterThan(previous.width)
    expect(frame.height).toBeGreaterThan(previous.height)
    const bounds = sceneContentBounds(after, id), moved = after.nodes.get('w')!
    expect(moved.x).toBeGreaterThanOrEqual(bounds.x - 1e-9)
    expect(moved.y).toBeGreaterThanOrEqual(bounds.y - 1e-9)
    expect(moved.x + moved.width).toBeLessThanOrEqual(bounds.x + bounds.width + 1e-9)
    expect(moved.y + moved.height).toBeLessThanOrEqual(bounds.y + bounds.height + 1e-9)
    rectEqual(live.groups.get(id)!, frame)
  }
  expect(after.scales).toEqual(before.scales)
  for (let count = 0, current = edited; count < 4; count++) {
    current = { ...current, view: captureCanvasLayout(current) }
    const restored = layoutContinuousScene(cloneGraph(JSON.parse(JSON.stringify(current))))
    for (const [id, rect] of after.nodes) rectEqual(restored.nodes.get(id)!, rect)
    for (const [id, rect] of after.groups) rectEqual(restored.groups.get(id)!, rect)
    expect(restored.scales).toEqual(after.scales)
  }
  const ungrouped = layoutContinuousScene(ungroupPreservingLayout(edited, inner.group!.id))
  expect(ungrouped.scales.get('w')).toBe(CONTAINMENT_SCALE)
  expect(ungrouped.nodes.get('w')!.width).toBeCloseTo(source.width / CONTAINMENT_SCALE, 8)
  expect(forwardPass(edited).loss).toEqual(forwardPass(graph).loss)
  expect(parameterValues(edited)).toEqual(parameterValues(graph))
  expect(graph).toEqual(original)
})

it.each([1, -1])('shrinks saved nested containers as their contents move closer together (direction %s)', direction => {
  const outer = mergePreservingLayout(createStarterGraph(), ['x', 'w', 'mul', 'b', 'add', 'pred'])
  const inner = mergePreservingLayout(outer.graph, ['x', 'w', 'mul'])
  const original = layoutContinuousScene(inner.graph), source = original.nodes.get('w')!
  const spread: GraphModel = { ...inner.graph, view: { ...inner.graph.view!, layoutOffsets: { w: { x: 450 * direction, y: 300 * direction } } } }
  let graph: GraphModel = { ...spread, view: captureCanvasLayout(spread) }
  const expanded = layoutContinuousScene(graph)
  const parameters = parameterValues(graph), model = structuredClone(graph)
  // Simulate several inward drag updates, then capture/reopen the result as
  // subsequent edits, save files, and local recovery do.
  for (const fraction of [.75, .5, 0]) {
    const current = layoutContinuousScene(graph), start = current.nodes.get('w')!
    const destination = { x: source.x + 450 * direction * fraction, y: source.y + 300 * direction * fraction }
    const delta = { x: destination.x - start.x, y: destination.y - start.y }
    const live = sceneAtPositions(current, new Map([['w', destination]]))
    const edited = { ...graph, view: { ...graph.view!, layoutOffsets: { w: delta } } }
    const after = layoutContinuousScene(edited)
    rectEqual(after.nodes.get('w')!, { ...source, ...destination })
    for (const [id, rect] of current.nodes) if (id !== 'w') rectEqual(after.nodes.get(id)!, rect)
    for (const id of [inner.group!.id, outer.group!.id]) {
      const frame = after.groups.get(id)!, previous = current.groups.get(id)!
      expect(frame.width).toBeLessThan(previous.width)
      expect(frame.height).toBeLessThan(previous.height)
      rectEqual(live.groups.get(id)!, frame)
    }
    expect(after.scales).toEqual(original.scales)
    graph = cloneGraph(JSON.parse(JSON.stringify({ ...edited, view: captureCanvasLayout(edited) })))
    const reopened = layoutContinuousScene(graph)
    for (const kind of ['nodes', 'groups'] as const) for (const [id, rect] of after[kind]) rectEqual(reopened[kind].get(id)!, rect)
    expect(parameterValues(graph)).toEqual(parameters)
    expect(forwardPass(graph).loss).toEqual(forwardPass(model).loss)
    // Resized boundary ports remain shared by the internal/external wires.
    const wires = routeContinuousScene(graph, reopened)
    for (const wire of wires.filter(wire => wire.parentId)) {
      const frame = reopened.groups.get(wire.parentId!)!
      for (const point of wire.route) {
        expect(point.x).toBeGreaterThanOrEqual(frame.x - 1e-7)
        expect(point.x).toBeLessThanOrEqual(frame.x + frame.width + 1e-7)
        expect(point.y).toBeGreaterThanOrEqual(frame.y - 1e-7)
        expect(point.y).toBeLessThanOrEqual(frame.y + frame.height + 1e-7)
      }
    }
  }
  const restored = layoutContinuousScene(graph)
  for (const [id, rect] of original.groups) rectEqual(restored.groups.get(id)!, rect)
  expect(expanded.groups.get(outer.group!.id)!.width).toBeGreaterThan(restored.groups.get(outer.group!.id)!.width)
})
