import { renderHook } from '@testing-library/react'
import { expect, it } from 'vitest'
import { useModelGraph } from './useModelGraph'
import { canvasPerformanceModel } from '../test/canvasPerformanceModel'

it('keeps model derivations stable across viewport-only updates, but observes edits and group navigation', () => {
  const graph = canvasPerformanceModel()
  const { result, rerender } = renderHook(useModelGraph, { initialProps: graph })
  const original = result.current
  for (let x = 0; x < 10; x++) {
    rerender({ ...graph, view: { ...graph.view!, expandedGroupIds: [], viewport: { x, y: 50, zoom: 1 + x / 10 } } })
    expect(result.current).toBe(original)
  }
  expect(graph.view!.viewport).toEqual({ x: 40, y: 30, zoom: .55 })
  rerender({ ...graph, view: { ...graph.view!, expandedGroupIds: ['group'] } })
  expect(result.current).not.toBe(original)
  rerender({ ...graph, nodes: graph.nodes.map(n => n.id === 'w1' ? { ...n, params: { ...n.params, value: { shape: [1], data: [1] } } } : n) })
  expect(result.current.nodes).not.toBe(original.nodes)
})
