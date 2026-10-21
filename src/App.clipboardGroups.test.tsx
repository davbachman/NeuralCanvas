import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import { createStarterGraph } from './domain/examples'
import { mergePreservingLayout } from './domain/mergeLayout'
import { compactVisualHierarchy, layoutContinuousScene } from './domain/continuousScene'
import * as session from './domain/session'

afterEach(() => vi.restoreAllMocks())

it.each(['keyboard', 'Paste', 'Duplicate'])('keeps a nested parameter inside its module using %s, including after saving', action => {
  const outer = mergePreservingLayout(createStarterGraph(), ['x', 'w', 'mul', 'b', 'add', 'pred'])
  const inner = mergePreservingLayout(outer.graph, ['x', 'w', 'mul'])
  const download = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  const { container } = render(<App initialGraph={inner.graph} />)
  const menu = (name: string) => {
    fireEvent.click(screen.getByRole('button', { name: 'Edit' }))
    fireEvent.click(screen.getByRole('menuitem', { name }))
  }
  const save = () => {
    fireEvent.click(screen.getByRole('button', { name: 'File' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
    return download.mock.calls.at(-1)![0]
  }
  fireEvent.click(container.querySelector('.react-flow__node[data-id="w"]')!)
  const before = save()
  if (action === 'keyboard') {
    fireEvent.keyDown(document, { key: 'c', ctrlKey: true })
    fireEvent.keyDown(document, { key: 'v', ctrlKey: true })
  } else {
    if (action === 'Paste') menu('Copy')
    menu(action)
  }
  const file = save(), graph = file.state.graph
  const id = file.state.selectedNodeIds[0]
  expect(id).toBe('weight-1')
  for (const group of [inner.group!, outer.group!]) {
    expect(graph.groups!.find(candidate => candidate.id === group.id)!.nodeIds).toContain(id)
  }
  const parsed = session.parseProjectStateFile(JSON.stringify(file))
  if (!parsed.ok) throw new Error(parsed.error)
  const scene = layoutContinuousScene(compactVisualHierarchy(parsed.file.state.graph))
  expect(scene.parents.get(id)).toBe(inner.group!.id)
  expect(scene.nodes.get(id)!.width).toBeCloseTo(scene.nodes.get('w')!.width, 8)
  expect(graph.nodes.slice(0, before.state.graph.nodes.length).map(node => ({ id: node.id, position: node.position, params: node.params })))
    .toEqual(before.state.graph.nodes.map(node => ({ id: node.id, position: node.position, params: node.params })))
  expect(graph.nodes.find(node => node.id === id)!.params).toEqual(graph.nodes.find(node => node.id === 'w')!.params)
  expect(graph.edges.map(edge => ({ id: edge.id, source: edge.source, target: edge.target, inputSlot: edge.inputSlot, sourceSlot: edge.sourceSlot })))
    .toEqual(before.state.graph.edges.map(edge => ({ id: edge.id, source: edge.source, target: edge.target, inputSlot: edge.inputSlot, sourceSlot: edge.sourceSlot })))
  fireEvent.keyDown(document, { key: 'z', ctrlKey: true })
  expect(save().state.graph).toEqual(before.state.graph)
})
