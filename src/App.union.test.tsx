import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import { createStarterGraph } from './domain/examples'
import { mergePreservingLayout } from './domain/mergeLayout'
import { compactVisualHierarchy, layoutContinuousScene, CONTAINMENT_SCALE } from './domain/continuousScene'
import * as session from './domain/session'

afterEach(() => vi.restoreAllMocks())

it.each([
  { nested: false, mixed: false, union: false }, { nested: false, mixed: false, union: true },
  { nested: true, mixed: false, union: false }, { nested: true, mixed: false, union: true },
  { nested: false, mixed: true, union: false }, { nested: false, mixed: true, union: true },
])('offers Merge and Union for selected modules or a module and block: %j', async ({ nested, mixed, union }) => {
  const user = userEvent.setup()
  let graph = createStarterGraph(true)
  if (nested) graph = mergePreservingLayout(graph, ['x', 'w', 'mul', 'b', 'add', 'pred']).graph
  const first = mergePreservingLayout(graph, ['x', 'w', 'mul'])
  graph = first.graph
  let secondId = 'b'
  const groups = [first.group!.id]
  if (!mixed) {
    const second = mergePreservingLayout(graph, ['b', 'add', 'pred'])
    graph = second.graph
    secondId = `visual-group:${second.group!.id}`
    groups.push(second.group!.id)
  }
  graph.view = { ...graph.view!, viewport: { x: 0, y: 0, zoom: nested ? 4 : 1 } }
  const download = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  const { container } = render(<App initialGraph={graph} />)
  const node = (id: string) => container.querySelector(`.react-flow__node[data-id="${id}"]`)!
  fireEvent.click(node(`visual-group:${first.group!.id}`))
  await user.keyboard('{Shift>}')
  fireEvent.click(node(secondId), { shiftKey: true })
  await user.keyboard('{/Shift}')
  expect(screen.getByRole('button', { name: 'Merge selection' })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Union selection' })).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: '2 blocks selected' })).toBeInTheDocument()
  const save = () => {
    fireEvent.click(screen.getByRole('button', { name: 'File' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
    return download.mock.calls.at(-1)![0]
  }
  const before = save()
  expect(new Set(before.state.selectedGroupIds)).toEqual(new Set(groups))
  fireEvent.click(screen.getByRole('button', { name: union ? 'Union selection' : 'Merge selection' }))
  const after = save(), groupId = after.state.selectedGroupId!
  expect(groupId).toBeTruthy()
  const combined = after.state.graph.groups!.find(group => group.id === groupId)!
  expect(combined.parentId).toBe(first.group!.parentId)
  for (const id of groups) {
    const oldGroup = after.state.graph.groups!.find(group => group.id === id)
    if (union) expect(oldGroup).toBeUndefined()
    else expect(oldGroup!.parentId).toBe(groupId)
  }
  expect(after.state.graph.nodes).toEqual(before.state.graph.nodes)
  expect(after.state.graph.edges).toEqual(before.state.graph.edges)
  const oldScene = layoutContinuousScene(compactVisualHierarchy(before.state.graph))
  const newScene = layoutContinuousScene(compactVisualHierarchy(after.state.graph))
  expect(newScene.scales.get('mul')).toBe(oldScene.scales.get('mul')! * (union ? 1 : CONTAINMENT_SCALE))
  expect(session.parseProjectStateFile(JSON.stringify(after)).ok).toBe(true)
  fireEvent.keyDown(document, { key: 'z', ctrlKey: true })
  const undone = save()
  expect(undone.state.graph).toEqual(before.state.graph)
  expect(undone.state.selectedGroupIds).toEqual(before.state.selectedGroupIds)
  // The saved selection retains which containers were selected as units.
  fireEvent.change(screen.getByLabelText('Import state file'), { target: { files: [new File([JSON.stringify(before)], 'selection.json', { type: 'application/json' })] } })
  await waitFor(() => expect(screen.getByRole('button', { name: 'Union selection' })).toBeInTheDocument())
})
