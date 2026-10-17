import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { useLocalRecovery } from './useLocalRecovery'
import * as storage from '../domain/localRecovery'
import { recoveryWorkspace } from '../test/recoveryWorkspace'

beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(storage, 'readRecovery').mockResolvedValue(undefined)
  vi.spyOn(storage, 'writeRecovery').mockResolvedValue()
})
afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers() })

it('coalesces live navigation but flushes the latest viewport on pagehide and preserves model edits', async () => {
  const workspace = recoveryWorkspace()
  const { result, rerender } = renderHook(({ value }) => useLocalRecovery(value, false), { initialProps: { value: workspace } })
  await act(async () => {})
  await act(async () => { await vi.advanceTimersByTimeAsync(3000) })
  expect(storage.writeRecovery).toHaveBeenCalledTimes(1)
  for (let x = 1; x <= 3; x++) {
    act(() => result.current.onNavigationChange(true, { x, y: 20, zoom: .75 }))
    await act(async () => { await vi.advanceTimersByTimeAsync(1000) })
  }
  expect(storage.writeRecovery).toHaveBeenCalledTimes(1)
  await act(async () => { window.dispatchEvent(new Event('pagehide')) })
  expect(storage.writeRecovery).toHaveBeenCalledTimes(2)
  const saved = vi.mocked(storage.writeRecovery).mock.lastCall![0]
  expect(saved.state.graph.view!.viewport).toEqual({ x: 3, y: 20, zoom: .75 })
  expect(saved.state.graph.nodes).toBe(workspace.state.graph.nodes)
  const edited = { ...workspace, state: { ...workspace.state, epoch: 501, graph: { ...workspace.state.graph, learningRate: .25 } } }
  rerender({ value: edited })
  await act(async () => { await vi.advanceTimersByTimeAsync(3000) })
  expect(storage.writeRecovery).toHaveBeenLastCalledWith(edited)
})

it('bounds navigation deferral and saves training checkpoints immediately and in order', async () => {
  const workspace = recoveryWorkspace()
  const { result } = renderHook(() => useLocalRecovery(workspace, false))
  await act(async () => {})
  await act(async () => { await result.current.save(workspace) })
  act(() => result.current.onNavigationChange(true, { x: 42, y: 1, zoom: 2 }))
  await act(async () => { await vi.advanceTimersByTimeAsync(12000) })
  expect(storage.writeRecovery).toHaveBeenCalledTimes(2)
  const checkpoint = { ...workspace, state: { ...workspace.state, epoch: 600 } }
  await act(async () => { await result.current.save(checkpoint) })
  expect(vi.mocked(storage.writeRecovery).mock.lastCall![0].state).toMatchObject({ epoch: 600, graph: { view: { viewport: { x: 42, y: 1, zoom: 2 } } } })
})


it('preserves new checkpoint weights and the live viewport when a camera flush queues behind training', async () => {
  const workspace = recoveryWorkspace()
  const { result } = renderHook(() => useLocalRecovery(workspace, true))
  await act(async () => {})
  await act(async () => { await result.current.save(workspace) })
  let finish!: () => void
  vi.mocked(storage.writeRecovery).mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  const checkpoint = { ...workspace, state: { ...workspace.state, epoch: 700, graph: {
    ...workspace.state.graph, nodes: workspace.state.graph.nodes.map(node => node.type === 'weight'
      ? { ...node, params: { ...node.params, value: 42 } } : node),
  } } }
  let saving!: Promise<void>
  await act(async () => { saving = result.current.save(checkpoint) })
  act(() => result.current.onNavigationChange(true, { x: 99, y: 100, zoom: .5 }))
  act(() => { window.dispatchEvent(new Event('pagehide')) })
  await act(async () => { finish(); await saving; await Promise.resolve() })
  const saved = vi.mocked(storage.writeRecovery).mock.lastCall![0]
  expect(saved.state.epoch).toBe(700)
  expect(saved.state.graph.nodes).toBe(checkpoint.state.graph.nodes)
  expect(saved.state.graph.view!.viewport).toEqual({ x: 99, y: 100, zoom: .5 })
})
