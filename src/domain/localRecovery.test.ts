import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { IDBFactory } from 'fake-indexeddb'
import { deleteRecovery, parseRecovery, readRecovery, writeRecovery } from './localRecovery'
import { createProjectStateFile } from './session'
import { recoveryWorkspace } from '../test/recoveryWorkspace'

beforeEach(() => vi.stubGlobal('indexedDB', new IDBFactory()))
afterEach(() => vi.unstubAllGlobals())

it('round-trips the model, parameters, reports, and controls through IndexedDB and deletes them', async () => {
  expect(await readRecovery()).toBeUndefined()
  const workspace = recoveryWorkspace()
  await writeRecovery(workspace)
  const loaded = await readRecovery()
  expect(loaded?.file.state).toEqual(createProjectStateFile(workspace.state).state)
  expect(loaded?.lossReports).toEqual(workspace.lossReports)
  expect(loaded?.shuffleEachEpoch).toBe(false)
  await deleteRecovery()
  expect(await readRecovery()).toBeUndefined()
})
it('rejects corrupted recovery records without treating them as a workspace', async () => {
  await writeRecovery(recoveryWorkspace())
  const record = await readRecovery()
  expect(() => parseRecovery({ ...record, lossReports: [{ epoch: -1, loss: 2 }] })).toThrow('invalid')
  expect(() => parseRecovery({ ...record, file: {} })).toThrow('invalid')
})
it('reports unavailable storage', async () => {
  vi.stubGlobal('indexedDB', undefined)
  await expect(writeRecovery(recoveryWorkspace())).rejects.toThrow('does not support')
})

it('keeps the previous good copy when a new snapshot is invalid', async () => {
  await writeRecovery(recoveryWorkspace())
  const invalid = recoveryWorkspace()
  invalid.state.graph.learningRate = NaN
  await expect(writeRecovery(invalid)).rejects.toThrow('invalid')
  expect((await readRecovery())?.file.state.graph.learningRate).toBe(recoveryWorkspace().state.graph.learningRate)
})

it('persists viewport-only changes without serializing tensors, then fully saves subsequent model edits', async () => {
  const workspace = recoveryWorkspace()
  await writeRecovery(workspace)
  const stringify = vi.spyOn(JSON, 'stringify')
  const viewport = { x: 120, y: -200, zoom: .75 }
  const moved = { ...workspace, state: { ...workspace.state, graph: { ...workspace.state.graph,
    view: { ...workspace.state.graph.view!, viewport },
  } } }
  await writeRecovery(moved)
  const serialized = [...stringify.mock.calls]
  stringify.mockRestore()
  expect(serialized.every(([value]) => !JSON.stringify(value).includes('learningRate'))).toBe(true)
  const recovered = (await readRecovery())!
  expect(recovered.file.state.graph.view!.viewport).toEqual(viewport)
  expect(recovered.file.state.graph.nodes).toEqual(createProjectStateFile(workspace.state).state.graph.nodes)
  const edited = { ...moved, state: { ...moved.state, epoch: 501, graph: { ...moved.state.graph, learningRate: .123 } } }
  await writeRecovery(edited)
  expect((await readRecovery())!.file.state.epoch).toBe(501)
  expect((await readRecovery())!.file.state.graph.learningRate).toBe(.123)
  await deleteRecovery()
  expect(await readRecovery()).toBeUndefined()
})

it('does not confuse selection, settings, or tensor edits with a viewport-only update', async () => {
  const workspace = recoveryWorkspace()
  await writeRecovery(workspace)
  const selected = { ...workspace, state: { ...workspace.state, selectedGroupId: 'network' } }
  await writeRecovery(selected)
  expect((await readRecovery())!.file.state.selectedGroupId).toBe('network')
  const settings = { ...selected, state: { ...selected.state, runSettings: { ...selected.state.runSettings!, epochsPerRun: '77' } } }
  await writeRecovery(settings)
  expect((await readRecovery())!.file.state.runSettings!.epochsPerRun).toBe('77')
  const withViewport = { ...settings, state: { ...settings.state, graph: { ...settings.state.graph, view: { ...settings.state.graph.view!, viewport: { x: 1, y: 2, zoom: 1 } } } } }
  await writeRecovery(withViewport)
  await writeRecovery(settings)
  expect((await readRecovery())!.file.state.graph.view!.viewport).toBeUndefined()
})
