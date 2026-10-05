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
