import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { IDBFactory } from 'fake-indexeddb'
import App from './App'
import { cloneGraph, parameterValues } from './domain/engine'
import * as storage from './domain/localRecovery'
import { recoveryWorkspace } from './test/recoveryWorkspace'

beforeEach(() => vi.stubGlobal('indexedDB', new IDBFactory()))
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

it('offers recovery without overwriting it and restores training paused', async () => {
  const workspace = recoveryWorkspace()
  await storage.writeRecovery(workspace)
  const write = vi.spyOn(storage, 'writeRecovery')
  render(<App />)
  expect(await screen.findByRole('dialog', { name: 'Restore previous workspace?' })).toBeInTheDocument()
  fireEvent(window, new Event('pagehide'))
  expect(write).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Restore workspace' }))
  expect(screen.getAllByText('Epoch 500').length).toBeGreaterThan(0)
  expect(screen.getByLabelText('Epochs per run')).toHaveValue(2000)
  expect(screen.getByLabelText(/Report loss every/)).toHaveValue(500)
  expect(screen.getByLabelText('Examples per update')).toHaveValue(3)
  expect(screen.getByLabelText('Reshuffle training examples each epoch')).not.toBeChecked()
  expect(screen.queryByRole('button', { name: 'Stop training' })).not.toBeInTheDocument()
  expect(screen.getByRole('list', { name: 'Reported losses' }).children).toHaveLength(2)
  // Flush the restored workspace and verify it still contains the checkpoint weights.
  fireEvent(window, new Event('pagehide'))
  await waitFor(() => expect(write).toHaveBeenCalled())
  await waitFor(() => expect(screen.getByText(/Saved locally at/)).toBeInTheDocument())
  const restored = (await storage.readRecovery())!.file.state.graph
  expect(parameterValues(restored)).toEqual(parameterValues(workspace.state.graph))
  expect(restored.nodes).toEqual(cloneGraph(workspace.state.graph).nodes)
})
it('discards the old workspace and periodically saves subsequent edits', async () => {
  await storage.writeRecovery(recoveryWorkspace())
  const write = vi.spyOn(storage, 'writeRecovery')
  render(<App />)
  fireEvent.click(await screen.findByRole('button', { name: 'Discard recovery' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  expect(await storage.readRecovery()).toBeUndefined()
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.change(screen.getByLabelText('Epochs per run'), { target: { value: '42' } })
  await waitFor(() => expect(write).toHaveBeenCalled(), { timeout: 4500 })
  await waitFor(() => expect(screen.getByText(/Saved locally at/)).toBeInTheDocument())
  expect((await storage.readRecovery())?.file.state.runSettings?.epochsPerRun).toBe('42')
})
it('shows a warning when saving fails without interrupting editing', async () => {
  vi.spyOn(storage, 'writeRecovery').mockRejectedValue(new Error('QuotaExceededError'))
  render(<App />)
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 20)) })
  fireEvent(window, new Event('pagehide'))
  expect(await screen.findByText(/Local recovery could not be saved/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  expect(screen.getByLabelText('Epochs per run')).toBeEnabled()
})
