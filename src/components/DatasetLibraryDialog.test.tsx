import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { DatasetLibraryDialog } from './DatasetLibraryDialog'

afterEach(() => vi.unstubAllGlobals())
function mockAssets() {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const text = readFileSync(resolve('public/datasets', `${url.split('/').at(-1)}`), 'utf8')
    return { ok: true, text: async () => text, json: async () => JSON.parse(text) }
  }))
}
it('organizes built-in and real datasets by category, search, and family', () => {
  render(<DatasetLibraryDialog onUse={vi.fn()} onCustom={vi.fn()} onClose={vi.fn()} />)
  expect(screen.getByRole('dialog', { name: 'Dataset library' })).toBeVisible()
  const nav = within(screen.getByRole('navigation', { name: 'Dataset categories' }))
  fireEvent.click(nav.getByRole('button', { name: 'Images' }))
  expect(screen.getByRole('button', { name: /Handwritten digits/ })).toBeVisible()
  expect(screen.queryByRole('button', { name: /Auto MPG/ })).not.toBeInTheDocument()
  fireEvent.click(nav.getByRole('button', { name: 'All datasets' }))
  fireEvent.change(screen.getByLabelText('Find a dataset'), { target: { value: 'penguin' } })
  expect(screen.getByRole('button', { name: /Palmer penguins/ })).toBeVisible()
  expect(screen.queryByRole('button', { name: /Movie reviews/ })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: /Download/ })).not.toBeInTheDocument()
})
it('shows football coverage and shapes, supports validated combinations, and installs only on Use', async () => {
  mockAssets()
  const onUse = vi.fn()
  render(<DatasetLibraryDialog onUse={onUse} onCustom={vi.fn()} onClose={vi.fn()} />)
  fireEvent.click(screen.getByRole('button', { name: /Football stories/ }))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Use this dataset' })).toBeEnabled())
  expect(screen.getByLabelText('Football task')).toHaveValue('narrative')
  fireEvent.change(screen.getByLabelText('Narrative training targets'), { target: { value: 'all' } })
  await waitFor(() => expect(screen.getByRole('button', { name: 'Use this dataset' })).toBeEnabled())
  expect(screen.getAllByText('[32]').length).toBe(3)
  fireEvent.click(screen.getByRole('button', { name: 'Mixed training · football benchmark' }))
  await waitFor(() => expect(screen.getByText('3,840')).toBeVisible())
  expect(screen.getByText('240')).toBeVisible()
  expect(screen.getByLabelText('Question selection')).toHaveValue('benchmark')
  expect(screen.getByText('[32, 1]')).toBeVisible()
  fireEvent.change(screen.getByLabelText('Football model inputs'), { target: { value: 'answer-token' } })
  await waitFor(() => expect(screen.getByText(/The answer is excluded from the input/)).toBeVisible())
  await waitFor(() => expect(screen.getByRole('button', { name: 'Use this dataset' })).toBeEnabled())
  expect(screen.getAllByText('[23]')).toHaveLength(2)
  expect(onUse).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Use this dataset' }))
  expect(onUse).toHaveBeenCalledWith(expect.objectContaining({ dataset: 'custom-text', textData: expect.objectContaining({ maxLength: 23, targetMode: 'last' }) }))
})
it('retains a direct custom prepared-JSON/text importer and numeric CSV option', () => {
  const onCustom = vi.fn()
  render(<DatasetLibraryDialog onUse={vi.fn()} onCustom={onCustom} onClose={vi.fn()} />)
  fireEvent.click(screen.getByRole('button', { name: 'Use a custom dataset…' }))
  fireEvent.click(screen.getByRole('button', { name: 'Import prepared JSON or text…' }))
  expect(onCustom).toHaveBeenLastCalledWith('custom-text')
  fireEvent.click(screen.getByRole('button', { name: 'Import numeric CSV…' }))
  expect(onCustom).toHaveBeenLastCalledWith('custom-csv')
})
it('discards slow results after selection changes or closing, and traps keyboard focus', async () => {
  const pending: Array<(response: unknown) => void> = []
  vi.stubGlobal('fetch', vi.fn(() => new Promise(resolve => pending.push(resolve))))
  const close = vi.fn(), use = vi.fn()
  const { unmount } = render(<DatasetLibraryDialog onUse={use} onCustom={vi.fn()} onClose={close} />)
  fireEvent.click(screen.getByRole('button', { name: /Football stories/ }))
  fireEvent.click(screen.getByRole('button', { name: 'Mixed-question classifier' }))
  const data = JSON.parse(readFileSync(resolve('public/datasets/football-narratives-last.json'), 'utf8'))
  await act(async () => { pending[0]({ ok: true, json: async () => data }) })
  expect(screen.getByRole('button', { name: 'Use this dataset' })).toBeDisabled()
  const closeButton = screen.getByRole('button', { name: 'Close dataset library' })
  closeButton.focus()
  fireEvent.keyDown(closeButton, { key: 'Tab', shiftKey: true })
  expect(screen.getByRole('button', { name: 'Use a custom dataset…' })).toHaveFocus()
  fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
  expect(close).toHaveBeenCalledOnce()
  unmount()
  await act(async () => { pending[1]({ ok: true, json: async () => data }) })
  expect(use).not.toHaveBeenCalled()
})
