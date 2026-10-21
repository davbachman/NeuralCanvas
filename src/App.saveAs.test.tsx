import { fireEvent, render, screen, within, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import { createStarterGraph } from './domain/examples'
import * as session from './domain/session'

afterEach(() => vi.restoreAllMocks())

it.each(['Assignment 10', 'Assignment 10.json'])('saves the complete project as %s with a single JSON extension', async name => {
  const graph = createStarterGraph(true)
  const download = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  render(<App initialGraph={graph} />)
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.change(screen.getByLabelText('Epochs per run'), { target: { value: '200' } })
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
  const original = download.mock.calls[0][0]
  download.mockClear()
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save as…' }))
  const dialog = screen.getByRole('dialog', { name: 'Save as' })
  const input = within(dialog).getByLabelText('File name')
  expect(input).toHaveFocus()
  fireEvent.change(input, { target: { value: name } })
  fireEvent.submit(dialog)
  expect(download).toHaveBeenCalledTimes(1)
  const [file, filename] = download.mock.calls[0]
  expect(filename).toBe('Assignment 10.json')
  expect(file.state).toEqual(original.state)
  expect(file.state.runSettings?.epochsPerRun).toBe('200')
  expect(session.parseProjectStateFile(JSON.stringify(file)).ok).toBe(true)
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  await waitFor(() => expect(screen.getByRole('button', { name: 'File' })).toHaveFocus())
})

it('validates names, traps keyboard focus, and cancels without downloading', async () => {
  const download = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  const user = userEvent.setup()
  render(<App initialGraph={createStarterGraph()} />)
  await user.click(screen.getByRole('button', { name: 'File' }))
  await user.click(screen.getByRole('menuitem', { name: 'Save as…' }))
  const dialog = screen.getByRole('dialog', { name: 'Save as' })
  const input = within(dialog).getByLabelText('File name')
  for (const name of ['', '  ', '.json', 'folder/model']) {
    fireEvent.change(input, { target: { value: name } })
    expect(within(dialog).getByRole('button', { name: 'Save' })).toBeDisabled()
    fireEvent.submit(dialog)
  }
  await user.tab({ shift: true })
  expect(within(dialog).getByRole('button', { name: 'Cancel' })).toHaveFocus()
  await user.tab()
  expect(input).toHaveFocus()
  await user.keyboard('{Escape}')
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  expect(download).not.toHaveBeenCalled()
  await user.click(screen.getByRole('button', { name: 'File' }))
  await user.click(screen.getByRole('menuitem', { name: 'Save' }))
  expect(download).toHaveBeenCalledTimes(1)
  expect(download.mock.calls[0][1]).toBeUndefined()
})
