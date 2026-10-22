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
  expect(screen.getByTitle('untitled')).toHaveTextContent('untitled')
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.change(screen.getByLabelText('Epochs per run'), { target: { value: '200' } })
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
  const original = download.mock.calls[0][0]
  const defaultFilename = `neural-canvas-state-${original.savedAt.slice(0, 10)}.json`
  expect(download.mock.calls[0][1]).toBe(defaultFilename)
  expect(screen.getByTitle(defaultFilename)).toHaveTextContent(defaultFilename)
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
  expect(screen.getByTitle(filename!)).toHaveTextContent('Assignment 10.json')
  expect(file.state).toEqual(original.state)
  expect(file.state.runSettings?.epochsPerRun).toBe('200')
  expect(session.parseProjectStateFile(JSON.stringify(file)).ok).toBe(true)
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  await waitFor(() => expect(screen.getByRole('button', { name: 'File' })).toHaveFocus())
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
  expect(download.mock.calls.at(-1)![1]).toBe('Assignment 10.json')
})

it('reuses an imported filename after edits and preserves it when Save as or an invalid import is canceled', async () => {
  const download = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  render(<App initialGraph={createStarterGraph(true)} />)
  const menu = (name: string) => {
    fireEvent.click(screen.getByRole('button', { name: 'File' }))
    fireEvent.click(screen.getByRole('menuitem', { name }))
  }
  menu('Save')
  const original = download.mock.calls[0][0]
  original.state.runSettings = { epochsPerRun: '123', reportEvery: '1', examplesPerUpdate: '' }
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.change(screen.getByLabelText('Import state file'), { target: { files: [new File([JSON.stringify(original)], 'My assignment.json', { type: 'application/json' })] } })
  await waitFor(() => expect(screen.getByLabelText('Epochs per run')).toHaveValue(123))
  expect(screen.getByTitle('My assignment.json')).toHaveTextContent('My assignment.json')
  fireEvent.change(screen.getByLabelText('Epochs per run'), { target: { value: '456' } })
  menu('Save')
  expect(download.mock.calls.at(-1)![1]).toBe('My assignment.json')
  expect(download.mock.calls.at(-1)![0].state.runSettings?.epochsPerRun).toBe('456')
  menu('Save as…')
  expect(screen.getByLabelText('File name')).toHaveValue('My assignment.json')
  fireEvent.change(screen.getByLabelText('File name'), { target: { value: 'Canceled name' } })
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  fireEvent.change(screen.getByLabelText('Import state file'), { target: { files: [new File(['invalid'], 'Broken.json', { type: 'application/json' })] } })
  await screen.findByText('Import failed: choose a valid Neural Canvas state JSON file.')
  expect(screen.getByTitle('My assignment.json')).toHaveTextContent('My assignment.json')
  menu('Save')
  expect(download.mock.calls.at(-1)![1]).toBe('My assignment.json')
  menu('Save as…')
  fireEvent.change(screen.getByLabelText('File name'), { target: { value: 'Revised assignment' } })
  fireEvent.submit(screen.getByRole('dialog', { name: 'Save as' }))
  expect(screen.getByTitle('Revised assignment.json')).toHaveTextContent('Revised assignment.json')
  menu('Save')
  expect(download.mock.calls.at(-1)![1]).toBe('Revised assignment.json')
})

it.each(['New', 'Starter'])('clears the previous filename when starting %s', name => {
  const download = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  render(<App initialGraph={createStarterGraph()} />)
  const menu = (item: string) => {
    fireEvent.click(screen.getByRole('button', { name: 'File' }))
    fireEvent.click(screen.getByRole('menuitem', { name: item }))
  }
  menu('Save as…')
  fireEvent.change(screen.getByLabelText('File name'), { target: { value: 'Old project' } })
  fireEvent.submit(screen.getByRole('dialog', { name: 'Save as' }))
  menu(name)
  expect(screen.getByTitle('untitled')).toHaveTextContent('untitled')
  menu('Save')
  const [file, filename] = download.mock.calls.at(-1)!
  expect(filename).toBe(`neural-canvas-state-${file.savedAt.slice(0, 10)}.json`)
  expect(screen.getByTitle(filename!)).toHaveTextContent(filename!)
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
  expect(screen.getByTitle('untitled')).toHaveTextContent('untitled')
  expect(download).not.toHaveBeenCalled()
  await user.click(screen.getByRole('button', { name: 'File' }))
  await user.click(screen.getByRole('menuitem', { name: 'Save' }))
  expect(download).toHaveBeenCalledTimes(1)
  const [file, filename] = download.mock.calls[0]
  expect(filename).toBe(`neural-canvas-state-${file.savedAt.slice(0, 10)}.json`)
  expect(screen.getByTitle(filename!)).toHaveTextContent(filename!)
})
