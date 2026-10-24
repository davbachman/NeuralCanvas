import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import { createNode, createSingleNeuronGraph } from './domain/examples'
import { datasetForNode } from './domain/datasets'
import * as session from './domain/session'

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })
function graph() {
  const result = createSingleNeuronGraph('identity', { x: 1, w: 2, b: 0, target: 3 })
  result.nodes.push(createNode('dataset', 1))
  result.edges.push({ id: 'data-x', source: 'dataset-1', sourceSlot: 0, target: 'x', inputSlot: 0 }, { id: 'data-y', source: 'dataset-1', sourceSlot: 1, target: 'target', inputSlot: 0 })
  return result
}
it('installs a library CSV with fixed splits, remaps its target wire, saves, and supports undo', async () => {
  const text = readFileSync(resolve('public/datasets/iris.csv'), 'utf8')
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, text: async () => text })))
  const save = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  const { container } = render(<App initialGraph={graph()} />)
  fireEvent.click(container.querySelector('[data-id="dataset-1"]')!)
  fireEvent.click(screen.getByRole('button', { name: 'Browse datasets…' }))
  fireEvent.click(screen.getByRole('button', { name: /Iris flowers/ }))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Use this dataset' })).toBeEnabled())
  fireEvent.click(screen.getByRole('button', { name: 'Use this dataset' }))
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  expect(screen.getByRole('combobox', { name: 'CSV task' })).toHaveValue('classification')
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
  const file = save.mock.calls.at(-1)![0]
  const data = file.state.graph.nodes.find(n => n.id === 'dataset-1')!
  expect(data.params.customCsv?.fileName).toBe('iris.csv')
  expect(data.params.datasetLibraryPreset).toBe('iris')
  expect(screen.getByLabelText('Library dataset')).toHaveTextContent('Iris flowers')
  expect(data.params.customCsv?.splits?.filter(s => s === 'train')).toHaveLength(120)
  expect(file.state.graph.edges.find(e => e.id === 'data-y')?.sourceSlot).toBe(4)
  const restored = session.parseProjectStateFile(JSON.stringify(file))
  expect(restored.ok).toBe(true)
  if (!restored.ok) throw Error(restored.error)
  expect(restored.file.state.graph.nodes.find(n => n.id === 'dataset-1')?.params.customCsv).toEqual(data.params.customCsv)
  fireEvent.keyDown(document, { key: 'z', metaKey: true })
  expect(screen.getByRole('combobox', { name: 'Dataset selection' })).toHaveDisplayValue(datasetForNode(graph().nodes.find(n => n.id === 'dataset-1')!).label)
})
it.each(['imdb.json', 'football-counts.json'])('preserves the existing prepared JSON import route for %s', async filename => {
  const { container } = render(<App initialGraph={graph()} />)
  const card = container.querySelector<HTMLElement>('[data-id="dataset-1"]')!
  fireEvent.click(card)
  const selector = within(card).getByLabelText('Dataset for dataset')
  expect(within(selector).getByRole('option', { name: 'Text / sentiment…' })).toBeInTheDocument()
  fireEvent.change(selector, { target: { value: 'custom-text' } })
  const text = readFileSync(resolve('public/datasets', `${filename}`), 'utf8')
  const file = Object.assign(new File([text], filename, { type: 'application/json' }), { text: async () => text })
  fireEvent.change(screen.getByLabelText('Choose text data file'), { target: { files: [file] } })
  // Leave every importer setting at its default: prepared JSON is authoritative.
  fireEvent.click(screen.getByRole('button', { name: 'Import text dataset' }))
  await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Import text data' })).not.toBeInTheDocument())
  expect(selector).toHaveDisplayValue(filename)
  expect(screen.getByRole('button', { name: 'Replace text dataset' })).toBeVisible()
  expect(screen.getByText(/Context limit:/)).toBeVisible()
}, 30000)
it('opens the same prepared JSON importer through Custom data in the library', () => {
  const { container } = render(<App initialGraph={graph()} />)
  fireEvent.click(container.querySelector('[data-id="dataset-1"]')!)
  fireEvent.click(screen.getByRole('button', { name: 'Browse datasets…' }))
  fireEvent.click(screen.getByRole('button', { name: 'Use a custom dataset…' }))
  fireEvent.click(screen.getByRole('button', { name: 'Import prepared JSON or text…' }))
  expect(screen.queryByRole('dialog', { name: 'Dataset library' })).not.toBeInTheDocument()
  expect(screen.getByRole('dialog', { name: 'Import text data' })).toBeVisible()
  expect(screen.getByLabelText('Choose text data file')).toHaveAttribute('accept', expect.stringContaining('.json'))
})
