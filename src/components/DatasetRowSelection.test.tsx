import { useState } from 'react'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { expect, it } from 'vitest'
import { prepareTextDocuments } from '../domain/textData'
import type { GraphNode, TextDatasetData } from '../domain/types'
import { DatasetWorkbench } from './DatasetWorkbench'
import { DataInspector } from './DataInspector'

function DatasetTabs({ data }: { data: TextDatasetData }) {
  const [node, setNode] = useState<GraphNode>({ id: 'data', type: 'dataset', label: 'Text', position: { x: 0, y: 0 }, params: { dataset: 'custom-text', datasetMode: 'sample', datasetIndex: 0, textData: data } })
  const [tab, setTab] = useState('Details')
  const graph = { nodes: [node], edges: [], learningRate: .1 }
  return <>
    {['Details', 'Data'].map(name => <button key={name} onClick={() => setTab(name)}>{name}</button>)}
    {tab === 'Details' ? <DatasetWorkbench graph={graph} node={node} onDataset={() => {}} onParams={(_, params) => setNode(previous => ({ ...previous, params: { ...previous.params, ...params } }))}/> : <DataInspector graph={graph} node={node}/>}
  </>
}

it('uses the same original row numbers in Details and Data, including filtered held-out rows', () => {
  const data = prepareTextDocuments([
    { text: 'good', label: 'positive', split: 'train' },
    { text: 'bad', label: 'negative', split: 'test' },
    { text: 'good', label: 'positive', split: 'train' },
    { text: 'bad', label: 'negative', split: 'test' },
  ], 'text.csv', { task: 'sentiment' })
  const { container } = render(<DatasetTabs data={data}/> )
  for (const row of [0, 1, 3]) {
    if (row === 1) fireEvent.change(screen.getByLabelText('Dataset split'), { target: { value: 'test' } })
    const selector = screen.getByLabelText('Dataset row')
    fireEvent.change(selector, { target: { value: String(row) } })
    expect(within(selector).getByRole('option', { selected: true })).toHaveTextContent(`Row ${row} ·`)
    fireEvent.click(screen.getByRole('button', { name: 'Data' }))
    expect(screen.getByLabelText('Current dataset row')).toHaveTextContent(`Current row: ${row}.`)
    expect(container.querySelector('tr.is-current th')).toHaveTextContent(String(row))
    fireEvent.click(screen.getByRole('button', { name: 'Details' }))
    expect(screen.getByLabelText('Dataset row')).toHaveValue(String(row))
  }
})

it('uses zero-based bounds for the numeric selector on large datasets', () => {
  const data = prepareTextDocuments(Array.from({ length: 502 }, (_, i) => ({ text: 'good', label: 'positive', split: i === 501 ? 'test' as const : 'train' as const })), 'large.csv', { task: 'sentiment' })
  render(<DatasetTabs data={data}/> )
  const selector = screen.getByRole('spinbutton', { name: 'Dataset row' })
  expect(selector).toHaveAttribute('min', '0')
  expect(selector).toHaveAttribute('max', '501')
  fireEvent.change(selector, { target: { value: '501' } })
  expect(selector).toHaveValue(501)
  for (const value of ['-1', '502', '', '1.5']) {
    fireEvent.change(selector, { target: { value } })
    expect(selector).toHaveValue(501)
  }
  fireEvent.click(screen.getByRole('button', { name: 'Data' }))
  expect(screen.getByLabelText('Current dataset row')).toHaveTextContent('Current row: 501.')
  fireEvent.click(screen.getByRole('button', { name: 'Details' }))
  fireEvent.change(screen.getByLabelText('Dataset row'), { target: { value: '0' } })
  expect(screen.getByLabelText('Dataset row')).toHaveValue(0)
})

it('distinguishes a language-model dataset row from its source document and token offset', () => {
  const data = prepareTextDocuments([{ text: 'abcdefghij', split: 'train' }, { text: 'abcdefghij', split: 'test' }], 'corpus.txt', { task: 'language', tokenizer: 'character', maxLength: 3, stride: 3 })
  render(<DatasetTabs data={data}/> )
  fireEvent.change(screen.getByLabelText('Dataset row'), { target: { value: '1' } })
  expect(screen.getByText('Original text · dataset row 1')).toBeInTheDocument()
  expect(screen.getByText(/^Source document 0 · token offset 3\./)).toBeInTheDocument()
  expect(within(screen.getByLabelText('Dataset row')).getByRole('option', { selected: true })).toHaveTextContent('Row 1 · Source document 0 · offset 3')
  fireEvent.click(screen.getByRole('button', { name: 'Data' }))
  expect(screen.getByLabelText('Current dataset row')).toHaveTextContent('Current row: 1.')
})
