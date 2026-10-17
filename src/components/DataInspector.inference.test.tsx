import { fireEvent, render, screen, within } from '@testing-library/react'
import { expect, it } from 'vitest'
import { DataInspector } from './DataInspector'
import { InferenceInspection } from '../domain/inferenceInspection'
import { sentimentModel } from '../test/inferenceInspectionModel'
import { forwardPass } from '../domain/engine'
import { prepareTextDocuments } from '../domain/textData'
import { buildTextModel } from '../test/textModels'

it('pages original dataset rows, hides gradients, and switches back to the unchanged canvas output', async () => {
  const graph = forwardPass(sentimentModel(17)).graph
  const node = graph.nodes.find(node => node.id === 'positive-probability')!
  const inference = new InferenceInspection(graph, 'text-data', 'test')
  render(<DataInspector graph={graph} node={node} inference={inference}/>)
  expect(screen.getByRole('combobox', { name: 'Inspection context' })).toHaveValue('inference')
  expect(screen.getByText('Held-out inference — 15 examples', { selector: 'p' })).toHaveTextContent('Held-out inference — 15 examples')
  const table = await screen.findByRole('columnheader', { name: 'Dataset row' })
  expect(within(table.closest('table')!).getAllByRole('rowheader').map(cell => cell.textContent)).toEqual(['1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'])
  expect(screen.queryByRole('tab', { name: '← Gradient' })).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Next examples' }))
  expect(await screen.findByRole('rowheader', { name: '16' })).toBeInTheDocument()
  expect(screen.queryByRole('rowheader', { name: '1' })).not.toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('Inspection context'), { target: { value: 'canvas' } })
  expect(screen.getByRole('tab', { name: '← Gradient' })).toBeInTheDocument()
  expect(screen.getByRole('rowheader', { name: '0' })).toBeInTheDocument()
  expect(graph.nodes.find(node => node.type === 'dataset')!.params.datasetIndex).toBe(0)
})

it('shows multidimensional tensors for the selected example with shape and axis information', async () => {
  const data = prepareTextDocuments([
    { text: 'good bad good', label: 'positive', split: 'train' },
    { text: 'bad good', label: 'negative', split: 'test' },
    { text: 'good bad good', label: 'positive', split: 'test' },
  ], 'tokens.csv', { task: 'sentiment', representation: 'tokens', maxLength: 4 })
  const graph = forwardPass(buildTextModel(data, 'attention', 4)).graph
  const node = graph.nodes.find(node => node.type === 'softmax')!
  const { container } = render(<DataInspector graph={graph} node={node} inference={new InferenceInspection(graph, 'text-data', 'test')}/>)
  expect(await screen.findByLabelText('Inference dataset row')).toHaveValue('1')
  expect(container.querySelector('.data-stats')).toHaveTextContent('[2 × 2]')
  expect(screen.getByText(/Axis 0: tensor index \(2\)/)).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('Inference dataset row'), { target: { value: '2' } })
  expect(await screen.findByLabelText('Inference dataset row')).toHaveValue('2')
  expect(container.querySelector('.data-stats')).toHaveTextContent('[3 × 3]')
  expect(screen.queryByRole('columnheader', { name: 'Dataset row' })).not.toBeInTheDocument()
})

it('displays shared parameter tensors once and defaults a new run to inference context', async () => {
  const graph = sentimentModel(), node = graph.nodes.find(node => node.type === 'weight')!
  const { rerender, container } = render(<DataInspector graph={graph} node={node} inference={new InferenceInspection(graph, 'text-data', 'test')}/>)
  expect(await screen.findByText(/Shared tensor/)).toBeInTheDocument()
  expect(screen.queryByLabelText('Inference dataset row')).not.toBeInTheDocument()
  expect(container.querySelector('.data-stats')).toHaveTextContent('[2 × 1]')
  fireEvent.change(screen.getByLabelText('Inspection context'), { target: { value: 'canvas' } })
  rerender(<DataInspector graph={graph} node={node} inference={new InferenceInspection(graph, 'text-data', 'train')}/>)
  expect(screen.getByLabelText('Inspection context')).toHaveValue('inference')
  expect(await screen.findByText(/Shared tensor/)).toBeInTheDocument()
})
