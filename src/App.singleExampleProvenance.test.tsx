import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import * as session from './domain/session'
import { parameterValues } from './domain/engine'
import { datasetExamplesForNode } from './domain/datasets'
import { singleExampleTextModel } from './test/inferenceInspectionModel'

const download = () => vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
afterEach(() => vi.restoreAllMocks())

it('preserves row 2 and its text experiment through Run forward, labels intermediate signals, and restores the original', () => {
  const graph = singleExampleTextModel(), spy = download()
  const { container } = render(<App initialGraph={graph}/>)
  const save = () => {
    fireEvent.click(screen.getByRole('button', { name: 'File' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
    return spy.mock.calls.at(-1)![0].state.graph
  }
  const select = (id: string, tab: 'Data' | 'Details') => {
    fireEvent.click(container.querySelector(`[data-id="${id}"]`)!)
    fireEvent.click(screen.getByRole('tab', { name: tab }))
  }
  select('text-data', 'Details')
  fireEvent.change(screen.getByLabelText('Dataset split'), { target: { value: 'test' } })
  fireEvent.change(screen.getByLabelText('Dataset row'), { target: { value: '2' } })
  fireEvent.click(screen.getByText('Word-order experiment'))
  fireEvent.click(screen.getByRole('button', { name: 'Reverse token order' }))
  const reversed = save().nodes.find(node => node.type === 'dataset')!.params
  expect(reversed.datasetIndex).toBe(2)
  expect(reversed.datasetSplit).toBe('test')
  expect(reversed.datasetValues).toHaveLength(3)
  const original = datasetExamplesForNode(graph.nodes.find(node => node.type === 'dataset')!)[2]
  // Reversal preserves counts, but an explicit override must still be retained.
  expect(reversed.datasetValues![0]).toEqual(original.features[0])
  expect(reversed.datasetValues![1].data).toEqual([0, 1, 2])
  expect(reversed.datasetValues![2]).toEqual(original.target)
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.click(screen.getByText('Step through a lesson'))
  fireEvent.click(screen.getByRole('button', { name: 'Run forward' }))
  const first = save()
  expect(first.nodes.find(node => node.type === 'dataset')!.params).toEqual(reversed)
  expect(parameterValues(first)).toEqual(parameterValues(graph))
  const originalProbability = first.nodes.find(node => node.id === 'positive-probability')!.value!.data[0]
  expect(originalProbability).toBeCloseTo(1 / (1 + Math.exp(-1)), 12)
  for (const id of ['review-counts', 'sentiment-product', 'sentiment-bias-add', 'positive-probability']) {
    select(id, 'Data')
    expect(screen.getByLabelText('Dataset inspection context')).toHaveTextContent(/^Dataset row 2 — modified input$/)
    expect(screen.getByRole('columnheader', { name: 'Dataset row' })).toBeInTheDocument()
    expect(screen.getByRole('rowheader', { name: '2' })).toBeInTheDocument()
  }
  select('sentiment-weights', 'Data')
  expect(screen.queryByLabelText('Dataset inspection context')).not.toBeInTheDocument()
  expect(screen.queryByRole('columnheader', { name: 'Dataset row' })).not.toBeInTheDocument()

  select('text-data', 'Details')
  fireEvent.change(screen.getByLabelText('Review experiment'), { target: { value: 'good good good film' } })
  fireEvent.click(screen.getByRole('button', { name: 'Apply text' }))
  const edited = save().nodes.find(node => node.type === 'dataset')!.params
  expect(edited.datasetIndex).toBe(2)
  const good = edited.textData!.vocabulary.indexOf('good')
  expect(edited.datasetValues![0].data[good]).toBe(3)
  expect(edited.datasetValues![0]).not.toEqual(original.features[0])
  for (const tab of ['Test', 'Build', 'Train']) fireEvent.click(screen.getByRole('tab', { name: tab }))
  fireEvent.click(screen.getByRole('button', { name: 'Run forward' }))
  const second = save()
  expect(second.nodes.find(node => node.type === 'dataset')!.params).toEqual(edited)
  const probability = second.nodes.find(node => node.id === 'positive-probability')!.value!.data[0]
  expect(probability).toBeCloseTo(1 / (1 + Math.exp(-2.3)), 12)
  expect(probability).not.toBeCloseTo(originalProbability, 5)
  expect(parameterValues(second)).toEqual(parameterValues(graph))
  select('positive-probability', 'Data')
  expect(screen.getByLabelText('Dataset inspection context')).toHaveTextContent('Dataset row 2 — modified input')
  expect(screen.getByRole('button', { name: `Dataset row 2, column 0: ${probability}` })).toBeInTheDocument()

  select('text-data', 'Details')
  fireEvent.change(screen.getByLabelText('Dataset row'), { target: { value: '3' } })
  fireEvent.change(screen.getByLabelText('Dataset row'), { target: { value: '2' } })
  expect(save().nodes.find(node => node.type === 'dataset')!.params.datasetValues).toBeUndefined()
  fireEvent.click(screen.getByRole('button', { name: 'Run forward' }))
  const restored = save()
  expect(restored.nodes.find(node => node.id === 'positive-probability')!.value!.data[0]).toBe(originalProbability)
  expect(parameterValues(restored)).toEqual(parameterValues(graph))
  select('positive-probability', 'Data')
  expect(screen.getByLabelText('Dataset inspection context')).toHaveTextContent(/^Dataset row 2$/)
})
