import { fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import * as session from './domain/session'
import { forwardPass, parameterValues } from './domain/engine'
import { datasetOutputValueForSlot } from './domain/datasets'
import { storyExperimentModel } from './test/storyExperimentModel'

const download = () => vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
function save(spy: ReturnType<typeof download>) {
  fireEvent.click(screen.getByRole('button', {name: 'File'}))
  fireEvent.click(screen.getByRole('menuitem', {name: 'Save'}))
  return spy.mock.calls.at(-1)![0]
}
afterEach(() => vi.restoreAllMocks())

it('preserves a temporary story across tabs and evaluates its real inputs without relabeling or updating parameters', () => {
  const graph = storyExperimentModel(), original = JSON.stringify(graph), spy = download()
  render(<App initialGraph={graph}/>)
  expect(screen.getByLabelText('Experiment facts')).toHaveValue('Mary is in kitchen.\nJohn is in garden.')
  expect(screen.getByLabelText('Experiment question')).toHaveValue('Where is Mary?')
  fireEvent.change(screen.getByLabelText('Experiment facts'), {target: {value: 'John is in garden.'}})
  fireEvent.change(screen.getByLabelText('Experiment question'), {target: {value: 'Where is John?'}})
  fireEvent.click(screen.getByRole('button', {name: 'Apply experiment'}))
  for (const name of ['Train', 'Test', 'Build', 'Train', 'Data', 'Code', 'Reporting', 'Details']) fireEvent.click(screen.getByRole('tab', {name}))
  expect(screen.getByLabelText('Experiment facts')).toHaveValue('John is in garden.')
  expect(screen.getByLabelText('Experiment question')).toHaveValue('Where is John?')
  expect(screen.getByText(/Original answer \(reference only\)/)).toHaveTextContent('kitchen')
  expect(screen.getByText(/Experimental answer: unknown/)).toBeInTheDocument()
  const update = screen.getByRole('button', {name: 'Run one full training step'})
  expect(update).toBeDisabled()
  fireEvent.click(update)
  expect(screen.getByText('Epoch 0')).toBeInTheDocument()
  fireEvent.click(screen.getByText('Step through a lesson'))
  fireEvent.click(screen.getByRole('button', {name: 'Run forward'}))
  const file = save(spy), inspected = file.state.graph
  expect(session.parseProjectStateFile(JSON.stringify(file)).ok).toBe(true)
  const source = inspected.nodes[0]
  expect(source.params.textData).toEqual(graph.nodes[0].params.textData)
  expect(source.params.datasetIndex).toBe(0)
  expect(source.params.datasetSplit).toBe(graph.nodes[0].params.datasetSplit)
  expect(parameterValues(inspected)).toEqual(parameterValues(graph))
  const expected = forwardPass({...graph, nodes: graph.nodes.map(n => n.type === 'dataset' ? {...n, params: {...n.params, storyExperiment: {facts: ['John is in garden.'], question: 'Where is John?'}}} : n)})
  for (const id of ['output', 'attention', 'question_lookup']) expect(inspected.nodes.find(n => n.id === id)!.value).toEqual(expected.graph.nodes.find(n => n.id === id)!.value)
  expect(screen.getByLabelText('attention experiment weights')).toHaveTextContent('1.000000, 0.000000, 0.000000')
  expect(within(screen.getByLabelText('Experimental class probabilities')).getAllByRole('listitem')).toHaveLength(2)
  expect(JSON.stringify(graph)).toBe(original)
  fireEvent.click(screen.getByRole('button', {name: 'Restore original'}))
  expect(screen.getByLabelText('Experiment facts')).toHaveValue('Mary is in kitchen.\nJohn is in garden.')
  expect(screen.queryByLabelText('Experimental result')).not.toBeInTheDocument()
  const restored = save(spy).state.graph
  expect(restored.nodes[0].params.storyExperiment).toBeUndefined()
  expect(datasetOutputValueForSlot(restored.nodes[0], 0)).toEqual(datasetOutputValueForSlot(graph.nodes[0], 0))
  expect(parameterValues(restored)).toEqual(parameterValues(graph))
  expect(screen.getByRole('button', {name: 'Run one full training step'})).toBeEnabled()
})

it('exposes unknown words, rejects truncation, and clears experiments and drafts when selecting another example', () => {
  const graph = storyExperimentModel(), spy = download()
  render(<App initialGraph={graph}/>)
  fireEvent.change(screen.getByLabelText('Experiment facts'), {target: {value: 'Mary is in Atlantis.'}})
  expect(screen.getByText(/Unknown words/)).toHaveTextContent('atlantis')
  fireEvent.click(screen.getByRole('button', {name: 'Apply experiment'}))
  const applied = save(spy).state.graph
  expect(datasetOutputValueForSlot(applied.nodes[0], 0).data[3]).toBe(0)
  fireEvent.change(screen.getByLabelText('Experiment question'), {target: {value: 'Mary '.repeat(7)}})
  expect(screen.getByRole('alert')).toHaveTextContent('Nothing was truncated')
  expect(screen.getByRole('button', {name: 'Apply experiment'})).toBeDisabled()
  expect(save(spy).state.graph.nodes[0].params.storyExperiment).toEqual(applied.nodes[0].params.storyExperiment)
  fireEvent.change(screen.getByLabelText('Dataset row'), {target: {value: '3'}})
  expect(screen.getByLabelText('Experiment question')).toHaveValue('Where is Mary?')
  expect(screen.getByLabelText('Experiment facts')).toHaveValue('Mary is in garden.\nJohn is in kitchen.')
  expect(screen.queryByLabelText('Experimental result')).not.toBeInTheDocument()
  expect(save(spy).state.graph.nodes[0].params.storyExperiment).toBeUndefined()
  expect(parameterValues(save(spy).state.graph)).toEqual(parameterValues(graph))
})
