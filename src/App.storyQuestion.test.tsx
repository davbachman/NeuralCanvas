import { fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import * as session from './domain/session'
import { storyQuestionData, storyQuestionModel } from './test/storyQuestionModel'
import { forwardPass, parameterValues } from './domain/engine'
import { withDatasetExample } from './domain/datasetTraining'
import { datasetOutputValueForSlot } from './domain/datasets'

afterEach(() => vi.restoreAllMocks())

it('preserves a custom held-out question across tabs, runs its actual inputs, inspects probabilities and restores the original', () => {
  const graph = storyQuestionModel(), original = structuredClone(graph)
  const download = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  const {container} = render(<App initialGraph={graph}/>)
  const save = () => {
    fireEvent.click(screen.getByRole('button', {name: 'File'}))
    fireEvent.click(screen.getByRole('menuitem', {name: 'Save'}))
    return download.mock.calls.at(-1)![0]
  }
  const select = (id: string, tab: 'Data' | 'Details') => {
    fireEvent.click(container.querySelector(`[data-id="${id}"]`)!)
    fireEvent.click(screen.getByRole('tab', {name: tab}))
  }
  select('text-data', 'Details')
  fireEvent.change(screen.getByLabelText('Dataset split'), {target: {value: 'test'}})
  fireEvent.change(screen.getByLabelText('Dataset row'), {target: {value: '2'}})
  expect(screen.getByLabelText('Experiment story')).toHaveValue('Mary garden')
  expect(screen.getByLabelText('Experiment question')).toHaveValue('where Mary ?')
  expect(within(screen.getByRole('list', {name: 'Story token IDs'})).getAllByRole('listitem')).toHaveLength(32)
  expect(within(screen.getByRole('list', {name: 'Question token IDs'})).getAllByRole('listitem')).toHaveLength(8)
  expect(screen.queryByLabelText('Review representation')).not.toBeInTheDocument()
  const before = forwardPass(withDatasetExample(graph, 'text-data', 2)).graph
  fireEvent.change(screen.getByLabelText('Experiment question'), {target: {value: 'where John ?'}})
  fireEvent.click(screen.getByRole('button', {name: 'Apply experiment'}))
  for (const name of ['Train', 'Test', 'Build', 'Train', 'Data', 'Code', 'Reporting', 'Details']) fireEvent.click(screen.getByRole('tab', {name}))
  expect(screen.getByLabelText('Dataset row')).toHaveValue('2')
  expect(screen.getByLabelText('Dataset split')).toHaveValue('test')
  expect(screen.getByLabelText('Experiment story')).toHaveValue('Mary garden')
  expect(screen.getByLabelText('Experiment question')).toHaveValue('where John ?')
  expect(screen.getByRole('button', {name: 'Run one full training step'})).toBeDisabled()
  expect(screen.getByText(/Original answer \(reference only\)/)).toHaveTextContent('garden')
  expect(screen.getByText(/Experimental answer: unknown/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', {name: 'Run forward'}))
  const file = save(), actual = file.state.graph, source = actual.nodes[0]
  expect(session.parseProjectStateFile(JSON.stringify(file)).ok).toBe(true)
  expect(source.params.storyExperiment).toEqual({text: 'Mary garden', question: 'where John ?'})
  expect(source.params.datasetValues).toBeUndefined()
  expect(source.params.datasetIndex).toBe(2)
  expect(source.params.datasetSplit).toBe('test')
  expect(source.params.textData).toEqual(graph.nodes[0].params.textData)
  expect(datasetOutputValueForSlot(source, 0)).toEqual(datasetOutputValueForSlot(before.nodes[0], 0))
  expect(datasetOutputValueForSlot(source, 1)).not.toEqual(datasetOutputValueForSlot(before.nodes[0], 1))
  for (const id of ['probabilities', 'attention', 'question_lookup']) expect(actual.nodes.find(n => n.id === id)!.value).not.toEqual(before.nodes.find(n => n.id === id)!.value)
  const expected = forwardPass({...graph, nodes: graph.nodes.map(n => n.type === 'dataset' ? {...n, params: source.params} : n)}).graph
  for (const id of ['probabilities', 'attention', 'question_lookup']) expect(actual.nodes.find(n => n.id === id)!.value).toEqual(expected.nodes.find(n => n.id === id)!.value)
  expect(parameterValues(actual)).toEqual(parameterValues(graph))
  expect(within(screen.getByLabelText('Experimental class probabilities')).getAllByRole('listitem')).toHaveLength(2)
  expect(screen.getByLabelText('attention experiment weights')).toBeInTheDocument()
  select('probabilities', 'Data')
  expect(screen.getByLabelText('Dataset inspection context')).toHaveTextContent('Dataset row 2 — modified input')
  select('text-data', 'Details')
  fireEvent.click(screen.getByRole('button', {name: 'Restore original'}))
  expect(screen.getByLabelText('Experiment question')).toHaveValue('where Mary ?')
  expect(screen.queryByLabelText('Experimental result')).not.toBeInTheDocument()
  const restored = save().state.graph
  expect(restored.nodes[0].params.storyExperiment).toBeUndefined()
  expect(datasetOutputValueForSlot(restored.nodes[0], 1)).toEqual(datasetOutputValueForSlot(before.nodes[0], 1))
  expect(parameterValues(restored)).toEqual(parameterValues(graph))
  select('probabilities', 'Data')
  expect(screen.getByLabelText('Dataset inspection context')).toHaveTextContent(/^Dataset row 2$/)
  expect(graph).toEqual(original)
})

it('validates independently sized draft fields, exposes unknown words, and clears applied edits when selecting another row', () => {
  const graph = storyQuestionModel(storyQuestionData(5, 4)), download = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  const {container} = render(<App initialGraph={graph}/>)
  fireEvent.click(container.querySelector('[data-id="text-data"]')!)
  fireEvent.click(screen.getByRole('tab', {name: 'Details'}))
  fireEvent.change(screen.getByLabelText('Experiment story'), {target: {value: 'Mary Atlantis garden'}})
  expect(screen.getByRole('status')).toHaveTextContent('atlantis')
  fireEvent.click(screen.getByRole('button', {name: 'Apply experiment'}))
  expect(within(screen.getByRole('list', {name: 'Story token IDs'})).getAllByRole('listitem')[1]).toHaveTextContent('"<unk>"')
  fireEvent.change(screen.getByLabelText('Experiment question'), {target: {value: 'where '.repeat(5)}})
  expect(screen.getByRole('alert')).toHaveTextContent('Question has 5 tokens; the limit is 4. Nothing was truncated.')
  expect(screen.getByRole('button', {name: 'Apply experiment'})).toBeDisabled()
  fireEvent.change(screen.getByLabelText('Experiment question'), {target: {value: 'where Mary ?'}})
  fireEvent.change(screen.getByLabelText('Experiment story'), {target: {value: 'Mary '.repeat(6)}})
  expect(screen.getByRole('alert')).toHaveTextContent('Story has 6 tokens; the limit is 5. Nothing was truncated.')
  expect(screen.getByRole('button', {name: 'Apply experiment'})).toBeDisabled()
  fireEvent.change(screen.getByLabelText('Dataset row'), {target: {value: '1'}})
  expect(screen.getByLabelText('Experiment story')).toHaveValue('John garden')
  expect(screen.getByLabelText('Experiment question')).toHaveValue('where John ?')
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('Experimental result')).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', {name: 'File'}))
  fireEvent.click(screen.getByRole('menuitem', {name: 'Save'}))
  const saved = download.mock.calls.at(-1)![0].state.graph
  expect(saved.nodes[0].params.storyExperiment).toBeUndefined()
  expect(saved.nodes[0].params.textData).toEqual(graph.nodes[0].params.textData)
  expect(parameterValues(saved)).toEqual(parameterValues(graph))
})
