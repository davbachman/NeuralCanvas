import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import * as session from './domain/session'
import { parameterValues, runTrainingStep } from './domain/engine'
import { withDatasetExample } from './domain/datasetTraining'
import { scalarValue } from './domain/tensor'
import { heldOutInspectionModel } from './test/heldOutInspectionModel'

const download = () => vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
function saveGraph(spy: ReturnType<typeof download>) {
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
  return spy.mock.calls.at(-1)![0].state.graph
}
function chooseHeldOut(container: HTMLElement, index: number) {
  fireEvent.click(container.querySelector('[data-id="starter-data"]')!)
  fireEvent.change(screen.getByLabelText('Dataset split'), { target: { value: 'test' } })
  fireEvent.change(screen.getByLabelText('Dataset example'), { target: { value: String(index) } })
}
function openLesson() {
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.click(screen.getByText('Step through a lesson'))
}
afterEach(() => vi.restoreAllMocks())

it.each([2, 3])('preserves held-out row %i across tabs and inspects its probability without updates', index => {
  const graph = heldOutInspectionModel(), spy = download()
  const { container } = render(<App initialGraph={graph} />)
  chooseHeldOut(container, index)
  const before = saveGraph(spy)
  for (const name of ['Train', 'Test', 'Build', 'Train']) fireEvent.click(screen.getByRole('tab', { name }))
  expect(screen.getByLabelText('Dataset split')).toHaveValue('test')
  expect(screen.getByLabelText('Dataset example')).toHaveValue(String(index))
  expect(saveGraph(spy).nodes.find(n => n.type === 'dataset')!.params).toEqual(before.nodes.find(n => n.type === 'dataset')!.params)
  fireEvent.click(screen.getByText('Step through a lesson'))
  expect(screen.getByRole('button', { name: 'Run forward' })).toBeEnabled()
  const update = screen.getByRole('button', { name: 'Run one full training step' })
  expect(update).toBeDisabled()
  fireEvent.click(update)
  fireEvent.click(screen.getByRole('button', { name: 'Run forward' }))
  const inspected = saveGraph(spy)
  const x = index === 2 ? -2 : 2
  expect(inspected.nodes.find(n => n.id === 'pred')!.value!.data[0]).toBeCloseTo(1 / (1 + Math.exp(-(.5 * x - .3))), 12)
  expect(parameterValues(inspected)).toEqual(parameterValues(before))
  expect(screen.getByLabelText('Dataset example')).toHaveValue(String(index))
  // Even repeatedly stepping beyond the backward pass must not update held-out parameters.
  for (let step = 0; step < 20; step++) fireEvent.click(screen.getByRole('button', { name: /^Step$/ }))
  expect(parameterValues(saveGraph(spy))).toEqual(parameterValues(before))
  expect(screen.getByText('Epoch 0')).toBeInTheDocument()
})

it('preserves temporary dataset overrides and uses them for forward inspection', () => {
  const graph = heldOutInspectionModel(), spy = download()
  const source = graph.nodes.find(n => n.type === 'dataset')!
  source.params = { ...source.params, datasetSplit: 'test', datasetIndex: 3, datasetValues: [scalarValue(4), scalarValue(1)] }
  render(<App initialGraph={graph} />)
  openLesson()
  fireEvent.click(screen.getByRole('button', { name: 'Run forward' }))
  fireEvent.click(screen.getByRole('tab', { name: 'Test' }))
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  const inspected = saveGraph(spy)
  expect(inspected.nodes.find(n => n.type === 'dataset')!.params).toEqual(source.params)
  expect(inspected.nodes.find(n => n.id === 'pred')!.value!.data[0]).toBeCloseTo(1 / (1 + Math.exp(-1.7)), 12)
  expect(parameterValues(inspected)).toEqual(parameterValues(graph))
})

it('runs epochs on training rows only while leaving the selected held-out row intact', async () => {
  const graph = heldOutInspectionModel(), spy = download()
  const { container } = render(<App initialGraph={graph} />)
  chooseHeldOut(container, 3)
  const before = saveGraph(spy)
  let expected = before
  for (const index of [0, 1]) expected = runTrainingStep(withDatasetExample(expected, 'starter-data', index)).graph
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.change(screen.getByLabelText('Epochs per run'), { target: { value: '1' } })
  fireEvent.change(screen.getByLabelText('Examples per update'), { target: { value: '1' } })
  fireEvent.click(screen.getByLabelText('Reshuffle training examples each epoch'))
  expect(screen.getByRole('button', { name: /Run 1 epoch/ })).toBeEnabled()
  fireEvent.click(screen.getByRole('button', { name: /Run 1 epoch/ }))
  await waitFor(() => expect(screen.getByText('Completed 1 epoch.')).toBeInTheDocument())
  const after = saveGraph(spy)
  expect(parameterValues(after)).toEqual(parameterValues(expected))
  expect(parameterValues(after)).not.toEqual(parameterValues(before))
  expect(after.nodes.find(n => n.type === 'dataset')!.params).toEqual(before.nodes.find(n => n.type === 'dataset')!.params)
  expect(screen.getByLabelText('Dataset split')).toHaveValue('test')
  expect(screen.getByLabelText('Dataset example')).toHaveValue('3')
})
