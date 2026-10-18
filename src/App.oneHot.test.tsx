import { initializeTensor, initializerForNode } from './domain/authoring'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import { oneHotModel } from './test/oneHotModel'
import { forwardPass, parameterValues } from './domain/engine'
import * as session from './domain/session'

const download = () => vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
function saved(spy: ReturnType<typeof download>) {
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
  return spy.mock.calls.at(-1)![0].state
}
function lesson() {
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.click(screen.getByText('Step through a lesson'))
}
afterEach(() => vi.restoreAllMocks())

it('runs forward and randomizes One-hot models with the preferred parameter initialization', () => {
  const graph = oneHotModel(), spy = download()
  render(<App initialGraph={graph}/>)
  lesson()
  fireEvent.click(screen.getByRole('button', { name: 'Run forward' }))
  const before = saved(spy)
  expect(before.traceSteps.find(step => step.nodeId === 'encoding')!.calculation).toContain('one-hot rows')
  expect(before.graph.nodes.find(node => node.id === 'probability')!.value).toEqual(forwardPass(oneHotModel(true), true).graph.nodes.find(node => node.id === 'probability')!.value)
  expect(parameterValues(before.graph)).toEqual(parameterValues(graph))
  vi.spyOn(Math, 'random').mockReturnValue(.75)
  fireEvent.click(screen.getByRole('button', { name: /Randomize parameters/ }))
  const after = saved(spy)
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(after.epoch).toBe(0)
  expect(after.traceSteps).toEqual([])
  for (const node of graph.nodes.filter(node => node.type === 'weight')) {
    expect(parameterValues(after.graph)[node.id]).toEqual(initializeTensor(parameterValues(graph)[node.id].shape, initializerForNode(node), 0xc0000000))
  }
  expect(after.initialParameterValues).toEqual(parameterValues(after.graph))
  expect(after.graph.nodes.find(node => node.id === 'ids')!.params).toEqual(graph.nodes.find(node => node.id === 'ids')!.params)
  expect(after.graph.nodes.find(node => node.id === 'probability')!.value).toEqual(forwardPass(after.graph, false).graph.nodes.find(node => node.id === 'probability')!.value)
  fireEvent.click(screen.getByRole('button', { name: 'Run forward' }))
  expect(saved(spy).traceSteps.some(step => step.nodeId === 'encoding')).toBe(true)
})

it('keeps parameters, execution state, reports, and undo history when randomized IDs fail evaluation', async () => {
  // Valid initial IDs become nonintegers when randomized. This exercises a real
  // evaluation error rather than mocking the forward evaluator.
  const graph = oneHotModel(false, true), spy = download()
  const { container } = render(<App initialGraph={graph}/>)
  lesson()
  fireEvent.click(screen.getByRole('button', { name: 'Run one full training step' }))
  expect(screen.getByText('Epoch 1')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('tab', { name: 'Reporting' }))
  await waitFor(() => expect(screen.getByRole('list', { name: 'Reported losses' }).children).toHaveLength(2))
  const reports = screen.getByRole('list', { name: 'Reported losses' }).textContent
  const before = saved(spy)
  vi.spyOn(Math, 'random').mockReturnValue(.25)
  fireEvent.click(screen.getByRole('button', { name: /Randomize parameters/ }))
  expect(screen.getByRole('alert')).toHaveTextContent('Could not randomize parameters: One-hot needs integer IDs')
  expect(screen.getByRole('alert')).toHaveTextContent('previous parameters and execution state were kept')
  const after = saved(spy)
  for (const key of ['graph', 'visualizationGraph', 'initialParameterValues', 'phase', 'traceSteps', 'traceIndex', 'epoch', 'currentLoss'] as const) expect(after[key]).toEqual(before[key])
  fireEvent.click(screen.getByRole('tab', { name: 'Reporting' }))
  expect(screen.getByRole('list', { name: 'Reported losses' })).toHaveTextContent(reports!)
  // Failure must not add a dummy undo checkpoint ahead of the training step.
  fireEvent.keyDown(document, { key: 'z', ctrlKey: true })
  expect(screen.getByText('Epoch 0')).toBeInTheDocument()
  expect(parameterValues(saved(spy).graph)).toEqual(parameterValues(graph))
  fireEvent.click(screen.getByRole('button', { name: 'Run forward' }))
  expect(saved(spy).traceSteps.some(step => step.nodeId === 'encoding')).toBe(true)
  // Choose a valid initialization for the discrete IDs and retry.
  fireEvent.click(container.querySelector('[data-id="ids"]')!)
  fireEvent.click(screen.getByRole('tab', { name: 'Details' }))
  fireEvent.change(screen.getByLabelText('Tensor initializer'), { target: { value: 'zeros' } })
  fireEvent.click(screen.getByRole('button', { name: /Randomize parameters/ }))
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(parameterValues(saved(spy).graph).ids.data).toEqual([0, 0, 0])
})
