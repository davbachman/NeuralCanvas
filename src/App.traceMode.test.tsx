import { fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import { createStarterGraph } from './domain/examples'
import { backwardPass, forwardPass, parameterValues, runTrainingStep } from './domain/engine'
import { formatCompactTensor } from './domain/tensor'
import { DEFAULT_TRAINING } from './domain/trainingSettings'
import * as session from './domain/session'

afterEach(() => vi.restoreAllMocks())

it('switches between compact tensor cards with optimizer controls and Trace readouts with SGD', () => {
  const graph = createStarterGraph(true)
  graph.training = { ...DEFAULT_TRAINING, optimizer: 'adamw' }
  const save = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  const { container } = render(<App initialGraph={graph} />)
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  expect(screen.getByLabelText('Optimizer')).toHaveValue('adamw')
  expect(screen.getByLabelText('Weight decay')).toBeInTheDocument()
  expect(container.querySelector('.node-trace-metrics')).toBeNull()
  const engine = screen.getByLabelText('Training execution')
  fireEvent.change(engine, { target: { value: 'trace' } })
  expect(container.querySelectorAll('.node-trace-metrics')).toHaveLength(graph.nodes.length)
  for (const label of ['Optimizer', 'Weight decay', 'Training backend', 'Gradient norm limit', 'Early stopping patience', 'Minimum validation improvement', 'Exact learning rate']) {
    expect(screen.queryByLabelText(label)).not.toBeInTheDocument()
  }
  expect(screen.getByLabelText('Learning rate')).toBeEnabled()
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
  const saved = save.mock.calls.at(-1)![0].state.graph
  expect(saved.training).toMatchObject({ engine: 'trace', optimizer: 'sgd' })
  expect(parameterValues(saved)).toEqual(parameterValues(graph))
  fireEvent.change(engine, { target: { value: 'tensor' } })
  expect(screen.getByLabelText('Optimizer')).toBeInTheDocument()
  expect(container.querySelector('.node-trace-metrics')).toBeNull()
})

it('shows step-visible outputs and gradients without a dataset and updates parameters with SGD', () => {
  const graph = createStarterGraph()
  const forward = forwardPass(graph, true), backward = backwardPass(forward.graph)
  const expected = runTrainingStep(graph).graph
  const save = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  const { container } = render(<App initialGraph={graph} />)
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.change(screen.getByLabelText('Training execution'), { target: { value: 'trace' } })
  const node = (id: string) => container.querySelector<HTMLElement>(`.react-flow__node[data-id="${id}"]`)!
  expect(within(node('w')).getByLabelText('Gradient preview')).toHaveTextContent('--')
  fireEvent.click(screen.getByRole('button', { name: 'Run forward' }))
  expect(within(node('mul')).getByLabelText('Output preview')).toHaveTextContent(formatCompactTensor(forward.graph.nodes.find(n => n.id === 'mul')!.value))
  expect(within(node('w')).getByLabelText('Gradient preview')).toHaveTextContent('--')
  fireEvent.click(screen.getByText('Step through a lesson'))
  for (let i = 0; i < backward.steps.length; i++) fireEvent.click(screen.getByRole('button', { name: /^Step$/ }))
  expect(within(node('w')).getByLabelText('Gradient preview')).toHaveTextContent(formatCompactTensor(backward.graph.nodes.find(n => n.id === 'w')!.grad))
  fireEvent.click(screen.getByRole('button', { name: /^Step$/ }))
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
  expect(parameterValues(save.mock.calls.at(-1)![0].state.graph)).toEqual(parameterValues(expected))
})
