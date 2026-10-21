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

it('shows applicable controls for each Tensor optimizer and preserves hidden settings', () => {
  const graph = createStarterGraph(true)
  graph.training = { ...DEFAULT_TRAINING, weightDecay: .025, minDelta: .004 }
  render(<App initialGraph={graph} />)
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  for (const optimizer of ['adamw', 'adam', 'sgd', 'adamw']) {
    fireEvent.change(screen.getByLabelText('Optimizer'), { target: { value: optimizer } })
    if (optimizer === 'adamw') expect(screen.getByLabelText('Weight decay')).toHaveValue(.025)
    else expect(screen.queryByLabelText('Weight decay')).not.toBeInTheDocument()
    expect(Boolean(screen.queryByText(/Running gradient averages start fresh/))).toBe(optimizer !== 'sgd')
    expect(screen.getByLabelText('Learning rate')).toBeEnabled()
    expect(screen.getByLabelText('Examples per update')).toBeEnabled()
    expect(screen.getByLabelText('Gradient norm limit')).toBeEnabled()
    expect(screen.getByLabelText('Early stopping patience')).toBeEnabled()
  }
  expect(screen.queryByLabelText('Minimum validation improvement')).not.toBeInTheDocument()
  for (const patience of [3, 0, 3]) {
    fireEvent.change(screen.getByLabelText('Early stopping patience'), { target: { value: String(patience) } })
    if (patience) expect(screen.getByLabelText('Minimum validation improvement')).toHaveValue(.004)
    else expect(screen.queryByLabelText('Minimum validation improvement')).not.toBeInTheDocument()
    expect(Boolean(screen.queryByText(/Early stopping checks the held-out/))).toBe(patience > 0)
  }
})
