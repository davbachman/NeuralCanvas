import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import { createStarterGraph } from './domain/examples'
import { parameterValues } from './domain/engine'
import * as session from './domain/session'

afterEach(() => vi.restoreAllMocks())

it('allows replacing training settings without applying an empty draft to the model', () => {
  const graph = createStarterGraph(true)
  const save = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  render(<App initialGraph={graph} />)
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  const patience = screen.getByLabelText('Early stopping patience')
  fireEvent.change(patience, { target: { value: '' } })
  expect(patience).toHaveValue(null)
  fireEvent.change(patience, { target: { value: '1' } })
  expect(patience).toHaveValue(1)
  const rate = screen.getByLabelText('Exact learning rate')
  fireEvent.change(rate, { target: { value: '' } })
  expect(rate).toHaveValue(null)
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
  const saved = save.mock.calls.at(-1)![0].state.graph
  expect(saved.learningRate).toBe(graph.learningRate)
  expect(saved.training?.patience).toBe(1)
  expect(parameterValues(saved)).toEqual(parameterValues(graph))
  fireEvent.blur(rate)
  expect(rate).toHaveValue(graph.learningRate)
})

it('allows replacing tensor coordinates in Details without writing zero while clearing', () => {
  const graph = createStarterGraph()
  const weight = graph.nodes.find(node => node.id === 'w')!
  weight.params.value = weight.value = { shape: [1, 2], data: [.5, .2] }
  const save = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  const { container } = render(<App initialGraph={graph} />)
  fireEvent.click(container.querySelector('[data-id="w"]')!)
  fireEvent.click(screen.getByRole('tab', { name: 'Details' }))
  const coordinate = screen.getByLabelText('Selected matrix coordinate')
  fireEvent.change(coordinate, { target: { value: '' } })
  expect(coordinate).toHaveValue(null)
  fireEvent.blur(coordinate)
  expect(coordinate).toHaveValue(.5)
  fireEvent.change(coordinate, { target: { value: '' } })
  fireEvent.change(coordinate, { target: { value: '-1.5' } })
  expect(coordinate).toHaveValue(-1.5)
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
  const saved = save.mock.calls.at(-1)![0].state.graph
  expect(saved.nodes.find(node => node.id === 'w')!.params.value).toEqual({ shape: [1, 2], data: [-1.5, .2] })
})
