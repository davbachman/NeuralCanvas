import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import { forwardPass } from './domain/engine'
import { createNode, createStarterGraph } from './domain/examples'
import * as session from './domain/session'

afterEach(() => vi.restoreAllMocks())

it.each(['unconnected loss', 'extra loss', 'missing model input'])(
  'runs forward with an %s while keeping training disabled', problem => {
    const graph = forwardPass(createStarterGraph(true)).graph
    const expected = graph.nodes.find(node => node.id === 'pred')!.value
    if (problem === 'unconnected loss') graph.edges = graph.edges.filter(edge => edge.target !== 'loss')
    if (problem === 'extra loss') graph.nodes.unshift(createNode('loss', 99))
    if (problem === 'missing model input') graph.edges = graph.edges.filter(edge => edge.id !== 'w-mul')
    const download = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
    const { container } = render(<App initialGraph={graph} />)
    fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
    fireEvent.click(screen.getByText('Step through a lesson'))

    const run = screen.getByRole('button', { name: 'Run forward' })
    expect(run).toBeEnabled()
    expect(screen.getByRole('button', { name: /Run 10 epochs/ })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Run one full training step' })).toBeDisabled()
    fireEvent.click(run)
    fireEvent.click(screen.getByRole('button', { name: 'File' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))

    const state = download.mock.calls[0][0].state
    const lossReached = problem === 'extra loss'
    expect(state.phase).toBe(lossReached ? 'loss' : 'forward')
    expect(state.currentLoss).toEqual(lossReached ? graph.nodes.find(node => node.id === 'loss')!.value!.data[0] : null)
    expect(state.graph.nodes.find(node => node.id === 'pred')?.value).toEqual(problem === 'missing model input' ? undefined : expected)
    expect(state.visualizationGraph).toEqual(state.graph)
    expect(state.traceSteps.map(step => step.nodeId)).toEqual(problem === 'missing model input' ? [] : lossReached ? ['mul', 'add', 'pred', 'loss'] : ['mul', 'add', 'pred'])
    expect(state.epoch).toBe(0)

    fireEvent.click(container.querySelector('[data-id="pred"]')!)
    fireEvent.click(screen.getByRole('tab', { name: 'Data' }))
    if (problem === 'missing model input') {
      expect(screen.getByText(/No value has reached this signal yet/)).toBeInTheDocument()
    } else {
      expect(screen.queryByText(/No value has reached this signal yet/)).not.toBeInTheDocument()
    }
  },
)

it('displays a partial forward pass with an unreachable cycle of input blocks', () => {
  const graph = createStarterGraph(true)
  const first = { ...createNode('input', 98), id: 'first' }
  const second = { ...createNode('input', 99), id: 'second' }
  graph.nodes.push(first, second)
  graph.edges.push(
    { id: 'first-second', source: 'first', target: 'second', inputSlot: 0 },
    { id: 'second-first', source: 'second', target: 'first', inputSlot: 0 },
  )
  const { container } = render(<App initialGraph={graph} />)
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.click(screen.getByRole('button', { name: 'Run forward' }))
  fireEvent.click(container.querySelector('[data-id="pred"]')!)
  fireEvent.click(screen.getByRole('tab', { name: 'Data' }))
  expect(screen.queryByText(/No value has reached this signal yet/)).not.toBeInTheDocument()
})
