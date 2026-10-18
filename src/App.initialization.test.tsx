import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import { oneHotModel } from './test/oneHotModel'
import { createNode } from './domain/examples'
import { initializeTensor, INITIALIZERS } from './domain/authoring'
import { parameterValues } from './domain/engine'
import * as session from './domain/session'

afterEach(() => vi.restoreAllMocks())
const download = () => vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
function save(spy: ReturnType<typeof download>) {
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
  return spy.mock.calls.at(-1)![0]
}

it('remembers each block’s method across selection, tabs and saved models without changing current parameters or execution', () => {
  const graph = oneHotModel(), spy = download()
  const { container, unmount } = render(<App initialGraph={graph} />)
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.click(screen.getByText('Step through a lesson'))
  fireEvent.click(screen.getByRole('button', { name: 'Run forward' }))
  const before = save(spy).state
  const select = (id: string) => fireEvent.click(container.querySelector(`[data-id="${id}"]`)!)
  select('table')
  fireEvent.click(screen.getByRole('tab', { name: 'Details' }))
  expect(screen.getByLabelText('Tensor initializer')).toHaveValue('xavier')
  fireEvent.change(screen.getByLabelText('Tensor initializer'), { target: { value: 'ones' } })
  select('head')
  fireEvent.change(screen.getByLabelText('Tensor initializer'), { target: { value: 'zeros' } })
  fireEvent.click(screen.getByRole('tab', { name: 'Data' }))
  fireEvent.click(screen.getByRole('tab', { name: 'Details' }))
  expect(screen.getByLabelText('Tensor initializer')).toHaveValue('zeros')
  select('table')
  expect(screen.getByLabelText('Tensor initializer')).toHaveValue('ones')
  const file = save(spy)
  expect(parameterValues(file.state.graph)).toEqual(parameterValues(before.graph))
  expect(file.state.traceSteps).toEqual(before.traceSteps)
  expect(file.state.phase).toBe(before.phase)
  expect(file.state.currentLoss).toBe(before.currentLoss)
  const parsed = session.parseProjectStateFile(JSON.stringify(file))
  expect(parsed.ok).toBe(true)
  if (!parsed.ok) throw Error(parsed.error)
  unmount()
  render(<App initialGraph={parsed.file.state.graph} />)
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  fireEvent.click(screen.getByRole('button', { name: /Randomize parameters/ }))
  const params = parameterValues(save(spy).state.graph)
  expect(params.table).toEqual({ shape: [3, 2], data: [1, 1, 1, 1, 1, 1] })
  expect(params.head).toEqual({ shape: [2, 1], data: [0, 0] })
})

it('uses all five initialization methods with fresh draws, retaining scalar, matrix and filter shapes', () => {
  const graph = oneHotModel(), spy = download()
  const shapes = [[], [2, 3], [4, 3, 3, 1], [3], [2, 2]]
  const params = INITIALIZERS.map((initializer, index) => ({ ...createNode('weight', index), id: initializer, label: initializer,
    params: { initializer, value: initializeTensor(shapes[index], 'ones') },
  }))
  graph.nodes.push(...params, { ...createNode('bias', 0), id: 'default-bias', label: 'default-bias', params: { value: 9 } })
  render(<App initialGraph={graph} />)
  fireEvent.click(screen.getByRole('tab', { name: 'Train' }))
  const random = vi.spyOn(Math, 'random').mockReturnValue(.25)
  for (const draw of [.25, .75]) {
    random.mockReturnValue(draw)
    fireEvent.click(screen.getByRole('button', { name: /Randomize parameters/ }))
    const state = save(spy).state, values = parameterValues(state.graph)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    params.forEach((node, i) => expect(values[node.id]).toEqual(initializeTensor(shapes[i], node.params.initializer, draw * 0x100000000)))
    expect(values['default-bias']).toEqual({ shape: [], data: [0] })
    expect(state.initialParameterValues).toEqual(values)
    expect(state.epoch).toBe(0)
  }
})
