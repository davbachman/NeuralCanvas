import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import { CodeOutline } from './CodeOutline'
import { createNode } from '../domain/examples'
import type { GraphModel } from '../domain/types'

it('lazily expands parameter values, preserves navigation, and shows updated values', () => {
  const node = createNode('weight', 1)
  const data = Array.from({ length: 20 }, (_, index) => index + .125)
  const read = vi.fn(() => 10.125)
  Object.defineProperty(data, 10, { get: read })
  node.params.value = { shape: [4, 5], data }
  node.params.outputName = 'kernel'
  const graph: GraphModel = { nodes: [node], edges: [], learningRate: .1 }
  const onNavigate = vi.fn()
  const { container, rerender } = render(<CodeOutline graph={graph} active onNavigate={onNavigate}/>)
  const toggle = screen.getByRole('button', { name: 'Expand parameter values for kernel' })
  expect(toggle).toHaveAttribute('aria-expanded', 'false')
  expect(container).toHaveTextContent('kernel = [4, 5] [0.125,...]')
  expect(container).not.toHaveTextContent('10.125')
  expect(read).not.toHaveBeenCalled()
  fireEvent.click(toggle)
  expect(read).toHaveBeenCalled()
  expect(container).toHaveTextContent('10.125')
  expect(onNavigate).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: /^Param 1: kernel =/ }))
  expect(onNavigate).toHaveBeenCalledWith({ kind: 'node', id: node.id })
  const changed = { ...graph, nodes: [{ ...node, params: { ...node.params, value: { shape: [4, 5], data: Array(20).fill(99.875) } } }] }
  rerender(<CodeOutline graph={changed} active onNavigate={onNavigate}/>)
  expect(container).toHaveTextContent('99.875, 99.875')
  expect(container).not.toHaveTextContent('10.125')
  fireEvent.click(screen.getByRole('button', { name: 'Collapse parameter values for kernel' }))
  expect(container).toHaveTextContent('kernel = [4, 5] [99.875,...]')
  expect(container).not.toHaveTextContent('99.875, 99.875')
})

it('opens selected parameter ancestors without automatically expanding their values', async () => {
  const node = createNode('bias', 1)
  node.params.value = { shape: [16], data: Array(16).fill(.5) }
  const graph: GraphModel = { nodes: [node], edges: [], learningRate: .1, groups: [
    { id: 'group', label: 'Layer', nodeIds: [node.id], position: { x: 0, y: 0 }, dimensions: { width: 200, height: 200 } },
  ] }
  render(<CodeOutline graph={graph} active selected={{ kind: 'node', id: node.id }} onNavigate={vi.fn()}/>)
  await waitFor(() => expect(screen.getByRole('button', { name: 'Collapse Layer' })).toHaveAttribute('aria-expanded', 'true'))
  expect(screen.getByRole('button', { name: 'Expand parameter values for p1' })).toHaveAttribute('aria-expanded', 'false')
  expect(screen.getByRole('button', { name: /^b1: p1 =/ })).toHaveAttribute('aria-current', 'true')
})

it('keeps short scalar parameter assignments visible', () => {
  const node = createNode('weight', 1)
  render(<CodeOutline graph={{ nodes: [node], edges: [], learningRate: .1 }} active onNavigate={vi.fn()}/>)
  expect(screen.getByRole('button', { name: 'Param 1: p1 = 0.500' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: /Expand parameter/ })).not.toBeInTheDocument()
})
