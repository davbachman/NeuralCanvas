import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { expect, it } from 'vitest'
import App from './App'
import { createModelPreset } from './domain/modelPresets'

it.each(['Enter', 'click'])('creates an editable addition block from the canvas shortcut using %s', async method => {
  const { container } = render(<App initialGraph={createModelPreset('blank')} />)
  fireEvent.doubleClick(container.querySelector('.react-flow__pane')!, { clientX: 300, clientY: 200 })
  const search = within(screen.getByRole('dialog', { name: 'Add a block' })).getByRole('searchbox', { name: 'Search blocks' })
  fireEvent.change(search, { target: { value: '+' } })
  if (method === 'Enter') fireEvent.keyDown(search, { key: 'Enter' })
  else fireEvent.click(screen.getByRole('option', { name: /Arithmetic/ }))
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Arithmetic expression' })).toHaveValue('x1 + x2'))
  expect(container.querySelectorAll('.builder-node .node-handle.target')).toHaveLength(2)
  expect(screen.queryByRole('dialog', { name: 'Add a block' })).not.toBeInTheDocument()
})

it.each([
  ['resh', 'reshape', 'Enter'],
  [' TRANSPOSE ', 'transpose', 'click'],
  ['sli', 'slice', 'Enter'],
  ['mean', 'mean', 'click'],
])('maps %s to Tensor transform %s using %s', async (query, operation, method) => {
  const { container } = render(<App initialGraph={createModelPreset('blank')} />)
  fireEvent.doubleClick(container.querySelector('.react-flow__pane')!, { clientX: 300, clientY: 200 })
  const dialog = screen.getByRole('dialog', { name: 'Add a block' })
  const search = within(dialog).getByRole('searchbox', { name: 'Search blocks' })
  fireEvent.change(search, { target: { value: query } })
  if (method === 'Enter') fireEvent.keyDown(search, { key: 'Enter' })
  else fireEvent.click(within(dialog).getByRole('option', { name: /Tensor transform/ }))
  await waitFor(() => expect(screen.getByLabelText('Tensor transform operation')).toHaveValue(operation))
  expect(screen.queryByRole('dialog', { name: 'Add a block' })).not.toBeInTheDocument()
})
