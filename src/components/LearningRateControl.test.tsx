import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it, vi } from 'vitest'
import { LearningRateControl } from './LearningRateControl'

function Harness() {
  const [value, setValue] = useState(.1)
  return <LearningRateControl value={value} onChange={setValue} tensorTraining={false} disabled={false}/>
}

it('edits the number on double-click, applies Enter or blur, and keeps the slider in sync', async () => {
  const user = userEvent.setup()
  render(<Harness/> )
  await user.dblClick(screen.getByRole('button', { name: 'Edit learning rate: 0.1' }))
  const input = screen.getByRole('spinbutton', { name: 'Learning rate value' })
  await user.clear(input)
  await user.type(input, '0.025{Enter}')
  expect(screen.getByRole('button', { name: 'Edit learning rate: 0.025' })).toBeInTheDocument()
  expect(screen.getByRole('slider', { name: 'Learning rate' })).toHaveValue('0.025')
  await user.dblClick(screen.getByRole('button', { name: 'Edit learning rate: 0.025' }))
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '1e-5' } })
  await user.tab()
  expect(screen.getByRole('button', { name: 'Edit learning rate: 0.00001' })).toBeInTheDocument()
  fireEvent.change(screen.getByRole('slider'), { target: { value: '0.2' } })
  expect(screen.getByRole('button', { name: 'Edit learning rate: 0.2' })).toBeInTheDocument()
})

it('cancels with Escape and rejects empty or nonpositive values', async () => {
  const user = userEvent.setup()
  render(<Harness/> )
  for (const value of ['', '0', '-1']) {
    await user.dblClick(screen.getByRole('button', { name: 'Edit learning rate: 0.1' }))
    const input = screen.getByRole('spinbutton')
    fireEvent.change(input, { target: { value } })
    fireEvent.blur(input)
    expect(screen.getByRole('alert')).toHaveTextContent('positive, finite')
    await user.click(input)
    await user.keyboard('{Escape}')
    expect(screen.getByRole('button', { name: 'Edit learning rate: 0.1' })).toBeInTheDocument()
  }
  await user.dblClick(screen.getByRole('button', { name: 'Edit learning rate: 0.1' }))
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '0.3' } })
  await user.keyboard('{Escape}')
  expect(screen.getByRole('slider')).toHaveValue('0.1')
})

it('disables the editor and slider during training', () => {
  const onChange = vi.fn()
  render(<LearningRateControl value={.1} onChange={onChange} tensorTraining disabled/>)
  expect(screen.getByRole('button')).toBeDisabled()
  expect(screen.getByRole('slider')).toBeDisabled()
  fireEvent.doubleClick(screen.getByRole('button'))
  expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument()
  expect(onChange).not.toHaveBeenCalled()
})
