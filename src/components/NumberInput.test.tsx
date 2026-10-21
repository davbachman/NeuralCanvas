import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it, vi } from 'vitest'
import { NumberInput } from './NumberInput'

it('allows clearing zero before typing a replacement without changing the saved number', async () => {
  const changed = vi.fn()
  function Field() {
    const [value, setValue] = useState(0)
    return <NumberInput aria-label="Number" value={value} onChange={event => {
      changed(Number(event.target.value))
      setValue(Number(event.target.value))
    }} />
  }
  render(<Field />)
  const user = userEvent.setup(), input = screen.getByLabelText('Number')
  await user.clear(input)
  expect(input).toHaveValue(null)
  expect(changed).not.toHaveBeenCalled()
  await user.type(input, '1')
  expect(input).toHaveValue(1)
  expect(changed).toHaveBeenLastCalledWith(1)
  await user.clear(input)
  await user.tab()
  expect(input).toHaveValue(1)
  expect(changed).toHaveBeenCalledTimes(1)
})

it.each(['-2.5', '0.001', '1e-3', '0'])('keeps immediate numeric updates for %s', text => {
  const changed = vi.fn()
  render(<NumberInput value={7} aria-label="Number" onChange={event => changed(Number(event.target.value))} />)
  fireEvent.change(screen.getByLabelText('Number'), { target: { value: text } })
  expect(changed).toHaveBeenCalledWith(Number(text))
})

it('lets a rejected intermediate value be completed and restores accepted values on blur or Enter', () => {
  function Field() {
    const [value, setValue] = useState(20)
    return <NumberInput aria-label="Number" value={value} min={10} max={99} onChange={event => {
      const next = Number(event.target.value)
      if (next >= 10 && next <= 99) setValue(next)
    }} />
  }
  render(<Field />)
  const input = screen.getByLabelText('Number')
  fireEvent.change(input, { target: { value: '3' } })
  expect(input).toHaveValue(3)
  fireEvent.change(input, { target: { value: '35' } })
  fireEvent.blur(input)
  expect(input).toHaveValue(35)
  fireEvent.change(input, { target: { value: '4' } })
  fireEvent.keyDown(input, { key: 'Enter' })
  expect(input).toHaveValue(35)
})

it('restores unfinished edits with Escape and follows external model changes', () => {
  const { rerender } = render(<NumberInput aria-label="Number" value={5} />)
  const input = screen.getByLabelText('Number')
  fireEvent.change(input, { target: { value: '' } })
  fireEvent.keyDown(input, { key: 'Escape' })
  expect(input).toHaveValue(5)
  fireEvent.change(input, { target: { value: '' } })
  rerender(<NumberInput aria-label="Number" value={12} />)
  expect(input).toHaveValue(12)
})
