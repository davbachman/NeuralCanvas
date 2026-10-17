import { act, fireEvent, render } from '@testing-library/react'
import { useRef } from 'react'
import { expect, it } from 'vitest'
import { useCanvasSelectionStyle } from './useCanvasSelectionStyle'

function Canvas({ zoom }: { zoom: number }) {
  const shell = useRef<HTMLDivElement>(null)
  const update = useCanvasSelectionStyle(shell, zoom)
  return <div ref={shell}><button onClick={() => update(10)}>Zoom</button><div className="react-flow__nodesselection-rect" /></div>
}

it('scales only selection outlines and handles outlines mounted by React Flow later', async () => {
  const { container, rerender } = render(<Canvas zoom={2} />)
  const shell = container.firstElementChild as HTMLElement
  expect(shell.style.length).toBe(0)
  expect((shell.querySelector('.react-flow__nodesselection-rect') as HTMLElement).style.getPropertyValue('--selection-border-width')).toBe('0.5px')
  rerender(<Canvas zoom={20} />)
  expect(shell.style.length).toBe(0)
  expect((shell.querySelector('.react-flow__nodesselection-rect') as HTMLElement).style.getPropertyValue('--selection-border-width')).toBe('0.05px')
  fireEvent.click(shell.querySelector('button')!)
  const outline = document.createElement('div')
  outline.className = 'react-flow__nodesselection-rect'
  await act(async () => { shell.append(outline) })
  expect(outline.style.getPropertyValue('--selection-border-width')).toBe('0.1px')
})
