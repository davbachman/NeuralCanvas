import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { expect, it } from 'vitest'
import { TextGenerationControls } from './TextGenerationControls'
import { generationConsistencyModel } from '../test/generationConsistencyModel'
import { generateText } from '../domain/textGeneration'

it('retains token context across Generate clicks and probability inspection; editing resets that context', async () => {
  const graph = generationConsistencyModel(), before = JSON.stringify(graph)
  render(<TextGenerationControls graph={graph}/>)
  const prompt = screen.getByLabelText('Text generation prompt')
  const button = screen.getByRole('button',{name:'Generate one token'})
  async function click(name: string) {
    fireEvent.click(screen.getByRole('button',{name}))
    await waitFor(()=>expect(button).toBeEnabled())
  }
  fireEvent.change(prompt,{target:{value:'alice was in the'}})
  await click('Generate one token')
  await click('Inspect next-token probabilities')
  await click('Generate one token')
  expect(prompt).toHaveValue(generateText(graph,'alice was in the',2))
  expect(screen.getByLabelText('Text next-token probabilities')).not.toHaveTextContent('"<pad>"')
  fireEvent.change(prompt,{target:{value:'<'}})
  for(let i=0;i<4;i++) await click('Generate one token')
  expect(prompt).toHaveValue('<unk>')
  await click('Inspect next-token probabilities')
  await click('Generate one token')
  expect(prompt).toHaveValue('<unk>l')
  // Pasting the same spelling is an intentional new prompt: reserved ID 0.
  fireEvent.change(prompt,{target:{value:'<unk>'}})
  await click('Generate one token')
  expect(prompt).toHaveValue('<unk>,')
  expect(JSON.stringify(graph)).toBe(before)
})
