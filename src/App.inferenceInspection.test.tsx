import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import { sentimentModel } from './test/inferenceInspectionModel'
import { downloadProjectStateFile } from './domain/session'
import * as session from './domain/session'
import { parameterValues } from './domain/engine'

afterEach(() => vi.restoreAllMocks())
it('opens all evaluated sigmoid outputs from the report while preserving canvas selection and parameters', async () => {
  const graph = sentimentModel()
  graph.nodes.find(node => node.type === 'dataset')!.params.datasetIndex = 2
  const save = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
  const { container } = render(<App initialGraph={graph}/>)
  fireEvent.click(screen.getByRole('tab', { name: 'Test' }))
  fireEvent.click(screen.getByRole('button', { name: 'Run inference' }))
  await screen.findByRole('region', { name: 'Inference report' })
  fireEvent.click(container.querySelector('[data-id="positive-probability"]')!)
  fireEvent.click(screen.getByRole('tab', { name: 'Data' }))
  expect(screen.getByLabelText('Inspection context')).toHaveValue('inference')
  expect(await screen.findByRole('rowheader', { name: '3' })).toBeInTheDocument()
  expect(screen.getAllByLabelText('Incorrect')).toHaveLength(2)
  expect(screen.queryByRole('tab', { name: '← Gradient' })).not.toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('Inspection context'), { target: { value: 'canvas' } })
  expect(screen.getByRole('rowheader', { name: '2' })).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'File' }))
  fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
  expect(downloadProjectStateFile).toHaveBeenCalled()
  const saved = save.mock.calls.at(-1)![0].state.graph
  expect(parameterValues(saved)).toEqual(parameterValues(graph))
  expect(saved.nodes.find(node => node.type === 'dataset')!.params).toEqual(graph.nodes.find(node => node.type === 'dataset')!.params)
  fireEvent.change(screen.getByLabelText('Inspection context'), { target: { value: 'inference' } })
  await screen.findByRole('rowheader', { name: '3' })
  // Editing the activation invalidates both the report and its inspection context.
  fireEvent.change(container.querySelector('[data-id="positive-probability"] select')!, { target: { value: 'tanh' } })
  await waitFor(() => expect(screen.queryByLabelText('Inspection context')).not.toBeInTheDocument())
  expect(screen.queryByRole('region', { name: 'Inference report' })).not.toBeInTheDocument()
})
