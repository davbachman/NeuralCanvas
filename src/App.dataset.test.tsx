import * as session from './domain/session'
import { builderCardWidth } from './domain/builderGeometry'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { DATASET_MENU_OPTIONS, DATASET_OPTIONS, datasetExamples } from './domain/datasets'
import { DATASET_LIBRARY } from './domain/datasetLibrary'
import type { DatasetKind } from './domain/types'
import { createNode, createSingleNeuronGraph } from './domain/examples'
import { tensorValue } from './domain/tensor'

function openCustomImport(selector: Element, kind: 'custom-csv' | 'custom-text') {
  fireEvent.change(selector, { target: { value: 'browse' } })
  fireEvent.click(screen.getByRole('button', { name: 'Use a custom dataset…' }))
  fireEvent.click(screen.getByRole('button', { name: kind === 'custom-csv' ? 'Import numeric CSV…' : 'Import prepared JSON or text…' }))
}

afterEach(() => vi.restoreAllMocks())

function datasetModel(semanticZoom: boolean) {
  const graph = createSingleNeuronGraph('identity', { x: 1, w: 2, b: 0, target: 3 })
  graph.nodes.push(createNode('dataset', 1))
  graph.edges.push(
    { id: 'dataset-feature', source: 'dataset-1', sourceSlot: 0, target: 'x', inputSlot: 0 },
    { id: 'dataset-target', source: 'dataset-1', sourceSlot: 1, target: 'target', inputSlot: 0 },
  )
  graph.view = { semanticZoom, expandedGroupIds: [] }
  return graph
}

async function chooseBuiltin(selector: HTMLElement, kind: DatasetKind) {
  fireEvent.change(selector, { target: { value: 'browse' } })
  const dialog = screen.getByRole('dialog', { name: 'Dataset library' })
  const family = DATASET_LIBRARY.find(item => item.presets.some(preset => preset.builtin === kind))!
  fireEvent.click(within(dialog).getByRole('button', { name: new RegExp(family.label) }))
  if (family.presets.length > 1) fireEvent.change(within(dialog).getByLabelText('Dataset configuration preset'), { target: { value: kind } })
  const useDataset = within(dialog).getByRole('button', { name: 'Use this dataset' })
  await waitFor(() => expect(useDataset).toBeEnabled())
  fireEvent.click(useDataset)
}

describe.each([true, false])('Dataset controls with continuous zoom %s', semanticZoom => {
  it('opens a CSV picker from the Dataset block and installs editable feature and target ports', async () => {
    const { container } = render(<App initialGraph={datasetModel(semanticZoom)} />)
    const card = container.querySelector<HTMLElement>('[data-id="dataset-1"]')!
    fireEvent.click(card)
    const selector = within(card).getByLabelText('Dataset for dataset')
    openCustomImport(selector, 'custom-csv')
    const chooser = screen.getByRole('dialog', { name: 'Choose a CSV file' })
    const input = within(chooser).getByLabelText('Choose custom CSV file') as HTMLInputElement
    expect(input).toBeVisible()
    const text = 'height,width,y\n1,2,3\n2,3,5\n3,4,7\n4,5,9\n'
    const file = Object.assign(new File([text], 'measurements.csv', { type: 'text/csv' }), { text: async () => text })
    fireEvent.change(input, { target: { files: [file] } })
    await waitFor(() => expect(selector).toHaveDisplayValue('Custom CSV · measurements.csv'))
    expect(screen.queryByRole('dialog', { name: 'Choose a CSV file' })).not.toBeInTheDocument()
    expect(card.querySelector('.semantic-operation, .builder-node')).toHaveStyle({ width: `${builderCardWidth(createNode('dataset', 1))}px` })
    expect(within(card).getByLabelText('height output')).toHaveAttribute('data-handleid', 'out-0')
    expect(within(card).getByLabelText('width output')).toHaveAttribute('data-handleid', 'out-1')
    expect(within(card).getByLabelText('y output')).toHaveAttribute('data-handleid', 'out-2')
    expect(screen.queryByRole('combobox', { name: 'CSV target column' })).not.toBeInTheDocument()
    expect(screen.getAllByText(/Connect a column to a Target block/).length).toBeGreaterThan(0)
    expect(screen.getByRole('combobox', { name: 'CSV task' })).toHaveValue('regression')
    expect(screen.getByRole('button', { name: 'Replace CSV file' })).toBeInTheDocument()
  })

  it('offers simple numeric shortcuts and browser access to the other built-in datasets', async () => {
    const { container } = render(<App initialGraph={datasetModel(semanticZoom)} />)
    const card = container.querySelector<HTMLElement>('[data-id="dataset-1"]')!
    const selector = within(card).getByLabelText('Dataset for dataset')
    expect(selector).toHaveClass('nodrag', 'nowheel')
    expect(within(selector).getAllByRole('option').map(option => option.textContent)).toEqual([...DATASET_MENU_OPTIONS.filter(option => !option.kind.startsWith('custom-')).map(option => option.label), 'Browse datasets…'])
    expect(within(selector).getAllByRole('option').map(option => (option as HTMLOptionElement).value)).toEqual([
      'line-1d', 'cubic-1d', 'plane-2d', 'threshold-1d', 'circle-center',
      'parabola-boundary', 'neuron-basics', 'xor', 'browse',
    ])
    expect(selector).toHaveDisplayValue(DATASET_OPTIONS[0].label)
    expect(Array.from(selector.querySelectorAll('option')).filter(option => option.textContent === DATASET_OPTIONS[0].label)).toHaveLength(1)
    // Opening and cancelling the browser must not replace the current source.
    fireEvent.change(selector, { target: { value: 'browse' } })
    fireEvent.click(screen.getByRole('button', { name: 'Close dataset library' }))
    expect(selector).toHaveDisplayValue(DATASET_OPTIONS[0].label)
    for (const dataset of DATASET_OPTIONS) {
      if (DATASET_MENU_OPTIONS.some(option => option.kind === dataset.kind)) {
        fireEvent.change(selector, { target: { value: dataset.kind } })
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
        expect(selector).toHaveValue(dataset.kind)
        expect(Array.from(selector.querySelectorAll('option')).filter(option => option.textContent === dataset.label)).toHaveLength(1)
      } else await chooseBuiltin(selector, dataset.kind)
      expect(selector).toHaveDisplayValue(dataset.label)
      expect(card.querySelectorAll('.source-handle')).toHaveLength(dataset.featureLabels.length + 1)
      expect(Array.from(card.querySelectorAll('.node-output-labels span')).map(label => label.textContent)).toEqual([...dataset.featureLabels, dataset.targetLabel])
      dataset.featureLabels.forEach((label, slot) => {
        expect(within(card).getByLabelText(`Feature ${label} output`)).toHaveAttribute('data-handleid', `out-${slot}`)
      })
      expect(within(card).getByLabelText(`Target ${dataset.targetLabel} output`)).toHaveAttribute('data-handleid', `out-${dataset.featureLabels.length}`)
    }
  }, 30000)

  it('keeps feature and target connections correct when either selector changes feature count, and supports undo', async () => {
    const user = userEvent.setup()
    const { container } = render(<App initialGraph={datasetModel(semanticZoom)} />)
    const card = container.querySelector<HTMLElement>('[data-id="dataset-1"]')!
    fireEvent.click(card)
    const nodeSelector = within(card).getByLabelText('Dataset for dataset')
    const inspectorSelector = () => screen.getByRole('combobox', { name: 'Dataset selection' })
    await user.selectOptions(nodeSelector, 'circle-center')
    expect(inspectorSelector()).toHaveDisplayValue(DATASET_OPTIONS.find(option => option.kind === 'circle-center')!.label)
    expect(screen.queryByText('Inside the calculation')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Dataset outputs')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Dataset configuration')).toBeInTheDocument()
    expect(screen.queryByText('Ready to evaluate')).not.toBeInTheDocument()
    const save = vi.spyOn(session, 'downloadProjectStateFile').mockImplementation(() => {})
    const expectResults = (kind: string) => {
      const dataset = DATASET_OPTIONS.find(option => option.kind === kind)!
      // Opening Train preserves the selected all-examples split, including after changing datasets or undoing.
      const examples = datasetExamples(dataset)
      const features = tensorValue([examples.length], examples.map(example => example.features[0].data[0]))
      const targets = tensorValue([examples.length], examples.map(example => example.target.data[0]))
      fireEvent.click(screen.getByRole('button', { name: 'File' }))
      fireEvent.click(screen.getByRole('menuitem', { name: 'Save' }))
      const evaluated = save.mock.calls.at(-1)![0].state.graph
      expect(evaluated.nodes.find(node => node.id === 'x')!.value).toEqual(features)
      expect(evaluated.nodes.find(node => node.id === 'target')!.value).toEqual(targets)
      expect(card.querySelector('.node-metrics')).toBeNull()
    }
    await user.click(screen.getByRole('tab', { name: 'Train' }))
    await user.click(screen.getByRole('button', { name: 'Run forward' }))
    expectResults('circle-center')
    fireEvent.click(card)
    await user.selectOptions(inspectorSelector(), 'cubic-1d')
    expect(nodeSelector).toHaveDisplayValue(DATASET_OPTIONS.find(option => option.kind === 'cubic-1d')!.label)
    expect(within(card).getByLabelText('Target y output')).toHaveAttribute('data-handleid', 'out-1')
    await user.click(screen.getByRole('button', { name: 'Run forward' }))
    expectResults('cubic-1d')
    // Undo the forward pass, then the dataset change, restoring the old ports.
    fireEvent.keyDown(document, { key: 'z', metaKey: true })
    fireEvent.keyDown(document, { key: 'z', metaKey: true })
    expect(nodeSelector).toHaveDisplayValue(DATASET_OPTIONS.find(option => option.kind === 'circle-center')!.label)
    expect(within(card).getByLabelText('Target y output')).toHaveAttribute('data-handleid', 'out-2')
    await user.click(screen.getByRole('button', { name: 'Run forward' }))
    expectResults('circle-center')
  })

  it('renames the selected dataset from its title', () => {
    const { container } = render(<App initialGraph={datasetModel(semanticZoom)} />)
    fireEvent.click(container.querySelector('[data-id="dataset-1"]')!)
    fireEvent.click(screen.getByRole('button', { name: 'Rename dataset' }))
    const name = screen.getByRole('textbox', { name: 'Block name' })
    fireEvent.change(name, { target: { value: 'Training data' } })
    fireEvent.blur(name)
    expect(container.querySelector('[data-id="dataset-1"]')).toHaveTextContent('Training data')
    expect(screen.queryByText('Block name')).not.toBeInTheDocument()
  })
})
