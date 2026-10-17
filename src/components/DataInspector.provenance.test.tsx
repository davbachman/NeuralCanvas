import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it } from 'vitest'
import { DataInspector } from './DataInspector'
import { singleExampleTextModel } from '../test/inferenceInspectionModel'
import { buildTextModel } from '../test/textModels'
import { withDatasetExample } from '../domain/datasetTraining'
import { datasetExamplesForNode } from '../domain/datasets'
import { forwardPass } from '../domain/engine'

it('shows signal-specific origin on wires and groups without labeling shared inputs or feature columns as examples', () => {
  const graph = withDatasetExample(singleExampleTextModel(), 'text-data', 2)
  const source = graph.nodes.find(node => node.type === 'dataset')!
  const example = datasetExamplesForNode(source)[2]
  source.params.datasetValues = [...example.features, example.target]
  const evaluated = forwardPass(graph).graph
  const node = evaluated.nodes.find(node => node.id === 'sentiment-product')!
  const { rerender } = render(<DataInspector graph={evaluated} node={node}/>)
  expect(screen.getByLabelText('Dataset inspection context')).toHaveTextContent('Dataset row 2 — modified input')
  fireEvent.change(screen.getByLabelText('Inspect signal'), { target: { value: '2' } })
  expect(screen.queryByLabelText('Dataset inspection context')).not.toBeInTheDocument()
  expect(screen.queryByRole('columnheader', { name: 'Dataset row' })).not.toBeInTheDocument()
  const edge = evaluated.edges.find(edge => edge.source === 'review-counts')!
  rerender(<DataInspector graph={evaluated} edge={edge}/>)
  expect(screen.getByRole('columnheader', { name: 'good' })).toBeInTheDocument()
  expect(screen.getByRole('rowheader', { name: '2' })).toBeInTheDocument()
  expect(screen.getByLabelText('Dataset inspection context')).toHaveTextContent('Dataset row 2 — modified input')
  rerender(<DataInspector graph={evaluated} group={{ id: 'classifier', label: 'Classifier', nodeIds: ['sentiment-bias-add', 'positive-probability'], position: { x: 0, y: 0 }, dimensions: { width: 200, height: 200 } }}/>)
  expect(screen.getByLabelText('Dataset inspection context')).toHaveTextContent('Dataset row 2 — modified input')
})

it('shows the selected text origin while token and attention coordinates retain their own indexing', () => {
  const data = singleExampleTextModel().nodes.find(node => node.type === 'dataset')!.params.textData!
  const graph = withDatasetExample(buildTextModel(data, 'attention', 4), 'text-data', 2)
  const source = graph.nodes.find(node => node.type === 'dataset')!
  const example = datasetExamplesForNode(source)[2]
  source.params.datasetValues = [...example.features, example.target]
  const evaluated = forwardPass(graph).graph
  const node = evaluated.nodes.find(node => node.type === 'softmax')!
  render(<DataInspector graph={evaluated} node={node}/>)
  expect(screen.getByLabelText('Dataset inspection context')).toHaveTextContent('Dataset row 2 — modified input')
  expect(screen.queryByRole('columnheader', { name: 'Dataset row' })).not.toBeInTheDocument()
  expect(screen.getByRole('rowheader', { name: '0: "<unk>"' })).toBeInTheDocument()
  expect(screen.getByRole('columnheader', { name: '1: "good"' })).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: /^Row 0, column 1:/ }))
  expect(screen.getByText('[0, 1]')).toBeInTheDocument()
})
