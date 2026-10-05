import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { tensorValue } from '../domain/tensor'
import type { GraphModel } from '../domain/types'
import { exampleRowsModel } from '../test/exampleRowsModel'
import { backwardPass, forwardPass } from '../domain/engine'
import { withDatasetIndices } from '../domain/datasetTraining'
import { DataInspector } from './DataInspector'

describe('DataInspector', () => {
  it('shows original dataset rows on outputs, input signals, wires, gradients, and group outputs', async () => {
    const user = userEvent.setup()
    const graph = backwardPass(forwardPass(withDatasetIndices(exampleRowsModel().graph, 'data', [4,1,3])).graph).graph
    const node = graph.nodes.find(node => node.id === 'biased')!
    const {container, rerender} = render(<DataInspector graph={graph} node={node}/>)
    const rowNumbers = () => within(container.querySelector('.data-value-table') as HTMLElement).getAllByRole('rowheader').map(row => row.textContent)
    expect(rowNumbers()).toEqual(['4','1','3'])
    expect(screen.getByRole('columnheader', {name:'Dataset row'})).toBeInTheDocument()
    await user.click(screen.getByRole('button', {name:/Dataset row 1, column 0:/}))
    expect(container.querySelector('.data-exact span')).toHaveTextContent('Dataset row 1 · [1, 0]')
    await user.click(screen.getByRole('tab', {name:'← Gradient'}))
    expect(rowNumbers()).toEqual(['4','1','3'])
    await user.selectOptions(screen.getByRole('combobox', {name:'Inspect signal'}), '1')
    expect(rowNumbers()).toEqual(['0','1']) // Bias parameters have no example identity.
    await user.selectOptions(screen.getByRole('combobox', {name:'Inspect signal'}), '2')
    expect(rowNumbers()).toEqual(['4','1','3'])
    const edge = graph.edges.find(edge => edge.source === 'biased' && edge.target === 'probabilities')!
    rerender(<DataInspector graph={graph} edge={edge}/>)
    expect(rowNumbers()).toEqual(['4','1','3'])
    rerender(<DataInspector graph={graph} group={{id:'group',label:'Group',nodeIds:['probabilities'],position:{x:0,y:0},dimensions:{width:100,height:100}}}/>)
    expect(rowNumbers()).toEqual(['4','1','3'])
  })

  it('keeps original row labels after paging, split changes, and returning from a shuffled batch', async () => {
    const user = userEvent.setup()
    const {graph} = exampleRowsModel()
    const order = [4,1,3,4,1,3,4,1,3,4,1,3,3,4]
    let evaluated = forwardPass(withDatasetIndices(graph,'data',order)).graph
    const {container,rerender} = render(<DataInspector graph={evaluated} node={evaluated.nodes.find(node=>node.id==='probabilities')}/>)
    await user.click(screen.getByRole('button',{name:'Next rows'}))
    const rows = () => within(container.querySelector('.data-value-table') as HTMLElement).getAllByRole('rowheader').map(row=>row.textContent)
    expect(rows()).toEqual(['3','4'])
    graph.nodes[0].params.datasetSplit='test'
    evaluated=forwardPass(graph).graph
    rerender(<DataInspector graph={evaluated} node={evaluated.nodes.find(node=>node.id==='probabilities')}/>)
    expect(rows()).toEqual(['0','2','5'])
    graph.nodes[0].params.datasetSplit='train'
    evaluated=forwardPass(graph).graph
    rerender(<DataInspector graph={evaluated} node={evaluated.nodes.find(node=>node.id==='probabilities')}/>)
    expect(rows()).toEqual(['1','3','4'])
  })

  it('shows the selected wire’s values, exact cell, and backward contribution', async () => {
    const user = userEvent.setup()
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'source', type: 'input', label: 'feature', position: { x: 0, y: 0 }, params: {} },
        { id: 'target', type: 'activation', label: 'ReLU', position: { x: 200, y: 0 }, params: { activation: 'relu' } },
      ],
      edges: [{ id: 'wire', source: 'source', target: 'target', value: tensorValue([2], [1.234567, -2]), grad: tensorValue([2], [0.5, -0.25]) }],
    }
    render(<DataInspector graph={graph} edge={graph.edges[0]} />)

    expect(screen.getByRole('heading', { name: 'feature → ReLU' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Index 1: -2' }))
    expect(screen.getByText('-2', { selector: '.data-exact strong' })).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: '← Gradient' }))
    expect(screen.getByRole('button', { name: 'Index 1: -0.25' })).toBeInTheDocument()
  })

  it('previews dataset columns and pages through all rows', async () => {
    const user = userEvent.setup()
    const dataset = { id: 'dataset', type: 'dataset' as const, label: 'Training data', position: { x: 0, y: 0 }, params: { dataset: 'line-1d' as const, datasetMode: 'batch' as const } }
    const graph: GraphModel = { learningRate: 0.1, nodes: [dataset], edges: [] }
    render(<DataInspector graph={graph} node={dataset} />)

    const preview = screen.getByRole('region', { name: 'Dataset rows' })
    expect(within(preview).getByRole('columnheader', { name: 'x' })).toBeInTheDocument()
    expect(within(preview).getByRole('columnheader', { name: 'y' })).toBeInTheDocument()
    expect(within(preview).getAllByRole('row')).toHaveLength(13)
    await user.click(within(preview).getByRole('button', { name: 'Next rows' }))
    expect(within(preview).getByRole('rowheader', { name: '19' })).toBeInTheDocument()
    await user.selectOptions(screen.getByRole('combobox', { name: 'Inspect column' }), '1')
    expect(screen.getByText('[20]', { selector: '.data-stats strong' })).toBeInTheDocument()
  })

  it('lets a selected calculation inspect each incoming signal', async () => {
    const user = userEvent.setup()
    const calculation = { id: 'sum', type: 'add' as const, label: 'sum', position: { x: 200, y: 0 }, params: {}, value: tensorValue([], [5]) }
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'left', type: 'input', label: 'left', position: { x: 0, y: 0 }, params: {} },
        { id: 'right', type: 'input', label: 'right', position: { x: 0, y: 100 }, params: {} },
        calculation,
      ],
      edges: [
        { id: 'left-wire', source: 'left', target: 'sum', inputSlot: 0, value: tensorValue([], [2]) },
        { id: 'right-wire', source: 'right', target: 'sum', inputSlot: 1, value: tensorValue([], [3]) },
      ],
    }
    render(<DataInspector graph={graph} node={calculation} />)

    await user.selectOptions(screen.getByRole('combobox', { name: 'Inspect signal' }), '2')
    expect(screen.getByLabelText('Scalar value')).toHaveTextContent('3')
  })

  it('slices a higher-rank tensor while retaining exact coordinates', async () => {
    const user = userEvent.setup()
    const node = { id: 'image', type: 'input' as const, label: 'image', position: { x: 0, y: 0 }, params: {}, value: tensorValue([2, 2, 2], [0, 1, 2, 3, 4, 5, 6, 7]) }
    render(<DataInspector graph={{ learningRate: 0.1, nodes: [node], edges: [] }} node={node} />)

    await user.selectOptions(screen.getByRole('combobox', { name: 'Axis 2 index' }), '1')
    await user.click(screen.getByRole('button', { name: 'Row 1, column 1: 7' }))
    expect(screen.getByText('[1, 1, 1]', { selector: '.data-exact span' })).toBeInTheDocument()
    expect(screen.getByText('7', { selector: '.data-exact strong' })).toBeInTheDocument()
  })
})
