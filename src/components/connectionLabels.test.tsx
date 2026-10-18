import { render, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import { ReactFlowProvider, type NodeProps } from '@xyflow/react'
import { BuilderNode, type BuilderNodeData } from './BuilderNode'
import { createNode } from '../domain/examples'
import { inputArityByType, inputArityForNode, outputArityForNode, LOSS_OPTIONS } from '../domain/engine'
import { inputConnectionLabels } from '../domain/connectionLabels'
import { builderCardHeight, builderInputPortY, builderOutputPortY } from '../domain/builderGeometry'
import type { GraphNode, NodeType } from '../domain/types'

afterEach(cleanup)
function card(node: GraphNode) {
  const data = { graphNode: node, showMath: true, showGradient: true, formula: 'formula', fullFormula: 'formula', active: false, hasIncomingValue: false } as BuilderNodeData
  return render(<ReactFlowProvider><BuilderNode {...{ id: node.id, data, selected: false } as unknown as NodeProps}/></ReactFlowProvider>)
}
it.each(Object.keys(inputArityByType) as NodeType[])('keeps %s sockets accessible, shows input labels, and only labels dataset outputs', type => {
  const node = createNode(type, 1), view = card(node)
  const inputs = view.container.querySelectorAll('.react-flow__handle.target')
  const outputs = view.container.querySelectorAll('.react-flow__handle.source')
  expect(view.container.querySelector('.node-formula, .node-metrics, .node-kind')).toBeNull()
  expect(view.container.querySelector('.node-title-row')).toHaveTextContent(node.label)
  expect(view.container.querySelector('.node-icon')).toHaveAttribute('aria-hidden', 'true')
  expect(view.container.querySelector('.node-icon svg')).not.toBeNull()
  expect(inputs).toHaveLength(inputArityForNode(node))
  expect(outputs).toHaveLength(outputArityForNode(node))
  for (const [handles, selector, portY] of [
    [inputs, '.node-input-labels span', builderInputPortY],
    [outputs, '.node-output-labels span', builderOutputPortY],
  ] as const) {
    const labels = view.container.querySelectorAll<HTMLElement>(selector)
    const visibleLabels = selector === '.node-input-labels span' || type === 'dataset'
    expect(labels).toHaveLength(visibleLabels ? handles.length : 0)
    handles.forEach((handle, i) => {
      expect(handle.getAttribute('aria-label')).toBeTruthy()
      expect(handle.getAttribute('title')).toBeTruthy()
      expect((handle as HTMLElement).style.top).toBe(`${portY(node, i)}px`)
      if (visibleLabels) {
        expect(labels[i].textContent).toBeTruthy()
        expect(labels[i].style.top).toBe(`${portY(node, i)}px`)
      }
      expect(portY(node, i)).toBeLessThan(builderCardHeight(node))
    })
  }
})
it('uses operation roles in slot order, including all loss conventions', () => {
  expect(inputConnectionLabels(createNode('embedding', 1))).toEqual(['table', 'token IDs'])
  expect(inputConnectionLabels(createNode('conv2d', 1))).toEqual(['image', 'kernel', 'bias'])
  expect(inputConnectionLabels(createNode('layer-norm', 1))).toEqual(['input', 'scale γ', 'offset β'])
  for (const { kind } of LOSS_OPTIONS) {
    const labels = inputConnectionLabels({ ...createNode('loss', 1), params: { loss: kind } })
    expect(labels).toEqual([kind === 'cross-entropy' || kind === 'binary-cross-entropy-with-logits' ? 'logits' : kind === 'binary-cross-entropy' ? 'probabilities' : 'prediction', 'target'])
  }
})
it('keeps all variable inputs and the add button inside the reserved connection area', () => {
  const node = { ...createNode('concat', 1), params: { inputCount: 12 } }
  const { container } = card(node)
  expect(container.querySelectorAll('.node-input-labels span')).toHaveLength(12)
  expect(container.querySelector('.node-add-input-button')).toHaveStyle({ top: `${builderInputPortY(node, 12)}px` })
  expect(builderInputPortY(node, 12)).toBeLessThan(builderCardHeight(node))
})

it.each(['tokens', 'counts', 'facts'] as const)('shows only labeled ports beneath the title for %s datasets', async representation => {
  const { prepareTextDocuments } = await import('../domain/textData')
  const { builderCardWidth } = await import('../domain/builderGeometry')
  const textData = prepareTextDocuments([
    { text: 'Mary went home', facts: ['Mary home'], question: 'where Mary', label: 'positive', split: 'train' },
    { text: 'John went away', facts: ['John away'], question: 'where John', label: 'negative', split: 'train' },
    { text: 'Mary went away', facts: ['Mary away'], question: 'where Mary', label: 'positive', split: 'test' },
  ], 'stories.csv', { task: representation === 'facts' ? 'classification' : 'sentiment', representation, maxFacts: 2, factWords: 3 })
  const node = { ...createNode('dataset', 1), params: { dataset: 'custom-text' as const, textData } }
  const { container } = card(node)
  const labels = [...container.querySelectorAll<HTMLElement>('.node-output-labels span')]
  expect(labels.map(label => label.textContent)).toEqual(representation === 'facts'
    ? ['fact token IDs', 'question token IDs', 'fact position IDs', 'fact mask', 'fact word mask', 'question word mask', 'answer']
    : [representation === 'counts' ? 'word counts' : 'token IDs', 'position IDs', 'sentiment'])
  expect(container.querySelector('.builder-node')).toHaveClass('is-compact')
  expect(container.querySelector('.node-metrics,.node-formula')).toBeNull()
  expect(builderCardWidth(node)).toBeGreaterThanOrEqual(176)
  expect(builderOutputPortY(node, 0)).toBeLessThan(120)
  expect(builderOutputPortY(node, labels.length - 1)).toBeLessThan(builderCardHeight(node))
  // Nested geometry uses only metadata, never copies the vocabulary or documents.
  const geometryNode = { ...node, params: { ...node.params, textData: { representation, task: textData.task } as typeof textData } }
  expect(builderCardWidth(geometryNode)).toBe(builderCardWidth(node))
  expect(builderCardHeight(geometryNode)).toBe(builderCardHeight(node))
})
