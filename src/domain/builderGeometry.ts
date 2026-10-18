import { datasetOutputCountForNode, datasetOutputLabelForSlot } from './datasets'
import { NODE_WIDTH, inputArityForNode, isFlexibleInputNodeType, outputArityForNode } from './engine'
import type { GraphNode } from './types'

/** Dimensions and port locations shared by BuilderNode, the zoom scene, and
 * its wire router. A card must keep the same connection geometry at every scale. */
// Reserve a title row above the connection labels.
export const BUILDER_PORT_TOP = 48
export const BUILDER_PORT_STEP = 30
export function builderCardWidth(node: GraphNode): number {
  if (node.type !== 'dataset') return NODE_WIDTH
  const longest = Math.max(0, ...Array.from({ length: datasetOutputCountForNode(node) }, (_, slot) => datasetOutputLabelForSlot(node, slot).length))
  return Math.min(420, Math.max(NODE_WIDTH, 28 + longest * 6))
}

export function builderCardHeight(node: GraphNode): number {
  const inputs = inputArityForNode(node)
  const addInput = node.type === 'arithmetic' || isFlexibleInputNodeType(node.type)
  return builderInputPortY(node, 0) + Math.max(inputs + Number(addInput), outputArityForNode(node)) * BUILDER_PORT_STEP
}

export function builderInputPortY(node: GraphNode, index: number): number {
  const controls = ['dataset', 'loss', 'activation', 'arithmetic', 'tensor-transform'].includes(node.type)
  return (controls ? 108 : BUILDER_PORT_TOP) + index * BUILDER_PORT_STEP
}

export function builderOutputPortY(node: GraphNode, index: number): number {
  return builderInputPortY(node, index)
}
