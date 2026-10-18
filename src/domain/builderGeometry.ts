import { customCsvCardHeight, customCsvCardWidth, customCsvOutputTop } from './datasets'
import { NODE_WIDTH, inputArityForNode, isFlexibleInputNodeType, outputArityForNode } from './engine'
import type { GraphNode } from './types'

/** Dimensions and port locations shared by BuilderNode, the zoom scene, and
 * its wire router. A card must keep the same connection geometry at every scale. */
// Keep controls and tensor previews above a dedicated connection area.
export const BUILDER_PORT_TOP = 215
export const BUILDER_PORT_STEP = 30
export function builderCardWidth(node: GraphNode): number {
  return node.type === 'dataset' ? customCsvCardWidth(node) : NODE_WIDTH
}

export function builderCardHeight(node: GraphNode): number {
  if (node.type === 'dataset') return customCsvCardHeight(node, true)
  const inputs = inputArityForNode(node)
  const addInput = node.type === 'arithmetic' || isFlexibleInputNodeType(node.type)
  return BUILDER_PORT_TOP + Math.max(inputs + Number(addInput), outputArityForNode(node)) * BUILDER_PORT_STEP
}

export function builderInputPortY(_node: GraphNode, index: number): number {
  return BUILDER_PORT_TOP + index * BUILDER_PORT_STEP
}

export function builderOutputPortY(node: GraphNode, index: number): number {
  return node.type === 'dataset' ? customCsvOutputTop(index, true) : BUILDER_PORT_TOP + index * BUILDER_PORT_STEP
}
