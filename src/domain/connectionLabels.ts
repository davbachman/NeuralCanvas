import { datasetOutputLabelForSlot, inputArityForNode, isLossNode, lossKindForNode } from './engine'
import type { GraphNode, LossKind, NodeType } from './types'

/** Names describe the role of each socket, independent of upstream variable names. */
export function inputConnectionLabels(node: GraphNode, lossKind?: LossKind): string[] {
  if (isLossNode(node)) {
    const kind = lossKind ?? lossKindForNode(node)
    return [node.type === 'cross-entropy' || kind === 'cross-entropy' || kind === 'binary-cross-entropy-with-logits' ? 'logits' : kind === 'binary-cross-entropy' ? 'probabilities' : 'prediction', 'target']
  }
  const roles: Partial<Record<NodeType, string[]>> = {
    embedding: ['table', 'token IDs'], 'one-hot': ['token IDs'],
    conv2d: ['image', 'kernel', 'bias'], 'layer-norm': ['input', 'scale γ', 'offset β'],
    matmul: ['left', 'right'], softmax: ['logits'], 'causal-mask': ['scores'], target: ['target'],
  }
  return Array.from({ length: inputArityForNode(node) }, (_, index) =>
    roles[node.type]?.[index] ?? (inputArityForNode(node) === 1 ? 'input' : `x${index + 1}`))
}

export function outputConnectionLabel(node: GraphNode, slot = 0): string {
  if (node.type === 'dataset') return datasetOutputLabelForSlot(node, slot)
  const roles: Partial<Record<NodeType, string>> = {
    weight: 'parameter', bias: 'parameter', target: 'target', embedding: 'embeddings',
    'one-hot': 'one-hot', softmax: 'probabilities',
  }
  return roles[node.type] ?? (node.type === 'activation' && node.params.activation === 'sigmoid' ? 'probabilities' : 'output')
}
