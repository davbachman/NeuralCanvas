import { datasetExampleIndex, datasetExamplesForNode, datasetMode, datasetOutputValueForSlot, datasetTargetSlotForNode } from './datasets'
import type { GraphModel, TensorValue } from './types'

interface Identity { datasetId: string; indices: number[]; axis: number }
interface Signal { shape?: number[]; identity?: Identity }
const sameRows = (a: Identity, b: Identity) => a.datasetId === b.datasetId && a.indices.length === b.indices.length && a.indices.every((row, index) => row === b.indices[index])

/** Track example identity through wiring and axis semantics, never by size alone. */
export function tensorExampleRows(graph: GraphModel, id: string, slot = 0): number[] | undefined {
  const nodes = new Map(graph.nodes.map(node => [node.id, node]))
  const cache = new Map<string, Signal>()
  const visit = (id: string, slot: number): Signal => {
    const key = JSON.stringify([id, slot])
    const cached = cache.get(key)
    if (cached) return cached
    cache.set(key, {}) // Stop malformed cycles.
    const node = nodes.get(id)
    if (!node) return {}
    const edges = graph.edges.filter(edge => edge.target === id).sort((a, b) => (a.inputSlot ?? 0) - (b.inputSlot ?? 0))
    const inputs = edges.map(edge => visit(edge.source, edge.sourceSlot ?? 0))
    const value = node.type === 'dataset' ? datasetOutputValueForSlot(node, slot)
      : node.value ?? (typeof node.params.value === 'number' ? { shape: [], data: [node.params.value] } : node.params.value)
        ?? graph.edges.find(edge => edge.source === id && (edge.sourceSlot ?? 0) === slot)?.value
    const shape = value?.shape
    const result: Signal = { shape }
    const keep = (identity?: Identity) => {
      if (identity && shape && (identity.axis === -1 ? shape.length === 0 : shape[identity.axis] === identity.indices.length)) result.identity = identity
    }
    if (node.type === 'dataset' && shape) {
      const examples = datasetExamplesForNode(node)
      const override: TensorValue | undefined = node.params.datasetValues?.[slot]
      if (override) {
        const indices = override.exampleIndices
        if (indices && indices.length === shape[0] && indices.every(index => Number.isInteger(index) && index >= 0 && index < examples.length)) {
          keep({ datasetId: id, indices, axis: 0 })
        }
      } else if (datasetMode(node) === 'batch') {
        const indices = examples.flatMap((example, index) => !node.params.datasetSplit || node.params.datasetSplit === 'all' || example.split === node.params.datasetSplit ? [index] : [])
        keep({ datasetId: id, indices, axis: 0 })
      } else {
        const counts = node.params.textData?.representation === 'counts' && slot === 0
        const singleTarget = slot === datasetTargetSlotForNode(node) && value?.data.length === 1 && node.params.textData?.task !== 'language'
        // Token positions, image rows, and arbitrary feature-vector axes are not examples.
        if (!shape.length || counts || singleTarget) keep({ datasetId: id, indices: [datasetExampleIndex(node)], axis: shape.length ? 0 : -1 })
      }
    } else if (node.type !== 'weight' && node.type !== 'bias' && shape) {
      const kind = node.type === 'tensor-transform' ? node.params.transform ?? 'reshape' : node.type
      const first = inputs[0]
      if (['input', 'target', 'standardize', 'dropout', 'activation', 'softmax', 'causal-mask', 'layer-norm'].includes(kind)) keep(first?.identity)
      else if (kind === 'matmul' && first?.shape?.length === 2 && first.identity?.axis === 0) keep(first.identity)
      else if (kind === 'one-hot' || kind === 'embedding') keep(inputs[kind === 'embedding' ? 1 : 0]?.identity)
      else if (kind === 'transpose' && first?.shape && first.identity && first.identity.axis >= 0) {
        const axes = node.params.axes ?? first.shape.map((_, index) => first.shape!.length - index - 1)
        keep({ ...first.identity, axis: axes.indexOf(first.identity.axis) })
      } else if (kind === 'reshape' && first?.identity) {
        const identity = first.identity
        if ((identity.axis === 0 || identity.axis === -1) && shape[0] === identity.indices.length &&
            (first.shape?.reduce((a, b) => a * b, 1) === shape.reduce((a, b) => a * b, 1))) keep({ ...identity, axis: 0 })
      } else if (kind === 'mean' && first?.identity && node.params.axis !== undefined && node.params.axis !== first.identity.axis) {
        const identity = first.identity
        if (identity.axis >= 0) keep({ ...identity, axis: identity.axis - (!node.params.keepDims && node.params.axis < identity.axis ? 1 : 0) })
      } else if (kind === 'slice' && first?.identity) {
        const identity = first.identity
        keep((node.params.axis ?? 0) === identity.axis
          ? { ...identity, indices: identity.indices.slice(node.params.start ?? 0, node.params.end) } : identity)
      } else if (kind === 'concat') {
        const axis = node.params.axis ?? 1
        const identities = inputs.flatMap(input => input.identity ? [input.identity] : [])
        const identity = identities[0]
        if (identity && axis === identity.axis) {
          if (identities.length === inputs.length && identities.every(item => item.axis === axis && item.datasetId === identity.datasetId)) {
            keep({ ...identity, indices: identities.flatMap(item => item.indices) })
          }
        } else if (identity && identities.length === inputs.length && identities.every(item => item.axis === identity.axis && sameRows(item, identity))) keep(identity)
      } else if (['arithmetic', 'add', 'multiply'].includes(kind)) {
        const identities = inputs.flatMap(input => input.identity && input.shape ? [{ ...input.identity,
          axis: input.identity.axis < 0 ? -1 : input.identity.axis + shape.length - input.shape.length }] : [])
        const identity = identities[0]
        // Broadcasting a feature bias is fine; mismatched example orders or axes are not.
        if (identity && identities.every(item => item.axis === identity.axis && sameRows(item, identity))) keep(identity)
      }
      // Reducing the example axis, losses, and unsupported transforms carry no identity.
    }
    cache.set(key, result)
    return result
  }
  const result = visit(id, slot)
  return result.identity?.axis === 0 ? result.identity.indices : undefined
}
