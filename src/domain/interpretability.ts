import { datasetOutputCountForNode, datasetOutputValueForSlot } from './datasets'
import { toTensor } from './tensor'
import type { GraphModel, GraphNode, TensorValue, ValidationIssue } from './types'

export function isParameter(node: GraphNode): boolean { return node.type === 'weight' || node.type === 'bias' }

/** Unit decoder directions remove the encoder-shrink/decoder-grow loophole.
 * A zero direction becomes e0; max scaling avoids overflow/underflow. */
export function normalizeParameterValue(node: GraphNode, value: TensorValue): TensorValue {
  if (node.params.constraint !== 'unit-row-norm' || node.params.trainable === false) return value
  if (value.shape.length !== 2 || value.shape.some(n => n < 1)) throw Error('Unit row norm requires a nonempty matrix parameter.')
  const width = value.shape[1], data = [...value.data]
  for (let start = 0; start < data.length; start += width) {
    const row = data.slice(start, start + width), scale = row.reduce((maximum,n)=>Math.max(maximum,Math.abs(n)),0)
    if (!Number.isFinite(scale)) throw Error('Parameter directions must be finite.')
    const norm = scale ? Math.sqrt(row.reduce((sum, n) => sum + (n / scale) ** 2, 0)) : 0
    row.forEach((n, j) => { data[start + j] = norm ? (n / scale) / norm : Number(j === 0) })
  }
  return { ...value, shape: [...value.shape], data }
}

/** Saved Loss references are dependencies even though they do not occupy prediction/target ports. */
export function objectiveDependencies(node: GraphNode): string[] {
  if (node.type !== 'loss' && node.type !== 'cross-entropy') return []
  return [...new Set([node.params.activationPenaltyNodeId, node.params.lossMaskNodeId].filter((id): id is string => !!id))]
}

export function graphAncestors(graph: GraphModel, id: string, includeObjective = true): Set<string> {
  const seen = new Set<string>(), stack = [id]
  while (stack.length) {
    const current = stack.pop()!
    if (seen.has(current)) continue
    seen.add(current)
    stack.push(...graph.edges.filter(edge => edge.target === current).map(edge => edge.source))
    const node = graph.nodes.find(node => node.id === current)
    if (includeObjective && node) stack.push(...objectiveDependencies(node))
  }
  return seen
}

export function interpretabilityIssues(graph: GraphModel): ValidationIssue[] {
  const issues: ValidationIssue[] = []
  const invalid = (node: GraphNode, message: string) => issues.push({code:'invalid-value', nodeId:node.id, message})
  for (const node of graph.nodes) {
    if (node.params.trainable !== undefined && typeof node.params.trainable !== 'boolean') invalid(node, 'Trainable must be a boolean.')
    if (node.params.inferenceOnly !== undefined && typeof node.params.inferenceOnly !== 'boolean') invalid(node, 'Inference only must be a boolean.')
    if (node.params.constraint !== undefined && (node.params.constraint !== 'unit-row-norm' || !isParameter(node) || (toTensor(node.params.value).shape.length !== 2 || toTensor(node.params.value).shape.some(n=>n<1)))) invalid(node, 'Unit row norm requires a matrix parameter.')
    if (node.type !== 'loss' && node.type !== 'cross-entropy') continue
    if (node.params.targetMode !== undefined && node.params.targetMode !== 'internal') invalid(node, 'Unknown target mode.')
    for (const id of objectiveDependencies(node)) {
      const selected = graph.nodes.find(n => n.id === id)
      if (!selected || selected.type === 'loss' || selected.type === 'cross-entropy') invalid(node, 'Select an existing non-Loss signal for the objective.')
    }
    if (node.params.activationPenaltyNodeId && graph.nodes.find(n=>n.id===node.params.activationPenaltyNodeId)?.type === 'dataset') invalid(node, 'Choose a computed activation, not a Dataset block, for activation L1.')
    const strength = node.params.activationPenaltyStrength ?? 0
    if (!Number.isFinite(strength) || strength < 0 || strength > 0 && !node.params.activationPenaltyNodeId) invalid(node, 'Activation L1 needs a nonnegative finite strength and a selected signal.')
    if (node.params.lossMaskSourceSlot !== undefined && (!Number.isInteger(node.params.lossMaskSourceSlot) || node.params.lossMaskSourceSlot < 0)) invalid(node, 'Mask output must be a nonnegative integer.')
    if (node.params.lossMaskNodeId && !['mse','mae'].includes(node.params.loss ?? 'squared-error')) invalid(node, 'A row mask currently supports Mean squared error or Mean absolute error.')
    if (node.params.lossMaskNodeId) {
      const selected = graph.nodes.find(n => n.id === node.params.lossMaskNodeId)
      const outputs = selected?.type === 'dataset' ? datasetOutputCountForNode(selected) : 1
      if ((node.params.lossMaskSourceSlot ?? 0) >= outputs) invalid(node, 'The selected mask output does not exist.')
      const maskAncestors = graphAncestors(graph,node.params.lossMaskNodeId)
      if (graph.nodes.some(n => maskAncestors.has(n.id) && isParameter(n) && n.params.trainable !== false)) invalid(node, 'Freeze every parameter feeding the loss mask; masks are fixed, not learned.')
      if (graph.nodes.some(n => maskAncestors.has(n.id) && n.type === 'dropout' && (n.params.dropoutRate ?? .1) > 0 && !n.params.inferenceOnly)) invalid(node, 'The loss mask must be deterministic; set its Dropout to Inference only.')
    }
    const prediction = graph.edges.find(edge => edge.target === node.id && (edge.inputSlot ?? 0) === 0)
    if (node.params.activationPenaltyNodeId && (!prediction || !graphAncestors(graph, prediction.source, false).has(node.params.activationPenaltyNodeId))) invalid(node, 'The activation penalty signal must feed the loss prediction.')
    if (node.params.targetMode === 'internal') {
      if (!['mse','mae'].includes(node.params.loss ?? 'squared-error')) invalid(node, 'Internal reconstruction requires Mean squared error or Mean absolute error.')
      const target = graph.edges.find(edge => edge.target === node.id && edge.inputSlot === 1)
      if (target) {
        const ancestors = graphAncestors(graph, target.source)
        if (graph.nodes.some(n => ancestors.has(n.id) && isParameter(n) && n.params.trainable !== false)) invalid(node, 'Freeze every parameter feeding the internal target.')
        if (graph.nodes.some(n => ancestors.has(n.id) && n.type === 'dropout' && (n.params.dropoutRate ?? .1) > 0 && !n.params.inferenceOnly)) invalid(node, 'Set target-side Dropout to Inference only to keep reconstruction targets stable.')
      }
    }
  }
  return issues
}

export function objectiveSignal(graph: GraphModel, id: string, slot = 0): TensorValue {
  const node = graph.nodes.find(n => n.id === id)
  if (!node) throw Error('The objective signal no longer exists.')
  const value = node.type === 'dataset' ? datasetOutputValueForSlot(node, slot) : slot === 0 ? node.value : undefined
  if (!value) throw Error('Evaluate the objective signal before Loss.')
  return value
}

/** Last axis is features; remaining axes enumerate token/example rows. */
export function objectiveRows(graph: GraphModel, loss: GraphNode, value: TensorValue): {width:number; rows:number; weights:number[]; count:number} {
  const width = value.shape.at(-1) ?? 1
  if (!width || !value.data.length) throw Error('The objective needs a nonempty feature tensor.')
  const rows = value.data.length / width
  let weights = Array(rows).fill(1) as number[]
  if (loss.params.lossMaskNodeId) {
    const mask = objectiveSignal(graph, loss.params.lossMaskNodeId, loss.params.lossMaskSourceSlot ?? 0)
    const rowShape = value.shape.slice(0,-1)
    const matches = (shape:number[]) => shape.length === rowShape.length && shape.every((n,i) => n === rowShape[i])
    if (!matches(mask.shape) && !(mask.shape.at(-1) === 1 && matches(mask.shape.slice(0,-1)))) throw Error('The loss mask must have one binary value for each token/example row.')
    if (mask.data.length !== rows || mask.data.some(n => n !== 0 && n !== 1)) throw Error('The loss mask must contain only 0 and 1, one per token/example row.')
    weights = mask.data
  }
  const source = graph.nodes.find(n => n.type === 'dataset')
  if ((loss.params.targetMode === 'internal' || loss.params.activationPenaltyNodeId) && source?.params.textData?.representation === 'tokens') {
    const ids = datasetOutputValueForSlot(source,0)
    if (value.shape.length !== 2 || ids.data.length !== rows) throw Error('Internal token reconstruction and activation penalties require [tokens, features], preserving the dataset token positions.')
    if (source.params.textData.fixedLength) weights = weights.map((n,i) => n * Number(ids.data[i] !== 1))
  }
  const count = weights.reduce((sum,n) => sum+n,0)
  if (!count || !width) throw Error('The loss mask needs at least one valid token/example row.')
  return {width,rows,weights,count}
}

export function activationObjective(graph: GraphModel, loss = graph.nodes.find(n => n.type === 'loss' || n.type === 'cross-entropy')) {
  if (!loss?.params.activationPenaltyNodeId) return undefined
  const value = objectiveSignal(graph, loss.params.activationPenaltyNodeId)
  const {width,rows,weights,count} = objectiveRows(graph, loss, value)
  const activeCounts = Array(width).fill(0) as number[]
  let magnitude = 0, active = 0
  for (let row=0;row<rows;row++) if(weights[row]) for(let col=0;col<width;col++) {
    const entry = value.data[row*width+col]
    if (!Number.isFinite(entry)) throw Error('Activation penalty needs finite signals.')
    magnitude += Math.abs(entry)
    if (Math.abs(entry)>0) {active++;activeCounts[col]++}
  }
  const strength = loss.params.activationPenaltyStrength ?? 0
  return {value, penalty:strength*magnitude/count, activePerToken:active/count,
    inactiveFeatures:activeCounts.filter(n=>n===0).length, activeCounts, validTokens:count, sparsityTolerance:0,
    gradient:{shape:[...value.shape],data:value.data.map((n,i)=>strength*weights[Math.floor(i/width)]*Math.sign(n)/count)} as TensorValue}
}
