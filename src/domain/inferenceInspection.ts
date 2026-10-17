import { createForwardEvaluator, inputArityForNode } from './engine'
import { withDatasetExample } from './datasetTraining'
import { datasetExamplesForNode, datasetMode, datasetOutputValueForSlot } from './datasets'
import { tensorExampleRows } from './tensorExampleRows'
import { tensorAxisLabels } from './tensorLabels'
import { toTensor } from './tensor'
import type { GraphModel, TensorValue } from './types'

export type InspectionValue = {
  status: 'available'; value: TensorValue; labels: Array<string[] | undefined>; axes: string[]
  kind: 'example' | 'shared' | 'aggregate'; simple: boolean
} | { status: 'not-evaluated' | 'disconnected'; message: string }
type Captured = { value: TensorValue; rows?: number[]; axes: string[]; labels: Array<string[] | undefined>; aggregate?: boolean }

/** Execution results, layout, and the canvas selection do not alter a frozen
 * inference run. Numerical settings, dataset contents, and wiring do. */
export function inferenceSignature(graph: GraphModel, datasetId = graph.nodes.find(node => node.type === 'dataset')?.id): string {
  return JSON.stringify({
    nodes: graph.nodes.map(node => ({ id: node.id, type: node.type, params: Object.fromEntries(Object.entries(node.params).filter(([key]) => key !== 'outputName' && !(node.id === datasetId && ['datasetIndex', 'datasetSplit', 'datasetValues', 'storyExperiment'].includes(key)))) })),
    edges: graph.edges.map(({ source, target, sourceSlot, inputSlot }) => ({ source, target, sourceSlot, inputSlot })),
  })
}

let nextRunId = 0

/** A snapshot of model/data, plus a bounded LRU of individual signals. Full
 * execution graphs are transient, never retained for every evaluated example. */
export class InferenceInspection {
  readonly runId = ++nextRunId
  readonly datasetId: string
  readonly split: 'train' | 'test'
  readonly backend?: string
  readonly batchSize: number
  readonly graph: GraphModel
  readonly indices: number[]
  readonly signature: string
  readonly title: string
  private cache = new Map<string, Captured>()
  private elements = 0
  private queue: Promise<unknown> = Promise.resolve()
  private evaluate?: ReturnType<typeof createForwardEvaluator>
  static readonly maxElements = 100_000
  static readonly maxEntries = 24
  constructor(graph: GraphModel, datasetId: string, split: 'train' | 'test', backend?: string, batchSize = 64) {
    this.datasetId = datasetId; this.split = split; this.backend = backend; this.batchSize = batchSize
    this.signature = inferenceSignature(graph, datasetId)
    this.graph = structuredClone({ ...graph,
      nodes: graph.nodes.map(node => ({ ...node, value: undefined, grad: undefined, cache: undefined, localDerivative: undefined,
        params: node.id === datasetId ? { ...node.params, datasetValues: undefined, storyExperiment: undefined } : node.params })),
      edges: graph.edges.map(edge => ({ ...edge, value: undefined, grad: undefined })),
    })
    const source = this.graph.nodes.find(node => node.id === datasetId)!
    this.indices = datasetExamplesForNode(source).flatMap((row, i) => row.split === split ? [i] : [])
    this.title = `${split === 'test' ? 'Held-out' : 'Training-set'} inference — ${this.indices.length} examples`
  }
  get cacheSize() { return { entries: this.cache.size, elements: this.elements } }
  private remember(key: string, value: Captured) {
    if (value.value.data.length > InferenceInspection.maxElements) return
    while (this.cache.size >= InferenceInspection.maxEntries || this.elements + value.value.data.length > InferenceInspection.maxElements) {
      const oldest = this.cache.keys().next().value!
      this.elements -= this.cache.get(oldest)!.value.data.length
      this.cache.delete(oldest)
    }
    this.cache.set(key, value); this.elements += value.value.data.length
  }
  read(id: string, slot: number, row: number): Promise<InspectionValue> {
    // Serialize device access and avoid duplicate work during rapid navigation.
    const work = this.queue.then(() => this.readSignal(id, slot, row))
    this.queue = work.then(() => undefined, () => undefined)
    return work
  }
  private async readSignal(id: string, slot: number, row: number): Promise<InspectionValue> {
    const node = this.graph.nodes.find(node => node.id === id)
    if (!node || !this.indices.includes(row)) return { status: 'not-evaluated', message: 'This output or dataset row was not evaluated in this inference run.' }
    const ancestors = new Set<string>()
    const visit = (id: string) => { if (ancestors.has(id)) return; ancestors.add(id); this.graph.edges.filter(edge => edge.target === id).forEach(edge => visit(edge.source)) }
    visit(id)
    const disconnected = this.graph.nodes.some(candidate => ancestors.has(candidate.id) && !['input', 'target'].includes(candidate.type) &&
      Array.from({ length: inputArityForNode(candidate) }, (_, slot) => slot).some(slot => !this.graph.edges.some(edge => edge.target === candidate.id && (edge.inputSlot ?? 0) === slot)))
    if (disconnected) return { status: 'disconnected', message: 'This output is unavailable because the block or one of its inputs is disconnected.' }
    const dependent = ancestors.has(this.datasetId)
    const source = this.graph.nodes.find(node => node.id === this.datasetId)!
    const examples = datasetExamplesForNode(source)
    const batch = !this.backend && datasetMode(source) === 'batch'
    const offset = this.backend ? Math.floor(this.indices.indexOf(row) / this.batchSize) * this.batchSize : 0
    const rows = this.backend ? this.indices.slice(offset, offset + this.batchSize) : batch ? this.indices : [row]
    const key = JSON.stringify([id, slot, dependent ? (this.backend ? offset : batch ? 'split' : row) : 'shared'])
    let captured = this.cache.get(key)
    if (captured) { this.cache.delete(key); this.cache.set(key, captured) }
    else {
      // Parameters are shared model tensors, not repeated example outputs.
      if (node.type === 'weight' || node.type === 'bias') captured = { value: toTensor(node.params.value), axes: [], labels: tensorAxisLabels(this.graph, id, slot) }
      else if (this.backend) {
        const { TensorGraph, tf } = await import('./tensorTraining')
        if (tf.getBackend() !== this.backend) {
          if (!await tf.setBackend(this.backend)) throw Error(`Inference backend ${this.backend} is unavailable.`)
          await tf.ready()
        }
        const model = new TensorGraph(this.graph)
        try {
          const result = await model.inspectSignal(rows.map(index => model.examples[index]), id, slot)
          if (result) captured = { ...result, rows: result.axes[0] === 'batch' ? rows : undefined,
            aggregate: dependent && result.axes[0] !== 'batch', labels: tensorAxisLabels(withDatasetExample(this.graph, this.datasetId, row), id, slot) }
        } finally { model.dispose() }
      } else {
        this.evaluate ??= createForwardEvaluator(this.graph)
        const input = batch ? { ...this.graph, nodes: this.graph.nodes.map(n => n.id === source.id ? { ...n, params: { ...n.params, datasetSplit: this.split } } : n) } : withDatasetExample(this.graph, this.datasetId, row)
        const evaluated = this.evaluate(input).graph
        const evaluatedNode = evaluated.nodes.find(n => n.id === id)!
        const value = evaluatedNode.type === 'dataset' ? datasetOutputValueForSlot(evaluatedNode, slot) : evaluatedNode.value
        if (value) captured = { value, rows: batch ? tensorExampleRows(evaluated, id, slot) : undefined,
          aggregate: batch && dependent && !tensorExampleRows(evaluated, id, slot), axes: [], labels: tensorAxisLabels(evaluated, id, slot) }
      }
      if (!captured) return { status: dependent ? 'not-evaluated' : 'disconnected', message: dependent ? 'This output was not evaluated in this inference run.' : 'This output is unavailable because the block is disconnected from the evaluated dataset.' }
      this.remember(key, captured)
    }
    let value = captured.value, axes = captured.axes, labels = captured.labels
    if (captured.rows) {
      const index = captured.rows.indexOf(row)
      if (index < 0) return { status: 'not-evaluated', message: 'This output does not contain the selected dataset row.' }
      const width = value.data.length / captured.rows.length
      value = { shape: value.shape.slice(1), data: value.data.slice(index * width, (index + 1) * width),
        ...(value.excluded ? { excluded: value.excluded.slice(index * width, (index + 1) * width) } : {}) }
      axes = axes.slice(1)
      if (!this.backend) labels = labels.slice(1)
    }
    if (this.backend && axes.includes('token')) {
      // Expose padding as masked cells, keeping the run's original padded shape.
      labels = tensorAxisLabels(withDatasetExample(this.graph, this.datasetId, row), id, slot)
      const length = examples[row].features[1].data.length
      value = { ...value, excluded: value.data.map((_, flat) => value.shape.some((size, axis) => axes[axis] === 'token' && Math.floor(flat / value.shape.slice(axis + 1).reduce((a, b) => a * b, 1)) % size >= (labels[axis]?.length ?? length))) }
    }
    const tokenAxis = axes.includes('token') || labels.some(axis => axis?.some(label => /^\d+: /.test(label)))
    return { status: 'available', value, axes, labels, kind: !dependent ? 'shared' : captured.aggregate ? 'aggregate' : 'example',
      simple: !tokenAxis && (value.shape.length <= 1 || value.shape.length === 2 && value.shape[0] === 1) }
  }
}
