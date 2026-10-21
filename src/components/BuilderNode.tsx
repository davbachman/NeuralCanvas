import { ARITHMETIC_OPERATIONS, arithmeticOperation, arithmeticOperationExpression, type ArithmeticOperation } from '../domain/arithmetic'
import { NodeIcon } from './NodeIcon'
import { Handle, Position, useUpdateNodeInternals, type NodeProps } from '@xyflow/react'
import { Plus } from 'lucide-react'
import { useEffect, type ReactElement } from 'react'
import { LOSS_OPTIONS, TENSOR_TRANSFORM_OPTIONS, lossKindForNode, inputArityForNode, outputArityForNode, isFlexibleInputNodeType, MAX_FLEX_INPUT_COUNT } from '../domain/engine'
import { DATASET_MENU_OPTIONS, datasetTargetSlotForNode } from '../domain/datasets'
import { inputConnectionLabels, outputConnectionLabel } from '../domain/connectionLabels'
import { builderCardHeight, builderCardWidth, builderInputPortY, builderOutputPortY } from '../domain/builderGeometry'
import { formatCompactTensor, formatShape } from '../domain/tensor'
import type { DatasetKind, GraphNode, LossKind, TensorTransformKind, TensorValue } from '../domain/types'

export interface BuilderNodeData extends Record<string, unknown> {
  graphNode: GraphNode
  showMath: boolean
  showGradient: boolean
  showTraceMetrics?: boolean
  formula: string
  fullFormula: string
  arithmeticInputLabels?: string[]
  outputVariable?: string
  lossKind?: LossKind
  lossOptions?: Array<{ kind: LossKind; label: string }>
  active: boolean
  validationError?: boolean
  accessible?: boolean
  coordinate?: boolean
  hasIncomingValue: boolean
  onFlexibleInputAdd: (nodeId: string) => void
  onValueChange: (nodeId: string, value: TensorValue) => void
  onActivationChange: (nodeId: string, value: string) => void
  onExpressionChange: (nodeId: string, expression: string) => void
  onTransformChange: (nodeId: string, transform: TensorTransformKind) => void
  onLossChange: (nodeId: string, value: LossKind) => void
  onDatasetChange: (nodeId: string, value: DatasetKind) => void
}

/** Trace cards add bounded value previews; complete tensors live in the sidebar. */
export function BuilderNode(props: NodeProps): ReactElement {
  const data = props.data as BuilderNodeData
  const node = data.graphNode
  const updateNodeInternals = useUpdateNodeInternals()
  const inputCount = inputArityForNode(node), outputCount = outputArityForNode(node)
  const lossKind = data.lossKind ?? lossKindForNode(node)
  const inputLabels = inputConnectionLabels(node, lossKind)
  const canAddInput = !data.coordinate && (isFlexibleInputNodeType(node.type) || node.type === 'arithmetic') && inputCount < MAX_FLEX_INPUT_COUNT
  const accessible = data.accessible !== false
  const width = builderCardWidth(node), height = builderCardHeight(node, data.showTraceMetrics)
  const parameterShape = node.type === 'weight' || node.type === 'bias'
    ? (typeof node.params.value === 'object' ? node.params.value.shape : []) : undefined
  const shapeDescription = parameterShape ? `${formatShape(parameterShape)}${parameterShape.length === 0 ? ' (scalar)' : ''}` : undefined
  useEffect(() => { updateNodeInternals(node.id) }, [inputCount, outputCount, width, height, node.id, updateNodeInternals])
  return <div className={`builder-node is-compact node-${node.type} ${data.active ? 'is-active' : ''} ${props.selected ? 'is-selected' : ''} ${data.validationError ? 'has-error' : ''}`}
    aria-hidden={!accessible} inert={!accessible} aria-invalid={data.validationError || undefined}
    style={{ width, height, ...(typeof data.sceneScale === 'number' ? { transform: `scale(${data.sceneScale})`, transformOrigin: 'top left' } : {}) }}>
    <div className="node-title-row"><NodeIcon type={node.type}/><strong title={node.label}>{node.type === 'weight' || node.type === 'bias' ? 'Param' : node.label}</strong></div>
    {shapeDescription && <div className="node-parameter-shape" title={shapeDescription}>{shapeDescription}</div>}
    {node.type === 'activation' && !data.coordinate && <label className="node-field">activation<select aria-label="activation" className="nodrag nowheel" value={node.params.activation ?? 'identity'} onChange={event => data.onActivationChange(node.id, event.target.value)}>
      {['identity', 'relu', 'sigmoid', 'tanh'].map(kind => <option key={kind} value={kind}>{kind === 'relu' ? 'ReLU' : kind}</option>)}
    </select></label>}
    {node.type === 'loss' && <label className="node-field">loss<select aria-label="loss" className="nodrag nowheel" value={lossKind} onChange={event => data.onLossChange(node.id, event.target.value as LossKind)}>
      {(data.lossOptions ?? LOSS_OPTIONS).map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}
    </select></label>}
    {node.type === 'tensor-transform' && <label className="node-field">transform<select aria-label="Tensor transform" className="nodrag nowheel" value={node.params.transform ?? 'reshape'} onChange={event => data.onTransformChange(node.id, event.target.value as TensorTransformKind)}>
      {TENSOR_TRANSFORM_OPTIONS.map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}
    </select></label>}
    {node.type === 'dataset' && <label className="node-field">dataset<select aria-label={`Dataset for ${node.label}`} className="nodrag nowheel" value={node.params.dataset ?? 'line-1d'} onChange={event => data.onDatasetChange(node.id, event.target.value as DatasetKind)}>
      {DATASET_MENU_OPTIONS.map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}
    </select></label>}
    {node.type === 'arithmetic' && <label className="node-field">operation<select aria-label="Arithmetic operation" className="nodrag nowheel"
      value={arithmeticOperation(node.params.expression ?? 'x1 * x2')}
      onChange={event => data.onExpressionChange(node.id, arithmeticOperationExpression(event.target.value as ArithmeticOperation, inputCount))}>
      {ARITHMETIC_OPERATIONS.map(operation => <option key={operation.kind} value={operation.kind}>{operation.label}</option>)}
      <option value="custom" disabled>Custom (see Details)</option>
    </select></label>}
    {inputLabels.map((label, index) => <Handle key={`in-${index}`} id={`in-${index}`} type="target" position={Position.Left}
      className="node-handle" aria-label={`${label} input`} title={`${label} input`} style={{ top: builderInputPortY(node, index) }}/>) }
    <div className="node-input-labels" aria-hidden="true">{inputLabels.map((label, index) =>
      <span key={index} style={{ top: builderInputPortY(node, index) }} title={label}>{label}</span>)}</div>
    {canAddInput && <button type="button" className="node-add-input-button nodrag nopan" style={{ top: builderInputPortY(node, inputCount) }}
      aria-label={`Add input to ${node.label}`} onPointerDown={event => event.stopPropagation()}
      onClick={event => { event.stopPropagation(); data.onFlexibleInputAdd(node.id) }}><Plus size={12} strokeWidth={3}/></button>}
    {Array.from({ length: outputCount }, (_, index) => {
      const label = outputConnectionLabel(node, index, data.outputVariable)
      const target = node.type === 'dataset' && node.params.dataset !== 'custom-csv' && index === datasetTargetSlotForNode(node)
      const description = node.type === 'dataset' && node.params.dataset !== 'custom-csv' ? `${target ? 'Target' : 'Feature'} ${label} output` : `${label} output`
      return <Handle key={`out-${index}`} id={outputCount === 1 ? 'out' : `out-${index}`} type="source" position={Position.Right}
        className={`node-handle source-handle${target ? ' dataset-target-handle' : ''}`} aria-label={description} title={description}
        style={{ top: builderOutputPortY(node, index) }}/>
    })}
    <div className="node-output-labels" aria-hidden="true" style={node.type === 'dataset' ? { width: width - 28 } : undefined}>
      {Array.from({ length: outputCount }, (_, index) => <span key={index} style={{ top: builderOutputPortY(node, index) }} title={outputConnectionLabel(node, index, data.outputVariable)}>{outputConnectionLabel(node, index, data.outputVariable)}</span>)}
    </div>
    {data.showTraceMetrics && <div className="node-trace-metrics" style={{ top: builderCardHeight(node) }}>
      {([['Output', node.value], ['Gradient', node.grad]] as const).map(([label, value]) => {
        const preview = `${value?.shape.length ? `${formatShape(value.shape)} ` : ''}${formatCompactTensor(value)}`
        return <div key={label}><span>{label}</span><code title={preview} aria-label={`${label} preview`}>{preview}</code></div>
      })}
    </div>}
  </div>
}
