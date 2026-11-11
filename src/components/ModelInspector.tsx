import { NumberInput } from './NumberInput'
import { normalizeParameterValue } from '../domain/interpretability'
import { datasetOutputCountForNode, datasetOutputLabelForSlot } from '../domain/datasets'
import { ArithmeticExpressionInput } from './ArithmeticExpressionInput'
import {StandardizationInspector} from './StandardizationInspector'
import { useState } from 'react'
import { initializeTensor, initializerForNode, INITIALIZERS, operationHelp, type Initializer } from '../domain/authoring'
import { ConvolutionInspector } from './ConvolutionInspector'
import { TensorHeatmap } from './TensorHeatmap'
import { EditableBlockTitle } from './EditableBlockTitle'
import { arithmeticInputLabels, isLossNode, lossFormulaForNode, lossKindForNode, lossOptionsForNode, TENSOR_TRANSFORM_OPTIONS } from '../domain/engine'
import { formatFullTensor, toTensor } from '../domain/tensor'
import { formatInputValues, parseInputValues } from '../domain/inputValues'
import type { CoordinateBinding } from '../domain/neuronProjection'
import type {
  GraphGroup,
  GraphModel,
  GraphNode,
  NodeParams,
  TensorValue,
  TensorTransformKind,
} from '../domain/types'

interface Props {
  graph: GraphModel
  node?: GraphNode
  binding?: CoordinateBinding
  group?: GraphGroup
  onGroupChange?: (id: string, changes: Partial<Pick<GraphGroup, 'label' | 'kind'>>) => void
  onRename?: (id: string, label: string) => void
  onParams: (id: string, params: NodeParams) => void
  onValue: (id: string, value: TensorValue) => void
  onOpen: (id: string) => void
  onInspectNeuron: (id: string, unit: number, row?: number) => void
  onGroup: () => void
  onUnion?: () => void
  selectionCount: number
}

export function ModelInspector({
  graph,
  node,
  binding,
  group,
  onParams,
  onGroupChange,
  onRename,
  onValue,
  onOpen,
  onInspectNeuron,
  onGroup,
  onUnion,
  selectionCount,
}: Props) {
  const canonical = binding
    ? graph.nodes.find((candidate) => candidate.id === binding.nodeId)
    : node
  const weight = group?.detail?.weightNodeId
    ? graph.nodes.find(
        (candidate) => candidate.id === group.detail?.weightNodeId,
      )
    : undefined
  const width = weight ? toTensor(weight.params.value).shape[1] : 0
  const children =
    graph.groups?.filter((candidate) => candidate.parentId === group?.id) ?? []
  const parameter = node?.type === 'weight' || node?.type === 'bias'
  return (
    <section className="model-inspector">
      <EditableBlockTitle
        key={group?.id ?? node?.id ?? 'selection'}
        label={group?.label ?? (parameter ? 'Param' : node?.label) ?? `${selectionCount} blocks selected`}
        onRename={group && onGroupChange ? label => onGroupChange(group.id, { label }) : node && onRename && !binding && !parameter ? label => onRename(node.id, label) : undefined}
      />
      {group && (
        <>
          {onGroupChange && <details className="inspector-disclosure"><summary>Group settings</summary><label className="inspector-field">Block kind<select aria-label="Block kind" value={group.kind ?? 'module'} onChange={event => onGroupChange(group.id, { kind: event.target.value })}>{['module','neuron','layer','mlp','head','attention','transformer-block','cnn','convolution','network','embedding','normalization','projection'].map(kind=><option key={kind}>{kind}</option>)}</select></label></details>}
          <button className="inspector-wide" onClick={() => onOpen(group.id)}>
            Zoom into {group.kind ?? 'block'} ↗
          </button>
          {children.length > 0 && (
            <div className="inspector-children">
              {children.map((child) => (
                <button key={child.id} onClick={() => onOpen(child.id)}>
                  {child.label}
                  <span>↗</span>
                </button>
              ))}
            </div>
          )}
          {width > 0 && (
            <label className="inspector-field">
              Inspect a neuron
              <select
                value={
                  graph.view?.inspectedNeuron?.groupId === group.id
                    ? graph.view.inspectedNeuron.unitIndex
                    : ''
                }
                onChange={(event) =>
                  onInspectNeuron(group.id, Number(event.target.value))
                }
              >
                <option value="" disabled>
                  Choose a neuron
                </option>
                {Array.from({ length: width }, (_, i) => (
                  <option value={i} key={i}>
                    Neuron {i + 1}
                  </option>
                ))}
              </select>
            </label>
          )}
        </>
      )}
      {graph.view?.inspectedNeuron && (
        <label className="inspector-field">
          Token row
          <NumberInput
            aria-label="Inspected token row"
            type="number"
            min="1"
            max={
              toTensor(
                graph.nodes.find(candidate => candidate.id === graph.groups?.find(group => group.id === graph.view?.inspectedNeuron?.groupId)?.detail?.inputNodeId)?.value,
              ).shape[0] ?? 1
            }
            value={graph.view.inspectedNeuron.row + 1}
            onChange={(event) => {
              const focus = graph.view!.inspectedNeuron!
              onInspectNeuron(
                focus.groupId,
                focus.unitIndex,
                Math.max(0, Number(event.target.value) - 1),
              )
            }}
          />
        </label>
      )}
      {node && (
        <>
          {!binding && !node.id.startsWith('inspect:') && !['dataset', 'input', 'target', 'loss', 'cross-entropy'].includes(node.type) &&
            <label className="inspector-field">Output variable
              <input aria-label="Output variable" placeholder={parameter ? 'Automatic (p1, p2, …)' : 'Automatic (z1, z2, …)'} value={node.params.outputName ?? ''}
                onChange={event => {
                  const name = event.target.value
                  if (!name || /^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) onParams(node.id, { outputName: name || undefined })
                }} />
              <span>Used in formulas and pseudocode. Clear for automatic naming. Duplicate names get a numeric suffix.</span>
            </label>}
          {operationHelp[node.type] && <p className="coordinate-note">{operationHelp[node.type]}</p>}
          {parameter && !binding && <fieldset><legend>Parameter training</legend>
            <label className="inspector-field">Update setting<select aria-label="Parameter update setting" value={node.params.trainable === false ? 'frozen' : 'trainable'} onChange={event => onParams(node.id, { trainable: event.target.value === 'trainable' })}><option value="trainable">Trainable</option><option value="frozen">Frozen</option></select></label>
            <p>Frozen values stay fixed during training and initialization. Gradients still flow through their calculations.</p>
            <label className="inspector-field">Norm constraint<select aria-label="Parameter norm constraint" value={node.params.constraint ?? 'none'} disabled={node.params.trainable === false} onChange={event => onParams(node.id, { constraint: event.target.value === 'unit-row-norm' ? 'unit-row-norm' : undefined })}><option value="none">None</option><option value="unit-row-norm" disabled={toTensor(node.params.value).shape.length !== 2}>Unit row norm · matrix directions</option></select></label>
            {toTensor(node.params.value).shape.length !== 2 && <p>Initialize a matrix shape before choosing a row-norm constraint.</p>}
            {node.params.constraint && <p>Each matrix row has Euclidean norm 1 after initialization and every update. A zero row becomes [1, 0, …].</p>}
          </fieldset>}
          {node.type === 'arithmetic' && !node.id.startsWith('inspect:') && <ArithmeticEditor key={node.id} graph={graph} node={node} onParams={onParams}/>}
          {node.type === 'one-hot' && <label className="inspector-field">Vocabulary size<NumberInput aria-label="One-hot vocabulary size" type="number" min="2" max="8192" value={node.params.numClasses ?? 2} onChange={event => onParams(node.id, {numClasses: Number(event.target.value)})}/></label>}
          {node.type === 'argsort' && <label className="inspector-field">Sort order<select aria-label="Argsort order" value={node.params.descending === false ? 'ascending' : 'descending'} onChange={event => onParams(node.id, {descending: event.target.value === 'descending'})}><option value="descending">Descending</option><option value="ascending">Ascending</option></select></label>}
          {node.type === 'index-mask' && <label className="inspector-field">Index selection<select aria-label="Index mask mode" value={node.params.indexMaskMode ?? 'keep'} onChange={event => onParams(node.id, {indexMaskMode: event.target.value as NodeParams['indexMaskMode']})}><option value="keep">Keep selected</option><option value="exclude">Exclude selected</option></select></label>}
          {node.type === 'conv2d' && <ConvolutionInspector graph={graph} node={node} onValue={onValue}/>}
          {node.type === 'loss' && <label className="inspector-field">Loss<select aria-label="Loss function" value={lossKindForNode(node, graph)} onChange={event => onParams(node.id,{loss:event.target.value as NodeParams['loss']})}>{lossOptionsForNode(node, graph).map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}</select></label>}
          {isLossNode(node) && <div className="inspector-field">Formula<code className="inspector-formula" aria-label="Loss formula">{lossFormulaForNode(node, graph, lossKindForNode(node, graph) === 'cross-entropy' ? 'target' : undefined)}</code>
            {lossKindForNode(node, graph) === 'cross-entropy' && <span>target is the correct-class index supplied to the target input. [target] selects that class’s probability.</span>}
          </div>}
          {(node.type==='loss'||node.type==='cross-entropy') && <fieldset><legend>Parameter regularization</legend>
            <label className="inspector-field">Penalty<select aria-label="Regularization penalty" value={node.params.regularization??'none'} onChange={event=>onParams(node.id,{regularization:event.target.value as NodeParams['regularization']})}><option value="none">None</option><option value="l1">L1 · λ Σ |w|</option><option value="l2">L2 · λ/2 Σ w²</option></select></label>
            {node.params.regularization && node.params.regularization!=='none' && <>
              <label className="inspector-field">Strength λ<NumberInput aria-label="Regularization strength" type="number" min="0" step="0.001" value={node.params.regularizationStrength??0} onChange={event=>onParams(node.id,{regularizationStrength:Number(event.target.value)})}/></label>
              <p>The Loss block shows data loss plus this penalty. Dataset evaluation reports data loss. This is separate from AdamW weight decay; use decay 0 to isolate the penalty.</p>
              {graph.nodes.filter(n=>n.type==='weight'||n.type==='bias').map(n=>{const ids=node.params.regularizationParameterIds??graph.nodes.filter(p=>p.type==='weight').map(p=>p.id);return <label key={n.id} style={{display:'block'}}><input type="checkbox" checked={ids.includes(n.id)} onChange={event=>onParams(node.id,{regularizationParameterIds:event.target.checked?[...ids,n.id]:ids.filter(id=>id!==n.id)})}/>{n.label}</label>})}
            </>}
          </fieldset>}
          {isLossNode(node) && <fieldset><legend>Activation objective</legend>
            <label className="inspector-field">Target source<select aria-label="Loss target source" value={node.params.targetMode ?? 'dataset'} onChange={event => onParams(node.id, { targetMode: event.target.value === 'internal' ? 'internal' : undefined })}><option value="dataset">Dataset target</option><option value="internal">Frozen internal activation</option></select></label>
            {node.params.targetMode === 'internal' && <p>Use mean squared error. Connect a frozen, deterministic activation to target port 2. Target-side gradients are stopped; the predictor still receives gradients.</p>}
            <label className="inspector-field">Valid-token mask<select aria-label="Loss valid-token mask" value={node.params.lossMaskNodeId ? `${node.params.lossMaskNodeId}:${node.params.lossMaskSourceSlot ?? 0}` : ''} onChange={event => {
              const option = event.target.selectedOptions[0]
              onParams(node.id, { lossMaskNodeId: option.dataset.node || undefined, lossMaskSourceSlot: option.dataset.node ? Number(option.dataset.slot) : undefined })
            }}><option value="">All valid positions</option>{graph.nodes.filter(candidate => !isLossNode(candidate)).flatMap(candidate => Array.from({ length: candidate.type === 'dataset' ? datasetOutputCountForNode(candidate) : 1 }, (_, slot) => <option key={`${candidate.id}:${slot}`} value={`${candidate.id}:${slot}`} data-node={candidate.id} data-slot={slot}>{candidate.label}{candidate.type === 'dataset' ? ` · ${datasetOutputLabelForSlot(candidate, slot)}` : ''}</option>))}</select></label>
            <p>Token padding is excluded automatically. An optional fixed binary mask further restricts positions: 1 includes, 0 excludes. Reconstruction averages over valid tokens and coordinates.</p>
            <label className="inspector-field">Penalized activation<select aria-label="Penalized activation" value={node.params.activationPenaltyNodeId ?? ''} onChange={event => onParams(node.id, { activationPenaltyNodeId: event.target.value || undefined })}><option value="">None</option>{graph.nodes.filter(candidate => !['dataset', 'weight', 'bias', 'loss', 'cross-entropy'].includes(candidate.type)).map(candidate => <option key={candidate.id} value={candidate.id}>{candidate.label} · {candidate.id}</option>)}</select></label>
            {node.params.activationPenaltyNodeId && <><label className="inspector-field">Activation L1 strength λ<NumberInput aria-label="Activation L1 strength" type="number" min="0" step="0.001" value={node.params.activationPenaltyStrength ?? 0} onChange={event => onParams(node.id, { activationPenaltyStrength: Number(event.target.value) })}/></label><p>λ × mean over valid tokens of the sum of absolute feature activations. This is separate from parameter regularization. The derivative at zero is zero.</p></>}
          </fieldset>}
          {node.type === 'loss' && lossKindForNode(node, graph) === 'binary-cross-entropy-with-logits' && <p className="coordinate-note">Connect raw scores directly. This loss includes sigmoid; use a separate sigmoid only to inspect probabilities.</p>}
          {node.type === 'loss' && lossKindForNode(node, graph) === 'binary-cross-entropy' && <p className="coordinate-note">Legacy probability loss: keep the sigmoid before this block. To use logits, remove that sigmoid connection and choose Binary cross entropy (logits).</p>}
          {node.type === 'tensor-transform' && <label className="inspector-field">Transform<select aria-label="Tensor transform operation" value={node.params.transform ?? 'reshape'} onChange={event => onParams(node.id, { transform: event.target.value as TensorTransformKind })}>{TENSOR_TRANSFORM_OPTIONS.map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}</select></label>}
          {binding && canonical && (
            <p className="coordinate-note">
              Coordinate {binding.index + 1} of {canonical.label}. This is the
              model’s shared parameter
              {canonical.type === 'input' ? ' or input' : ''}; its gradient
              includes all uses.
            </p>
          )}
          {canonical && binding?.editable && (
            <label className="inspector-field">
              Edit coordinate
              <NumberInput
                aria-label="Edit tensor coordinate"
                type="number"
                step="0.05"
                value={toTensor(canonical.params.value).data[binding.index]}
                onChange={(event) => {
                  const value = toTensor(canonical.params.value)
                  onValue(canonical.id, {
                    ...value,
                    data: value.data.map((item, i) =>
                      i === binding.index ? Number(event.target.value) : item,
                    ),
                  })
                }}
              />
            </label>
          )}
          {canonical &&
            !binding &&
            !graph.edges.some(edge => edge.target === canonical.id) &&
            ['input', 'target', 'weight', 'bias'].includes(canonical.type) && (
              <TensorEditor
                key={canonical.id}
                node={canonical}
                onValue={onValue}
                onParams={onParams}
              />
            )}
          {node.type === 'standardize' && !node.id.startsWith('inspect:') && <StandardizationInspector key={node.id} graph={graph} node={node} onParams={onParams}/>}
          {node.type === 'dropout' && !node.id.startsWith('inspect:') && <><label className="inspector-field">Dropout probability<NumberInput aria-label="Dropout probability" type="number" min="0" max="0.99" step="0.05" value={node.params.dropoutRate ?? 0.1} onChange={event => onParams(node.id,{dropoutRate:Number(event.target.value)})}/><span>Training drops activations and scales survivors. Evaluation passes them unchanged.</span></label><label><input type="checkbox" checked={node.params.inferenceOnly ?? false} onChange={event => onParams(node.id, { inferenceOnly: event.target.checked })}/>Always use evaluation mode</label><p>Enable this for stable frozen representations, including during autoencoder training.</p></>}
          {node.type === 'activation' &&
            !binding &&
            !node.id.startsWith('inspect:') && (
              <label className="inspector-field">
                Activation
                <select
                  value={node.params.activation ?? 'identity'}
                  onChange={(event) =>
                    onParams(node.id, {
                      activation: event.target.value as NodeParams['activation'],
                    })
                  }
                >
                  {['identity', 'relu', 'sigmoid', 'tanh'].map((name) => (
                    <option key={name}>{name}</option>
                  ))}
                </select>
              </label>
            )}
          {[
            'slice',
            'transpose',
            'concat',
            'reshape',
            'mean',
            'layer-norm',
            'add',
            'tensor-transform',
          ].includes(node.type) &&
            !node.id.startsWith('inspect:') && (
              <OperationEditor
                key={`${node.id}:${JSON.stringify(node.params)}`}
                node={node}
                onParams={onParams}
              />
            )}
          {node.value && node.value.data.length > 1 && (
            <TensorHeatmap
              key={`${node.id}:${node.value.shape.join(',')}`}
              value={node.value}
              onEdit={
                canonical &&
                !binding &&
                !graph.edges.some(edge => edge.target === canonical.id) &&
                ['weight', 'bias', 'input', 'target'].includes(canonical.type)
                  ? (index, next) => {
                      const value = toTensor(canonical.params.value)
                      onValue(canonical.id, {
                        ...value,
                        data: value.data.map((item, i) =>
                          i === index ? next : item,
                        ),
                      })
                    }
                  : undefined
              }
            />
          )}
          {(node.value || node.grad) && <details className="inspector-disclosure"><summary>Exact values and gradients</summary><div className="inspector-values">
            <span>
              Value<strong>{formatFullTensor(node.value)}</strong>
            </span>
            <span>
              Gradient<strong>{formatFullTensor(node.grad)}</strong>
            </span>
          </div></details>}
        </>
      )}
      {selectionCount > 1 && (
        <div className="inspector-actions">
          <button title="Create an outer block around the selection." onClick={onGroup}>Merge blocks</button>
          {onUnion && <button title="Combine the selected modules’ contents in one block." onClick={onUnion}>Union contents</button>}
        </div>
      )}
    </section>
  )
}

function TensorEditor({
  node,
  onValue,
  onParams,
}: {
  node: GraphNode
  onValue: Props['onValue']
  onParams: Props['onParams']
}) {
  const value = toTensor(node.params.value)
  const isInput = node.type === 'input'
  const frozen = (node.type === 'weight' || node.type === 'bias') && node.params.trainable === false
  const manualInput = isInput && node.params.inputMode !== 'random'
  const editableData = isInput ? formatInputValues(value) : value.data.map(String).join(', ')
  const [shape, setShape] = useState(value.shape.join(', ')),
    [data, setData] = useState(editableData),
    [error, setError] = useState('')
  const initializer = initializerForNode(node)
  const [parameterSeed, setParameterSeed] = useState(() => [...node.id].reduce((seed,letter) => (seed*31+letter.charCodeAt(0)) >>> 0,42))
  const seed = isInput ? node.params.inputSeed ?? parameterSeed : parameterSeed
  const signature = JSON.stringify(value)
  const [source, setSource] = useState(signature)
  if (source !== signature) {
    setSource(signature)
    setShape(value.shape.join(', '))
    setData(editableData)
  }
  const initializeShape = () => {
    if (frozen) return
    try {
      onValue(node.id, normalizeParameterValue(node, initializeTensor(parseTensorShape(shape), initializer, seed)))
      setError('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Invalid shape.')
    }
  }
  return (
    <form
      className="tensor-editor"
      onSubmit={(event) => {
        event.preventDefault()
        if (frozen) return
        if (manualInput) {
          try {
            onValue(node.id, parseInputValues(data))
            setError('')
          } catch (cause) {
            setError(cause instanceof Error ? cause.message : 'Invalid input values.')
          }
          return
        }
        if (isInput) {
          initializeShape()
          return
        }
        let dimensions: number[]
        try {
          dimensions = parseTensorShape(shape)
        } catch (cause) {
          setError(cause instanceof Error ? cause.message : 'Invalid shape.')
          return
        }
        const numbers = data
          .split(/[\s,]+/)
          .filter(Boolean)
          .map(Number)
        if (
          dimensions.some((size) => !Number.isInteger(size) || size <= 0) ||
          dimensions.reduce((a, b) => a * b, 1) !== numbers.length ||
          numbers.some((item) => !Number.isFinite(item))
        ) {
          setError(
            'Use positive dimensions matching the number of finite values.',
          )
          return
        }
        try {
          onValue(node.id, normalizeParameterValue(node, { shape: dimensions, data: numbers }))
          setError('')
        } catch (cause) {
          setError(cause instanceof Error ? cause.message : 'Invalid parameter values.')
        }
      }}
    >
      {isInput && <>
        <label className="inspector-field">Input values
          <select aria-label="Input value mode" value={node.params.inputMode ?? 'manual'} onChange={event => {
            onParams(node.id, { inputMode: event.target.value as NodeParams['inputMode'] })
            setError('')
          }}>
            <option value="manual">Manual Value Entry</option>
            <option value="random">Random Inputs</option>
          </select>
        </label>
        <p className="coordinate-note">Input values stay fixed during training. Param values are updated by training.</p>
      </>}
      {!manualInput && <>
        <label className="inspector-field">
          Shape <span>e.g. 3, 1 or (3, 1); empty = scalar</span>
          <input
            aria-label="Tensor shape"
            value={shape}
            onChange={(event) => setShape(event.target.value)}
            onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); initializeShape() } }}
          />
        </label>
        <p className="coordinate-note">{isInput ? 'Choose a shape, initialization method, and seed, then generate fixed input values.' : 'Enter a shape, then initialize its values or enter all values below and apply them.'}</p>
        <div className="tensor-initializer"><label className="inspector-field">Initialize<select aria-label="Tensor initializer" value={initializer} onChange={event => onParams(node.id, { initializer: event.target.value as Initializer })}>{INITIALIZERS.map(kind => <option key={kind} value={kind}>{kind === 'xavier' ? 'Xavier · matrices' : kind === 'he' ? 'He · ReLU / filters' : kind}</option>)}</select></label><button type="button" disabled={frozen} onClick={initializeShape}>{isInput ? 'Generate inputs' : 'Initialize tensor'}</button></div>
        {frozen && <p className="coordinate-note">Unfreeze this parameter before changing or initializing its values.</p>}
        {(node.type === 'weight' || node.type === 'bias') && <p className="coordinate-note">Saved with this block. Train → Randomize parameters uses this method with fresh random draws.</p>}
        <label className="inspector-field">Random seed<NumberInput aria-label="Initializer seed" type="number" min={isInput ? 0 : undefined} max={isInput ? 0xffffffff : undefined} step="1" value={seed} onChange={event => {
          const next = Number(event.target.value)
          if (isInput) {
            if (Number.isInteger(next) && next >= 0 && next <= 0xffffffff) onParams(node.id, { inputSeed: next })
          } else setParameterSeed(next)
        }}/></label>
      </>}
      {(!isInput || manualInput) && <>
        <label className="inspector-field">
          Values
          <textarea
            aria-label="Tensor values"
            rows={Math.min(6, Math.max(2, Math.ceil(value.data.length / 6)))}
            value={data}
            placeholder={isInput ? '2 or 1, 2, 3 or [[1, 2], [3, 4]]' : undefined}
            onChange={(event) => setData(event.target.value)}
          />
        </label>
        {manualInput && <p className="coordinate-note">Enter a number, a comma-separated list, or nested arrays such as [[1, 2], [3, 4]]. Brackets determine matrix and tensor shapes.</p>}
        <button type="submit" className="inspector-wide" disabled={frozen}>
          Apply values
        </button>
      </>}
      {error && <p role="alert">{error}</p>}
    </form>
  )
}

function parseTensorShape(text: string): number[] {
  const trimmed = text.trim()
  const unwrapped = trimmed.startsWith('(') && trimmed.endsWith(')') || trimmed.startsWith('[') && trimmed.endsWith(']')
    ? trimmed.slice(1, -1).trim()
    : trimmed
  if (/[()[\]]/.test(unwrapped)) throw new Error('Use dimensions such as 3, 1 or (3, 1).')
  const dimensions = unwrapped ? unwrapped.split(/[\s,×x]+/).filter(Boolean).map(Number) : []
  if (dimensions.some(size => !Number.isInteger(size) || size <= 0)) throw new Error('Use positive dimensions (whole numbers) such as 3, 1.')
  return dimensions
}

function OperationEditor({
  node,
  onParams,
}: {
  node: GraphNode
  onParams: Props['onParams']
}) {
  const operation = node.type === 'tensor-transform' ? node.params.transform ?? 'reshape' : node.type
  const fields =
    operation === 'slice'
      ? ['axis', 'start', 'end']
      : operation === 'transpose'
        ? ['axes']
        : operation === 'reshape'
          ? ['shape']
          : operation === 'layer-norm'
            ? ['epsilon']
            : operation === 'mean'
              ? ['axis']
              : operation === 'concat'
                ? ['axis', 'inputCount']
                : ['inputCount']
  const [draft, setDraft] = useState<Record<string, string>>(
    Object.fromEntries(
      fields.map((field) => [
        field,
        Array.isArray(node.params[field as keyof NodeParams])
          ? (node.params[field as keyof NodeParams] as number[]).join(', ')
          : String(
              node.params[field as keyof NodeParams] ??
                (field === 'inputCount'
                  ? 2
                  : field === 'epsilon'
                    ? 0.00001
                    : ['axis', 'start'].includes(field) && operation !== 'mean' ? (field === 'axis' && operation === 'concat' ? 1 : 0) : ''),
            ),
      ]),
    ),
  )
  const [error, setError] = useState('')
  const [keepDims, setKeepDims] = useState(node.params.keepDims ?? false)
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        const params = Object.fromEntries(
          fields.map((field) => [
            field,
            draft[field].trim() === '' && (field !== 'shape' || node.type === 'tensor-transform') ? undefined : ['axes', 'shape'].includes(field)
              ? draft[field]
                  .split(/[\s,]+/)
                  .filter(Boolean)
                  .map(Number)
              : Number(draft[field]),
          ]),
        )
        const numbers = Object.values(params).flat().filter(value => value !== undefined) as number[]
        if (numbers.some(value => !Number.isFinite(value) || (value < 0 && !(operation === 'reshape' && value === -1))) || (params.epsilon !== undefined && Number(params.epsilon) <= 0) || (params.inputCount !== undefined && (!Number.isInteger(params.inputCount) || Number(params.inputCount) < 2 || Number(params.inputCount) > 16)) || fields.some(field => field !== 'epsilon' && params[field] !== undefined && (Array.isArray(params[field]) ? params[field] as number[] : [params[field] as number]).some(n => !Number.isInteger(n))) || (Array.isArray(params.shape) && (params.shape.some(n => n < 1 && n !== -1) || params.shape.filter(n => n === -1).length > 1))) {
          setError('Use whole nonnegative indices, positive dimensions, 2–16 inputs, and a positive epsilon.'); return
        }
        onParams(node.id, {...params, ...(operation === 'mean' ? {keepDims} : {})})
        setError('')
      }}
    >
      <div className="operation-fields">
        {fields.map((field) => (
          <label className="inspector-field" key={field}>
            {field}
            <input
              aria-label={`Operation ${field}`}
              value={draft[field]}
              onChange={(event) =>
                setDraft({ ...draft, [field]: event.target.value })
              }
            />
          </label>
        ))}
      </div>
      {operation === 'mean' && <label className="inspector-field"><span><input type="checkbox" checked={keepDims} onChange={event => setKeepDims(event.target.checked)}/> Keep dimensions</span></label>}
      {error && <p role="alert">{error}</p>}
      <button className="inspector-wide">Apply operation</button>
    </form>
  )
}

function ArithmeticEditor({ graph, node, onParams }: { graph: GraphModel; node: GraphNode; onParams: Props['onParams'] }) {
  return <label className="inspector-field">Expression<ArithmeticExpressionInput node={node} inputLabels={arithmeticInputLabels(node, graph)} onChange={expression => onParams(node.id, { expression })}/></label>
}
