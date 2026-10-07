import { LOSS_OPTIONS, TENSOR_TRANSFORM_OPTIONS } from './engine'
import type { ActivationKind, NodeParams, NodeType } from './types'

export const blockPalette: Array<{ type: NodeType; label: string }> = [
  { type: 'dataset', label: 'Dataset' },
  { type: 'input', label: 'Input' },
  { type: 'weight', label: 'Param' },
  { type: 'arithmetic', label: 'Arithmetic' },
  { type: 'matmul', label: 'Matrix product' },
  { type: 'activation', label: 'Activation' },
  { type: 'standardize', label: 'Standardize features' },
  { type: 'dropout', label: 'Dropout' },
  { type: 'target', label: 'Target' },
  { type: 'loss', label: 'Loss' },
  { type: 'embedding', label: 'Embedding lookup' },
  { type: 'one-hot', label: 'One-hot' },
  { type: 'tensor-transform', label: 'Tensor transform' },
  { type: 'concat', label: 'Concatenate' },
  { type: 'softmax', label: 'Softmax' },
  { type: 'causal-mask', label: 'Causal mask' },
  { type: 'layer-norm', label: 'Layer norm' },
  { type: 'conv2d', label: 'Convolution' },
  { type: 'avgpool2d', label: 'Average pooling' },
]

export const blockCategories: Array<{ id: string; label: string; types: NodeType[] }> = [
  { id: 'core', label: 'Core model', types: ['dataset', 'input', 'weight', 'arithmetic', 'matmul', 'target', 'loss'] },
  { id: 'features', label: 'Features and tensors', types: ['standardize', 'tensor-transform', 'concat', 'one-hot'] },
  { id: 'neural', label: 'Neural networks', types: ['activation', 'softmax', 'dropout', 'layer-norm'] },
  { id: 'sequences', label: 'Sequences and attention', types: ['embedding', 'causal-mask'] },
  { id: 'images', label: 'Images', types: ['conv2d', 'avgpool2d'] },
]

const arithmeticShortcuts: Record<string, string> = {
  '+': 'x1 + x2',
  '-': 'x1 - x2',
  '−': 'x1 - x2',
  '*': 'x1 * x2',
  '×': 'x1 * x2',
  '·': 'x1 * x2',
  '/': 'x1 / x2',
  '÷': 'x1 / x2',
  '^': 'x1 ^ 2',
  '**': 'x1 ^ 2',
}

const activationFunctions: Array<{ kind: ActivationKind; label: string }> = [
  { kind: 'sigmoid', label: 'Sigmoid' },
  { kind: 'relu', label: 'ReLU' },
  { kind: 'tanh', label: 'Tanh' },
  { kind: 'identity', label: 'Identity' },
]

const lossAliases: Record<string, string[]> = {
  mse: ['mse'],
  mae: ['mae'],
  'binary-cross-entropy-with-logits': ['bce', 'bce with logits', 'binary cross entropy with logits'],
  'cross-entropy': ['ce', 'cce', 'categorical cross entropy', 'multiclass cross entropy'],
}
const normalizeSearch = (value: string) => value.trim().toLowerCase().replace(/[-_]+/g, ' ').replace(/\s+/g, ' ')

export function blockSuggestions(query: string): Array<{ type: NodeType; label: string; params?: NodeParams }> {
  const search = query.trim().toLowerCase()
  const expression = Object.hasOwn(arithmeticShortcuts, search) ? arithmeticShortcuts[search] : undefined
  if (expression) return [{ type: 'arithmetic', label: `Arithmetic · ${expression}`, params: { expression } }]
  const transforms = search ? TENSOR_TRANSFORM_OPTIONS
    .filter(option => option.label.toLowerCase().includes(search))
    .map(option => ({ type: 'tensor-transform' as const, label: `Tensor transform · ${option.label}`, params: { transform: option.kind } })) : []
  const activations = search ? activationFunctions
    .filter(option => option.label.toLowerCase().includes(search))
    .map(option => ({ type: 'activation' as const, label: `Activation · ${option.label}`, params: { activation: option.kind } })) : []
  const lossSearch = normalizeSearch(search)
  const losses = search ? LOSS_OPTIONS
    .filter(option => option.kind !== 'binary-cross-entropy')
    .map(option => ({ option, names: [option.label, option.label.replace(/ \(.*\)$/, ''), option.kind, ...(lossAliases[option.kind] ?? [])].map(normalizeSearch) }))
    .filter(({ names }) => names.some(name => name.includes(lossSearch)))
    .sort((a, b) => Number(b.names.includes(lossSearch)) - Number(a.names.includes(lossSearch)))
    .map(({ option }) => ({ type: 'loss' as const, label: `Loss · ${option.label}`, params: { loss: option.kind } })) : []
  const blocks = blockPalette.filter(item => `${item.label} ${item.type}`.toLowerCase().includes(search))
  return [...activations, ...transforms, ...losses, ...blocks.filter(item =>
    (!transforms.length || item.type !== 'tensor-transform') && (!activations.length || item.type !== 'activation') && (!losses.length || item.type !== 'loss'))]
}
