import {
  ArrowDownToLine, ArrowRightLeft, Blend, Braces, Calculator, ChartNoAxesColumn,
  CirclePlus, Combine, Database, Equal, Grid2X2, Grid3X3, Layers, ListFilter,
  ListOrdered, Maximize, Minimize2, Scale, Scissors, Shapes, Sigma, SlidersHorizontal,
  Spline, Table2, Target, Triangle, X, type LucideIcon,
} from 'lucide-react'
import type { NodeType } from '../domain/types'

const icons: Record<NodeType, LucideIcon> = {
  dataset: Database,
  input: ArrowDownToLine,
  weight: SlidersHorizontal,
  bias: Equal,
  multiply: X,
  matmul: Grid2X2,
  add: CirclePlus,
  arithmetic: Calculator,
  activation: Spline,
  target: Target,
  loss: Scale,
  embedding: Table2,
  'one-hot': ListOrdered,
  transpose: ArrowRightLeft,
  slice: Scissors,
  concat: Combine,
  softmax: ChartNoAxesColumn,
  'causal-mask': Triangle,
  'layer-norm': Layers,
  standardize: Blend,
  dropout: ListFilter,
  reshape: Maximize,
  'tensor-transform': Shapes,
  mean: Sigma,
  'cross-entropy': Braces,
  conv2d: Grid3X3,
  avgpool2d: Minimize2,
}

export function NodeIcon({ type }: { type: NodeType }) {
  const Icon = icons[type]
  return <span className="node-icon" aria-hidden="true"><Icon size={16}/></span>
}
