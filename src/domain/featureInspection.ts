import type { InferenceInspection, InspectionValue } from './inferenceInspection'
import { isPaddingTokenLabel } from './tensorLabels'

export interface FeatureContext { example: number; token: number; activation: number; context: string }
export interface FeatureScan {
  contexts: FeatureContext[]; examples: number; tokens: number; activePerToken: number; inactiveFeatures: number
}

/** Bounded, cancellable inspection of the saved inference snapshot, never cached
 * scores from a submission. Exact zero is the inactivity convention. */
export async function scanFeatureContexts(
  inspection: Pick<InferenceInspection, 'indices' | 'read'>, id: string, slot: number,
  feature: number, maximumExamples: number, signal?: AbortSignal,
  onProgress?: (completed: number) => void,
): Promise<FeatureScan> {
  if (!Number.isInteger(feature) || feature < 0 || !Number.isInteger(maximumExamples) || maximumExamples < 1) throw Error('Choose a nonnegative feature index and a positive example limit.')
  let tokens = 0, active = 0, examples = 0
  let fired: boolean[] | undefined
  const contexts: FeatureContext[] = []
  for (const example of inspection.indices.slice(0, maximumExamples)) {
    signal?.throwIfAborted()
    const output: InspectionValue = await inspection.read(id, slot, example)
    signal?.throwIfAborted()
    if (output.status !== 'available') throw Error(output.message)
    const { value, labels } = output
    if (output.kind !== 'example' || value.shape.length !== 2 || feature >= value.shape[1]) throw Error('Choose a feature from a token × feature output.')
    const [rows, width] = value.shape
    if (fired && fired.length !== width) throw Error('The feature width changed across examples.')
    fired ??= Array<boolean>(width).fill(false)
    const names = labels[0]
    for (let token = 0; token < rows; token++) {
      if (names && token >= names.length || isPaddingTokenLabel(names?.[token]) || value.excluded?.[token * width + feature]) continue
      const score = value.data[token * width + feature]
      if (!Number.isFinite(score)) throw Error('Feature activations must be finite.')
      tokens++
      for (let column = 0; column < width; column++) if (!value.excluded?.[token * width + column] && value.data[token * width + column] !== 0) { active++; fired[column] = true }
      if (score > 0) contexts.push({ example, token, activation: score,
        context: names ? names.slice(Math.max(0, token - 4), Math.min(names.length, token + 5)).map((name, offset) => Math.max(0, token - 4) + offset === token ? `[${name}]` : name).join(' · ') : `Token position ${token}` })
    }
    contexts.sort((a, b) => b.activation - a.activation || a.example - b.example || a.token - b.token)
    contexts.splice(20)
    examples++
    onProgress?.(examples)
    if (examples % 8 === 0) await new Promise(resolve => setTimeout(resolve, 0))
  }
  return { contexts, examples, tokens, activePerToken: tokens ? active / tokens : 0, inactiveFeatures: fired?.filter(value => !value).length ?? 0 }
}
