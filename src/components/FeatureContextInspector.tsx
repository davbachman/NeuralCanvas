import { useEffect, useRef, useState } from 'react'
import { scanFeatureContexts, type FeatureScan } from '../domain/featureInspection'
import type { InferenceInspection } from '../domain/inferenceInspection'
import { NumberInput } from './NumberInput'

export function FeatureContextInspector({ inspection, id, slot, width }: { inspection: InferenceInspection; id: string; slot: number; width: number }) {
  const [feature, setFeature] = useState(0)
  const [limit, setLimit] = useState(Math.min(256, inspection.indices.length))
  const [progress, setProgress] = useState<number>()
  const [result, setResult] = useState<FeatureScan>()
  const [error, setError] = useState<string>()
  const controller = useRef<AbortController | undefined>(undefined)
  useEffect(() => () => controller.current?.abort(), [])
  const scan = async () => {
    const current = new AbortController()
    controller.current = current
    setProgress(0); setResult(undefined); setError(undefined)
    try { setResult(await scanFeatureContexts(inspection, id, slot, feature, limit, current.signal, setProgress)) }
    catch (cause) { if (!current.signal.aborted) setError(cause instanceof Error ? cause.message : String(cause)) }
    finally { if (controller.current === current) { controller.current = undefined; setProgress(undefined) } }
  }
  return <details className="feature-context-inspector"><summary>Inspect feature contexts</summary>
    <p>Rank positive activations in this saved inference run. Feature indices start at 0. This is evidence about the selected examples, not proof of a feature's meaning.</p>
    <label className="data-signal-select">Feature index<NumberInput aria-label="Feature index" type="number" min="0" max={width - 1} value={feature} disabled={progress !== undefined} onChange={event => { setFeature(Number(event.target.value)); setResult(undefined) }}/></label>
    <label className="data-signal-select">Examples to inspect<NumberInput aria-label="Feature scan example limit" type="number" min="1" max={inspection.indices.length} value={limit} disabled={progress !== undefined} onChange={event => { setLimit(Number(event.target.value)); setResult(undefined) }}/></label>
    <button type="button" disabled={progress !== undefined || !Number.isInteger(feature) || feature < 0 || feature >= width || !Number.isInteger(limit) || limit < 1 || limit > inspection.indices.length} onClick={() => { void scan() }}>Find high activations</button>
    {progress !== undefined && <><p role="status">Inspected {progress} of {limit} examples…</p><button type="button" onClick={() => controller.current?.abort()}>Stop feature scan</button></>}
    {error && <p role="alert">{error}</p>}
    {result && <><p role="status">{result.examples} examples · {result.tokens} valid tokens · {result.activePerToken.toFixed(2)} active features/token · {result.inactiveFeatures} inactive features. Active means exactly nonzero (tolerance 0); padding is excluded.</p>
      {result.contexts.length ? <ol>{result.contexts.map(item => <li key={`${item.example}:${item.token}`}><strong>{item.activation.toPrecision(5)}</strong> · dataset row {item.example}, token {item.token}<br/>{item.context}</li>)}</ol> : <p>This feature had no positive activations in the inspected examples.</p>}</>}
  </details>
}
