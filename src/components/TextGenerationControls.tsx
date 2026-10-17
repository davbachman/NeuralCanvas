import { useState } from 'react'
import { textGenerationStates, predictText, textSamplingDistribution, type TextGenerationState } from '../domain/textGeneration'
import './decoderControls.css'
import type { GraphModel } from '../domain/types'

export function TextGenerationControls({graph}: {graph: GraphModel}) {
  const [prompt, setPrompt] = useState('alice'), [sample, setSample] = useState(false), [temperature, setTemperature] = useState(1), [topK, setTopK] = useState(10)
  const [error, setError] = useState(''), [busy, setBusy] = useState(false), [rows, setRows] = useState<{token: string; probability: number}[]>([])
  const [continuation, setContinuation] = useState<TextGenerationState>()
  const data = graph.nodes.find(node => node.type === 'dataset' && node.params.textData?.task === 'language')?.params.textData
  const [vocabularySource, setVocabularySource] = useState(data)
  if (vocabularySource !== data) { setVocabularySource(data); setContinuation(undefined) }
  const [signature, setSignature] = useState(graph)
  if (signature !== graph) { setSignature(graph); setRows([]) }
  async function run(count: number) {
    setBusy(true); setError('')
    try {
      let state: string | TextGenerationState = continuation ?? prompt
      let i = 0
      for (const next of count ? textGenerationStates(graph, state, count, {sample, temperature, topK}) : []) {
        state = next
        setContinuation(next)
        setPrompt(next.text)
        if (i++ % 4 === 0) await new Promise(resolve => setTimeout(resolve, 0))
      }
      const result = predictText(graph, state)
      const distribution = textSamplingDistribution(result.logits, result.vocabulary, sample ? temperature : 1, sample ? topK : result.logits.length)
      setRows(result.vocabulary.map((token, i) => ({token, probability: distribution[i]})).filter(row => row.token !== '<pad>').sort((a,b) => b.probability-a.probability).slice(0, 12))
    } catch (cause) {setError(cause instanceof Error ? cause.message : 'Generation failed.')}
    finally {setBusy(false)}
  }
  return <section className="decoder-controls" aria-label="Imported text generation"><h3>Continue the text</h3>
    <p className="decoder-explanation">Uses your graph and current weights. The most recent context window is used after it fills. Generation does not train or calculate a loss. Padding (&lt;pad&gt;) is never generated; probabilities are normalized over the remaining tokens.</p>
    <label className="decoder-field-label">Prompt and generated text<textarea aria-label="Text generation prompt" value={prompt} disabled={busy} onChange={event => {setPrompt(event.target.value); setContinuation(undefined); setRows([])}} rows={5}/></label>
    <div className="decoder-sampling-controls"><label>Decoding<select aria-label="Text decoding" disabled={busy} value={sample ? 'sample' : 'greedy'} onChange={event => {setSample(event.target.value === 'sample'); setRows([])}}><option value="greedy">Most likely</option><option value="sample">Sample</option></select></label>
    {sample && <div className="decoder-sampling-fields"><label>Temperature<input aria-label="Text temperature" type="number" min="0.1" max="3" step="0.1" value={temperature} onChange={event => {setTemperature(Number(event.target.value)); setRows([])}}/></label><label>Top-k<input aria-label="Text top-k" type="number" min="1" max="4096" value={topK} onChange={event => {setTopK(Number(event.target.value)); setRows([])}}/></label></div>}</div>
    <button className="decoder-apply text-generation-action" disabled={busy} onClick={() => void run(0)}>Inspect next-token probabilities</button><button className="decoder-apply text-generation-action" disabled={busy} onClick={() => void run(1)}>Generate one token</button><button className="decoder-generate text-generation-action" disabled={busy} onClick={() => void run(40)}>{busy ? 'Generating…' : 'Generate 40 tokens'}</button>
    <div className="decoder-distribution" aria-label="Text next-token probabilities">{rows.map(row => <div className="decoder-probability-row" key={row.token}><code title={row.token}>{JSON.stringify(row.token)}</code><div className="decoder-probability-track"><div style={{width: `${row.probability * 100}%`}}/></div><output>{(row.probability * 100).toFixed(2)}%</output></div>)}</div>
    {error && <p className="decoder-error" role="alert">{error}</p>}
  </section>
}
