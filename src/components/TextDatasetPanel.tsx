import { useState } from 'react'
import { datasetExampleIndex, datasetExamplesForNode, datasetOutputValueForSlot } from '../domain/datasets'
import { activeStoryExperiment, decodeTokens, encodeText, encodeStory, tokenize } from '../domain/textData'
import { StoryExperimentEditor } from './StoryExperimentEditor'
import { predictionNode } from '../domain/datasetTraining'
import type { GraphModel, GraphNode, NodeParams } from '../domain/types'

export function TextDatasetPanel({ node, graph, onParams, onReplace }: { node: GraphNode; graph?: GraphModel; onParams: (id: string, params: NodeParams) => void; onReplace: () => void }) {
  const [search, setSearch] = useState(''), [draft, setDraft] = useState(''), [error, setError] = useState('')
  const data = node.params.textData!
  const example = datasetExamplesForNode(node)[datasetExampleIndex(node)]
  const document = data.documents[example.documentIndex ?? 0]
  const wholeStory = data.representation === 'story-question'
  const structured = data.representation === 'facts' || wholeStory
  const story = activeStoryExperiment(node)
  const originalStory = wholeStory ? {text: document.text, question: document.question!} : {facts: document.facts ?? [], question: document.question ?? ''}
  const visibleStory = story ?? originalStory
  const prediction = graph && predictionNode(graph)
  const scores = story ? prediction?.value?.data : undefined
  const probabilities = scores && (prediction?.type === 'softmax' ? scores : (() => {
    const max = Math.max(...scores)
    const exp = scores.map(value => Math.exp(value - max))
    const sum = exp.reduce((a, b) => a + b, 0)
    return exp.map(value => value / sum)
  })())
  const ids = data.representation === 'tokens' || wholeStory ? datasetOutputValueForSlot(node, 0).data : encodeText(document.text, data).slice(0, data.maxLength)
  const tokens = data.vocabulary.map((token, id) => ({token, id})).filter(({token, id}) => token.includes(search) || String(id) === search).slice(0, 40)
  const apply = (reverse = false) => {
    let next = encodeText(draft || document.text, data).slice(0, data.maxLength)
    if (reverse) next = ids.filter(id=>!data.fixedLength || id!==1).reverse()
    if (!next.length) { setError('Enter at least one token.'); return }
    const counts = Array(data.vocabulary.length).fill(0) as number[]
    next.forEach(id => counts[id]++)
    if(data.fixedLength && data.representation==='tokens')next=[...next,...Array(data.maxLength-next.length).fill(1)]
    onParams(node.id, {datasetValues: [data.representation === 'counts' ? {shape: [1, counts.length], data: counts} : {shape: [next.length], data: next}, {shape: [next.length], data: next.map((_, i) => i)}, example.target]})
    setError('')
  }
  return <section className="text-dataset-panel" aria-label="Text data inspection">
    <button type="button" onClick={onReplace}>Replace text dataset</button>
    {data.task !== 'language' && !structured && <label className="inspector-field">Representation<select aria-label="Review representation" value={data.representation} onChange={event => onParams(node.id, {textData: {...data, representation: event.target.value as 'tokens' | 'counts'}, datasetValues: undefined})}><option value="counts">Word counts</option><option value="tokens">Token IDs</option></select></label>}
    <p>{wholeStory ? `Story: ${data.maxLength} padded tokens. Question: ${data.questionLength} padded tokens. Separate word masks identify padding. Overlong inputs are rejected without truncation.` : data.representation === 'facts'
      ? `Up to ${data.maxFacts} facts, ${data.factWords} tokens per fact or question. Fact order is preserved. Separate masks identify padded facts and words.`
      : `Context limit: ${data.maxLength}. Vocabulary and document splits stay fixed when switching representation. ${data.fixedLength ? 'Fixed inputs use a distinct padding token.' : 'Tensor training masks padding within each batch.'}`}</p>
    <details><summary>Original text · dataset row {datasetExampleIndex(node)}</summary><p>Source document {example.documentIndex ?? 0} · token offset {example.offset ?? 0}. Dataset rows and source documents are numbered from 0; one document can supply multiple context windows.</p><p className="text-original">{document.text}</p><p>{tokenize(document.text, data.tokenizer, data.lowercase).length} original tokens; {ids.length} displayed input tokens.</p></details>
    {structured ? <details open><summary>{story ? 'Applied experiment · ' : ''}{wholeStory ? 'Story and question tokens' : 'Fact and question tokens'}</summary>
      {wholeStory ? <div><strong>Story</strong><p>{visibleStory.text}</p><div role="list" aria-label="Story token IDs" className="text-token-list">{datasetOutputValueForSlot(node, 0).data.map((id, position) => <span role="listitem" key={position}>{position}: {JSON.stringify(data.vocabulary[id])} <small>#{id}</small></span>)}</div></div>
        : visibleStory.facts!.map((fact, index) => <div key={index}><strong>Fact {index + 1}</strong><p>{fact}</p><div className="text-token-list">{datasetOutputValueForSlot(node, 0).data.slice(index * data.factWords!, (index + 1) * data.factWords!).map((id, position) => <span key={position}>{position}: {JSON.stringify(data.vocabulary[id])} <small>#{id}</small></span>)}</div></div>)}
      <strong>Question</strong><p>{visibleStory.question}</p><div role="list" aria-label="Question token IDs" className="text-token-list">{datasetOutputValueForSlot(node, 1).data.map((id, position) => <span role="listitem" key={position}>{position}: {JSON.stringify(data.vocabulary[id])} <small>#{id}</small></span>)}</div>
    </details> : <details open><summary>{data.representation === 'counts' ? 'Original review tokens and IDs' : 'Input tokens and IDs'}</summary><div className="text-token-list">{ids.map((id, position) => <span key={position} title={'Position ' + position + ', token ID ' + id}>{position}: {JSON.stringify(data.vocabulary[id])} <small>#{id}</small></span>)}</div></details>}
    {data.task==='classification' && <p>{story ? 'Original answer (reference only)' : 'Answer'}: <strong>{document.label}</strong>{document.question && !story && <> · Question: {document.question}</>}</p>}
    {structured && <StoryExperimentEditor key={`${node.id}:${example.documentIndex}:${document.text}:${JSON.stringify(story)}`} original={originalStory} applied={story} data={data}
      onApply={next => { encodeStory(next, data); onParams(node.id, {storyExperiment: next, datasetValues: undefined}) }}
      onRestore={() => onParams(node.id, {storyExperiment: undefined, datasetValues: undefined})}/>}
    {story && <section aria-label="Experimental result">
      <p>Experimental answer: unknown. Any loss shown uses the original answer as a reference only.</p>
      {probabilities && probabilities.length === data.classLabels?.length ? <>
        <p>Experimental prediction: <strong>{data.classLabels[probabilities.indexOf(Math.max(...probabilities))]}</strong></p>
        <ul aria-label="Experimental class probabilities">{probabilities.map((value, i) => <li key={i}>{data.classLabels![i]}: {(100 * value).toFixed(2)}%</li>)}</ul>
      </> : <p>Run forward to inspect the experimental prediction.</p>}
      <p>Select a Softmax or attention block and open Data to inspect its full output.</p>
      {graph?.nodes.filter(n => n.type === 'softmax' && n.id !== prediction?.id && n.value).map(n => <p key={n.id}>{n.label} · Softmax: <output aria-label={`${n.label} experiment weights`}>{n.value!.data.map(value => value.toFixed(6)).join(', ')}</output></p>)}
    </section>}
    {data.task === 'language' && <p>Targets: <code>{decodeTokens(example.target.data, data)}</code></p>}
    {data.task !== 'language' && !structured && <details><summary>Word-order experiment</summary><textarea aria-label="Review experiment" placeholder="Enter text or reverse the current token order" value={draft} onChange={event => setDraft(event.target.value)}/><button type="button" onClick={() => apply()}>Apply text</button><button type="button" onClick={() => apply(true)}>Reverse token order</button><p>Experiment keeps the selected example’s label for comparison. Choose a dataset row to restore its original input.</p>{error && <p role="alert">{error}</p>}</details>}
    <details><summary>Vocabulary · {data.vocabulary.length} entries</summary><input aria-label="Search vocabulary" placeholder="Find a word or ID" value={search} onChange={event => setSearch(event.target.value)}/><div className="text-token-list">{tokens.map(({token, id}) => <span key={id}>{id}: {JSON.stringify(token)}</span>)}</div><p>Showing up to 40 matches. ID 0 means an unknown token. Words and punctuation tokenize separately; internal apostrophes stay within words.</p></details>
  </section>
}
