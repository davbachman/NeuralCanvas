import { useState } from 'react'
import { encodeStory } from '../domain/textData'
import type { StoryExperiment, TextDatasetData } from '../domain/types'

export function StoryExperimentEditor({ original, applied, data, onApply, onRestore }: {
  original: StoryExperiment
  applied?: StoryExperiment
  data: TextDatasetData
  onApply: (story: StoryExperiment) => void
  onRestore: () => void
}) {
  const [facts, setFacts] = useState((applied ?? original).facts.join('\n'))
  const [question, setQuestion] = useState((applied ?? original).question)
  const story = {facts: facts.split('\n'), question}
  let error = '', unknownWords: string[] = []
  try { unknownWords = encodeStory(story, data).unknownWords }
  catch (e) { error = e instanceof Error ? e.message : 'Invalid story.' }
  return <details open>
    <summary>Structured-story experiment</summary>
    <label className="inspector-field">Facts · one per line<textarea aria-label="Experiment facts" rows={Math.min(8, Math.max(3, story.facts.length))} value={facts} onChange={event => setFacts(event.target.value)}/></label>
    <label className="inspector-field">Question<textarea aria-label="Experiment question" value={question} onChange={event => setQuestion(event.target.value)}/></label>
    <p>Change a question or fact, then apply and run forward. Draft edits do not affect the model until applied.</p>
    {error && <p role="alert">{error}</p>}
    {unknownWords.length > 0 && <p role="status">Unknown words → &lt;unk&gt; (ID 0): {unknownWords.join(', ')}</p>}
    <button type="button" disabled={Boolean(error)} onClick={() => onApply(story)}>Apply experiment</button>
    <button type="button" onClick={() => { setFacts(original.facts.join('\n')); setQuestion(original.question); onRestore() }}>Restore original</button>
    <p>Temporary inference only. Vocabulary, class labels, splits, and learned parameters stay fixed. The original answer may no longer apply; no experimental correctness is reported. Run epochs and Test inference use the original dataset.</p>
  </details>
}
