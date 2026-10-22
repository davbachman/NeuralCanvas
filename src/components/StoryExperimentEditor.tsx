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
  const wholeStory = data.representation === 'story-question'
  const bodyFor = (story: StoryExperiment) => wholeStory ? story.text! : story.facts!.join('\n')
  const [body, setBody] = useState(bodyFor(applied ?? original))
  const [question, setQuestion] = useState((applied ?? original).question)
  const story: StoryExperiment = wholeStory ? {text: body, question} : {facts: body.split('\n'), question}
  let error = '', unknownWords: string[] = []
  try { unknownWords = encodeStory(story, data).unknownWords }
  catch (e) { error = e instanceof Error ? e.message : 'Invalid story.' }
  return <details open>
    <summary>Structured-story experiment</summary>
    <label className="inspector-field">{wholeStory ? 'Story' : 'Facts · one per line'}<textarea aria-label={wholeStory ? 'Experiment story' : 'Experiment facts'} rows={Math.min(8, Math.max(3, body.split('\n').length))} value={body} onChange={event => setBody(event.target.value)}/></label>
    <label className="inspector-field">Question<textarea aria-label="Experiment question" value={question} onChange={event => setQuestion(event.target.value)}/></label>
    <p>Change a question or {wholeStory ? 'story' : 'fact'}, then apply and run forward. Draft edits do not affect the model until applied.</p>
    {error && <p role="alert">{error}</p>}
    {unknownWords.length > 0 && <p role="status">Unknown words → &lt;unk&gt; (ID 0): {unknownWords.join(', ')}</p>}
    <button type="button" disabled={Boolean(error)} onClick={() => onApply(story)}>Apply experiment</button>
    <button type="button" onClick={() => { setBody(bodyFor(original)); setQuestion(original.question); onRestore() }}>Restore original</button>
    <p>Temporary inference only. Vocabulary, class labels, splits, and learned parameters stay fixed. The original answer may no longer apply; no experimental correctness is reported. Run epochs and Test inference use the original dataset.</p>
  </details>
}
