import { useEffect, useRef, useState } from 'react'
import { DATASET_LIBRARY, LIBRARY_CATEGORIES, libraryPreset, libraryPreview, loadLibraryDataset } from '../domain/datasetLibrary'
import type { FootballCoverage, FootballInput, LibraryCategory, LibrarySelection } from '../domain/datasetLibrary'
import './datasetLibrary.css'

interface Props {
  onUse: (selection: LibrarySelection) => void
  onCustom: (type: 'custom-csv' | 'custom-text') => void
  onClose: () => void
}
interface Resource {
  id: string
  selection?: LibrarySelection
  preview?: ReturnType<typeof libraryPreview>
  error?: string
}
const inputLabels: Record<FootballInput, string> = {
  counts: 'Word counts → answer class', tokens: 'One ordered token sequence → answer class',
  separate: 'Separate story and question inputs → answer class', 'answer-token': 'Story and question together → answer token',
}
const coverageLabels: Record<FootballCoverage, string> = {
  football: 'Football location only · smaller question set', mixed: 'Mixed questions in training and evaluation',
  benchmark: 'Mixed training · football-only evaluation',
}

export function DatasetLibraryDialog({ onUse, onCustom, onClose }: Props) {
  const dialog = useRef<HTMLElement>(null)
  const content = useRef<HTMLElement>(null)
  const [category, setCategory] = useState<LibraryCategory>('All datasets')
  const [search, setSearch] = useState(''), [familyId, setFamilyId] = useState<string>()
  const [presetId, setPresetId] = useState<string>(), [resource, setResource] = useState<Resource>({ id: '' })
  const [retry, setRetry] = useState(0)
  const family = DATASET_LIBRARY.find(item => item.id === familyId)
  const preset = presetId ? libraryPreset(presetId) : undefined
  const ready = resource.id === presetId ? resource : undefined
  useEffect(() => { if (content.current) content.current.scrollTop = 0 }, [familyId, category])
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    dialog.current?.querySelector<HTMLInputElement>('input')?.focus()
    return () => { if (previous?.isConnected) previous.focus() }
  }, [])
  useEffect(() => {
    if (!presetId) return
    const controller = new AbortController()
    let active = true
    void loadLibraryDataset(presetId, controller.signal).then(selection => {
      const preview = libraryPreview(selection)
      if (active) setResource({ id: presetId, selection, preview })
    }).catch((error: unknown) => {
      if (active) setResource({ id: presetId, error: error instanceof Error ? error.message : 'Could not load this dataset.' })
    })
    return () => { active = false; controller.abort() }
  }, [presetId, retry])
  const chooseFamily = (id: string) => {
    setFamilyId(id)
    setPresetId(DATASET_LIBRARY.find(item => item.id === id)!.presets[0].id)
  }
  const families = DATASET_LIBRARY.filter(item => (category === 'All datasets' || item.category === category)
    && `${item.label} ${item.description} ${item.presets.map(p => p.label).join(' ')}`.toLowerCase().includes(search.toLowerCase()))
  const football = preset?.football
  const preview = ready?.preview
  return <div className="dataset-library-backdrop">
    <section ref={dialog} role="dialog" aria-modal="true" aria-labelledby="dataset-library-title" className="dataset-library-dialog" onKeyDown={event => {
      event.stopPropagation()
      if (event.key === 'Escape') { event.preventDefault(); onClose() }
      if (event.key !== 'Tab') return
      const controls = [...event.currentTarget.querySelectorAll<HTMLElement>('button:not([disabled]), input, select, a[href]')]
      const first = controls[0], last = controls.at(-1)
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
    }}>
      <header className="dataset-library-header">
        <div><h2 id="dataset-library-title">Dataset library</h2><p>Choose a dataset family, explore its options, then add it to your Dataset block.</p></div>
        <button type="button" onClick={onClose} aria-label="Close dataset library">Close</button>
      </header>
      <div className="dataset-library-body">
        <nav className="dataset-library-categories" aria-label="Dataset categories">
          {LIBRARY_CATEGORIES.map(name => <button type="button" key={name} aria-pressed={category === name} onClick={() => { setCategory(name); setFamilyId(undefined); setPresetId(undefined) }}>{name}</button>)}
        </nav>
        <main ref={content} className="dataset-library-content">
          {category === 'Custom data' ? <section className="dataset-library-custom">
            <h3>Use a custom dataset</h3>
            <p>Import a file from your computer. Prepared JSON keeps its vocabulary, input lengths, targets, and splits automatically.</p>
            <button type="button" onClick={() => onCustom('custom-text')}>Import prepared JSON or text…</button>
            <p>The text importer accepts prepared dataset JSON, review or question CSV, and plain-text corpora.</p>
            <button type="button" onClick={() => onCustom('custom-csv')}>Import numeric CSV…</button>
            <p>For numeric features with regression or classification targets.</p>
          </section> : !family ? <>
            <label className="dataset-library-search">Find a dataset<input type="search" aria-label="Find a dataset" placeholder="Search by name or learning task" value={search} onChange={event => setSearch(event.target.value)} /></label>
            <div className="dataset-library-cards">
              {families.map(item => <button type="button" className="dataset-library-card" key={item.id} onClick={() => chooseFamily(item.id)}>
                <span className="dataset-library-category">{item.category}</span><strong>{item.label}</strong><span>{item.description}</span>
                <small>{item.presets.length === 1 ? 'Explore dataset' : `${item.presets.length} configurations`}</small>
              </button>)}
            </div>
            {!families.length && <p role="status">No matching datasets. Try another search or category.</p>}
          </> : <section className="dataset-library-detail" aria-label={`${family.label} options`}>
            <button type="button" className="dataset-library-back" onClick={() => { setFamilyId(undefined); setPresetId(undefined) }}>← Back to datasets</button>
            <div><span className="dataset-library-category">{family.category}</span><h3>{family.label}</h3><p>{family.description}</p></div>
            {family.id === 'football' ? <>
              <fieldset className="dataset-library-presets"><legend>Start with a preset</legend>
                {['narrative-last', 'football-counts', 'football-mixed', 'football-benchmark', 'football-qa'].map(id => <button type="button" key={id} aria-pressed={presetId === id || (id === 'narrative-last' && presetId === 'narrative-all') || (id === 'football-counts' && presetId === 'football-tokens')} onClick={() => setPresetId(id)}>{libraryPreset(id).label}</button>)}
              </fieldset>
              <div className="dataset-library-options">
                <label>Task<select aria-label="Football task" value={football!.task} onChange={event => setPresetId(event.target.value === 'narrative' ? 'narrative-last' : 'football-mixed')}><option value="narrative">Continue stories · no questions</option><option value="questions">Answer questions</option></select></label>
                {football?.task === 'narrative' ? <label>Training targets<select aria-label="Narrative training targets" value={football.targets} onChange={event => setPresetId(event.target.value === 'last' ? 'narrative-last' : 'narrative-all')}><option value="last">Final position only</option><option value="all">Every input position</option></select></label> : <>
                  <label>Question selection<select aria-label="Question selection" value={football!.coverage} onChange={event => setPresetId({ football: 'football-counts', mixed: 'football-mixed', benchmark: 'football-benchmark' }[event.target.value as FootballCoverage])}>{Object.entries(coverageLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
                  <label>Model inputs and prediction<select aria-label="Football model inputs" value={football!.input} onChange={event => setPresetId(family.presets.find(item => item.football?.coverage === football!.coverage && item.football?.input === event.target.value)!.id)}>{family.presets.filter(item => item.football?.task === 'questions' && item.football.coverage === football!.coverage).map(item => <option key={item.id} value={item.football!.input}>{inputLabels[item.football!.input!]}</option>)}</select></label>
                </>}
              </div>
              <p className="dataset-library-note">Options use fixed vocabularies and splits. Your graph keeps its blocks and learned parameters; check their shapes for the chosen data.</p>
            </> : family.presets.length > 1 && <label className="dataset-library-variant">Configuration<select aria-label="Dataset configuration preset" value={presetId} onChange={event => setPresetId(event.target.value)}>{family.presets.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>}
            <p className="dataset-library-description">{preset?.description}</p>
            {!ready && <p role="status">Loading dataset preview…</p>}
            {ready?.error && <div role="alert"><p>{ready.error}</p><button type="button" onClick={() => setRetry(value => value + 1)}>Try again</button></div>}
            {preview && <>
              <dl className="dataset-library-stats"><div><dt>Training examples</dt><dd>{preview.train.toLocaleString()}</dd></div><div><dt>Held-out examples</dt><dd>{preview.test.toLocaleString()}</dd></div>{preview.vocabulary && <div><dt>Vocabulary entries</dt><dd>{preview.vocabulary.toLocaleString()}</dd></div>}{preview.classes && <div><dt>Answer classes</dt><dd>{preview.classes.length}</dd></div>}</dl>
              <div className="dataset-library-preview"><section><h4>Example input</h4>{preview.input ? <p className="dataset-library-example">{preview.input}</p> : <dl>{preview.numeric?.map(row => <div key={row.label}><dt>{row.label}</dt><dd>{row.values}</dd></div>)}</dl>}{preview.question && <p><strong>Question:</strong> {preview.question}</p>}<p className="dataset-library-target"><strong>Target:</strong> {preview.target}</p></section>
                <section><h4>Outputs per example</h4><table><thead><tr><th>Output</th><th>Shape</th></tr></thead><tbody>{preview.outputs.map(output => <tr key={output.label}><td>{output.label}</td><td>{output.shape.length ? `[${output.shape.join(', ')}]` : 'scalar'}</td></tr>)}</tbody></table></section></div>
            </>}
            <section className="dataset-library-source"><h4>About this data</h4><p>{family.preparation}</p><p>{family.sourceUrl ? <a href={family.sourceUrl} target="_blank" rel="noreferrer">{family.source} ↗</a> : family.source}</p></section>
          </section>}
        </main>
      </div>
      <footer className="dataset-library-footer"><button type="button" onClick={() => { setCategory('Custom data'); setFamilyId(undefined); setPresetId(undefined) }}>Use a custom dataset…</button><button type="button" className="dataset-library-use" disabled={!family || !ready?.selection} onClick={() => { if (ready?.selection) onUse(ready.selection) }}>Use this dataset</button></footer>
    </section>
  </div>
}
