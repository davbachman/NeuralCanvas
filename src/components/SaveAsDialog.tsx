import { useState } from 'react'
import { projectStateFilename } from '../domain/session'

export function SaveAsDialog({ onSave, onCancel }: { onSave: (filename: string) => void; onCancel: () => void }) {
  const [name, setName] = useState(`neural-canvas-state-${new Date().toISOString().slice(0, 10)}`)
  const [error, setError] = useState('')
  const valid = name.trim().replace(/\.json$/i, '').length > 0 && !/[\\/:*?"<>|]/.test(name)
  return <div className="csv-picker-backdrop">
    <form role="dialog" aria-modal="true" aria-labelledby="save-as-title" aria-describedby="save-as-description" className="csv-picker-dialog"
      onSubmit={event => {
        event.preventDefault()
        if (!valid) return
        try { onSave(projectStateFilename(name)) }
        catch { setError('Could not save the file. Please try again.') }
      }}
      onKeyDown={event => {
        event.stopPropagation()
        if (event.key === 'Escape') { event.preventDefault(); onCancel() }
        if (event.key !== 'Tab') return
        const controls = [...event.currentTarget.querySelectorAll<HTMLElement>('input, button:not([disabled])')]
        const first = controls[0], last = controls.at(-1)
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
      }}>
      <h2 id="save-as-title">Save as</h2>
      <p id="save-as-description">Choose a filename. The .json extension is added automatically.</p>
      <label className="inspector-field">File name
        <input autoFocus value={name} onFocus={event => event.currentTarget.select()} onChange={event => { setName(event.target.value); setError('') }} />
      </label>
      {!valid && <p className="csv-picker-error">Enter a file name without \ / : * ? &quot; &lt; &gt; | characters.</p>}
      {error && <p role="alert" className="csv-picker-error">{error}</p>}
      <button type="submit" disabled={!valid}>Save</button>
      <button type="button" onClick={onCancel}>Cancel</button>
    </form>
  </div>
}
