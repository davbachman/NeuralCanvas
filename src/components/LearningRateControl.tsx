import { useRef, useState } from 'react'
import './learningRateControl.css'

export function LearningRateControl({ value, tensorTraining, disabled, onChange }: {
  value: number
  tensorTraining: boolean
  disabled: boolean
  onChange: (value: number) => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [error, setError] = useState('')
  const canceled = useRef(false)
  const start = () => {
    if (disabled) return
    setDraft(String(value))
    setError('')
    canceled.current = false
    setEditing(true)
  }
  const finish = () => {
    if (canceled.current) { setEditing(false); setError(''); return }
    const rate = Number(draft)
    if (!draft.trim() || !Number.isFinite(rate) || rate <= 0) {
      setError('Enter a positive, finite learning rate.')
      return
    }
    if (rate !== value) onChange(rate)
    setError('')
    setEditing(false)
  }
  return <div className="run-field">
    <div className="learning-rate-heading">Learning rate · η = {editing
      ? <input className="learning-rate-input" aria-label="Learning rate value" aria-invalid={Boolean(error)}
          type="number" step="any" value={draft} disabled={disabled} autoFocus onFocus={event => event.currentTarget.select()}
          onChange={event => { setDraft(event.target.value); setError('') }} onBlur={finish}
          onKeyDown={event => {
            if (event.key === 'Enter') { event.preventDefault(); event.currentTarget.blur() }
            if (event.key === 'Escape') { event.preventDefault(); canceled.current = true; event.currentTarget.blur() }
          }} />
      : <button className="learning-rate-value" type="button" aria-label={`Edit learning rate: ${value}`} title="Double-click to edit learning rate"
          disabled={disabled} onDoubleClick={start} onKeyDown={event => {
            if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); start() }
          }}>{value}</button>}
    </div>
    {error && <span role="alert">{error}</span>}
    <input aria-label="Learning rate" type="range" min={tensorTraining ? '0.0001' : '0.001'} max="0.5"
      step={tensorTraining ? '0.0001' : '0.001'} value={value} onChange={event => onChange(Number(event.target.value))} disabled={disabled}/>
  </div>
}
