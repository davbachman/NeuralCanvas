import { useState, type InputHTMLAttributes } from 'react'

type Props = Omit<InputHTMLAttributes<HTMLInputElement>, 'value' | 'defaultValue' | 'type'> & {
  value: number
  type?: 'number'
}

/** Keep unfinished edits local instead of converting an empty field to zero.
 * Complete numbers still reach the existing change handler immediately; that
 * handler remains responsible for accepting, rejecting, or clamping values. */
export function NumberInput({ value, onChange, onBlur, onKeyDown, ...props }: Props) {
  const [draft, setDraft] = useState({ value, text: String(value) })
  if (!Object.is(draft.value, value)) {
    setDraft({
      value,
      text: draft.text.trim() !== '' && Number(draft.text) === value ? draft.text : String(value),
    })
  }

  return <input {...props} type="number" value={draft.text}
    onChange={event => {
      const text = event.currentTarget.value
      setDraft({ value, text })
      if (text.trim() !== '' && Number.isFinite(Number(text))) onChange?.(event)
    }}
    onBlur={event => {
      setDraft({ value, text: String(value) })
      onBlur?.(event)
    }}
    onKeyDown={event => {
      onKeyDown?.(event)
      if (!event.defaultPrevented && (event.key === 'Enter' || event.key === 'Escape')) {
        event.preventDefault()
        setDraft({ value, text: String(value) })
        event.currentTarget.blur()
      }
    }}
  />
}
