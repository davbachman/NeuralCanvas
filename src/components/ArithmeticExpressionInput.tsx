import { useState } from 'react'
import { arithmeticVariableNames, canonicalArithmeticExpression, displayArithmeticExpression, parseArithmetic } from '../domain/arithmetic'
import type { GraphNode } from '../domain/types'

export function ArithmeticExpressionInput({ node, inputLabels, onChange }: { node: GraphNode; inputLabels: string[]; onChange: (expression: string) => void }) {
  const names = arithmeticVariableNames(inputLabels)
  const saved = node.params.expression ?? 'x1 * x2'
  const displayed = displayArithmeticExpression(saved, names)
  const source = JSON.stringify([saved, names])
  const [draft, setDraft] = useState({ source, text: displayed, valid: true })
  const text = draft.source === source ? draft.text : displayed
  const valid = draft.source !== source || draft.valid
  const commit = () => {
    if (!valid) return
    const expression = canonicalArithmeticExpression(text, names)
    if (expression !== saved) onChange(expression)
  }
  return <input aria-label="Arithmetic expression" aria-invalid={!valid || undefined} title={valid ? displayed : 'Enter a valid arithmetic expression.'}
    className={`nodrag nowheel ${valid ? '' : 'is-invalid'}`} value={text} spellCheck={false}
    onChange={event => {
      const value = event.target.value
      let valid = true
      try { parseArithmetic(canonicalArithmeticExpression(value, names)) } catch { valid = false }
      setDraft({ source, text: value, valid })
    }} onBlur={commit} onKeyDown={event => { if (event.key === 'Enter') { event.currentTarget.blur(); event.stopPropagation() } }}/>
}
