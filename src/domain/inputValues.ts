import { tensorValue } from './tensor'
import type { TensorValue } from './types'

const ENTRY_ERROR = 'Enter a number, a list, or rectangular nested arrays containing finite numbers.'

/** Infer manually entered input shapes without a separate shape control. */
export function parseInputValues(text: string): TensorValue {
  const trimmed = text.trim()
  if (!trimmed) throw new Error(ENTRY_ERROR)
  let entered: unknown
  if (trimmed.startsWith('[')) {
    try { entered = JSON.parse(trimmed) } catch { throw new Error(ENTRY_ERROR) }
  } else {
    const numbers = trimmed.split(/[\s,]+/).filter(Boolean).map(Number)
    entered = numbers.length === 1 ? numbers[0] : numbers
  }
  const data: number[] = []
  const visit = (value: unknown, depth = 0): number[] => {
    if (typeof value === 'number' && Number.isFinite(value)) {
      if (data.length >= 1_048_576) throw new Error('Use at most 1,048,576 input values.')
      data.push(value)
      return []
    }
    if (!Array.isArray(value) || !value.length || depth >= 32) throw new Error(ENTRY_ERROR)
    const shape = visit(value[0], depth + 1)
    for (let index = 1; index < value.length; index++) {
      const next = visit(value[index], depth + 1)
      if (next.length !== shape.length || next.some((size, axis) => size !== shape[axis])) throw new Error(ENTRY_ERROR)
    }
    return [value.length, ...shape]
  }
  return tensorValue(visit(entered), data)
}

/** Brackets retain matrix/tensor dimensions when returning from random mode. */
export function formatInputValues(value: TensorValue): string {
  if (value.shape.length <= 1) return value.shape.length === 1 && value.data.length === 1
    ? `[${value.data[0]}]` : value.data.join(', ')
  let index = 0
  const nest = (axis: number): unknown => axis === value.shape.length
    ? value.data[index++] : Array.from({ length: value.shape[axis] }, () => nest(axis + 1))
  return JSON.stringify(nest(0))
}
