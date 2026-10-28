import { expect, it } from 'vitest'
import { formatInputValues, parseInputValues } from './inputValues'
import { tensorValue } from './tensor'

it('infers scalar, list, and rectangular tensor shapes from manual entries', () => {
  expect(parseInputValues('-2.5e-3')).toEqual(tensorValue([], [-.0025]))
  expect(parseInputValues('1, 2\n3')).toEqual(tensorValue([3], [1, 2, 3]))
  expect(parseInputValues('[[1, 2], [3, 4]]')).toEqual(tensorValue([2, 2], [1, 2, 3, 4]))
})

it('preserves singleton axes and higher-rank shapes when editing generated values', () => {
  for (const shape of [[], [1], [3], [1, 3], [3, 1], [1, 2, 1, 2]]) {
    const value = tensorValue(shape, Array.from({ length: shape.reduce((n, d) => n * d, 1) }, (_, i) => i - .25))
    expect(parseInputValues(formatInputValues(value))).toEqual(value)
  }
})

it('rejects missing, non-finite, non-numeric, and ragged entries', () => {
  for (const text of ['', '1, Infinity', 'NaN', '[true]', '[null]', '[]', '[[1, 2], [3]]', '[1, [2]]', '[[1], [2, 3]]']) {
    expect(() => parseInputValues(text)).toThrow(/finite numbers/)
  }
})
