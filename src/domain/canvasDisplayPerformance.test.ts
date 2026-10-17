import { describe, expect, it, vi } from 'vitest'
import { formatCompactTensor, formatFullTensor } from './tensor'
import { edgeSignalIntensity } from './edgeSignal'
import { canvasPerformanceModel } from '../test/canvasPerformanceModel'
import { parameterValues } from './engine'

describe('tensor display work', () => {
  it('reads only the first entry for a compact preview, without copying or validating the buffer', () => {
    const data = [1, 2, 3]
    Object.defineProperty(data, 1, { get() { throw Error('Preview scanned a hidden value') } })
    expect(formatCompactTensor({ shape: [3], data })).toBe('[1.000,...]')
  })

  it('reuses full descriptions and wire intensity until the tensor identity changes', () => {
    const data = [1, 2, 3]
    const read = vi.fn(() => 2)
    Object.defineProperty(data, 1, { get: read })
    const tensor = { shape: [3], data }
    const description = formatFullTensor(tensor)
    const intensity = edgeSignalIntensity(tensor)
    read.mockClear()
    for (let frame = 0; frame < 20; frame++) {
      expect(formatFullTensor(tensor)).toBe(description)
      expect(edgeSignalIntensity(tensor)).toBe(intensity)
    }
    expect(read).not.toHaveBeenCalled()
    const changed = { shape: [3], data: [4, 5, 6] }
    expect(formatFullTensor(changed)).not.toBe(description)
    expect(edgeSignalIntensity(changed)).toBeGreaterThan(intensity)
    expect(formatFullTensor(tensor, 1)).toBe('[3] [1.0, 2.0, 3.0]')
  })

  it('benchmarks equivalent populated One-hot and Embedding models', () => {
    const oneHot = canvasPerformanceModel(), embedding = canvasPerformanceModel(true)
    expect(oneHot.nodes.find(n => n.id === 'encoding')!.value!.shape).toEqual([16, 27])
    expect(oneHot.nodes.find(n => n.id === 'flatten')!.value!.shape).toEqual([1, 128])
    for (const id of ['vectors', 'logits', 'probabilities', 'loss', 'table', 'w1', 'w2']) {
      const a = oneHot.nodes.find(n => n.id === id)!, b = embedding.nodes.find(n => n.id === id)!
      expect(a.value).toEqual(b.value)
      expect(a.grad!.data).toEqual(b.grad!.data)
      expect(a.cache).toBeDefined()
    }
    expect(parameterValues(oneHot)).toEqual(parameterValues(embedding))
  })
})
