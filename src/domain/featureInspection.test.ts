import { describe, expect, it, vi } from 'vitest'
import { scanFeatureContexts } from './featureInspection'
import type { InspectionValue } from './inferenceInspection'

describe('feature context inspection', () => {
  it('ranks valid token contexts and computes exact-zero sparsity across examples', async () => {
    const read = vi.fn(async (_id: string, _slot: number, row: number): Promise<InspectionValue> => ({
      status: 'available', kind: 'example', simple: false, axes: ['token', 'feature'],
      labels: [['0: jack', '1: moved'], undefined],
      value: { shape: [3, 3], data: row === 2 ? [1, 0, 0, 4, 2, 0, 100, 100, 100] : [0, 3, 0, 2, 0, 0, 200, 200, 200], excluded: [false, false, false, false, false, false, true, true, true] },
    }))
    const result = await scanFeatureContexts({ indices: [2, 5], read }, 'h', 0, 0, 2)
    expect(result).toMatchObject({ examples: 2, tokens: 4, activePerToken: 1.25, inactiveFeatures: 1 })
    expect(result.contexts.map(context => context.activation)).toEqual([4, 2, 1])
    expect(result.contexts[0]).toMatchObject({ example: 2, token: 1, context: '0: jack · [1: moved]' })
  })

  it('limits examples and never continues after cancellation', async () => {
    const controller = new AbortController()
    const read = vi.fn(async (): Promise<InspectionValue> => {
      controller.abort()
      return { status: 'available', kind: 'example', simple: false, axes: ['token', 'feature'], labels: [], value: { shape: [1, 1], data: [1] } }
    })
    await expect(scanFeatureContexts({ indices: [1, 2, 3], read }, 'h', 0, 0, 2, controller.signal)).rejects.toThrow()
    expect(read).toHaveBeenCalledTimes(1)
  })

  it('excludes labelled fixed-length padding even when the output has no mask', async () => {
    const read = async (): Promise<InspectionValue> => ({ status: 'available', kind: 'example', simple: false,
      axes: [], labels: [['0: "jack"', '1: "<pad>"']], value: { shape: [2, 2], data: [2, 0, 1000, 1000] } })
    const result = await scanFeatureContexts({ indices: [0], read }, 'h', 0, 0, 1)
    expect(result).toMatchObject({ tokens: 1, activePerToken: 1, inactiveFeatures: 1 })
    expect(result.contexts).toHaveLength(1)
    expect(result.contexts[0].activation).toBe(2)
  })
})
