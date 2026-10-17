import { renderHook } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import { useStableCallback } from './useStableCallback'

it('keeps canvas handler identities stable while calling the latest committed handler', () => {
  const first = vi.fn(), latest = vi.fn()
  const { result, rerender } = renderHook(useStableCallback<[string], void>, { initialProps: first })
  const handler = result.current
  rerender(latest)
  expect(result.current).toBe(handler)
  handler('new selection')
  expect(latest).toHaveBeenCalledWith('new selection')
  expect(first).not.toHaveBeenCalled()
})
