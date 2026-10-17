import type { TensorValue } from './types'

const intensities = new WeakMap<TensorValue, number>()

/** Fixed logarithmic scale makes intensity comparable across wires and runs.
 * Causal exclusions are display sentinels, not numerical signal. */
export function edgeSignalIntensity(value: TensorValue | undefined): number {
  if (!value) return 0.2
  const cached = intensities.get(value)
  if (cached !== undefined) return cached
  const count = value.excluded ? value.excluded.reduce((n, excluded) => n + Number(!excluded), 0) : value.data.length
  let magnitude = 0
  for (let i = 0; i < value.data.length; i++) {
    if (!value.excluded?.[i]) magnitude += Math.abs(value.data[i]) / Math.max(1, count)
  }
  const intensity = 0.24 + 0.76 * Math.min(1, Math.log1p(magnitude) / Math.log(11))
  intensities.set(value, intensity)
  return intensity
}
