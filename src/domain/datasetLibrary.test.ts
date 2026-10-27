import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, expect, it, vi } from 'vitest'
import assets from '../learning/dataset-library.json'
import { DATASET_OPTIONS } from './datasets'
import { DATASET_LIBRARY, libraryPreview, loadLibraryDataset } from './datasetLibrary'

afterEach(() => vi.unstubAllGlobals())
const read = (file: string) => readFileSync(resolve('public/datasets', `${file}`), 'utf8')
function mockAssets() {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const file = url.split('/').at(-1)!
    const text = read(file)
    return { ok: true, text: async () => text, json: async () => JSON.parse(text) }
  }))
}
it('includes every existing built-in source and every curated file exactly once', () => {
  const presets = DATASET_LIBRARY.flatMap(family => family.presets)
  expect(new Set(presets.map(p => p.id)).size).toBe(presets.length)
  expect(presets.filter(p => p.builtin).map(p => p.builtin).sort()).toEqual(DATASET_OPTIONS.map(d => d.kind).sort())
  expect(presets.filter(p => p.asset).map(p => p.asset).sort()).toEqual(Object.keys(assets).sort())
  for (const asset of Object.values(assets)) {
    expect(createHash('sha256').update(read(asset.file)).digest('hex')).toBe(asset.sha256)
  }
})
it('loads all configurations through existing dataset APIs, preserving declared splits and target ports', async () => {
  mockAssets()
  for (const preset of DATASET_LIBRARY.flatMap(f => f.presets)) {
    const selection = await loadLibraryDataset(preset.id)
    const preview = libraryPreview(selection)
    expect(preview.train, preset.id).toBeGreaterThan(0)
    expect(preview.test, preset.id).toBeGreaterThan(0)
    expect(preview.outputs.length, preset.id).toBeGreaterThan(1)
    if (preset.asset && selection.customCsv) {
      expect(preview.train, preset.id).toBe(assets[preset.asset].sourceRecords.train)
      expect(preview.test, preset.id).toBe(assets[preset.asset].sourceRecords.test)
      expect(selection.customCsv.task).toBe(preset.task)
      expect(selection.customCsv.splits).toHaveLength(preview.train + preview.test)
    }
    if (preset.asset && selection.textData?.task !== 'language' && selection.textData) {
      expect(preview.train).toBe(assets[preset.asset].sourceRecords.train)
      expect(preview.test).toBe(assets[preset.asset].sourceRecords.test)
    }
  }
}, 30000)
it('preserves the exact football tasks and their common benchmark across representations', async () => {
  mockAssets()
  const mixed = await loadLibraryDataset('football-mixed'), benchmark = await loadLibraryDataset('football-benchmark')
  const qa = await loadLibraryDataset('football-qa'), small = await loadLibraryDataset('football-counts')
  const ordered = await loadLibraryDataset('football-tokens')
  expect(small.textData?.documents).toEqual(ordered.textData?.documents)
  const train = (d: typeof mixed) => d.textData!.documents.filter(doc => doc.split === 'train')
  expect(train(mixed)).toEqual(train(benchmark))
  expect(new Set(train(mixed).map(doc => doc.question))).toEqual(new Set(['where is football', 'where is liam', 'where is nora', 'who has football']))
  expect(benchmark.textData!.documents.filter(doc => doc.split === 'test')).toHaveLength(240)
  expect(new Set(benchmark.textData!.documents.filter(doc => doc.split === 'test').map(doc => doc.question))).toEqual(new Set(['where is football']))
  expect(qa.textData!.documents).toEqual(benchmark.textData!.documents.map(doc => ({ text: `${doc.text} ${doc.question} ${doc.label}`, split: doc.split })))
  expect(libraryPreview(qa).outputs.map(output => output.shape)).toEqual([[23], [23], [1]])
  expect(libraryPreview(benchmark).outputs.map(output => output.shape)).toEqual([[32], [8], [32], [32, 1], [8, 1], [1]])
  expect(libraryPreview(benchmark).outputs[2].label).toBe('story position IDs')
  const last = await loadLibraryDataset('narrative-last'), all = await loadLibraryDataset('narrative-all')
  expect({ ...last.textData, targetMode: undefined }).toEqual({ ...all.textData, targetMode: undefined })
  expect(libraryPreview(last)).toMatchObject({ train: 1534, test: 765 })
  expect(libraryPreview(last).outputs.at(-1)?.shape).toEqual([1])
  expect(libraryPreview(all).outputs.at(-1)?.shape).toEqual([32])
})
it('fails clearly for unavailable or malformed assets and passes through cancellation', async () => {
  const signal = new AbortController().signal
  const fetch = vi.fn().mockResolvedValue({ ok: false })
  vi.stubGlobal('fetch', fetch)
  await expect(loadLibraryDataset('imdb', signal)).rejects.toThrow('Could not load')
  expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/datasets/imdb.json'), { signal })
  fetch.mockResolvedValue({ ok: true, json: async () => ({}) })
  await expect(loadLibraryDataset('imdb')).rejects.toThrow('invalid')
})
