import { createProjectStateFile, parseProjectStateFile } from './session'
import type { ProjectStateFile, ProjectStateSnapshot } from './types'
import type { LossReport } from '../components/LossReportPanel'

export interface RecoveryWorkspace {
  state: ProjectStateSnapshot
  lossReports: LossReport[]
  shuffleEachEpoch: boolean
  projectFilename?: string
}
export interface RecoveryRecord {
  version: 1
  file: ProjectStateFile
  lossReports: LossReport[]
  shuffleEachEpoch: boolean
  projectFilename?: string
}

const DATABASE = 'neural-canvas-recovery'
const STORE = 'workspace'
const KEY = 'latest'
const VIEWPORT = 'viewport'
let lastWritten: RecoveryWorkspace | undefined
let lastFileTime: string | undefined

async function transaction<T>(mode: IDBTransactionMode, request: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  if (!globalThis.indexedDB) throw Error('This browser does not support local recovery. Use File → Save for backups.')
  const db = await new Promise<IDBDatabase>((resolve, reject) => {
    const opening = indexedDB.open(DATABASE, 1)
    opening.onupgradeneeded = () => opening.result.createObjectStore(STORE)
    opening.onsuccess = () => resolve(opening.result)
    opening.onerror = () => reject(opening.error)
    opening.onblocked = () => reject(Error('Local recovery is blocked by another browser tab.'))
  })
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction(STORE, mode)
      const result = request(tx.objectStore(STORE))
      tx.oncomplete = () => resolve(result.result)
      tx.onabort = () => reject(tx.error ?? Error('Local recovery could not be saved.'))
      tx.onerror = () => reject(tx.error)
    })
  } finally { db.close() }
}

export function parseRecovery(value: unknown): RecoveryRecord {
  if (!value || typeof value !== 'object') throw Error('The local recovery copy could not be read. Use a saved model file.')
  const record = value as Partial<RecoveryRecord>
  const parsed = parseProjectStateFile(JSON.stringify(record.file))
  if (record.version !== 1 || !parsed.ok || typeof record.shuffleEachEpoch !== 'boolean' ||
      (record.projectFilename !== undefined && typeof record.projectFilename !== 'string') || !Array.isArray(record.lossReports) ||
      !record.lossReports.every(row => row && Number.isInteger(row.epoch) && row.epoch >= 0 && Number.isFinite(row.loss) &&
        (row.heldOutLoss === undefined || Number.isFinite(row.heldOutLoss)))) {
    throw Error('The local recovery copy is invalid. Use a saved model file.')
  }
  return { version: 1, file: parsed.file, lossReports: record.lossReports, shuffleEachEpoch: record.shuffleEachEpoch, projectFilename: record.projectFilename }
}

export async function readRecovery(): Promise<RecoveryRecord | undefined> {
  const value: unknown = await transaction('readonly', store => store.get(KEY))
  if (value === undefined) return undefined
  const record = parseRecovery(value)
  const patch = await transaction('readonly', store => store.get(VIEWPORT))
  if (patch?.savedAt === record.file.savedAt && ('viewport' in patch) && (patch.viewport === undefined ||
      (Number.isFinite(patch.viewport?.x) && Number.isFinite(patch.viewport?.y) && Number.isFinite(patch.viewport?.zoom) && patch.viewport.zoom > 0))) {
    record.file.state.graph.view = { ...record.file.state.graph.view, expandedGroupIds: record.file.state.graph.view?.expandedGroupIds ?? [], viewport: patch.viewport }
  }
  return record
}
export async function writeRecovery(workspace: RecoveryWorkspace): Promise<void> {
  if (lastWritten && lastFileTime && sameRecoveryContent(lastWritten, workspace)) {
    const viewport = workspace.state.graph.view?.viewport
    if (viewport && (!Number.isFinite(viewport.x) || !Number.isFinite(viewport.y) || !Number.isFinite(viewport.zoom) || viewport.zoom <= 0)) throw Error('The local recovery viewport is invalid.')
    await transaction('readwrite', store => store.put({ savedAt: lastFileTime, viewport }, VIEWPORT))
    lastWritten = workspace
    return
  }
  const record: RecoveryRecord = { version: 1, file: createProjectStateFile(workspace.state), lossReports: workspace.lossReports.map(row => ({ ...row })), shuffleEachEpoch: workspace.shuffleEachEpoch, projectFilename: workspace.projectFilename }
  // Keep the previous good copy if an in-progress edit is not serializable.
  const validated = parseRecovery(record)
  await transaction('readwrite', store => { store.delete(VIEWPORT); return store.put(validated, KEY) })
  lastWritten = workspace
  lastFileTime = validated.file.savedAt
}
export async function deleteRecovery(): Promise<void> {
  await transaction('readwrite', store => { store.delete(VIEWPORT); return store.delete(KEY) })
  lastWritten = undefined
  lastFileTime = undefined
}

/** Compare immutable workspace content without traversing tensor buffers.
 * Only the camera may differ; all edits, results and controls force a full save. */
export function sameRecoveryContent(a: RecoveryWorkspace, b: RecoveryWorkspace): boolean {
  const { graph: ag, display: ad, runSettings: ac, ...as } = a.state
  const { graph: bg, display: bd, runSettings: bc, ...bs } = b.state
  const { view: av, ...am } = ag
  const { view: bv, ...bm } = bg
  const { viewport: _ap, ...ar } = av ?? {}
  const { viewport: _bp, ...br } = bv ?? {}
  void _ap; void _bp
  const shallow = (x: object, y: object) => Object.keys(x).length === Object.keys(y).length && Object.entries(x).every(([key, value]) => value === (y as Record<string, unknown>)[key])
  return a.lossReports === b.lossReports && a.shuffleEachEpoch === b.shuffleEachEpoch && a.projectFilename === b.projectFilename && shallow(am, bm)
    && JSON.stringify({ expandedGroupIds: [], ...ar }) === JSON.stringify({ expandedGroupIds: [], ...br })
    && shallow(as, bs) && JSON.stringify(ad) === JSON.stringify(bd) && JSON.stringify(ac) === JSON.stringify(bc)
}
