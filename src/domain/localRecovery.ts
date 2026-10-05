import { createProjectStateFile, parseProjectStateFile } from './session'
import type { ProjectStateFile, ProjectStateSnapshot } from './types'
import type { LossReport } from '../components/LossReportPanel'

export interface RecoveryWorkspace {
  state: ProjectStateSnapshot
  lossReports: LossReport[]
  shuffleEachEpoch: boolean
}
export interface RecoveryRecord {
  version: 1
  file: ProjectStateFile
  lossReports: LossReport[]
  shuffleEachEpoch: boolean
}

const DATABASE = 'neural-canvas-recovery'
const STORE = 'workspace'
const KEY = 'latest'

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
  if (record.version !== 1 || !parsed.ok || typeof record.shuffleEachEpoch !== 'boolean' || !Array.isArray(record.lossReports) ||
      !record.lossReports.every(row => row && Number.isInteger(row.epoch) && row.epoch >= 0 && Number.isFinite(row.loss) &&
        (row.heldOutLoss === undefined || Number.isFinite(row.heldOutLoss)))) {
    throw Error('The local recovery copy is invalid. Use a saved model file.')
  }
  return { version: 1, file: parsed.file, lossReports: record.lossReports, shuffleEachEpoch: record.shuffleEachEpoch }
}

export async function readRecovery(): Promise<RecoveryRecord | undefined> {
  const value: unknown = await transaction('readonly', store => store.get(KEY))
  return value === undefined ? undefined : parseRecovery(value)
}
export async function writeRecovery(workspace: RecoveryWorkspace): Promise<void> {
  const record: RecoveryRecord = { version: 1, file: createProjectStateFile(workspace.state), lossReports: workspace.lossReports.map(row => ({ ...row })), shuffleEachEpoch: workspace.shuffleEachEpoch }
  // Keep the previous good copy if an in-progress edit is not serializable.
  const validated = parseRecovery(record)
  await transaction('readwrite', store => store.put(validated, KEY))
}
export async function deleteRecovery(): Promise<void> {
  await transaction('readwrite', store => store.delete(KEY))
}
