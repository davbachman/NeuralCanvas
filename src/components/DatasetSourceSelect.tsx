import { DATASET_MENU_OPTIONS } from '../domain/datasets'
import type { DatasetKind } from '../domain/types'

interface Props {
  label: string
  'aria-label': string
  className?: string
  onBrowse?: () => void
  onDataset: (kind: DatasetKind) => void
}

/** Show the current source when closed, and offer browser/import actions only. */
export function DatasetSourceSelect({ label, 'aria-label': ariaLabel, className, onBrowse, onDataset }: Props) {
  return <select aria-label={ariaLabel} title={label} className={className} value="" onChange={event => {
    const action = event.target.value
    if (action === 'browse') onBrowse?.()
    else if (action === 'custom-csv' || action === 'custom-text') onDataset(action)
  }}>
    <option value="" disabled hidden>{label}</option>
    <option value="browse">Browse datasets…</option>
    {DATASET_MENU_OPTIONS.map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}
  </select>
}
