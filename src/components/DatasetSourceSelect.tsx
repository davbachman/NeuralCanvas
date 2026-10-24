import { DATASET_MENU_OPTIONS } from '../domain/datasets'
import type { DatasetKind } from '../domain/types'

interface Props {
  kind: DatasetKind
  label: string
  'aria-label': string
  className?: string
  showImportOptions?: boolean
  onBrowse?: () => void
  onDataset: (kind: DatasetKind) => void
}

/** Show the current source when closed, with simple examples and source actions. */
export function DatasetSourceSelect({ kind, label, 'aria-label': ariaLabel, className, showImportOptions = false, onBrowse, onDataset }: Props) {
  const options = DATASET_MENU_OPTIONS.filter(option => showImportOptions || (option.kind !== 'custom-csv' && option.kind !== 'custom-text'))
  const selected = options.find(option => option.kind === kind && !option.kind.startsWith('custom-'))
  return <select aria-label={ariaLabel} title={label} className={className} value={selected?.kind ?? ''} onChange={event => {
    const action = event.target.value
    if (action === 'browse') onBrowse?.()
    else {
      const option = options.find(item => item.kind === action)
      if (option) onDataset(option.kind)
    }
  }}>
    {!selected && <option value="" disabled hidden>{label}</option>}
    {options.map(option => <option key={option.kind} value={option.kind}>{option.label}</option>)}
    <option value="browse">Browse datasets…</option>
  </select>
}
