"""Check current handouts against the app's shared dataset files and hashes."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from handout_markdown import Handout

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT.parent


def audit():
    grouped = defaultdict(list)
    library = json.loads((APP / 'src/learning/dataset-library.json').read_text())
    assignments = sorted(p for p in (ROOT / 'assignments').iterdir() if p.is_dir())
    for base in assignments:
        for name in sorted(set(Handout((base / 'description.html').read_text()).datasets)):
            if name == 'digits-8x8':
                continue
            path = APP / 'public/datasets' / name
            expected, = [entry['sha256'] for entry in library.values() if entry['file'] == name]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
            grouped[path.name].append((base.name, hashlib.sha256(path.read_bytes()).hexdigest()))
    records = []
    for name, copies in sorted(grouped.items()):
        assert len({digest for _, digest in copies}) == 1, (
            f'Different student datasets share {name}: ' + ', '.join(base for base, _ in copies))
        records.append(dict(filename=name, sha256=copies[0][1], assignments=[base for base, _ in copies]))
    return dict(scope='App datasets referenced by current handouts; excludes archived releases and private fixtures.',
                assignmentsChecked=[p.name for p in assignments],
                filesChecked=sum(len(copies) for copies in grouped.values()),
                distinctFilenames=len(records), conflictingFilenames=[], datasets=records)


if __name__ == '__main__':
    report = audit()
    (ROOT / 'docs/dataset-filenames.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Checked {report['filesChecked']} app dataset references across {len(report['assignmentsChecked'])} assignments: no filename conflicts.")
