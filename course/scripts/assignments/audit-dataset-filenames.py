"""Require identical bytes whenever current student datasets share a filename.

Canvas course uploads share a filename namespace. Archived releases and private
pilot fixtures are excluded; this checks the current Canvas materials for every
assignment, including already-issued assignments.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def audit():
    grouped = defaultdict(list)
    assignments = sorted(p for p in (ROOT / 'assignments').iterdir() if p.is_dir())
    for base in assignments:
        for path in sorted((base / 'Canvas').iterdir()):
            if path.name == 'description.html':
                continue
            grouped[path.name].append((base.name, hashlib.sha256(path.read_bytes()).hexdigest()))
    records = []
    for name, copies in sorted(grouped.items()):
        assert len({digest for _, digest in copies}) == 1, (
            f'Different student datasets share {name}: ' + ', '.join(base for base, _ in copies))
        records.append(dict(filename=name, sha256=copies[0][1], assignments=[base for base, _ in copies]))
    return dict(scope='Current Canvas dataset files only; excludes archived releases and private fixtures.',
                assignmentsChecked=[p.name for p in assignments],
                filesChecked=sum(len(copies) for copies in grouped.values()),
                distinctFilenames=len(records), conflictingFilenames=[], datasets=records)


if __name__ == '__main__':
    report = audit()
    (ROOT / 'docs/dataset-filenames.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Checked {report['filesChecked']} dataset files across {len(report['assignmentsChecked'])} assignments: no filename conflicts.")
