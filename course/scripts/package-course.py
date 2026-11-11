"""Rebuild every current Gradescope ZIP from committed public sources.

This command neither trains reference models nor regenerates dataset bytes.
Run sync-handouts.py afterward to update deployed student pages and redirects.
"""
import json
from pathlib import Path
import sys

COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE / 'scripts/assignments'))
from package_common import package_assignment


def package_course():
    sequence = json.loads((COURSE / 'sequence.json').read_text())['assignments']
    for entry in sequence:
        name = entry['slug']
        config = json.loads((COURSE / 'assignments' / name / 'reference/data.json').read_text())
        if config.get('status') == 'Pilot thresholds pending; do not package.':
            raise ValueError('Complete validation before packaging ' + name)
        package_assignment(name, config.get('version', name))
    print(f'Packaged {len(sequence)} current assignments.')


if __name__ == '__main__':
    package_course()
