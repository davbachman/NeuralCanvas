"""Prepare the additional public teaching datasets from vendored source bytes.

The original classroom datasets in public/datasets are retained unchanged.
No network access or third-party Python packages are required.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import random

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/dataset-sources'
OUT = ROOT / 'public/datasets'
manifest_path = ROOT / 'src/learning/dataset-library.json'
manifest = json.loads(manifest_path.read_text())


def write(name, headers, rows, classification=False):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row[-1] if classification else 'all', []).append(i)
    held_out = set()
    rng = random.Random(20261024)
    for indices in groups.values():
        held_out.update(rng.sample(indices, round(len(indices) * .2)))
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow([*headers, 'split'])
    for i, row in enumerate(rows):
        writer.writerow([*row, 'test' if i in held_out else 'train'])
    payload = output.getvalue().encode()
    (OUT / f'{name}.csv').write_bytes(payload)
    manifest[name] = dict(file=f'{name}.csv', sha256=hashlib.sha256(payload).hexdigest(),
                          sourceRecords=dict(train=len(rows)-len(held_out), test=len(held_out)))
    print(name, manifest[name]['sourceRecords'])


with (SOURCE / 'wine.csv').open() as file:
    rows = list(csv.reader(file))[1:]
write('wine', ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
               'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
               'color_intensity', 'hue', 'od280_od315', 'proline', 'cultivar'], rows, True)

features = [line.split() for line in gzip.decompress((SOURCE / 'diabetes-data.csv.gz').read_bytes()).decode().splitlines()]
targets = gzip.decompress((SOURCE / 'diabetes-target.csv.gz').read_bytes()).decode().split()
assert len(features) == len(targets) == 442
write('diabetes', ['age', 'sex', 'bmi', 'blood_pressure', 's1', 's2', 's3', 's4', 's5', 's6', 'progression'],
      [[*row, target] for row, target in zip(features, targets)])

with (SOURCE / 'penguins.csv').open() as file:
    records = list(csv.DictReader(file))
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
rows = [[*[r[k] for k in features], r['species']] for r in records if all(r[k] != 'NA' for k in features)]
write('penguins', [*features, 'species'], rows, True)

with (SOURCE / 'california-housing.csv').open() as file:
    records = list(csv.DictReader(file))
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
rows = [[*[r[k] for k in features], r['median_house_value']] for r in records if all(r[k] for k in features)]
write('california-housing', [*features, 'median_house_value'], rows)
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.iterdir()) if p.is_file()}
(ROOT / 'scripts/dataset-source-sha256.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
