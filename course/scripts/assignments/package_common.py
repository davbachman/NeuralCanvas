"""Package from canonical per-assignment sources, without duplicating Canvas files."""
import hashlib
import json
from pathlib import Path
import zipfile


def package_assignment(name, dataset, version):
    base = Path(__file__).resolve().parents[2] / 'assignments' / name
    source, reference = base / 'source', base / 'reference'
    grade = base / 'Gradescope' / f'gradescope-{version}.zip'
    grade.parent.mkdir(parents=True, exist_ok=True)
    files = ['setup.sh', 'run_autograder', 'runner.py', 'grader.py']
    if (source / 'graph_eval.py').exists():
        files.append('graph_eval.py')
    with zipfile.ZipFile(grade, 'w', zipfile.ZIP_DEFLATED) as archive:
        for filename in files:
            archive.write(source / filename, filename)
        archive.write(reference / 'data.json', 'data.json')
    with zipfile.ZipFile(grade) as archive:
        assert set(archive.namelist()) == set(files) | {'data.json'}
        assert (archive.getinfo('run_autograder').external_attr >> 16) & 0o111
    canvas = base / 'Canvas'
    datasets = [dataset] if isinstance(dataset, str) else list(dataset)
    assert {p.name for p in canvas.iterdir()} == {'description.html', *datasets}
    manifest = {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [grade, canvas / 'description.html', *(canvas / name for name in datasets)]}
    (reference / 'package-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(grade)
