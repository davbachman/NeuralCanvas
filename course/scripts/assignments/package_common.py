"""Package canonical grader sources and hash both student description formats."""
import hashlib
import json
from pathlib import Path
import zipfile
from handout_markdown import markdown


def write_manifest(base):
    packages = list((base / 'Gradescope').glob('*.zip'))
    assert len(packages) == 1, f'Expected one current package: {base}'
    files = [packages[0], base / 'description.html', base / 'description.md']
    config = json.loads((base / 'reference/data.json').read_text())
    if config.get('checkpoint'):
        files.append(base / config['checkpoint']['file'])
    manifest = {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (base / 'reference/package-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')


def package_assignment(name, version):
    base = Path(__file__).resolve().parents[2] / 'assignments' / name
    source, reference = base / 'source', base / 'reference'
    (base / 'description.md').write_text(markdown((base / 'description.html').read_text()))
    # The directory identifies a lesson; the config retains its recipe version.
    # This keeps upload names stable when a recipe is revised.
    grade = base / 'Gradescope' / f'gradescope-{name}.zip'
    grade.parent.mkdir(parents=True, exist_ok=True)
    files = ['setup.sh', 'run_autograder', 'runner.py', 'grader.py']
    for helper in ('graph_eval.py', 'reward_eval.py'):
        if (source / helper).exists():
            files.append(helper)
    with zipfile.ZipFile(grade, 'w', zipfile.ZIP_DEFLATED) as archive:
        for filename in files:
            archive.write(source / filename, filename)
        archive.write(reference / 'data.json', 'data.json')
    with zipfile.ZipFile(grade) as archive:
        assert set(archive.namelist()) == set(files) | {'data.json'}
        assert (archive.getinfo('run_autograder').external_attr >> 16) & 0o111
    write_manifest(base)
    print(grade)
