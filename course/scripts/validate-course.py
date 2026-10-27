"""Check the public teaching release without private fixtures or new training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent / 'assignments'))
from handout_markdown import Handout

COURSE = Path(__file__).resolve().parents[1]
APP = COURSE.parent
EARLY_PARTS = {
    '01-mpg': ['one_feature', 'two_features', 'quadratic'],
    '02-regularization': ['baseline', 'polynomial', 'regularized'],
    '03-classification': ['one_feature', 'two_features'],
    '04-matrices': ['matrix_regression', 'iris'],
    '05-mlp': ['linear', 'mlp'],
    '06-rings': ['linear', 'mlp'],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_templates(value, path):
    if isinstance(value, dict):
        if value.get('type') in ('weight', 'bias'):
            numbers = value.get('params', {}).get('value', {}).get('data', [])
            require(all(number in (0, 1) for number in numbers),
                    f'Possible trained parameters in {path}')
        for child in value.values():
            check_templates(child, path)
    elif isinstance(value, list):
        for child in value:
            check_templates(child, path)


def validate():
    assignments = sorted(path for path in (COURSE / 'assignments').iterdir() if path.is_dir())
    require(len(assignments) == 19, 'Expected all 19 current assignments')
    for path in COURSE.rglob('*'):
        if any(part in ('.venv', '__pycache__', 'node_modules', '.git')
               for part in path.relative_to(COURSE).parts):
            continue
        require(not path.is_symlink(), f'Public material depends on an external link: {path}')
        if path.suffix == '.json':
            check_templates(json.loads(path.read_text()), path)
    library = json.loads((APP / 'src/learning/dataset-library.json').read_text())
    dataset_count = 0
    for base in assignments:
        require((base / 'Instructor/INSTRUCTOR.md').is_file(), f'Missing instructor guide: {base}')
        require(not (base / 'INSTRUCTOR.md').exists(), f'Obsolete instructor guide path: {base}')
        require((base / 'description.html').is_file(), f'Missing HTML description: {base}')
        require((base / 'description.md').is_file(), f'Missing Markdown description: {base}')
        require(not (base / 'Canvas').exists(), f'Obsolete Canvas folder: {base}')
        require(not (base / 'archive').exists(), f'Unreviewed historical archive: {base}')
        require(not (base / 'reference/models').exists(), f'Trained reference models: {base}')
        require(not (base / 'reference/standardized-models').exists(), f'Trained reference models: {base}')
        packages = list((base / 'Gradescope').glob('*.zip'))
        require(len(packages) == 1, f'Expected one current package: {base}')
        package = packages[0]
        data_file = base / 'reference/data.json'
        data = json.loads(data_file.read_text())
        expected = {'setup.sh', 'run_autograder', 'runner.py', 'grader.py', 'data.json'}
        if (base / 'source/graph_eval.py').is_file():
            expected.add('graph_eval.py')
        with zipfile.ZipFile(package) as archive:
            require(set(archive.namelist()) == expected, f'Unexpected ZIP members: {package}')
            require(len(archive.namelist()) == len(expected), f'Duplicate ZIP members: {package}')
            require((archive.getinfo('run_autograder').external_attr >> 16) & 0o111,
                    f'Nonexecutable entry point: {package}')
            for name in expected:
                canonical = data_file if name == 'data.json' else base / 'source' / name
                require(archive.read(name) == canonical.read_bytes(), f'Package/source mismatch: {base.name}/{name}')
        manifest = json.loads((base / 'reference/package-sha256.json').read_text())
        require(set(manifest) == {str(package.relative_to(base)), 'description.html', 'description.md'},
                f'Incomplete release manifest: {base}')
        for relative, digest in manifest.items():
            require(hashlib.sha256((base / relative).read_bytes()).hexdigest() == digest,
                    f'Package manifest mismatch: {base.name}/{relative}')
        handout = Handout((base / 'description.html').read_text())
        require(handout.datasets, f'Missing dataset browser choices: {base}')
        for filename in set(handout.datasets):
            if filename == 'digits-8x8':
                continue  # The optional image preset has no separate classroom file.
            entries = [entry for entry in library.values() if entry['file'] == filename]
            require(len(entries) == 1, f'Unknown app dataset: {base.name}/{filename}')
            dataset = APP / 'public/datasets' / filename
            require(hashlib.sha256(dataset.read_bytes()).hexdigest() == entries[0]['sha256'],
                    f'App dataset hash mismatch: {dataset}')
            dataset_count += 1
        parts = EARLY_PARTS.get(base.name) or list(data['parts'])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with zipfile.ZipFile(package) as archive:
                archive.extractall(root / 'source')
            (root / 'submission').mkdir()
            for case in ('missing', 'malformed'):
                if case == 'malformed':
                    for part in parts:
                        (root / 'submission' / (part + '.json')).write_text('{broken')
                subprocess.run([sys.executable, str(root / 'source/runner.py')], check=True,
                               env=dict(os.environ, AUTOGRADER_ROOT=str(root), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
                               timeout=180, capture_output=True, text=True)
                result = json.loads((root / 'results/results.json').read_text())
                tests = result.get('tests', [])
                require(tests and result['score'] == 0, f'Invalid {case} feedback: {base.name}')
                require(sum(test['max_score'] for test in tests) == 100,
                        f'Incomplete rubric on {case} files: {base.name}')
                require(not any('Unable to evaluate' in test.get('output', '') for test in tests),
                        f'Grader subprocess failed: {base.name}/{case}')
        print(f'{base.name}: package, data, manifest, missing/malformed submissions OK', flush=True)
    subprocess.run([sys.executable, str(COURSE / 'scripts/sync-handouts.py'), '--check'], check=True)
    print(f'Validated 19 public teaching releases and {dataset_count} app dataset references.')


if __name__ == '__main__':
    validate()
