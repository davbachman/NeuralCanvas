"""Check the public teaching release without private fixtures or new training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

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
    dataset_count = 0
    for base in assignments:
        require((base / 'INSTRUCTOR.md').is_file(), f'Missing instructor guide: {base}')
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
        for relative, digest in json.loads((base / 'reference/package-sha256.json').read_text()).items():
            require(hashlib.sha256((base / relative).read_bytes()).hexdigest() == digest,
                    f'Package manifest mismatch: {base.name}/{relative}')
        for dataset in (base / 'Canvas').iterdir():
            if dataset.name != 'description.html':
                require(dataset.read_bytes() == (APP / 'public/datasets' / dataset.name).read_bytes(),
                        f'App/course dataset mismatch: {dataset}')
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
    print(f'Validated 19 public teaching releases and {dataset_count} dataset copies.')


if __name__ == '__main__':
    validate()
