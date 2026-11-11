"""Package the story/transformer sequence using committed public references."""
from pathlib import Path
from package_common import package_assignment
import json,os
root=Path(__file__).resolve().parents[2]
for base in sorted((root/'assignments').iterdir()):
    if base.is_dir() and 14<=int(base.name[:2])<=24:
        if os.getenv('RELEASE_ASSIGNMENTS') and base.name not in os.environ['RELEASE_ASSIGNMENTS'].split(','):continue
        config=json.loads((base/'reference/data.json').read_text())
        package_assignment(base.name,config['version'])
