"""Package the appended interpretability releases, using committed public fixtures."""
import json
import os
from pathlib import Path
from package_common import package_assignment

root=Path(__file__).resolve().parents[2]
for name in ('20-sparse-autoencoder','21-feature-interventions'):
    if os.getenv('RELEASE_ASSIGNMENTS') and name not in os.environ['RELEASE_ASSIGNMENTS'].split(','):continue
    config=json.loads((root/'assignments'/name/'reference/data.json').read_text())
    if config.get('status')=='Pilot thresholds pending; do not package.':
        raise SystemExit('Finish native pilots and finalize thresholds before packaging '+name)
    package_assignment(name,config['version'])
