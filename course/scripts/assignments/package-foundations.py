"""Build all five revised foundation releases from canonical committed sources."""
from pathlib import Path
from handout_markdown import markdown
from package_common import package_assignment
NAMES=['01-predictions','02-gradient-descent','03-features','04-generalization','05-regularization']
for name in NAMES:
 base=Path(__file__).resolve().parents[2]/'assignments'/name
 (base/'description.md').write_text(markdown((base/'description.html').read_text()))
 package_assignment(name,name)
