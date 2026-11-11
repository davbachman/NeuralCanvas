"""Package the finite-answer reward-learning course releases."""
from pathlib import Path
from handout_markdown import markdown
from package_common import package_assignment
root=Path(__file__).resolve().parents[2]
for name in ['25-rlvr','26-rlhf']:
    base=root/'assignments'/name
    (base/'description.md').write_text(markdown((base/'description.html').read_text()))
    package_assignment(name,name)
