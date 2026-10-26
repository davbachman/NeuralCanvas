# Assignment 12 — Control how the model chooses words

Public teaching release: [course setup and packaging](../../README.md). Wiring,
zeroed templates, grader sources, and this guide are public. Trained references,
solution-based tests, and browser model exports remain in NeuralCanvas-Coursework.

## Teaching purpose

Separate two choices in decoding: temperature changes relative probabilities;
top-k restricts eligible tokens. Students construct ranking and selection on
canvas, preserve token identities, and combine both controls without training.
The trained bigram still sees only the final token; decoding cannot add memory.

## Publish and handoff

Paste Canvas/description.html into Canvas and upload the current
Gradescope/gradescope-12-sampling.zip to a 100-point Programming Assignment.
Use Ubuntu 22.04 or 24.04, at least 1 GB RAM, and a grading timeout of at least
180 seconds. Setup pins NumPy 1.26.4. Hosted Gradescope has not been exercised.
Require base.json and sampling.json. No dataset attachment, starter model,
written answer, or screenshot is required. A Git push does not upload to either
course service.

Students import their own trained bigram.json, save an unchanged base.json, and
construct six parallel probability branches in sampling.json. Keep every
learned value, the original loss input, and the original output Softmax intact.
No randomization or additional training is required. The next assignment starts
from base.json; its architecture and training recipe remain unchanged.

## Required constructions and rubric

| Artifact or construction | Points |
| --- | ---: |
| base.json: canonical data, original model form, training loss, held-out loss | 5 + 5 + 5 + 5 |
| sampling.json: same four original-model checks | 5 + 5 + 5 + 5 |
| Full-vocabulary Softmax at T = 0.5 and T = 2 | 5 each |
| Top-k at T = 1, with k = 1 and k = 5 | 10 each |
| Combined top-5 distributions at T = 0.5 and T = 2 | 10 each |
| Identical learned parameters across both files | 10 |
| Total | 100 |

Both files retain 825 trainable values and the existing loss limits: training
cross-entropy <= 1.15 and held-out cross-entropy <= 1.20. Only the unscaled,
unmasked logits enter Loss. Construction points are awarded independently; a
missing branch loses its own points while valid original models retain theirs.
The original 40/50/10 split is superseded by this rubric.

## Construction details and expected observations

Argsort returns original final-axis indices, descending by default. Equal scores
prefer the lower original index. Slice must keep the first k indices along the
vocabulary axis. Index mask keeps the original logits shape, excluding every
unselected coordinate before Softmax. Its second input is integer IDs, not a
list of scores or probabilities. It supports a shared [k] index vector or
per-row indices with matching leading dimensions.

Use an Input tensor [1] containing padding ID 1 and Index mask / Exclude selected
before ranking. Excluded entries sort last. A single ranking can be shared by
all positive temperatures because positive scaling does not change rank. Use
separate slices for k = 1 and k = 5; the latter selection can serve the three
required top-5 distributions. Apply each selection to the appropriately scaled
original logits. All outputs retain 25 vocabulary coordinates. The grader
accepts equivalent constant Arithmetic scaling and ignores node names/layout.

At k = 1 the result is a point mass, independent of positive temperature. At
k = 5 the support is unchanged by temperature, while relative probabilities
change. Removing candidates requires renormalization; setting their logits to
zero is incorrect. Padding must be excluded before choosing k so it cannot take
one of the eligible places. The two plain temperature branches retain the full
25-entry vocabulary as a mathematical comparison; actual generation excludes
padding. The four top-k branches match the app's generation distributions.

The saved constructions are the required experiments. UI generation remains
optional: reset a common prompt, select Sample, and compare k = 1/5 at
T = 0.5/1/2. Canvas branches do not change those dialog settings, which are not
persisted in the saved project. Random continuations and claimed button clicks
are not grading evidence.

## Data, verification, and reproduction

Canonical data, vocabulary, history split, and prepared dataset bytes are
unchanged: Football stories / Narrative generation / Final position only,
25 vocabulary tokens, context length 32, 1,534 train windows and 765 validation
windows. The original five-epoch bigram pilots and their app revisions remain
valid because the trained values and original predictions are preserved.

The independent NumPy grader verifies the original model/data, then verifies
construction signatures and recomputed probability vectors for all 25 possible
final-token contexts. It rejects fixed favorite-token lists, ascending ranking,
incorrect k or temperature, missing padding exclusion, malformed/unsafe inputs,
and tampered model values. Separate checks preserve partial credit. Private
three-seed validation uses the retained starts 137, 211 and 307 with no training.

From the private Coursework root:

```sh
.venv/bin/python scripts/assignments/advanced/rebuild-sampling.py
RELEASE_ASSIGNMENTS=12-sampling .venv/bin/python scripts/assignments/package-advanced.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/advanced/validate-early-sampling.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/12-sampling/source -p test_grader.py -v
```

Public packaging needs only the public checkout. See the
[expanded sampling report](../../docs/TOP-K-SAMPLING.md) for validation scope,
app revision information, and browser coverage. Current trained models and
browser File Save exports remain private under reference/.
