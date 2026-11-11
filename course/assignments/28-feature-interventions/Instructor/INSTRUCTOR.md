# Assignment 28 — Discover and steer a final-model feature

The preceding lesson uses each student's own final preference-adapted policy
and SAE. This lesson supplies a common frozen final transformer plus an SAE
trained specifically on that checkpoint. Students construct the discovery and
steering branches; completed constructions remain private.

## Checkpoint compatibility

The supplied `student/feature-steering.json` contains an actual policy after
instruction tuning, verifiable rewards and preference rewards. Its narrative
dataset exposes varied contexts without updating its weights. The fitted SAE
uses the residual immediately after the second block's feed-forward residual
addition and before final output Layer norm. Every Param is frozen.

The student download is also deployed at `public/models/feature-steering.json`.
The package manifest and reference configuration record its exact hash. Base
and SAE fingerprints are derived independently from operations and values; the
grader accepts renamed nodes but rejects altered values, disconnected copied
decoder directions and the old narrative-only checkpoint.

The private reward reference uses explicitly synthetic checker-derived
preferences to exercise the app's preference-fitting and policy-update path.
It is an engineering fixture, not evidence of human preference collection.
Students' preceding preference lesson records their own actual judgments. This
provenance distinction carries into the supplied interpretability checkpoint.

## Guided discovery and causal intervention

The assigned feature index, input association and output effect come from a
fresh post-training screen. They are checkpoint-specific. Preserve the first
assignment's distinction between input association and causal output effects.
The handout initially withholds the interpretation behind separate collapsed
input and output hints.

Students inspect activation contexts, then build `h_j*m` and `h_j*(1-m)` where
m is a vocabulary One-hot Slice for their hypothesized input word. The grader
checks those connected operations and the association at fixed final positions
on both splits. It does not grade prose or trust a manually chosen set of
successful examples.

Students next build three copies: `r + 0*v_j`, `r + alpha*v_j`, and
`r - h_j*v_j`. Keep the original residual feeding the SAE encoder. Reconnect only
the vocabulary path to the intervened residual. Full canonical validation
windows determine distribution change; word-matched contexts determine the
supported output word's probability change. High- and low-activation controls
are ranked reproducibly with original-row tie breaking.

Feature selection and numerical targets use validation data, so this is not an
untouched test-set claim. An observed intervention effect neither establishes a
unique semantic feature nor proves improved factual accuracy. The final model
can retain poor narrative predictions after QA adaptation; narrative loss is
reported diagnostically and changes under steering are bounded.

## Assessment and validation

Submit discovery.json (20 points), zero.json (20), amplified.json (20),
ablated.json (30), with 10 further points for the common frozen checkpoint and
assigned feature across all files. The handout and reference/data.json contain
the measured feature-specific requirements. Each construction has independent
partial credit. Students may keep their own sparse.json for optional comparison,
but must use the supplied checkpoint in all four graded files.

See [Post-training interpretability](../../../docs/POSTTRAINING-INTERPRETABILITY.md)
for current pilot evidence and exact checkpoint provenance. The old feature-5
“now → the” result belongs to the earlier narrative-only checkpoint and must
not be carried forward without revalidation.

Public package: run `python scripts/assignments/package-interpretability.py`
from course/. Private reproduction uses
`scripts/assignments/interpretability/posttraining-pilot.mjs`,
`posttraining-screen.py`, and `build-posttraining-steering.py`, followed by the
private grader tests. Use the single Gradescope ZIP with Python 3.12/Ubuntu
24.04, 4 GB memory and a 20-minute timeout. Local browser and extracted-package
validation do not certify hosted Gradescope or a deployed website.
