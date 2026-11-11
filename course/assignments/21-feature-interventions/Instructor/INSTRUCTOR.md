# Assignment 21 — Test features through residual interventions

This lesson uses the instructor-authorized frozen student checkpoint
`../student/feature-steering.json`, also deployed at
`public/models/feature-steering.json`. It is a trained transformer plus SAE,
without completed discovery or intervention branches. The preceding lesson
continues to use students' own models. This exception to the normal no-starter
policy was explicitly requested on 2026-11-11; completed solutions remain private.

## Guided discovery and causal prediction

Specify zero-based feature 5, initially withholding its interpretation. Students
inspect activating contexts, then construct word-conditioned activation branches
`h_5*m` and `h_5*(1-m)`, where m is a vocabulary-column Slice of One-hot encoding
of actual input tokens. `discovery.json` records an objectively checkable
hypothesis, with no written response or quiz. They must derive the vocabulary
index and shapes from the Dataset. Hints in the handout are collapsed initially.

The input association is **now**. On final positions, this feature separates all
69 training occurrences from 1,465 other tokens and all 32 validation occurrences
from 733 other tokens: AUROC 1.0 in both splits, mean activation gaps 2.821/2.838.
Grade AUROC ≥0.98 and gap ≥0.5, with at least 10 observations per class. Inspection
can include other positions, but the association check holds absolute position
fixed at 31. Do not describe this as a universally monosemantic now detector.

The output supported in now-ending contexts is **the**, typically beginning a
consequence about the football. This second discovery distinguishes the context
that activates a feature from the continuation its direction supports. Do not
suggest that a now-associated feature should make the model generate now.

## Construction and rubric

All four files preserve every supplied transformer and SAE value and use feature
5. All Params are frozen. The supplied Loss already uses original vocabulary
logits, Dataset next-token targets and cross entropy, with no activation penalty.
Keep the full reconstruction and discovery branches. Only the input of final
output Layer norm changes: zero `r + 0*v_5`, amplified `r + alpha*v_5`, ablated
`r - h_5*v_5`. The encoder reads original r. Every position receives the change;
all behavioral checks score the final position. Reconstruction error is retained.

Points: discovery construction/association 10/10; zero construction/restoration
10/10; amplification construction/effect 10/10; ablation construction/effect
15/15; cross-file preserved checkpoint and feature 10. Independent valid parts
earn partial credit. No retraining or additional Params are needed.

Existing controls remain: zero probability error ≤1e-7; alpha in (0,1]; all-window
amplification TV ≥0.005; high-32 ablation TV ≥0.05; low-32 ablation TV ≤0.01;
maximum activation ≤4 across all validation positions; both nonzero interventions
mean TV ≤0.10 and final-position CE increase ≤0.30. High/low groups are the 32
largest/smallest final activations over all 765 validation windows, with stable
canonical-order ties. They are distinct from the 32 now-ending contexts.

The added directional checks use all 32 now-ending validation windows: mean
P(the) must increase ≥0.05 under amplification and decrease ≥0.15 under ablation.
Recommend alpha=1. The evaluator verifies connected operations, checkpoint values
using name-independent fingerprints, correct original encoder input, and exact
feature ancestry. It rejects changed/retrained models, copied masks, target-token
leakage, vocabulary biases, disconnected directions and unrelated features.

## Evidence and limits

Checkpoint: native narrative base 137 / SAE 5137, width 32, lambda 0.03; trained
with the recipe retained in the preceding lesson. The canonical checkpoint hash
and base/SAE fingerprints are in `reference/data.json`, and its file hash is in
the package manifest. Both public copies must be byte-identical. Preparation and
completed solutions live privately in `build-guided-steering.py` and
`assignments/21-feature-interventions/reference/models/`.

| On all 32 now-ending validation contexts | Zero/original | Amplified alpha=1 | Ablated |
|---|---:|---:|---:|
| Mean P(the) | 0.281991 | 0.418627 | 0.018053 |
| Most likely token is the | 6/32 | 32/32 | 0/32 |

Amplification changes 26 most-likely choices. Maximum feature activation across
all positions is 3.279897; all-window amplification TV is 0.025078. The same
feature passes the existing global and control requirements. Wider language
analysis across three seeds also found eligible now-associated continuation
features, but their indices and supported outputs differ. Those findings do not
guarantee this particular interpretation in students' own models, motivating the
common checkpoint here.

This is an exploratory result on the existing validation split, which was used
for recipe and feature selection. Repeated templates and overlapping windows
are correlated. No claim is made of improved factual football tracking, a unique
semantic concept, or an advantage from sparsity over dense/random encoders.
Optional generation is illustrative; it is not the acceptance test.

## Validation and release

From the private sibling repository:

```sh
.venv/bin/python scripts/assignments/interpretability/build-guided-steering.py
.venv/bin/python -m unittest discover -s assignments/21-feature-interventions/source -p test_grader.py -v
```

The builder reproduces the authorized public checkpoint and private completed
artifacts from the existing native model; it does not retrain. From the app root
sync HTML with `python3.12 course/scripts/sync-handouts.py`, then from `course/`
run `python3.12 scripts/assignments/package-interpretability.py` and
`python3.12 scripts/validate-course.py`. The grader ZIP contains only bounded
runtime and configuration, not completed solutions or the trained checkpoint.

Use the canonical HTML in Canvas and the stable
`Gradescope/gradescope-21-feature-interventions.zip` in a 100-point Gradescope
Programming Assignment (Ubuntu 24.04, Python 3.12, ≥4 GB memory, 20-minute timeout).
The handout's checkpoint link becomes available when the matching public site
revision deploys; GitHub's student/ copy is also downloadable. Test saved
instructor files on hosted Gradescope before issuing. A repository push does not
upload to Canvas/Gradescope or establish hosted validation. Historical engine
and browser checks are distinguished from this revision's checks in
[extension validation](../../../docs/INTERPRETABILITY-EXTENSION.md).
