# Temperature and top-k construction lab

Assignment 12 now extends its frozen bigram with six independently graded
probability constructions. This replaces the former two-branch temperature
exercise. Assignment names, filenames, datasets, trained bigram values, original
loss limits, and the base.json handoff to the context lesson are retained.

## App operations

Argsort returns final-axis integer indices, with configurable ascending or
descending order, stable lower-index tie breaking, and excluded entries last.
Index mask accepts values and selected integer indices, keeping or excluding
those original positions. A [k] vector can apply across rows, or each row can
supply its own indices. Index mask retains the original tensor shape and exact
exclusion metadata. Softmax normalizes only eligible entries, even when a valid
logit is below the finite display sentinel. IDs and ranks are discrete; selected
values retain gradients and unselected values receive zero.

Both blocks appear in Build / Features and tensors and the canvas search menu.
Details exposes Sort order and Index selection. File Save/Import preserves
operation settings and masks. Trace evaluation, Tensor CPU execution/inspection,
and PyTorch export support the operations. The existing generation controls
continue to work independently of canvas branches.

## Required work and assessment

| Construction in sampling.json | Points |
| --- | ---: |
| Full-vocabulary T = 0.5 | 5 |
| Full-vocabulary T = 2 | 5 |
| Top-1, T = 1 | 10 |
| Top-5, T = 1 | 10 |
| Top-5, T = 0.5 | 10 |
| Top-5, T = 2 | 10 |

Original-model data/form/train/held-out checks earn 20 points per file. Matching
parameters earn 10, totaling 100. Full-vocabulary branches preserve the existing
25-token mathematical comparison; top-k branches exclude padding before ranking
and match generation behavior. Every substantial required construction is saved
and automatically checked. Generation button clicks, prose, screenshots, and
unverifiable training history are not graded. No retraining is required.

The grader checks each construction across all 25 possible final-token contexts,
using independent NumPy computation. Its structural checks require dynamic
ranking and correct index selection, while allowing equivalent constant scaling
and arbitrary block names. This rejects fixed favorite-token lists and a
padding token that displaces a real candidate. Partial construction credit is
independent of the other branches. Existing model quality gates still apply.

## Validation

The three retained trained starts 137/211/307 all earn 100/100 with unchanged
base predictions and parameters. App trace probabilities match the independent
NumPy results across every token context to within 1e-12. Tensor CPU probability
inspection agrees within 1e-5. These are numerical tolerances, not new student
performance thresholds. The private sampling-engine-validation.json records
actual maximum errors and the app revision used.

Headless Chromium checks real generation controls at k = 1/5 and T = 0.5/1/2,
compares the canvas distributions to generation, edits and restores both new
Details controls, and exercises File Save and reimport without changing learned
values or epochs. Current handouts are checked at phone and desktop widths.
Private saved UI exports are also evaluated with the rebuilt upload ZIP.
App tests cover stable ties, exact exclusions, malformed indices/shapes, gradient
behavior, serialization, inspector controls, Tensor execution, and PyTorch
export. Local testing does not establish hosted Gradescope behavior or physical
GPU performance.

Source and browser evidence retain the tested app commit and dirty-checkout
status. Trained exports and reproduction tools remain in the private Coursework
repository; this report, zeroed templates, grader runtime, handout, and ZIP are
public.

## Reproduction

From the private Coursework root:

```sh
.venv/bin/python scripts/assignments/advanced/rebuild-sampling.py
RELEASE_ASSIGNMENTS=12-sampling .venv/bin/python scripts/assignments/package-advanced.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/advanced/validate-early-sampling.py
NEURALCANVAS_ROOT=/workspace/NeuralCanvas node scripts/assignments/advanced/validate-sampling-engine.mjs
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/12-sampling/source -p test_grader.py -v
PLAYWRIGHT_ROOT=<playwright-package> NEURALCANVAS_ROOT=<app-checkout> APP_URL=<local-app-url> node scripts/assignments/advanced/browser-early-sampling.cjs
```

The targeted builder preserves canonical handouts and dataset bytes and rebuilds
trained development references only from existing bigram pilots. Public users
can rebuild ZIPs and run public package checks without access to private models.
