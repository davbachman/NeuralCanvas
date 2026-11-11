# Assignment 20 — Interpret a transformer with sparse features

This appends the existing sequence without changing Assignments 1–19. Students
use their own preserved `multihead.json`, before QA fine-tuning. No trained
solution or starter model is supplied. Public handouts, graders, fixtures and
packages are self-contained; trained references and solution-producing pilots
remain in NeuralCanvas-Coursework.

## Purpose and handoff

Reconstruct the width-16 residual immediately after the second block's MLP
residual addition and before final output Layer norm. Compare an unpenalized
autoencoder with an otherwise controlled sparse variant. Both have 32 ReLU
features, four new Params, and unit-norm decoder rows. Students independently
derive shapes and build a selected feature's rank-one contribution with Slice
and Matrix product. Inspection supports that artifact; reading contexts or
claiming a semantic label earns no points.

The 48-run NumPy screening study compared narrative and QA bases, widths 32/64,
four penalties and three independent base/encoder initializations. Width 64
improved reconstruction but increased active features and inspection complexity.
Narrative width 32 had sparse validation MSE 0.0769–0.0941 and 9.12–9.52 active
features/token; QA width 32 had 0.1001–0.1115 and 11.15–11.61. These are different
representation distributions, not a head-to-head claim about language-model
quality. The narrative handoff offers varied next-token contexts and the simpler
32-feature construction on 1,534 training windows. Its exact app recipe was
then validated separately.

## Student workflow

1. Save the adequately trained narrative generator as `baseline.json`.
2. Freeze existing parameters and stable Dropout before adding the autoencoder.
3. Construct `h = ReLU(r We + be)` and `r_hat = h Wd + bd`, constrain decoder rows,
   and save a personally constructed untrained copy. Reuse the single Loss as
   masked mean squared error with a frozen internal target and selected `h`.
4. Train ten epochs at λ=0; save `reconstruction.json`.
5. Reopen the same untrained copy, use λ=0.03, train ten epochs, inspect features,
   add `h_j v_j`, and save `sparse.json`.

Both runs use Tensor CPU, Adam 0.003, batch 32, shuffle, report every epoch,
gradient norm limit 1, no early stopping, no parameter penalty and no decay.
Matrices use Xavier and biases zero; decoder rows are normalized on
initialization and updates. Frozen parameter values remain exact. The canonical
32-token windows have no input padding; general internal-target support excludes
padding in both engines. The validation split is used for choosing settings.

## Rubric and independently evaluated targets

| Component | Data | Form | Quality | Sparsity | Contribution |
|---|---:|---:|---:|---:|---:|
| baseline.json | 5 | 5 | 10 | — | — |
| reconstruction.json | 5 | 10 | 10 | — | — |
| sparse.json | 5 | 10 | 10 | 10 | 10 |

The remaining 10 points check the controlled comparison: identical adequately
trained transformer values and sparse validation activity at most 75% of the
unpenalized value. Generator CE limits remain 0.60 training / 0.65 validation.
Both-split reconstruction MSE limits are 0.10 unpenalized and 0.20 sparse.
Sparse validation activity must be at most 12 features/token, with at most 16
inactive features. Active means strictly positive ReLU output (tolerance zero).
Accept λ in [0.01, 0.10], recommending the tested 0.03. Decoder row norm tolerance
is 5e-5. Relative improvement alone never substitutes for absolute quality.

The evaluator is bounded NumPy with an allowlisted expression parser. It checks
canonical data, actual operations, preserved base values, deterministic frozen
targets, exact decoder ancestry and norms. Labels, caches and epoch counts are
ignored. Equivalent operation graphs within the stated constraints are accepted;
the feature index is student-selected. The original residual is anchored to the
preserved vocabulary path, not a shape-compatible duplicate branch.

## Evidence and validation

Six exact app Tensor CPU runs (three independently trained base models, each with
an independently initialized autoencoder and matched dense/sparse initial values)
gave dense validation MSE 0.0249–0.0280 and 19.00–19.20 active features/token;
sparse MSE 0.0789–0.0884 and 8.83–8.93 active features/token. No feature was inactive
over validation in these runs. Frozen values remained byte-for-byte identical;
the largest decoder norm error was below 8.6e-8.

Pilots used app `d856af3da2fc411eb58252138fd51117b4bb97b1` with the extension's
uncommitted engine changes. Each private report records the status and diff hash.
Runs took roughly 4–6 minutes each while three CPU jobs shared this cloud host;
this is not a student-laptop timing guarantee. Later validation preflight adds a
first-epoch forward pass for malformed-objective safety; the training recipe and
parameter updates are unchanged. See the detailed
[extension validation](../../../docs/INTERPRETABILITY-EXTENSION.md), including
browser, package and export limits.

Private reproduction (run in NeuralCanvas-Coursework):

```sh
.venv/bin/python scripts/assignments/interpretability/screen.py
node scripts/assignments/interpretability/pilot.mjs
.venv/bin/python -m unittest discover -s assignments/20-sparse-autoencoder/source -p test_grader.py -v
```

Public packaging (run in `course/`):

```sh
python3.12 scripts/assignments/package-interpretability.py
python3.12 scripts/validate-course.py
```

## Publish

Use `../description.html` in Canvas and
`Gradescope/gradescope-20-sparse-autoencoder.zip` for a 100-point Gradescope
Programming Assignment. Use Ubuntu 24.04 (Python 3.12), at least 4 GB
memory and a 20-minute external timeout. The runner additionally bounds workers.
Test instructor-owned saved files on hosted Gradescope before issuing. Local
extracted-package checks do not establish hosted behavior. No upload is performed
by these packaging commands or by pushing Git.
