# Assignment 2 — Overfitting and L2 regularization

Public teaching release: see [the course guide](../../../README.md) for standalone setup, packaging, and validation. Wiring, templates, grader sources, and these notes are public. Trained reference models, browser model exports, solution-based tests, and pilot/reproduction tools mentioned below remain in the private NeuralCanvas-Coursework checkout; their paths are historical development references, not required teaching files. Test your own saved models when configuring Gradescope.

Version: regularization-02-v1. Three saved model submissions, 30/30/40 points. No comprehension questions, L1 exercise, or student starter models.

## Files to publish

All files for this assignment live here, under `assignments/02-regularization/`:

1. Paste [description.html](../description.html) into Canvas's HTML editor. Set the due date and Gradescope link there.
2. Dataset attachments are unnecessary; students use the dataset browser as described in the handout.
3. Upload [Gradescope/gradescope-regularization-02-v1.zip](../Gradescope/gradescope-regularization-02-v1.zip) to the Gradescope autograder configuration for a 100-point Programming Assignment.

Read [description.md](../description.md) on GitHub; `../description.html` is the canonical handout for Canvas. `Gradescope/` is for instructor upload to Gradescope, not distribution to students. `source/` holds the public grader implementation; `reference/` holds public canonical grading data and package hashes. Trained models, solution-dependent tests, validation exports, and historical `archive/` releases remain in the private Coursework repository. There is no current student ZIP or duplicate handout to maintain.

Rebuild the current autograder with the packaging command below. A GitHub push updates this repository; it does not publish to Canvas or Gradescope.

Use Ubuntu 22.04 or 24.04, at least 1 GB memory, and a timeout of at least 60 seconds. Setup creates `/opt/regularization-grader` with NumPy 1.26.4. Students submit `baseline.json`, `polynomial.json`, and `regularized.json`. Test the three reference models under `reference/models/`: expect 100/100, or 70/100 without `polynomial.json`. The packaged runner is tested locally; hosted Gradescope remains unverified.

## Why this experiment

The lesson is about generalization, with L2 as a concrete intervention. Students first build a familiar one-feature linear baseline, then a fifth-degree polynomial, then penalize its coefficients. A deliberately small sample makes overfitting visible without manufacturing outcomes or altering real measurements. A too-strong penalty supplies the follow-up failure case.

The CSV contains 104 original Auto MPG observations: 24 training examples sampled without replacement by NumPy default_rng(20261001) from Assignment 1's training pool, and the same 80 validation examples used in Assignment 1. The other training examples are omitted. Source indices, sampling seed, and source archive hash are recorded in data.json. The rows are kept in source order. Weight and mpg stay in their original units; model year stays in the file for compatibility but is unused. This selection is a teaching experiment, not an unbiased benchmark. The validation set has already been used and is openly used again for model selection.

The higher-degree model need not beat the linear model after regularization. Here it does not. The main comparison is between the same fifth-degree architecture with and without L2. Explain this before students infer that every new technique must beat all simpler models.

## Feature construction and penalty convention

Fit z = (weight - training mean) / training population standard deviation using Standardize features. For k=2 through 5, create z^k with Arithmetic and fit a separate Standardize features block on that power. All statistics use exactly the 24 training observations; each fitted transformation is fixed thereafter. The first feature is z itself.

Prediction is b + sum(w_k v_k), with v_1=z and each higher v_k standardized separately. Standardizing all powers matters: an L2 penalty depends on feature units. Grading checks the resulting standardized feature basis, rather than merely trusting a stored count or that a Fit button was clicked. Each coefficient must be a separate scalar Param, including the intercept.

The app uses J = MSE + lambda/2 * sum(w_k^2), with no sample-count divisor on the penalty. In the Loss inspector students must explicitly exclude the intercept: all Params have type `weight`, so the default list can include b. Dataset evaluation reports data MSE, whereas the Loss block includes the penalty. No AdamW weight decay, dropout, early stopping, or extra regularization is used.

## Pilot evidence

Actual app trace-engine training and saved-project serialization were run with seeds 137, 211, and 307 for all required models. Every power standardizer was fitted using the app's fitStandardizer. A separate NumPy calculation solves the ordinary/ridge normal equations, including the app's factor of 1/2. App predictions and the Python grader agree within 1e-6.

| Model | Epochs | Rate | Training MSE | Validation MSE |
|---|---:|---:|---:|---:|
| Linear, no penalty | 1,000 | 0.1 | 22.01439 | 13.04985 |
| Degree 5, no penalty | 10,000 | 0.1 | 17.23190 | 28.59490 |
| Degree 5, L2=0.1 | 1,000 | 0.1 | 19.23691 | 13.99684 |
| Degree 5, L2=10 | 5,000 | 0.05 | 43.89897 | 43.92075 |

The required runs agree across all three seeds. The strength sweep also includes 0.001, 0.01, and 1, with results in app-pilot.json. Both 0.01 and 0.1 earn full credit. The degree-5 ordinary least-squares solution has training MSE 17.10578 and validation MSE 40.93367. The 10,000-epoch run is close in training objective, but the weaker directions converge slowly; the finite-run validation error should not be represented as the exact optimum's value.

The low-level Node trace-engine polynomial pilot took about 3 seconds on this machine. This excludes browser rendering, scheduling, evaluation/reporting overhead, and device variation; it is not a promised student runtime. The Chrome interface pilot ran roughly 35–50 epochs/second, so allow several minutes for 10,000 epochs on a comparable machine, with the tab foregrounded. Keep reporting at 500 epochs. Tensor/WebGL is unnecessary. A long run may be broken into smaller consecutive runs without resetting parameters.

Browser verification used Chrome: an unregularized run stopped at 11,042 epochs scored 30/30 (training MSE 17.20234, validation MSE 29.94107). A fresh zero-parameter L2=0.1 model trained for 1,000 epochs scored 40/40 (validation MSE 13.99684). Both were exported using File → Save and evaluated independently. The shorter 10,000-epoch recipe was verified with three engine seeds. Canvas HTML was rendered and visually checked; numeric HTML entities preserve mathematical symbols when previewed as a standalone fragment. Safari and hosted Gradescope were not exercised for this assignment.

## Grading and tests

A: 5 file/data + 15 structure/scaling + 5 training + 5 validation = 30.
B: the same 30.
C: the same 30 + 10 L2 configuration = 40.

Training credit uses objective <= the independently computed optimum for the submitted lambda + 0.20. A/B have lambda zero, so the objective is MSE. Validation limits are respectively 14, 60, and 16 mpg squared. The generous Part B limit allows the intended overfitting. Parts are independent and missing or broken files retain other parts' credit.

The grader symbolically evaluates restricted polynomial graph operations. It checks exact source rows and splits; rejects target leakage even if canceled; verifies the degree, separate trainable coefficients, and standardized feature basis; and recomputes predictions, penalty, and the reference minimum. It ignores cached values, reported losses, and epochs. It evaluates saved parameters; it cannot certify who trained them or prove an optimizer history. It does not execute student code.

Each model runs in a separate process with a 15-second timeout and Linux CPU/address-space limits. Limits include 4 MB per file, 100 nodes, 200 wires, and 64 values per tensor. The tests cover three reference models, alternate lambda, excessive penalty, undertraining, penalty membership/intercept handling, wrong basis/scaling, target leakage, renamed/grouped models, forged scores, modified datasets, malformed files, unsafe expressions, and complete/partial submissions using the actual ZIP. All 15 tests pass with NumPy 1.26.4.

## Reproduction

Run from the repository root with Python + NumPy and the app's npm dependencies installed:

- `python scripts/assignments/prepare-mpg.py` (only if Assignment 1 source data is absent; do not overwrite issued data against a changed source)
- `python scripts/assignments/prepare-regularization.py`
- `node scripts/assignments/pilot-regularization.mjs`
- `python scripts/assignments/package-regularization.py`
- `python -m unittest discover -s assignments/02-regularization/source -p test_grader.py -v`

`reference.json` holds independent fits; `app-pilot.json` holds actual app-engine runs. Canonical datasets and current ZIPs are public; trained models and raw pilot logs remain in the private Coursework repository. No application changes were required for this assignment.

## Default loss compatibility fix (2026-10-02)

The grader now accepts an omitted loss setting as Neural Canvas's default MSE for batch inputs. Explicit incompatible losses remain rejected, and L2 configuration, scaling, and grading thresholds are unchanged. Upload the rebuilt current ZIP to Gradescope and regrade affected submissions; the prior package is preserved in `archive/before-default-loss-fix/`.

A checked submission exposed both this compatibility issue and reused fitted statistics: the standardizers for the third, fourth, and fifth powers retained the second-power mean and scale. Duplicating a fitted block does not refit it when its input expression changes. Students must click Refit on training rows on each affected power's Standardize features block after wiring the intended power, then retrain and save both polynomial models. The baseline in that submission receives full credit after the grader fix. Student files are not committed.

## Saved-default audit (2026-10-02)

Checked supported saved defaults against Neural Canvas revision `3fa65974cb668346627b40935ce1abc0131ed104`. Tensor transform now defaults to Reshape when its transform setting is omitted, matching the app. Regression losses accept omitted MSE settings; classification still requires explicit BCE or cross-entropy (or the dedicated Cross entropy block for Iris). Omitted sigmoid activation is not accepted because the app defaults to identity. Default concatenation axis, reshape shape, transpose axes, and absent/zero penalties retain their existing handling. Fitted standardization statistics are still required.

Upload the rebuilt current Gradescope ZIP to apply these compatibility fixes. Data, thresholds, and rubrics are unchanged. Previous ZIPs are preserved in `archive/before-defaults-audit/`. Automated tests cover the omitted transform setting as well as the existing full and partial submissions. Hosted Gradescope was not exercised.

## Training defaults update (2026-10-19)

The handout now explicitly directs students to change the new Tensor default to **Trace engine · SGD**. Tensor backend and optimizer controls are hidden and inactive in this mode; the issued full-batch SGD recipe remains unchanged.

Verified the current defaults and these selections in the production UI on clean app `51e00c7c64d6b68624a7d02a246479530c88b09c`, then used File → Save and checked the serialized settings and unchanged learned parameters. This is a settings/save check, not a new training pilot; the original convergence evidence above still applies to the retained recipe. See [backend validation](../../../docs/BACKEND-VALIDATION.md) and `docs/early-training-defaults-browser.json`. Datasets, model requirements, grading thresholds, original reference models, and Gradescope ZIPs are unchanged. Only the Canvas handout and its manifest need refreshing.

## Dataset browser workflow

Students open **Dataset → Details → Browse datasets…** and click **Use this dataset** after selecting the following choices. Datasets live in the app’s `public/datasets/` directory; no assignment-local copies or Canvas attachments are needed. Read [description.md](../description.md) on GitHub or use `../description.html` in Canvas. Model JSON submissions and grading targets are unchanged.

- **Regression → Auto MPG**; Configuration: **Small training sample**.


Browser/grader compatibility is checked with app 417be2e and the original canonical data. See [the dataset browser update](../../../docs/DATASET-BROWSER-ASSIGNMENTS.md).
