# Assignment 4: Matrix models and multiclass classification

Public teaching release: see [the course guide](../../../README.md) for standalone setup, packaging, and validation. Wiring, templates, grader sources, and these notes are public. Trained reference models, browser model exports, solution-based tests, and pilot/reproduction tools mentioned below remain in the private NeuralCanvas-Coursework checkout; their paths are historical development references, not required teaching files. Test your own saved models when configuring Gradescope.

Version: matrices-04-v1. Two saved-model submissions, 40/60 points. This implements Day 4's bridge from a familiar scalar regression formula to matrix multiplication, then logits, softmax, and multiclass cross-entropy. It does not introduce hidden layers or change earlier assignments.

## Publishing

- Paste [description.html](../description.html) into Canvas’s HTML editor. Students select Auto MPG / Full sample and Iris flowers through the dataset browser; dataset attachments are unnecessary.
- Upload [Gradescope/gradescope-matrices-04-v1.zip](../Gradescope/gradescope-matrices-04-v1.zip) as a 100-point Gradescope Programming Assignment. Use Ubuntu 22.04 or 24.04, at least 1 GB memory, and a grading timeout of at least 60 seconds. Setup installs Python with NumPy 1.26.4 in `/opt/matrices-grader`.
- Students submit `matrix_regression.json` and `iris.json`, downloaded using File → Save. Each is graded independently; no written responses or starter files are required. Feedback is visible and supports resubmission.
- Before release, test the two models in `reference/models/` on hosted Gradescope: expect 100/100, or 40/100 without Iris. Local packaged-runner checks are recorded below; hosted Gradescope remains unverified.
- Keep trained reference models private; canonical grading data, source, and the autograder ZIP are public. GitHub pushes do not update Canvas or Gradescope.

## Teaching sequence and construction details

Use Part A as a short classroom bridge: Xw+b reproduces the two-feature regression from Assignment 1. The shared columns and learned weights have simply been organized into a matrix and a column vector. The 2-by-1 Param contains two independently learned entries. A scalar bias broadcasts across examples.

Then replace two input columns with four Iris measurements and one output column with three class-score columns. The 4-by-3 weight matrix has one column per class; the length-3 bias provides one intercept per class. The output is raw logits. Softmax is an inspection branch, while the loss consumes logits directly and applies its own softmax internally. There is no sigmoid or class-by-class 0.5 decision threshold. Four-feature models have no simple model plot; students should use loss history and inference accuracy.

Concatenate the raw feature vectors on axis 1 in the prescribed order, then fit one Standardize features block on the resulting matrix. Concatenate treats length-B vectors as B-by-1 columns automatically, so no feature reshapes are needed. With Numeric batch selected, the updated app preserves column shapes during fitting and standardizes columns independently using training-only means and population scales. MPG's target must still be reshaped to `-1, 1` to match B-by-1 predictions; Iris targets remain a length-B vector of class IDs. Earlier constructions using feature reshapes or separate per-feature standardizers remain accepted by the same grader.

In Details, students choose the initializer first, enter the Param shape, and press Enter in the Shape field. This invokes the same initialization as the Initialize tensor button. The tested initialization is uniform for the weight matrix and zeros for the bias. A browser check verified Enter initializes the prescribed shapes (`2, 1`, `4, 3`, and `3`) with two, twelve, and three values respectively, including zeros for the bias. Arithmetic's displayed expression may use connected variable names; the saved underlying addition is `x1 + x2`. Dataset feature and target connections are direct; Input/Target aliases are optional. The Softmax inspection branch is not separately graded.

## Dataset provenance

MPG reuses the original Assignment 1 CSV byte-for-byte, including its 318/80 split and source order. Assignment 4's canonical JSON copies its values and records the issued CSV and upstream source hashes. This is the same 398-row Auto MPG exercise (UCI DOI 10.24432/C5859H, CC BY 4.0), not Assignment 2's small training subset.

Iris comes from `sklearn/datasets/data/iris.csv` in the scikit-learn 1.5.2 wheel; `reference/raw-source/` retains the original file and description. Source: R. A. Fisher, Iris, UCI DOI 10.24432/C56C76, CC BY 4.0. The four measurements are sepal length, sepal width, petal length, and petal width in centimeters. Text labels sort as setosa, versicolor, virginica, giving class indices 0, 1, 2 in the app and grader.

NumPy `default_rng(20261004)` chooses 10 validation observations without replacement from each of the three classes; all other rows are training. Rows retain source order. There are 120 training examples (40 per class) and 30 validation examples (10 per class). `reference/data.json` records source indices, seed, source hash, labels, and the original rows. There is no separate hidden test set; this split is openly used as validation. The standardizer uses training means and population standard deviations only.

## Pilot evidence

App revision: `3fa65974cb668346627b40935ce1abc0131ed104`, validated in an isolated clean checkout. No app source changes were made. The actual app CSV parser, standardizer fitter, trace engine, evaluator, and project serializer were used. Seeds 137, 211, and 307 were piloted for both models using the revised concatenate → columnwise-standardize recipe without feature reshapes. Every saved reference model passes the app's state parser.

| Model | Learning rate | Full batch | Epochs | Training loss | Validation loss | Validation accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Matrix MPG, seed 137 | 0.05 | 318 | 1,000 | 12.757442 MSE | 7.555346 MSE | — |
| Iris, seed 137 | 0.1 | 120 | 2,000 | 0.095093 CE | 0.092154 CE | 29/30 (96.67%) |

All three MPG seeds agree to the shown precision. Iris validation CE ranges from 0.092154 to 0.093132, with 29/30 correct for every seed. Low-level training took about 1.3–2.4 seconds here; that excludes UI overhead and is not a promised classroom runtime. No optimality claim is made for multiclass CE. Report every 500 epochs and allow consecutive training chunks.

Independent Python graph evaluation agrees with app losses and accuracy to at least nine decimal places. The grader tests every seed, including cases where the trained matrix's individual entries differ. The regression errors agree with the familiar two-feature model because the model class, standardized inputs, split, and objective are the same.

## Browser verification

After moving standardization after concatenation, headless Chromium loaded the local app at the recorded revision, imported each revised graph with its parameters reset to the pilot's initial values, and completed the prescribed Train → Run epochs recipe (1,000 MPG / 2,000 Iris, full batches, reporting every 500). Training and held-out loss histories appeared. Test → Run inference reported MPG validation MSE 7.555346 and Iris validation CE 0.092154 with 96.67% accuracy. Both actual File → Save downloads received full credit from the independent grader (40/40 and 60/60). Saved files, screenshots, and grading results are in `reference/browser/`. The handout was also rendered and visually checked without horizontal overflow.

This browser check exercised imported graphs and the training/evaluation/save workflow; it did not build every block by clicking from scratch or test hosted Gradescope. The development server emitted ResizeObserver-loop notifications while laying out large blocks, but both runs completed and saved correctly. The deployed student site and Safari were not verified. For both revised graphs, a separate browser check imported unfitted standardizers, clicked Details → Fit on training rows, and verified the displayed observation and column counts (318/2 for MPG, 120/4 for Iris). Screenshots are saved as `reference/browser/*-fit.png`. Construction and fitting were also verified with the actual app engine in the pilot.

## Grading

Part A: 5 file/data + 20 matrix structure/scaling + 10 training MSE + 5 validation MSE = 40. Full performance credit requires training MSE ≤14 and validation MSE ≤10, preserving the earlier two-feature criteria.

Part B: 5 file/data + 30 matrix structure/scaling + 10 training CE + 10 validation CE + 5 accuracy = 60. Full performance credit requires training CE ≤0.15, validation CE ≤0.25, and validation accuracy ≥90% (27/30). These bounds leave margin around all pilot seeds. Cutoffs are inclusive; no exact coefficients or epoch count are graded.

The grader checks canonical data, labels, split, matrix/bias shapes, one Matrix product with the matrix Param directly in its second input, Concatenate in the prediction path, and independently trainable entries in the prescribed standardized feature basis. It compiles restricted arithmetic symbolically, checks the effect of every weight and bias, and independently computes predictions. MSE and numerically stable softmax CE/argmax accuracy are recomputed from the actual graph. Softmax in the loss path is rejected; a disconnected inspection branch is allowed. Target leakage is rejected even when canceled algebraically. Learned hidden layers, extra powers, scalar substitutions for the matrix, fixed fitted additions, modified labels, incorrect feature order/scaling, and missing regression-target reshape are rejected.

Renamed blocks, visual groups, Input/Target aliases, and a dedicated Cross entropy block equivalent to Loss → Cross entropy (logits) are accepted. Both current `neural-canvas-state` and legacy `backprop-builder-state` version-1 files are tested. Saved losses, cached outputs, and epoch counts are ignored. The grader does not execute student code or prove training history. Limits: 4 MB per file, 100 blocks, 200 wires, 64 values per Param, with separate bounded processes (15-second wall timeout, Linux CPU/memory limits) for each part.

The 15-test suite covers app parity across all seeds, both saved-file names, untrained partial credit, shapes, column order, standardization, losses, penalty, target reshape, leakage, changed data/labels/splits, renames/groups/passthroughs, unsupported expressions/cycles/oversized values, and the actual ZIP runner with complete and incomplete submissions. Complete reference submissions earn 100/100. The Canvas folder and unchanged MPG CSV are also checked.

## Reproduction

From the coursework root, using its NumPy environment and the app's installed dependencies:

```sh
.venv/bin/python scripts/assignments/prepare-matrices.py
NEURALCANVAS_ROOT=/path/to/verified/NeuralCanvas node scripts/assignments/pilot-matrices.mjs
.venv/bin/python scripts/assignments/package-matrices.py
.venv/bin/python -m unittest discover -s assignments/04-matrices/source -p test_grader.py -v
```

The default app root is the sibling NeuralCanvas checkout; set NEURALCANVAS_ROOT when validating a different revision. Preparation uses the vendored Iris source and Assignment 1's committed data and needs no download. Preparation and pilots overwrite this assignment's files and evidence; review the diff. Do not regenerate issued datasets or change grading criteria without versioning the release. To package existing evidence, only the packaging command is needed.

## Saved-default audit (2026-10-02)

Checked supported saved defaults against Neural Canvas revision `3fa65974cb668346627b40935ce1abc0131ed104`. Tensor transform now defaults to Reshape when its transform setting is omitted, matching the app. Regression losses accept omitted MSE settings; classification still requires explicit BCE or cross-entropy (or the dedicated Cross entropy block for Iris). Omitted sigmoid activation is not accepted because the app defaults to identity. Default concatenation axis, reshape shape, transpose axes, and absent/zero penalties retain their existing handling. Fitted standardization statistics are still required.

Upload the rebuilt current Gradescope ZIP to apply these compatibility fixes. Data, thresholds, and rubrics are unchanged. Previous ZIPs are preserved in `archive/before-defaults-audit/`. Automated tests cover the omitted transform setting as well as the existing full and partial submissions. Hosted Gradescope was not exercised.

## Held-out Data inspection update (2026-10-17)

The handout now follows the inference Data workflow verified locally against clean app commit `7ded9c7064a47c1672e4e0be32d6b74cdc45258a`: original zero-based Dataset row numbers, split-wide output browsing, and per-example multidimensional tensors. Dataset Details uses **Rows / Current row**; changing that selection clears inference results, so rerun inference afterward. Custom experiments use **Run forward**, which returns inspection to the current canvas example. See [the verification and remaining limitation](../../07-word-counts/Instructor/INSTRUCTOR.md#held-out-data-inspection-update-2026-10-17). These are instruction-only updates; data, thresholds, models, and Gradescope ZIPs are unchanged. The deployed student app was not checked.

## Training defaults update (2026-10-19)

The handout now explicitly directs students to change the new Tensor default to **Trace engine · SGD**. Tensor backend and optimizer controls are hidden and inactive in this mode; the issued full-batch SGD recipe remains unchanged.

Verified the current defaults and these selections in the production UI on clean app `51e00c7c64d6b68624a7d02a246479530c88b09c`, then used File → Save and checked the serialized settings and unchanged learned parameters. This is a settings/save check, not a new training pilot; the original convergence evidence above still applies to the retained recipe. See [backend validation](../../../docs/BACKEND-VALIDATION.md) and `docs/early-training-defaults-browser.json`. Datasets, model requirements, grading thresholds, original reference models, and Gradescope ZIPs are unchanged. Only the Canvas handout and its manifest need refreshing.

## Dataset browser workflow

Students open **Dataset → Details → Browse datasets…** and click **Use this dataset** after selecting the following choices. Datasets live in the app’s `public/datasets/` directory; no assignment-local copies or Canvas attachments are needed. Read [description.md](../description.md) on GitHub or use `../description.html` in Canvas. Model JSON submissions and grading targets are unchanged.

- **Regression → Auto MPG**; Configuration: **Full sample**.

- **Classification → Iris flowers**.


Browser/grader compatibility is checked with app 417be2e and the original canonical data. See [the dataset browser update](../../../docs/DATASET-BROWSER-ASSIGNMENTS.md).
