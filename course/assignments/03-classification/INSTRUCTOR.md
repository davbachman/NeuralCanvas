# Assignment 3: Binary classification

Public teaching release: see [the course guide](../../README.md) for standalone setup, packaging, and validation. Wiring, templates, grader sources, and these notes are public. Trained reference models, browser model exports, solution-based tests, and pilot/reproduction tools mentioned below remain in the private NeuralCanvas-Coursework checkout; their paths are historical development references, not required teaching files. Test your own saved models when configuring Gradescope.

Day 3 covers sigmoid, binary cross-entropy (BCE), and decision thresholds. Students build one- and two-feature logistic classifiers using the Wisconsin Diagnostic Breast Cancer dataset. No hidden layer, regularization, or matrix-authoring prerequisite is needed. Two saved models are graded automatically, 50 points each; threshold exploration has no separate written submission.

## Files to publish

All files for this assignment live here, under `assignments/03-classification/`:

1. Paste [Canvas/description.html](Canvas/description.html) into Canvas's HTML editor. Set the due date and Gradescope link there.
2. Dataset attachments are unnecessary; students use the dataset browser as described in the handout.
3. Upload [Gradescope/gradescope-classification-03-v1.zip](Gradescope/gradescope-classification-03-v1.zip) to the Gradescope autograder configuration for a 100-point Programming Assignment.

Only `Canvas/` contains student-facing files. `Gradescope/` is for instructor upload to Gradescope, not distribution to students. `source/` holds the public grader implementation; `reference/` holds public canonical grading data and package hashes. Trained models, solution-dependent tests, validation exports, and historical `archive/` releases remain in the private Coursework repository. There is no current student ZIP or duplicate handout to maintain.

Rebuild the current autograder with the packaging command below. A GitHub push updates this repository; it does not publish to Canvas or Gradescope.

Use Ubuntu 22.04. Setup installs Python and NumPy 1.26.4 in a virtual environment. Students submit `one_feature.json` and `two_features.json`. Test those files from `reference/models/`: expect 100/100. Partial submissions retain the available part’s credit. The packaged runner is tested locally; hosted Gradescope remains unverified.

## Learning progression

Part A feeds a familiar standardized linear score directly into Binary cross entropy (logits), which includes sigmoid internally. Students also add a separate Sigmoid branch to inspect probabilities in Data and keep it in the saved model. Part B adds texture to radius with the same two branches. The probability inspection branch is required by the instructions but has no separate rubric item; the grader validates the training path. Both models predict **P(malignant)**: 0=benign, 1=malignant. This deliberately reverses scikit-learn's original 0=malignant/1=benign encoding; numeric labels in the distributed CSV make the convention explicit and match the app's ascending label order.

The threshold exercise separates training probabilities from subsequent decisions. The app's evaluation report always classifies at 0.5; there is no threshold selector. The grader reports validation confusion counts at 0.25, 0.5, and 0.75 without changing coefficients or probabilities. Students first classify hypothetical probabilities by hand, then examine these counts. There is no written response and threshold reasoning is not separately assessed by the automated score; discuss it in class.

At thresholds 0.25, 0.5, and 0.75, the hypothetical probabilities [0.20, 0.40, 0.60, 0.80] produce [0,1,1,1], [0,0,1,1], and [0,0,0,1], respectively. A lower threshold reduces or preserves false negatives and increases or preserves false positives. BCE stays fixed. These tradeoffs do not justify a clinical threshold; this is an educational dataset exercise.

## Visualization and reporting wiring

Connect each used Dataset feature directly to Standardize features, and malignant directly to the second Loss input. App commit `e4a6859e19672bd3059e5f9381d9ac5cb4f3e9ee` infers feature ports by tracing the prediction path into Loss, excluding target ports, and creates temporary aliases internally for visualization. Students do not need Input or Target blocks. Existing models with passthrough blocks remain valid.

Checked that revision in an isolated clean app checkout: all 13 tests in `datasetInputs.test.ts` and `VisualizationPanel.test.tsx` passed. Both original Assignment 3 reference models, without Input/Target blocks, rendered plots and held-out readouts through the app's server-rendered VisualizationPanel. Their validation BCE remained 0.2613671209503326 and 0.22985307302838365. This checks the source revision and component behavior; deployment to the student site and interactive browser training were not verified.

Reporting shows the model plot and its training/held-out readouts. The separate Loss by epoch history reports at epoch 0 and every 500 epochs with this recipe. Test → Held-out test set → Run inference provides a separate validation evaluation. The earlier automatic-input instruction correction did not change the grader. The logits-loss update below requires replacing both the Canvas description and Gradescope ZIP; dataset and rubric remain unchanged.

## Data and provenance

The source is the 569-row Wisconsin Diagnostic Breast Cancer dataset, UCI DOI 10.24432/C5DW2B, CC BY 4.0. `reference/raw-source/breast_cancer.csv` and its description were extracted from the scikit-learn 1.5.2 wheel downloaded from PyPI. The canonical data JSON records the source CSV SHA-256, source indices, seed, labels, and split. Only the original mean radius and mean texture are distributed as inputs; no precomputed scaling or answer coefficients appear in student files.

NumPy `default_rng(20261003)` samples 20% of each class (rounded per class) for validation without replacement, preserving source order in the final CSV. There are 456 training observations (286 benign, 170 malignant) and 113 validation observations (71 benign, 42 malignant). There is no separate hidden test. This split differs from the older semester pilot, which used all 30 inputs and different splits; do not substitute that pilot's numerical results.

Standardizers use the population mean and standard deviation fitted on the 456 training observations. Raw texture and radius remain in the CSV. Do not fit on validation rows or normalize the labels.

## Verified recipe and evidence

Current app revision: `874de31b38523ce3f55ea63ade044651af8ac360` (isolated checkout; no app source modifications). Original probability-BCE evidence used `9c0608155c867be241072a2c10313f9cd7913759` and is preserved in `archive/before-logits-loss/`.

The app's actual trace engine, standardizer fitting, CSV parser, evaluator, and File → Save serialization were exercised through the Node pilot. For this revision, the linear score connects directly to `binary-cross-entropy-with-logits`; a separate sigmoid branch is retained for probability inspection. Each model was trained from seeds 137, 211, and 307 using learning rate 0.1, SGD, a full batch of 456 training examples, 2,000 epochs, and no penalty. Every serialized model parses successfully in the app. All three seeds pass the grader; independently recomputed losses and accuracy agree with the app within 1e-9.

| Model | Training BCE | Validation BCE | Validation accuracy at 0.5 |
|---|---:|---:|---:|
| One feature, seed 137 | 0.297176 | 0.261367 | 90.27% (102/113) |
| Two features, seed 137 | 0.262397 | 0.229853 | 89.38% (101/113) |

The two-feature model improves BCE but does not improve threshold accuracy on this split. Do not promise that both metrics improve. Near-boundary examples can change accuracy by one observation while BCE barely changes. The independent Newton solutions give training BCE 0.297157 and 0.262337; the finite app runs are sufficiently optimized. `reference.json` is mathematical evidence, not the source of saved trained parameters.

For the two-feature reference model, the threshold feedback is:

| Threshold | True positives | False positives | True negatives | False negatives |
|---|---:|---:|---:|---:|
| 0.25 | 38 | 10 | 61 | 4 |
| 0.50 | 34 | 4 | 67 | 8 |
| 0.75 | 31 | 0 | 71 | 11 |

The low-level engine runs took approximately 4–6 seconds per model here. Browser scheduling/rendering overhead and student hardware are not included; this is not a promised classroom runtime. Report every 500 epochs and allow consecutive training chunks. Browser checks for the logits update cover short 10-epoch UI training from initial parameters, importing the full engine-trained references, held-out inference, the separate Sigmoid Data view, and File → Save. The complete 2,000-epoch run was verified in the engine, not through the browser UI. Hosted Gradescope and the deployed student site remain unverified.

## Grading and limits

Each part: 5 file/data + 25 model/scaling + 10 training BCE + 5 validation BCE + 5 accuracy = 50. Part A accepts training BCE ≤0.31 and validation BCE ≤0.29; Part B accepts ≤0.28 and ≤0.26. Both require accuracy ≥87% at threshold 0.5. These bounds have margin around all three pilot seeds. The rubric does not require reproducing exact coefficients or an epoch count.

The evaluator parses restricted arithmetic into polynomials to check the linear logit, verifies each independently trainable scalar parameter's standardized feature direction, accepts a raw linear score into logits BCE or a final sigmoid directly into explicitly selected legacy probability BCE, checks the original dataset/labels/splits, and rejects target leakage even if canceled. It recomputes sigmoid probabilities, decisions, and confusion counts independently. Logits BCE uses the stable expression max(z,0) − yz + log1p(exp(−abs(z))); legacy probability BCE retains the app's 1e-7 probability clipping. Sigmoid into logits BCE and raw scores into legacy probability BCE are rejected. Node names, visual groups, cached outputs, and reported losses are not trusted. Separate scalar Params are required; equivalent small tensor wiring is accepted. Standardization may be algebraically equivalent to the fitted transform, but parameter units must match the required standardized basis.

The grader does not execute student code or prove an optimizer history. Limits: 4 MB per file, 100 blocks, 200 wires, 64 values per tensor, a separate 15-second process per part with Linux CPU and memory limits. Models with invalid form retain only file/data credit. Threshold feedback is informational, not a third graded part.

The automated suite covers every seed, independent app metric agreement, undertraining, incorrect loss/activation/penalty, incorrect feature family/scaling, target leakage, altered data/labels/splits, renamed/grouped models, forged cached results, malformed files, cycles, unsafe expressions, confusion counts including ties, and the actual packaged runner with full and partial submissions. The Canvas folder is checked to contain only the handout and CSV.

## Reproduction

Run from the coursework repository with its NumPy environment and the companion app's installed Node dependencies:

```sh
.venv/bin/python scripts/assignments/prepare-classification.py
node scripts/assignments/pilot-classification.mjs
.venv/bin/python scripts/assignments/package-classification.py
.venv/bin/python -m unittest discover -s assignments/03-classification/source -p test_grader.py -v
```

Preparation reads the vendored source and needs no download or scikit-learn installation. Pilots read the companion app and write only coursework artifacts. Regeneration overwrites this version's evidence and models; after issuing an assignment, version any changes to data, split, or rubric. To package existing artifacts, run only the packaging command; new training is unnecessary.

## Saved-default audit (2026-10-02)

Checked supported saved defaults against Neural Canvas revision `3fa65974cb668346627b40935ce1abc0131ed104`. Tensor transform now defaults to Reshape when its transform setting is omitted, matching the app. Regression losses accept omitted MSE settings; classification still requires explicit BCE or cross-entropy (or the dedicated Cross entropy block for Iris). Omitted sigmoid activation is not accepted because the app defaults to identity. Default concatenation axis, reshape shape, transpose axes, and absent/zero penalties retain their existing handling. Fitted standardization statistics are still required.

Upload the rebuilt current Gradescope ZIP to apply these compatibility fixes. Data, thresholds, and rubrics are unchanged. Previous ZIPs are preserved in `archive/before-defaults-audit/`. Automated tests cover the omitted transform setting as well as the existing full and partial submissions. Hosted Gradescope was not exercised.

## Logits-loss compatibility update (2026-10-07)

Confirmed the new loss menu label **Binary cross entropy (logits)**, stable forward computation, sigmoid-based gradient, and probability conversion for dataset evaluation in app commit `874de31b38523ce3f55ea63ade044651af8ac360`. All four app `binaryLogits.test.ts` tests passed. The old `binary-cross-entropy` serialized setting still means probabilities and is offered as a legacy option when loading those models; it was not silently reinterpreted.

Both student models now have two branches from the linear score: one directly into Loss, and one into Sigmoid for viewing probabilities. The student instructions explicitly require retaining the Sigmoid branch and inspecting its Data tab. The target remains the unchanged malignant column. Do not send the Sigmoid output into the logits loss.

Rebuilt the grader and reference models for the new loss while preserving support for issued legacy models. The 15-test grader suite covers both formats, incorrect pairings, extreme logits without probability clipping, unchanged legacy scores, all six new app pilots, and full/partial packaged submissions. Thresholds, data, and point allocations remain unchanged, so the assignment retains version classification-03-v1. Prior handout, ZIP, models, and evidence are archived under `archive/before-logits-loss/`.

Replace both the Canvas description and current Gradescope ZIP to apply this update. A repository push alone does not update either service. Browser evidence for this revision is under `reference/browser-logits/`.

## Held-out Data inspection update (2026-10-17)

The handout now follows the inference Data workflow verified locally against clean app commit `7ded9c7064a47c1672e4e0be32d6b74cdc45258a`: original zero-based Dataset row numbers, split-wide output browsing, and per-example multidimensional tensors. Dataset Details uses **Rows / Current row**; changing that selection clears inference results, so rerun inference afterward. Custom experiments use **Run forward**, which returns inspection to the current canvas example. See [the verification and remaining limitation](../07-word-counts/INSTRUCTOR.md#held-out-data-inspection-update-2026-10-17). These are instruction-only updates; data, thresholds, models, and Gradescope ZIPs are unchanged. The deployed student app was not checked.

## Training defaults update (2026-10-19)

The handout now explicitly directs students to change the new Tensor default to **Trace engine · SGD**. Tensor backend and optimizer controls are hidden and inactive in this mode; the issued full-batch SGD recipe remains unchanged.

Verified the current defaults and these selections in the production UI on clean app `51e00c7c64d6b68624a7d02a246479530c88b09c`, then used File → Save and checked the serialized settings and unchanged learned parameters. This is a settings/save check, not a new training pilot; the original convergence evidence above still applies to the retained recipe. See [backend validation](../../docs/BACKEND-VALIDATION.md) and `docs/early-training-defaults-browser.json`. Datasets, model requirements, grading thresholds, original reference models, and Gradescope ZIPs are unchanged. Only the Canvas handout and its manifest need refreshing.

## Dataset browser workflow

Students open **Dataset → Details → Browse datasets…** and click **Use this dataset** after selecting the following choices. Dataset files retained beside the handout are canonical provenance/compatibility copies; they do not need to be attached in Canvas. Model JSON submissions and grading targets are unchanged.

- **Classification → Breast cancer measurements**.


Browser/grader compatibility is checked with app 417be2e and the original canonical data. See [the dataset browser update](../../docs/DATASET-BROWSER-ASSIGNMENTS.md).
