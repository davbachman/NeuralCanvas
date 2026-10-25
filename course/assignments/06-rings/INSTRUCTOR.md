# Assignment 6: When a straight boundary is not enough

Public teaching release: see [the course guide](../../README.md) for standalone setup, packaging, and validation. Wiring, templates, grader sources, and these notes are public. Trained reference models, browser model exports, solution-based tests, and pilot/reproduction tools mentioned below remain in the private NeuralCanvas-Coursework checkout; their paths are historical development references, not required teaching files. Test your own saved models when configuring Gradescope.

Version **rings-06-v3**, replacing binary rings-06-v2; the original release was digits-06-v1. This is a consolidation exercise, not the introduction of a new model family. Students combine classification from Assignments 3–4 with the ReLU MLP from Assignment 5, then explain the difference between their decision boundaries. Familiar setup and parameter-shape steps are intentionally abbreviated. Two saved models are graded: `linear.json` (40 points) and `mlp.json` (60 points). No written submission is required.

## Publishing and prior release

Paste `Canvas/description.html` into Canvas. Dataset attachments are unnecessary; students use the dataset browser. Upload `Gradescope/gradescope-rings-06-v3.zip` as the autograder for a 100-point Programming Assignment, using Ubuntu 22.04 or 24.04, at least 1 GB memory, and at least a 60-second timeout. Setup installs NumPy 1.26.4 into `/opt/rings-grader`. Test the two `reference/models/` files on hosted Gradescope before issuing (expect 100/100); one linear file alone should score 40/100. Hosted Gradescope and the deployed student site have not been verified. A GitHub push does not publish to Canvas or Gradescope.

This is an explicit dataset/rubric version change. Previously saved digit or binary-ring models will not pass the new three-class grader; preserve any existing assignment/submission history separately before replacing its deployed grader. The complete binary-ring release is archived at `archive/rings-06-v2/`. The complete earlier release, including handout, grader ZIP, models, tests, data, and browser evidence, is preserved at `archive/digits-06-v1/`. Its archived scripts/documentation retain historical paths. The new handout includes an optional, ungraded digits extension; it does not ask students to submit digit models.

## Teaching focus and construction

The two coordinates are inputs; labels identify the lower outer semicircle (0), upper outer semicircle (1), and full inner ring (2). A three-score affine classifier has straight pairwise decision boundaries because differences of scores are affine. It can distinguish upper from lower but cannot cleanly isolate the inner ring: that class region is the intersection of only two half-planes. More training or another affine layer does not remove this restriction. ReLU changes this: a hidden layer can create a piecewise linear boundary enclosing the center. Inspect both decision regions in Reporting and compare held-out predictions. Do not imply that MLPs always outperform linear models on real data; this synthetic example deliberately isolates nonlinear representational capacity.

Select **Classification → Concentric rings** in the dataset browser; it supplies **Multiclass classification** and target `class`, Numeric batch mode, and Training examples. Concatenate raw x1 then x2 on axis 1. The inputs have comparable scale, so no standardization is used. Disallow squared features, radius, target leakage, or other feature engineering: those would solve a different exercise.

Linear: W `[2,3]`, b `[3]`, logits = XW+b. MLP: W1 `[2,16]`, b1 `[16]`, ReLU, W2 `[16,3]`, b2 `[3]`. Each matrix/bias uses its own Param. Use Xavier matrices and zero bias vectors. Arithmetic bias addition is `x1 + x2`; the grader also accepts a two-input Add and reversed addition operands. The student handout supplies output/hidden widths but asks students to work out parameter shapes.

Both models use **Cross entropy (logits)** with three scores, not one-score binary cross-entropy. Connect the Dataset's integer class labels directly to Loss; no target reshape or one-hot block. Keep a separate Softmax branch for viewing probabilities. Softmax is required by the instructions as an inspection tool, but has no separate grading item; the grader validates the training path. Send original logits, not Softmax probabilities, into Loss.

Use Tensor engine · minibatches, Tensor CPU, SGD, learning rate 0.1, 32 examples/update, reshuffle each epoch, 100 epochs, reporting every 10. Disable loss penalties, weight decay, gradient clipping, and early stopping. The held-out set is used as validation for model comparison, not an untouched final test.

## Data generation and provenance

Synthetic data generated for this course, not measurements of a real population. `scripts/assignments/prepare-rings.mjs` uses seed 20261007 and the unsigned 32-bit LCG recurrence `1664525*s + 1013904223`; uniforms are `(s+0.5)/2^32`. Cycle through classes 0, 1, 2. Draw the angle uniformly over [−π,0) for class 0, [0,π) for class 1, or [0,2π) for class 2. Add independent Box–Muller radial noise with standard deviation 0.12 to radius 2 for classes 0/1 or radius 1 for class 2, then convert to Cartesian coordinates. First 600 examples are training (200/class); final 300 are validation (100/class). No validation-fitted preprocessing or selection of easy held-out examples occurs.

`Canvas/rings.csv` contains x1, x2, class, split. `reference/data.json` freezes coordinates, labels, source indices, generation recipe, seed, and CSV SHA-256. Generation is deterministic and offline; do not regenerate an issued dataset from a different seed. Tests check canonical class counts, the CSV hash, parameter-to-metric agreement, and rejection of changed CSV values or splits.

## Rubric and rationale

| Model | File/data | Model form | Training CE | Held-out CE | Held-out accuracy |
|---|---:|---:|---:|---:|---:|
| Linear | 5 | 20 | 10 (≤0.56) | 5 (≤0.57) | Informational only |
| MLP | 5 | 25 | 10 (≤0.08) | 10 (≤0.10) | 10 (≥97%) |

CE is mean natural-log cross-entropy. Cutoffs are inclusive. Performance points require correct model form. Linear accuracy is intentionally ungraded: limited accuracy is the phenomenon being studied, not a student error. On the balanced three-class data, uniform predictions have CE log(3) ≈ 1.099 and do not satisfy the trained-baseline thresholds. A zero-parameter baseline retains file/form credit (25/40), while a zero-parameter MLP retains 30/60. The grader does not claim to prove that training happened or enforce a particular coefficient vector, epoch count, or optimizer. A zero-initialized MLP remains at chance and retains only architecture/file credit.

The independent NumPy evaluator recomputes matrix products, bias additions, ReLU, stable cross-entropy, and argmax decisions from canonical inputs and saved parameters. It verifies raw feature order, exact prescribed widths and bias shapes, distinct Params, original labels and explicit splits. It ignores cached predictions and losses, labels, and visual groups. Passthrough aliases, legacy/current save-format kind strings, omitted default concat axis, and dedicated Cross entropy nodes are accepted. Wrong activations, engineered features, output Softmax into Loss, incorrect target wiring, cycles, unsafe expressions, and malformed or oversized inputs are rejected. Limits: 32 MiB/file, 100 nodes, 200 edges, 4,096 values/Param; runner bounds time and memory per independent part.

## Validation

App revision `9a475583dcf1071fcc412c755f742041c6ec9034`, isolated unmodified checkout. The shared app repository was not changed. `pilot-rings.mjs` uses the actual CSV parser, builder/initializer, tensor training engine, dataset evaluator, and saved-project serializer/parser. It verifies three initializations, seeds 137, 211, 307 (successive Params increment the initialization seed). All six runs use the exact prescribed 100-epoch recipe and pass the independent grader.

| Model | Training CE | Validation CE | Validation accuracy |
|---|---|---|---|
| Linear, three seeds | 0.524927–0.524929 | 0.532615–0.532617 | 68.33% |
| MLP, three seeds | 0.023690–0.026396 | 0.020682–0.023421 | 100% |

The 68.33% is an empirical outcome on this finite split, not a theoretical maximum over linear classifiers. Engine-only runs took approximately four to five seconds per model here; UI overhead and student hardware add time. Independent grader metrics agree with app evaluation to eight decimal places.

`reference/browser/` preserves full 100-epoch CPU browser training from initial seed-137 graphs, Reporting decision regions, held-out inference, File → Save downloads, and independent grading of those downloads. This tests the training/save workflow, not construction of every block by mouse. The full browser runs took 80.7 seconds (linear) and 104.8 seconds (MLP) here and reproduced 68.33% and 100% accuracy; these are observations, not runtime guarantees. The development browser emits ResizeObserver layout warnings; verify completion and results rather than interpreting those warnings as training failures. The handout is rendered and reviewed for overflow. Other browsers, the deployed app, and hosted Gradescope remain unverified.

## Reproduce

```sh
node scripts/assignments/prepare-rings.mjs
NEURALCANVAS_ROOT=/path/to/NeuralCanvas node scripts/assignments/pilot-rings.mjs
.venv/bin/python scripts/assignments/package-rings.py
.venv/bin/python -m unittest discover -s assignments/06-rings/source -p test_grader.py -v
```

Run from the coursework repository. Preparation and pilots overwrite their outputs; inspect changes before replacing issued evidence. Existing committed fixtures are enough to package and test; new training is unnecessary. Browser evidence is preserved separately from engine pilot regeneration.

## Held-out Data inspection update (2026-10-17)

The handout now follows the inference Data workflow verified locally against clean app commit `7ded9c7064a47c1672e4e0be32d6b74cdc45258a`: original zero-based Dataset row numbers, split-wide output browsing, and per-example multidimensional tensors. Dataset Details uses **Rows / Current row**; changing that selection clears inference results, so rerun inference afterward. Custom experiments use **Run forward**, which returns inspection to the current canvas example. See [the verification and remaining limitation](../07-word-counts/INSTRUCTOR.md#held-out-data-inspection-update-2026-10-17). These are instruction-only updates; data, thresholds, models, and Gradescope ZIPs are unchanged. The deployed student app was not checked.

## Training defaults update (2026-10-19)

The handout retains **Tensor engine · minibatches**, but explicitly overrides Auto with **Tensor CPU**, Adam with **SGD**, and gradient norm limit 1 with **0**. Early-stopping patience and minimum validation improvement remain 0. The issued 100-epoch recipe remains unchanged; this update does not extend Assignment 6 to WebGL.

Verified the current defaults and these selections in the production UI on clean app `51e00c7c64d6b68624a7d02a246479530c88b09c`, then used File → Save and checked the serialized settings and unchanged learned parameters. This is a settings/save check, not a new training pilot; the original convergence evidence above still applies to the retained recipe. See [backend validation](../../docs/BACKEND-VALIDATION.md) and `docs/early-training-defaults-browser.json`. Datasets, model requirements, grading thresholds, original reference models, and Gradescope ZIPs are unchanged. Only the Canvas handout and its manifest need refreshing.

## Dataset browser workflow

Students open **Dataset → Details → Browse datasets…** and click **Use this dataset** after selecting the following choices. Dataset files retained beside the handout are canonical provenance/compatibility copies; they do not need to be attached in Canvas. Model JSON submissions and grading targets are unchanged.

- **Classification → Concentric rings**.

- **Images → Handwritten digits** (optional extension only).


Browser/grader compatibility is checked with app 417be2e and the original canonical data. See [the dataset browser update](../../docs/DATASET-BROWSER-ASSIGNMENTS.md).
