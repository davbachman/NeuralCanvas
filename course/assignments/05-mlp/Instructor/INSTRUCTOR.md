# Assignment 5: Linear models and a regression MLP

Public teaching release: see [the course guide](../../../README.md) for standalone setup, packaging, and validation. Wiring, templates, grader sources, and these notes are public. Trained reference models, browser model exports, solution-based tests, and pilot/reproduction tools mentioned below remain in the private NeuralCanvas-Coursework checkout; their paths are historical development references, not required teaching files. Test your own saved models when configuring Gradescope.

Version: mlp-05-v1. Day 5 introduces composition of linear maps, then a nonlinear hidden layer. Two saved models are graded: `linear.json` (40 points) and `mlp.json` (60 points). No written answers, starter models, or extra regularization exercise are required.

## Publishing

Paste `../description.html` into Canvas's HTML editor. Dataset attachments are unnecessary; students use the dataset browser. Upload `Gradescope/gradescope-mlp-05-v1.zip` as a 100-point Programming Assignment. Use Ubuntu 22.04 or 24.04, at least 1 GB memory, and a grading timeout of at least 60 seconds. Setup installs NumPy 1.26.4 in `/opt/mlp-grader`. Keep trained reference models private; grader source and the ZIP are public.

Test the two `reference/models/` files on hosted Gradescope before release: expect 100/100; omitting the MLP yields 40/100. The packaged runner has been exercised locally, but hosted Gradescope has not. Pushing to GitHub does not publish to Canvas or Gradescope.

## Teaching and construction

Start from Assignment 4's XW+b. On the board, expand `(XW1+b1)W2+b2` to `X(W1W2)+(b1W2+b2)`. Two affine layers can be collapsed to one; independently training a factorized model is a different optimization problem, so identical finite training trajectories are not promised. Inserting ReLU prevents this algebraic collapse.

The common input path is Dataset (Numeric batch, Training) → Concatenate (axis 1, eight original columns in order) → one Standardize features block fitted on training rows. The target is raw strength reshaped to `[-1,1]`. Both models use MSE without a loss penalty. The linear model uses W `[8,1]` and a scalar bias. The MLP uses W1 `[8,16]`, b1 `[16]`, ReLU, W2 `[16,1]`, and a scalar b2. Bias addition uses Arithmetic `x1 + x2`; equivalent two-input Add blocks and operand reversal are accepted. Matrix products keep the learned matrix in the second input. Output logits, sigmoid, and softmax are unnecessary for this regression task.

Weights use Xavier initialization; biases use zeros. Choose the initializer before pressing Enter in the Shape field. The handout supplies fixed seeds for a reproducible reference run; pilots also check other seeds. ReLU can produce dead units or different local solutions, so the numerical thresholds have margin. If shapes or scaling change, reset/reinitialize parameters and train again. Never refit on validation data. The app has no simple model plot for eight inputs; inspect Data, Reporting, and Test.

## Data provenance

I-Cheng Yeh's Concrete Compressive Strength dataset, UCI DOI 10.24432/C5PK67, CC BY 4.0. All 1,030 observations and eight predictors are retained. Ingredients are kg/m³; age is days; strength is MPa. The CSV source mirror is `https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/concrete.csv`. Direct UCI download was unavailable in this environment. The exact mirror bytes are vendored under `reference/raw-source/`; `reference/data.json` records their SHA-256 and source row indices. No source network access is needed to reproduce preparation.

There are 992 unique feature vectors. The split samples 20% of those groups with NumPy `default_rng(20261005)` and keeps identical feature vectors together, including rows with differing measured targets. Result: 828 training and 202 validation rows in source order. This avoids exact input duplicates crossing the split. The split is used for model comparison and is not an untouched final test. Standardization uses population mean/standard deviation of training observations, counting repeated observations as rows. Do not substitute the older semester pilot's scaled CSV or three-way split.

## Validation evidence

App revision: `a02dc9d89c92606aafd97893f52dd7a9d17485d0` (isolated checkout, no app source modifications). The app's CSV parser, standardizer fitter, trace SGD engine, split evaluation, and project serializer are used by `scripts/assignments/pilot-mlp.mjs`. Reference files pass the app's saved-project parser. Canonical settings: linear learning rate 0.03 / 4,000 epochs; MLP 0.003 / 6,000 epochs; both full batches of 828, no penalty, optimizer weight decay, clipping, or early stopping. Seeds 137, 211, and 307 are recorded in `reference/app-pilot.json`.

Across all three seeds, linear validation MSE is 135.91954–135.91962 and MLP validation MSE is 38.35605–45.44715. MLP training MSE ranges from 21.02747 to 29.38297. All six runs earn full credit.

The independent grader evaluates matrix operations with NumPy from canonical raw data and saved parameters, not cached predictions or losses. Its outputs agree with the app's metrics to eight decimal places. A separate least-squares check confirms that the linear reference's training MSE is within 0.01 of the optimum. ReLU-network performance is empirical, not an optimality claim.

Seed 137: linear training MSE 100.632204 and validation MSE 135.919624; MLP training MSE 21.027471 and validation MSE 38.356052. In original units these validation RMSEs are about 11.66 and 6.19 MPa. The low-level engine runs took approximately 29 seconds for the linear model and 150 seconds for the MLP on this environment; browser overhead is additional and machine-dependent. Students should allow sustained foreground training or consecutive chunks.

Browser evidence under `reference/browser/` checks imported initial graphs through short 10-epoch training runs, then imports the full engine-trained references, evaluates held-out data, and downloads File → Save projects for independent grading. This verifies the UI workflow and saved outputs, not the full 4,000/6,000-epoch browser runs or every block's construction by mouse. The handout is rendered for overflow and visual review. Standardizer fitting is exercised by the app-engine pilots. The deployed student site, Safari, and hosted Gradescope remain unverified.

## Grading and boundaries

Linear: 5 file/data + 15 model/scaling + 10 training MSE + 10 validation MSE = 40. MLP: 5 + 25 + 15 + 15 = 60. Full performance credit requires MSE ≤120 train / ≤150 validation for linear, and ≤45 train / ≤65 validation for MLP. Cutoffs are inclusive. These are MSE in MPa², not RMSE. No exact coefficients, epoch count, or minimum improvement over the student's own baseline is required.

The grader structurally checks the assigned feature path, column order, training-fitted means/scales, matrix/bias shapes, independent parameter blocks, and ReLU placement. It recomputes all predictions. It accepts renamed blocks, groups, connected Input/Target aliases, both saved-file kind names, omitted default MSE, default concat axis 1, and default Tensor transform → Reshape. It rejects target leakage, extra learned paths, missing ReLU, output activations, incorrect scaling, data changes, malformed graphs, cycles, and unsupported arithmetic. It intentionally does not accept alternative architectures, per-feature standardizer paths, or a flattened prediction/target convention; students are instructed to use the specific matrix construction. It does not certify the student's training history.

Bounds: 32 MiB per JSON, 100 blocks, 200 wires, 512 values per Param, and bounded intermediate shapes. Each model is graded in a separate process with a 15-second timeout, 12-second CPU limit, and 1 GiB address-space limit. Expressions are inspected with Python's AST; no submitted expressions or code are executed. The 14-test suite includes complete/partial packaged submissions, defaults, reference parity across seeds, undertraining, incorrect structure/scaling, changed data, and adversarial malformed inputs. Keep student files out of version control.

## Reproduction

From the coursework root:

```sh
.venv/bin/python scripts/assignments/prepare-mlp.py
NEURALCANVAS_ROOT=/path/to/verified/NeuralCanvas node scripts/assignments/pilot-mlp.mjs
.venv/bin/python scripts/assignments/package-mlp.py
.venv/bin/python -m unittest discover -s assignments/05-mlp/source -p test_grader.py -v
```

Preparation and pilots overwrite current fixtures. Existing fixtures are enough for packaging and grading tests; do not regenerate an issued dataset or change thresholds without versioning the release. Pilot environment overrides (`PILOT_SEED`, `PILOT_RATE`, `PILOT_EPOCHS`) are for experiments; reproduction of the issued recipe uses the defaults. Single-seed experiments write a separate report rather than replacing the combined evidence.

## Held-out Data inspection update (2026-10-17)

The handout now follows the inference Data workflow verified locally against clean app commit `7ded9c7064a47c1672e4e0be32d6b74cdc45258a`: original zero-based Dataset row numbers, split-wide output browsing, and per-example multidimensional tensors. Dataset Details uses **Rows / Current row**; changing that selection clears inference results, so rerun inference afterward. Custom experiments use **Run forward**, which returns inspection to the current canvas example. See [the verification and remaining limitation](../../07-word-counts/Instructor/INSTRUCTOR.md#held-out-data-inspection-update-2026-10-17). These are instruction-only updates; data, thresholds, models, and Gradescope ZIPs are unchanged. The deployed student app was not checked.

## Training defaults update (2026-10-19)

The handout now explicitly directs students to change the new Tensor default to **Trace engine · SGD**. Tensor backend and optimizer controls are hidden and inactive in this mode; the issued full-batch SGD recipe remains unchanged.

Verified the current defaults and these selections in the production UI on clean app `51e00c7c64d6b68624a7d02a246479530c88b09c`, then used File → Save and checked the serialized settings and unchanged learned parameters. This is a settings/save check, not a new training pilot; the original convergence evidence above still applies to the retained recipe. See [backend validation](../../../docs/BACKEND-VALIDATION.md) and `docs/early-training-defaults-browser.json`. Datasets, model requirements, grading thresholds, original reference models, and Gradescope ZIPs are unchanged. Only the Canvas handout and its manifest need refreshing.

## Dataset browser workflow

Students open **Dataset → Details → Browse datasets…** and click **Use this dataset** after selecting the following choices. Datasets live in the app’s `public/datasets/` directory; no assignment-local copies or Canvas attachments are needed. Read [description.md](../description.md) on GitHub or use `../description.html` in Canvas. Model JSON submissions and grading targets are unchanged.

- **Regression → Concrete strength**.


Browser/grader compatibility is checked with app 417be2e and the original canonical data. See [the dataset browser update](../../../docs/DATASET-BROWSER-ASSIGNMENTS.md).
