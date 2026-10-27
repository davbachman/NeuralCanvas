<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 2: When a better fit makes worse predictions

In [Neural Canvas](https://davbachman.github.io/NeuralCanvas/), build three models to investigate overfitting and L2 regularization. Start with a linear model, increase its flexibility, then penalize large coefficients. You will use validation data to judge whether a better training fit is useful.

**Submit three saved models to Gradescope:** `baseline.json`, `polynomial.json`, and `regularized.json`. Total: 100 points. No written answers or comprehension-question file is required.

## Data and the experiment

Use the **Auto MPG → Small training sample** configuration in the dataset browser. It contains 104 real cars from the [UCI Auto MPG dataset](https://archive.ics.uci.edu/dataset/9/auto%2Bmpg): **24 training rows and 80 validation rows**. This is a deliberately small training subset, chosen reproducibly to make the effect of model flexibility visible. It uses fewer training rows than the previous assignment’s Full sample; the 80 validation rows are the same.

Use `weight_lb` as your only input and `mpg` as the regression target. Leave `model_year` unused. Keep all rows, columns, and split labels unchanged. The app calls the other split “Held-out test set”; here it is **validation data**, because you use it to select a model. It is not an untouched final test set.

## Part A: Establish a linear baseline — 30 points

1. Start with a blank canvas and add a Dataset block.

   Select your Dataset block and open **Details → Browse datasets…**. Choose **Regression** → **Auto MPG**. Set **Configuration** to **Small training sample**. Click **Use this dataset**. Keep the supplied data and split unchanged.

   Confirm Regression and `mpg` as the target.
2. Connect `weight_lb` to **Standardize features**. Select the block, open Details, and click **Fit on training rows**. Confirm **24 training observations**. Call its output **z**. It subtracts the training mean weight and divides by the training standard deviation.
3. Build **prediction = b + w₁z**, with two separate scalar Param blocks. Give them useful names, such as “intercept b” and “weight coefficient w1.” Connect prediction and actual mpg to an MSE Loss block, with **Penalty: None**. Keep the target in mpg.
4. Train with the settings below. In Test, run inference separately on Training set and Held-out test set. Compare their MSE values and inspect the prediction line in Reporting.
5. Use **File → Save** and name the downloaded file `baseline.json`.

## Part B: Increase flexibility — 30 points

Keep your Part A file. Extend a copy of the model to include powers through degree five. Keep **z** as the first feature. For each k = 2, 3, 4, 5:

1. Branch from **z** to an Arithmetic block with expression `x1 ^ k`, replacing k with the actual integer. For example, use `x1 ^ 5` for the fifth power. Each power starts from z; do not repeatedly square the previous branch.
2. Connect that power to its own **Standardize features** block. Fit it on the 24 training rows. Call the output **vₖ**.
3. Multiply vₖ by a new scalar Param wₖ, and add the term to your prediction.

Your completed model is **prediction = b + w₁z + w₂v₂ + w₃v₃ + w₄v₄ + w₅v₅**, with six independently trainable scalar coefficients. Each vₖ is the standardized version of zᵏ. Centering and scaling each power preserves the polynomial family and puts the coefficient penalty on comparable feature scales. It does not make the features independent.

Use **Penalty: None**. Start with small coefficients using Randomize parameters, train, and save `polynomial.json`. This model needs substantially more epochs than the linear baseline because its power features are correlated. Continue the prescribed training even if validation performance gets worse: this part deliberately studies a well-fitted, unregularized model.

Compare the training and validation MSE with Part A, and look at the curve near the lightest and heaviest cars. A lower training error is not enough to choose a model.

## Part C: Add L2 regularization — 40 points

Keep the same fifth-degree architecture and fitted feature statistics. Do not remove power terms. Select the Loss block and open Details:

1. Set **Penalty** to **L2 · λ/2 Σ w²**.
2. Check exactly the five feature coefficients w₁ through w₅. **Uncheck the intercept b.** The app may initially include every Param, so inspect this list explicitly.
3. Try strengths **0.001, 0.01, 0.1, 1, and 10**. Reset the coefficients to the same small starting values for each comparison, then train with the settings below. Do not refit feature statistics between runs.
4. Compare validation MSE and curve shape. Observe both what a moderate penalty does and what happens when the penalty is too strong. These trials are for exploration; you only submit your chosen regularized model.
5. Choose a strength using validation MSE, retrain if needed, and save `regularized.json`. A useful starting candidate is **λ = 0.1**. Other positive strengths up to 10 are accepted if the resulting model meets the grading criteria.

The training objective is **J = training MSE + (λ/2)(w₁² + w₂² + w₃² + w₄² + w₅²)**. The Loss block includes this penalty. Dataset evaluation and the reporting curves use **data MSE alone**. Compare models using validation MSE, not the penalized Loss value. Increasing λ changes the objective; its value is not directly comparable across strengths.

## Training settings

Use Dataset output mode **Numeric batch** and Rows **Training**. In **Train → Training execution**, change the default Tensor engine to **Trace engine · SGD**. Use **24 examples per update** and report every **500 epochs**. These are full-batch updates. Leave regularization off except in Part C; do not add dropout in this experiment.

**Settings that worked in the instructor pilot**

| Model / strength | Learning rate | Epochs |
| --- | --- | --- |
| Linear baseline | 0.1 | 1,000 |
| Fifth degree, no penalty | 0.1 | 10,000 |
| Fifth degree, λ = 0.001 | 0.1 | 10,000 |
| Fifth degree, λ = 0.01 | 0.1 | 5,000 |
| Fifth degree, λ = 0.1 or 1 | 0.1 | 1,000 |
| Fifth degree, λ = 10 | **0.05** | 5,000 |

Allow several minutes for the longest runs in the browser; keep the app tab in the foreground. You can split long training runs into smaller runs whose epochs add to the suggested total. If loss becomes nonfinite, stop, check that every power feature has been standardized on training rows, reset to small coefficients, and reduce the learning rate. WebGL is unnecessary for these small models.

## What to notice

Look for a model whose training error decreases while validation error increases, and for a penalty that reverses some of that validation damage. A penalty can also be too strong. The regularized fifth-degree model does not have to beat the simple linear baseline: the goal is to control an overly flexible model, not to guarantee that complexity wins. These observations are tied to this dataset and split. You do not submit written responses.

## Grading

Parts A and B each award 5 points for the saved file and dataset, 15 for the assigned model and fitted feature scaling, 5 for training fit, and 5 for validation MSE. Part C adds 10 points for the correct L2 configuration, for 40 points total. Each part is graded independently, and you may resubmit.

**Full-credit fit criteria (inclusive)**

| Model | Training objective | Validation MSE ≤ |
| --- | --- | --- |
| Linear | Within 0.20 of minimum MSE 22.01439 | 14 mpg² |
| Fifth degree, no penalty | Within 0.20 of minimum MSE 17.10578 | 60 mpg² |
| Fifth degree with L2 | Within 0.20 of the minimum J for your λ; at λ = 0.1 the minimum is 21.86221 | 16 mpg² |

The grader recomputes the exact reference minimum and predictions independently from the data and saved coefficients. The reference numbers shown above are rounded. Gradescope feedback shows your objective and its reference. Part B deliberately has a generous validation bound: you are not penalized for the overfitting this experiment is designed to reveal. Performance credit requires the assigned model and penalty configuration.

Use at most 100 blocks and 200 wires. Use separate scalar Params, with one coefficient per standardized feature and one intercept. Allowed operations are Dataset, Input, Target, Param, Arithmetic, Standardize features, MSE Loss, and equivalent small tensor operations (Matrix product, Concatenate, Reshape, Transpose). Each tensor must have at most 64 values. Merged groups and renamed blocks are accepted. Do not add other predictors, activations, or higher-degree terms.

## Submission checklist

- Upload `baseline.json`, `polynomial.json`, and `regularized.json` together; do not upload a ZIP or screenshots.
- Use File → Import to check each saved model. All must contain this assignment’s unchanged dataset and split.
- Check that Part C penalizes the five feature coefficients, excludes the intercept, and saves your chosen λ.
- Review Gradescope feedback and resubmit before the Canvas deadline.

Assignment version: regularization-02-v1. Source: R. Quinlan, Auto MPG, UCI Machine Learning Repository, DOI 10.24432/C5859H; CC BY 4.0. This file retains the original measurements, expands model year to four digits, and supplies a reproducible teaching subset and split.
