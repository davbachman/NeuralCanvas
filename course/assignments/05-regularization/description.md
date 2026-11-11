<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 5: Regularization

Keep the flexible model from the generalization lesson and change its training objective. Compare a moderate L2 penalty with a much stronger one. Submit **regularized.json** and **strong.json**, 50 points each.

Use **Dataset → Details → Browse datasets… → Regression → Auto MPG**, with **Configuration → Small training sample**. Click **Use this dataset**. Preserve the 24 training and 80 validation rows. Use weight\_lb as the only predictor and mpg as the target; leave model\_year unused. These are original UCI Auto MPG observations, DOI 10.24432/C5859H, CC BY 4.0. The validation rows are used for model selection, not an untouched final test.

## Reuse your polynomial

Import your saved **polynomial.json**. Keep its architecture, six independent scalar Params, and fitted feature statistics. If necessary, rebuild it using the feature specification below. Keep baseline.json and polynomial.json as comparison checkpoints.

## Construct the feature family

Fit a **Standardize features** block on weight\_lb using **Fit on training rows**. Its output z uses exactly 24 training observations. Build a linear prediction b + w₁z with separate scalar Params.

For the degree-five model, retain z. For k = 2, 3, 4, 5, branch from z into an Arithmetic block computing zᵏ, then fit a separate Standardize features block to that power on the same 24 rows. Call each output vₖ, with v₁ = z. Build **prediction = b + Σ wₖvₖ** using six independent scalar Params. Connect prediction and the unchanged mpg target to MSE Loss. Do not use year, activations, or other features.

## Penalize feature coefficients

In **Loss → Details → Parameter regularization**, set **Penalty → L2 · λ/2 Σ w²**. Select exactly w₁ through w₅. **Uncheck the intercept b.** The app may initially include every Param, so inspect the selection.

The objective is **J = training MSE + (λ/2) Σ wₖ²**. The loss trace includes the penalty; Test reports and the learning curves report data MSE. Compare prediction quality using data MSE, not objective values from different penalty strengths.

Use **Dataset → Output mode → Numeric batch**, **Rows → Training**, and **Train → Training execution → Trace engine · SGD**. Use **24 examples per update**. Initialize the coefficients to small values with **Randomize parameters**. Each feature transformation stays fixed during training and evaluation. Run Test inference on Training set and Held-out test set; those reports show data MSE.

| File | Penalty strength λ | Learning rate | Epochs per run | Report loss every |
| --- | --- | --- | --- | --- |
| regularized.json | 0.1 | 0.1 | 1,000 | 50 |
| strong.json | 10 | 0.05 | 5,000 | 250 |

Start each comparison from a fresh copy of polynomial.json with small reset coefficients. Do not refit feature statistics. Save both outcomes even when the stronger penalty predicts poorly. Compare coefficient sizes, curves, and validation error with the unregularized model. A useful penalty need not make a polynomial outperform the simpler linear baseline.

## Grading

Each file: correct data 5, unchanged polynomial feature family and fitted scaling 15, correct L2 strength and coefficient selection 10, training objective 10, validation error 10. The objective must be within 0.20 of the independently calculated optimum for its specified λ. Validation MSE limits are **16 mpg²** for regularized.json and **45 mpg²** for strong.json. A poor but correctly optimized strong-penalty result is part of the experiment.

Use the generalization lesson's scalar-parameter and graph-size limits. No dropout, AdamW weight decay, extra predictors, or neural-network layers are used. Assignment version: regularization-05-v2. The previous issued regularization release is retained in Git history and private archives.

## Submit and verify

Use **File → Save** to save each requested model JSON. Import it again to check its contents, then upload the named files together to Gradescope. Files are graded independently. No written explanation, screenshot, Python program, or ZIP is submitted. The grader recomputes outputs from saved graph operations and parameters on canonical data; it ignores labels, cached metrics, and claimed training history.
