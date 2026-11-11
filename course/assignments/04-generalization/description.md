<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 4: Model flexibility and generalization

Can a model fit its training observations better while predicting new observations worse? Build a linear baseline and a more flexible polynomial on a small real training sample. Submit **baseline.json** and **polynomial.json**, 50 points each.

Use **Dataset → Details → Browse datasets… → Regression → Auto MPG**, with **Configuration → Small training sample**. Click **Use this dataset**. Preserve the 24 training and 80 validation rows. Use weight\_lb as the only predictor and mpg as the target; leave model\_year unused. These are original UCI Auto MPG observations, DOI 10.24432/C5859H, CC BY 4.0. The validation rows are used for model selection, not an untouched final test.

## Construct the feature family

Fit a **Standardize features** block on weight\_lb using **Fit on training rows**. Its output z uses exactly 24 training observations. Build a linear prediction b + w₁z with separate scalar Params.

For the degree-five model, retain z. For k = 2, 3, 4, 5, branch from z into an Arithmetic block computing zᵏ, then fit a separate Standardize features block to that power on the same 24 rows. Call each output vₖ, with v₁ = z. Build **prediction = b + Σ wₖvₖ** using six independent scalar Params. Connect prediction and the unchanged mpg target to MSE Loss. Do not use year, activations, or other features.

Use **Dataset → Output mode → Numeric batch**, **Rows → Training**, and **Train → Training execution → Trace engine · SGD**. Use **24 examples per update**. Initialize the coefficients to small values with **Randomize parameters**. Each feature transformation stays fixed during training and evaluation. Run Test inference on Training set and Held-out test set; those reports show data MSE.

| Model | Learning rate | Epochs per run | Report loss every |
| --- | --- | --- | --- |
| baseline.json, linear | 0.1 | 1,000 | 50 |
| polynomial.json, degree five | 0.1 | 10,000 | 500 |

Keep **Loss → Details → Penalty → None** for both. The polynomial's correlated features need more updates; training longer here is part of exposing the generalization problem. You may split the long run into consecutive runs without resetting. The learning curves should include intermediate reports.

Save both files and compare their prediction curves and errors. Preserve polynomial.json for the regularization lesson. A decrease in training error is evidence of better training fit; it does not by itself establish useful predictions on new data.

## Grading

Each file: correct data 5, specified model and fitted feature basis 25, training fit 10, validation error 10. Training MSE must be within 0.20 of the independently computed least-squares minimum for the specified feature family. Validation limits are **14 mpg²** for the linear model and **60 mpg²** for the polynomial. The generous polynomial limit allows the intended overfitting; the task does not ask the polynomial to beat the baseline on validation.

Use separate scalar coefficients, at most 100 blocks, 200 wires, and 64 values per tensor. The grader verifies original data and splits, independent polynomial coefficients, training-fitted standardization, and absence of target leakage. No written comparison is submitted. Assignment version: generalization-04-v1.

## Submit and verify

Use **File → Save** to save each requested model JSON. Import it again to check its contents, then upload the named files together to Gradescope. Files are graded independently. No written explanation, screenshot, Python program, or ZIP is submitted. The grader recomputes outputs from saved graph operations and parameters on canonical data; it ignores labels, cached metrics, and claimed training history.
