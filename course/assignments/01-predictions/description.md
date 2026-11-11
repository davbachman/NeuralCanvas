<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 1: Predictions, parameters, and loss

A model gives a prediction for each input. Build a line whose slope and intercept you can change, then use prediction error to judge your choices. Submit **zero.json** and **fitted.json**, worth 30 and 70 points.

Start with a blank canvas and add a Dataset. In **Details → Browse datasets…**, choose **Regression → Simple regression examples → Line regression (1D)**, then **Use this dataset**. Keep the default split: 15 training and 5 validation examples. The app calls validation rows the held-out test set; they are validation data when used to choose a model.

## Construct an editable line

1. Add two scalar **Param** blocks, one for slope w and one for intercept b. Select each and set its value to zero. Keep both trainable.
2. Use Arithmetic blocks to build **prediction = b + w × x**. The Dataset's x is an input; its y must not feed the prediction.
3. Connect prediction to the first port of a **Loss** block and Dataset y to its second port. Select **Mean squared error**, with no parameter or activation penalties.
4. Select the Dataset's **Numeric batch** output mode and **Rows → Training**. Choose **Train → Run forward** to compute predictions without changing the parameters. Save **zero.json**.

## Make the predictions useful

Make a copy of zero.json. Inspect the observed points in Reporting. Adjust w and b by editing the Param values, rerunning forward after each change. Use the Data tab to inspect a prediction and its target. Their difference is a residual; MSE averages the squared residuals. Large positive and negative residuals cannot cancel in MSE.

Changing the intercept shifts the line; changing the slope changes how prediction depends on x. Use those two effects to reduce error. Keep the same data and graph form. Automatic training will be the topic of the next lesson; here you make the adjustments yourself.

Evaluate both splits with **Test → Run inference**. Find a line with **training MSE ≤ 0.20** and **validation MSE ≤ 0.20**, then save **fitted.json**. You do not need an exact optimum. Inspection supports the saved result and is not separately graded as an alleged activity.

## Grading

Each file assigns 40% of its points to the original dataset, MSE wiring, and two independently adjustable coefficients; 60% checks its parameter result. zero.json must have effective slope and intercept zero. fitted.json must meet both error bounds. The grader accepts equivalent arithmetic wiring, not fixed predictions or target leakage. A saved result cannot establish whether adjustments were made manually.

Use at most 30 blocks and 50 wires. Allowed blocks: Dataset, Input, Target, scalar Param, Arithmetic (+, −, ×), and MSE Loss. No scaling, extra predictors, powers, or activations are needed. Assignment version: predictions-01-v1.

## Submit and verify

Use **File → Save** to save each requested model JSON. Import it again to check its contents, then upload the named files together to Gradescope. Files are graded independently. No written explanation, screenshot, Python program, or ZIP is submitted. The grader recomputes outputs from saved graph operations and parameters on canonical data; it ignores labels, cached metrics, and claimed training history.
