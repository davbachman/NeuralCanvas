<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 4: From weighted sums to matrix models

Build two models in [Neural Canvas](https://davbachman.github.io/NeuralCanvas/). First express the familiar two-feature MPG regression as a matrix calculation. Then give the matrix three output columns to classify Iris flowers. You will use concatenation, tensor Params, matrix multiplication, logits, softmax, and multiclass cross-entropy.

## Before you begin: rows, columns, and batches

A batch contains **B examples**. A feature matrix has one row per example and one column per feature. Multiplying a B-by-d matrix by a d-by-c weight matrix gives B-by-c outputs. The inner dimensions must match. A Param can store an entire learned matrix: its entries are separate trainable numbers, even though they share one block.

For two features and one output, **Xw + b** computes exactly the same predictions as **w₁u + w₂v + b**. Matrix notation changes the organization, not the model's expressive power.

## Part A: Rebuild MPG regression with a matrix — 40 points

Start a new project and add a Dataset block.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Regression** → **Auto MPG**. Set **Configuration** to **Full sample**. Click **Use this dataset**. Keep the supplied data and split unchanged.

This is the same dataset and fixed split as the initial fuel-economy assignment: **318 training rows and 80 validation rows**. Confirm **Regression**, with `mpg` as the target. Use `weight_lb` and `model_year` as inputs; keep all rows, columns, and split labels unchanged.

1. Set the Dataset output mode to **Numeric batch** and Rows to **Training**.
2. Add **Concatenate**. Set Axis to `1` and input count to `2`. Connect the Dataset's `weight_lb` output first, then `model_year`. Concatenate treats each length-B vector as a B-by-1 column automatically, producing a B-by-2 matrix. No feature reshapes are needed; axis 0 would join the vectors end to end instead.
3. Connect Concatenate to one **Standardize features** block. In Details, click **Fit on training rows**. Confirm 318 observations and two fitted means/scales: each column is standardized independently using training data. Call this block's output **X**. Keep these statistics fixed during training and validation.
4. Add a **Param** for **w**. In Details, choose the `uniform` initializer, enter Shape `2, 1`, and press **Enter** in the Shape field to initialize the tensor. Check that two values appear.
5. Add **Matrix product**: connect X to its first input and w to its second. The result has shape B-by-1.
6. Add a second Param, a **scalar bias b** initialized to 0. Add it to the product with **Arithmetic** addition. Connect the product first and b second; use the addition expression shown for those inputs (the underlying operation is `x1 + x2`). The same scalar is added to every example.
7. Reshape the Dataset's `mpg` output to `-1, 1` with a Tensor transform. The `-1` infers the row count, so this works for both 318 training rows and 80 validation rows. Connect predictions and this target column to the first and second inputs of **Loss → Mean squared error**, respectively. Choose **Penalty: None**. Do not standardize the target. Matching the B-by-1 prediction and target shapes avoids unintended broadcasting.
8. Train with the settings below, evaluate both splits, and save with **File → Save** as `matrix_regression.json`.

Compare with your earlier two-feature regression. The fitted predictions and errors should be very similar. Do not expect matrix multiplication alone to improve accuracy.

## Part B: Three-class Iris classification — 60 points

Start a separate project and add a Dataset block.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Classification** → **Iris flowers**. Click **Use this dataset**. Keep the supplied data and split unchanged.

It contains 150 real flowers, with **120 training and 30 validation rows**: 40 training and 10 validation examples of each species. Confirm **Multiclass classification**, with `species` as the target. The app orders the labels as **setosa = 0, versicolor = 1, virginica = 2**. Keep all labels and split assignments unchanged.

1. Set Numeric batch / Training. Connect the four Dataset feature outputs directly to **Concatenate**, with Axis `1` and input count `4`, in this exact order: **sepal\_length\_cm, sepal\_width\_cm, petal\_length\_cm, petal\_width\_cm**. It automatically treats the vectors as columns, producing a B-by-4 matrix without feature reshapes. All measurements are in centimeters.
2. Connect the matrix to one **Standardize features** block and click **Fit on training rows**. Confirm **120 training observations** and four fitted means/scales. Each column is standardized independently. Call the output **X**; its shape remains B-by-4. Keep the fitted statistics fixed during training and validation.
3. Create a weight Param **W**. Choose the `uniform` initializer, enter Shape `4, 3`, and press **Enter** in the Shape field; confirm 12 values. Connect X and W to Matrix product. Each of W's three columns supplies weights for one class.
4. Create a bias Param **b**. Choose the `zeros` initializer, enter Shape `3`, and press **Enter** in the Shape field; confirm three zeros. Add it to XW using Arithmetic addition. The bias vector broadcasts across the batch: one bias per class, reused for every flower.
5. The resulting B-by-3 values are **logits**: unrestricted class scores, not probabilities. Connect them directly to **Loss → Cross entropy (logits)**. Connect the Dataset's `species` output directly to the second Loss input. Leave the target as a length-B vector of class IDs; do not one-hot encode, standardize, or reshape it into three columns. Choose Penalty: None.
6. Branch the logits into a **Softmax** block to inspect probabilities. Keep the original logits-to-Loss connection. **Do not put Softmax between the logits and Cross entropy (logits)**: that loss already includes the softmax calculation.
7. Train, evaluate both splits, and save as `iris.json`.

Softmax converts a row of scores z into probabilities **pⱼ = exp(zⱼ) / Σₖ exp(zₖ)**. A row's three probabilities sum to 1. Cross-entropy is the mean of **−log(p of the true class)**, using natural logarithms. The predicted species is the class with the largest logit, equivalently the largest softmax probability. Unlike binary classification, there is no separate 0.5 threshold for each class.

## Shape checklist

| Quantity | MPG regression | Iris classification |
| --- | --- | --- |
| Standardized feature matrix X | B × 2 | B × 4 |
| Weight Param | 2 × 1 | 4 × 3 |
| Bias Param | Scalar | Vector of length 3 |
| Prediction / logits | B × 1 | B × 3 |
| Target into Loss | B × 1 | Vector of length B |

Concatenate the feature vectors first, then standardize the resulting matrix with one block. Each column gets its own training-fitted mean and scale. Only the MPG target needs an explicit reshape to match its B-by-1 predictions; leave the Iris target as a vector. Input and Target alias blocks are not required.

## Training and evaluation

For both models, in **Train → Training execution**, change the default Tensor engine to **Trace engine · SGD**. Use no loss penalty. Use the initializers specified above and these full-batch settings:

| Model | Learning rate | Examples per update | Epochs | Report every |
| --- | --- | --- | --- | --- |
| MPG | 0.05 | 318 | 1,000 | 500 epochs |
| Iris | 0.1 | 120 | 2,000 | 500 epochs |

Use **Train → Run epochs**. Consecutive runs continue training unless you reset parameters. Open **Reporting** for loss history: it updates at epoch 0 and at the chosen reporting interval. Use **Test → Held-out test set → Run inference** to evaluate validation data; choose Training set to evaluate training data. Here “held-out test” is validation because we use it to compare models. There is no hidden performance set.

The MPG model has two inputs and supports a simple model visualization. The Iris model uses four inputs, so no simple input-output plot is expected; use its loss history, inference results, and accuracy. After held-out inference, select Softmax and open **Data**. Keep **Inspection context** on held-out inference and use the arrows beside **Examples** to browse all evaluated rows. Match the original **Dataset row** numbers to the prediction report. Compare each probability row's largest entry with its predicted species, and check that the probabilities sum to 1. This exploration requires no written response. The probability branch is for inspection and is not separately graded.

## Grading and submission

Part A awards 5 points for the saved model and dataset, 20 for the assigned matrix model and scaling, 10 for training MSE, and 5 for validation MSE. Part B awards 5 for file/data, 30 for matrix model/scaling, 10 each for training and validation cross-entropy, and 5 for validation accuracy. Performance credit requires the assigned matrix construction. All cutoffs below are inclusive.

| Model | Training loss ≤ | Validation loss ≤ | Validation accuracy ≥ |
| --- | --- | --- | --- |
| MPG (MSE) | 14 | 10 | Not applicable |
| Iris (cross-entropy) | 0.15 | 0.25 | 90% (at least 27/30) |

Submit only **matrix\_regression.json** and **iris.json**, not a ZIP or screenshots. Reopen each with File → Import to verify its saved data and parameters. The grader recomputes outputs from the saved graph and parameters; reported losses and cached predictions do not determine the score. Each part is independent and you may resubmit.

Use one Dataset and one Loss per model, one learned weight matrix directly connected to Matrix product's second input, and one separate bias Param with the shapes specified above. Use Concatenate in the prediction path and the prescribed feature order and training-fitted scaling. Do not replace the matrix Param with separate scalar weights, add extra trainable layers, powers, or other features. Renamed blocks, visual groups, and optional Input/Target aliases are accepted. Keep at most 100 blocks, 200 wires, and 64 values in any Param tensor.

Version: matrices-04-v1. MPG: R. Quinlan, Auto MPG, UCI, DOI 10.24432/C5859H; the original Auto MPG data and split are reused unchanged. Iris: R. A. Fisher, Iris, UCI, DOI 10.24432/C56C76; raw measurements and labels from scikit-learn 1.5.2 with a reproducible stratified split. Both datasets are provided under CC BY 4.0.
