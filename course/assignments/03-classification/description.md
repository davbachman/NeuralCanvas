<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 3: From scores to probabilities

In [Neural Canvas](https://davbachman.github.io/NeuralCanvas/), build two binary classifiers. You will train linear scores (logits) with binary cross-entropy, interpret them as probabilities using sigmoid, and distinguish probability quality from decisions made at a threshold.

## Data and labels

Use **Breast cancer measurements** from the dataset browser. It contains 569 observations from the [Wisconsin Diagnostic Breast Cancer dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic): **456 training rows and 113 validation rows**. The two inputs are `mean_radius` (mean cell-nucleus radius) and `mean_texture` (variation in grayscale values within the image). The target `malignant` is **0 for benign and 1 for malignant**. These are measurements from cell-nucleus images, not patient ages or tumor stages.

Add a Dataset block.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Classification** → **Breast cancer measurements**. Click **Use this dataset**. Keep the supplied data and split unchanged.

Confirm **binary classification** with `malignant` as the target. Keep all rows, columns, and split labels unchanged. The app calls the other split “Held-out test set”; here it serves as **validation data** because we use it to compare models. No hidden performance set determines your grade.

## Part A: One-feature logistic classifier — 50 points

1. Connect the Dataset's `mean_radius` output directly to a **Standardize features** block. In Details, click **Fit on training rows** and confirm 456 observations. Call the output **u**. Leave `mean_texture` unused.
2. Use separate scalar Param blocks and Arithmetic blocks to build the linear score **z = w₁u + b**. You can use `x1 * x2` for a product and `x1 + x2` for a sum.
3. Add a separate branch to inspect probabilities: connect z to an **Activation** block set to **sigmoid**. Its output is **p = 1 / (1 + exp(−z))**, interpreted as the model's estimated probability of class 1 (malignant). After running the model, select this Sigmoid block and open its **Data** tab to inspect the probabilities. Keep this branch in your saved model.
4. Connect **z directly** to the first input of a **Loss** block set to **Binary cross entropy (logits)**. This loss includes sigmoid internally: **do not connect the separate sigmoid output to Loss**. Connect the Dataset's `malignant` output directly to the Loss block's second input. Choose **Penalty: None**. Do not standardize the target.
5. Train with the settings below, evaluate both splits, and use **File → Save**. Rename the downloaded file `one_feature.json`.

Binary cross-entropy is **−mean\[y log(p) + (1−y) log(1−p)\]**, where p = sigmoid(z), using natural logarithms. The logits Loss computes this directly from z in a numerically stable way. Smaller is better. A confidently wrong prediction is penalized more heavily than a tentative wrong prediction. Do not use MSE, multiclass cross-entropy, or the raw linear score as the probability.

## Part B: Add a second feature — 50 points

Keep the same dataset and split. Connect the Dataset's `mean_texture` output directly to its own **Standardize features** block. Fit it using the 456 training rows; call its output **v**. Build **z = w₁u + w₂v + b** with three separate scalar Params, connected directly to the same **Binary cross entropy (logits)** Loss, with a separate sigmoid branch for viewing probabilities. Keep the features linear: do not add powers or hidden layers. Reset the Params to small starting values, train, evaluate, and save as `two_features.json`.

Compare training BCE, validation BCE, and validation accuracy with Part A. Adding texture should help cross-entropy on this split, but **a lower BCE does not guarantee higher accuracy at threshold 0.5**. Accuracy counts decisions; BCE also measures how much probability the model assigns to the observed labels.

## Training settings for both models

- Dataset output mode: **Numeric batch**; Rows: **Training**.
- In **Train → Training execution**, change the default Tensor engine to **Trace engine · SGD**; learning rate **0.1**; **456 examples per update** (full batch).
- Start with small coefficients using Randomize parameters, or set all Params to zero. Run **2,000 epochs**, reporting every **500 epochs**. You may run consecutive chunks without resetting parameters.
- Use no loss penalty. Fit feature statistics only on training rows and keep them fixed for validation.

The validation loss here is BCE, not the MSE used in the earlier fuel-economy regression assignments. Their numerical values and grading cutoffs are not comparable.

## Viewing the model and validation loss

Open the **Reporting** tab to see the model visualization and its **Training loss** and **Held-out loss** readouts. Held-out loss is the validation BCE for this assignment. The app identifies the input features from the Dataset columns connected to the prediction path into Loss. No separate Input or Target blocks are needed for training, visualization, or validation reporting. Existing models with those passthrough blocks remain valid; you do not need to rebuild them.

The separate **Loss by epoch** chart records training and held-out loss at epoch 0 and at the reporting interval. With these settings, new reports appear every **500 epochs**, not every epoch. This history does not require Input or Target blocks. To evaluate a saved model separately, open **Test**, choose **Held-out test set**, and click **Run inference**. After inference, select the Sigmoid block and open **Data**. The **Inspection context** defaults to held-out inference; use the arrows beside **Examples** to browse probabilities. Match each **Dataset row** to the same row in the prediction report, including incorrect predictions. If held-out reporting is unavailable, check the warning in Reporting and confirm that the dataset still has its original training and test split.

## Threshold exploration — no additional submission

A probability becomes a decision only after choosing a threshold: **predict 1 if p ≥ t; otherwise predict 0**. The app's dataset accuracy uses **t = 0.5**. It does not provide a threshold control for that report.

1. For hypothetical probabilities **0.20, 0.40, 0.60, and 0.80**, work out the predicted classes at thresholds **0.25, 0.5, and 0.75**. Which predictions change? A probability exactly equal to the threshold counts as class 1.
2. After submitting your saved models, inspect the autograder's validation-accuracy feedback. It recomputes confusion counts at all three thresholds from your saved probabilities. A **false negative** is a malignant example predicted benign; a **false positive** is a benign example predicted malignant. Lowering the threshold cannot increase false negatives, but can increase false positives.
3. Compare the counts for your two models. Explain to yourself why changing only a decision threshold leaves the probabilities and BCE unchanged. Do not retrain or change the sigmoid to represent a different decision threshold.

This is an educational classification example, not a clinical decision tool. The exercise illustrates a tradeoff; it does not establish a medically appropriate threshold. No written response or threshold file is required.

## Grading

Each model earns 5 points for the readable saved file and original dataset, 25 for the correct model and training-fitted scaling, 10 for training BCE, 5 for validation BCE, and 5 for validation accuracy. All cutoffs are inclusive. Performance credit requires the assigned model form.

**Full-credit performance criteria**

| Model | Training BCE ≤ | Validation BCE ≤ | Validation accuracy at 0.5 ≥ |
| --- | --- | --- | --- |
| One feature | 0.31 | 0.29 | 87% |
| Two features | 0.28 | 0.26 | 87% |

The grader independently recomputes predictions from saved coefficients; reported losses, cached values, and epoch counts do not determine the grade. Both parts are independent and you may resubmit. Use at most 100 blocks and 200 wires, with tensors of at most 64 values. Renamed blocks and visual groups are accepted. Use Dataset, Standardize features, separate scalar Params, Arithmetic, a linear score directly connected to Binary cross entropy (logits) Loss, a separate sigmoid branch for inspection, and optional Input/Target passthrough blocks. Direct Dataset wiring is accepted by the grader and supports model visualization. Equivalent small tensor operations for the same model are also accepted.

Previously saved models using sigmoid directly into the legacy probability-based BCE Loss are still accepted. When building a new model, use the logits setup above.

## Submission checklist

- Upload **one\_feature.json** and **two\_features.json** together. Do not submit a ZIP or screenshots.
- Reopen both saved files with File → Import and check that their dataset, fitted statistics, logits-to-Loss connection, separate sigmoid branch, and trained coefficients are present.
- Review the performance and threshold feedback in Gradescope. Save again after any further training before resubmitting.

Assignment version: classification-03-v1 (logits-loss compatibility update). Data: W. H. Wolberg, W. N. Street, and O. L. Mangasarian, Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository, DOI 10.24432/C5DW2B, CC BY 4.0; obtained from scikit-learn 1.5.2. Two original features are retained, malignant is encoded as 1, and a reproducible stratified split is supplied.
