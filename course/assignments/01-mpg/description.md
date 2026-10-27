<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 1: Predicting fuel economy with models you build

Build and train three regression models in [Neural Canvas](https://davbachman.github.io/NeuralCanvas/). You will start with vehicle weight, add model year, and then add a squared feature. Use Standardize Features blocks to scale your inputs using training-data statistics.

**Submit to Gradescope:** `one_feature.json`, `two_features.json`, and `quadratic.json`. Total: 100 points.

## Choose the dataset

Start with a blank canvas and add a Dataset block.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Regression** → **Auto MPG**. Set **Configuration** to **Full sample**. Click **Use this dataset**. Keep the supplied data and split unchanged.

Submit only your three saved model JSON files.

## The data

The [UCI Auto MPG dataset](https://archive.ics.uci.edu/dataset/9/auto%2Bmpg) records city-cycle fuel economy. The Full sample configuration contains 398 cars. We use weight and model year, which have no missing values; missing horsepower values do not require deleting cars because horsepower is not used.

**Dataset columns and split**

| Column | Meaning |
| --- | --- |
| `weight_lb` | Vehicle weight in pounds; an input. |
| `model_year` | Four-digit model year; an input. |
| `mpg` | Miles per gallon; the prediction target. |
| `split` | 318 train rows and 80 test rows. The app reads this as a fixed split, not a model input. |

Keep the dataset and its split unchanged for all three models. Train only on training rows. The app calls the other rows the “held-out test set”; because you will use them to compare and improve models, they serve as **validation data** in this assignment. No separate hidden performance set determines your grade.

## Part A: One input (30 points)

1. Use the Dataset selected above. Confirm Regression and `mpg` as the target.
2. Add a **Standardize features** block from **Features and tensors**, and connect the Dataset’s `weight_lb` output to it. Select the block, open **Details**, and click **Fit on training rows**. Call its output **u**: `u = (weight_lb - training mean) / training standard deviation`. Confirm that it was fitted on **318 training observations**. Its statistics remain fixed during training and prediction; do not fit on validation data.
3. Build **prediction = b + w · u**. Both b and w must be trainable **Param** values. Use Arithmetic blocks for the multiplication and addition. An Input block containing a constant is not a trainable parameter.
4. Connect the prediction to Loss input 1 and the actual mpg to Loss input 2, optionally through a Target block. Choose **Mean squared error**, with no regularization. Never feed mpg into the prediction.
5. Train, evaluate the training and validation sets, and use **File → Save**. Rename the downloaded project `one_feature.json`.

## Part B: Two inputs (30 points)

Keep the standardized weight and add model year. Connect `model_year` to a second **Standardize features** block, select it, and click **Fit on training rows** in Details. Confirm 318 training observations. Call this standardized year **v**.

Build **prediction = b + w<sub>1</sub> · u + w<sub>2</sub> · v**. All three coefficients must be independently trainable. Train this model and save it as `two_features.json`. Keep your Part A file.

## Part C: Add a squared feature (40 points)

Square the *scaled weight* using an Arithmetic block, for example `x1 ^ 2`. Retain both original scaled inputs. Build:

**prediction = b + w<sub>1</sub> · u + w<sub>2</sub> · v + w<sub>3</sub> · u<sup>2</sup>**.

Make all four coefficients independently trainable. Train and save as `quadratic.json`. This model can curve with respect to vehicle weight while remaining linear in its learned coefficients.

## Training and troubleshooting

For the fitted Standardize features blocks and scalar Arithmetic construction, use **Dataset output mode: Numeric batch** and **Rows: Training**. In **Train → Training execution**, change the default Tensor engine to **Trace engine · SGD**. Then set learning rate **0.05**, **318 examples per update**, **1,000 epochs per run**, and reporting every **50 epochs**. This is full-batch gradient descent. The instructor pilot reached the targets with these settings; you may use other successful settings.

If your inputs use different scales, adjust the learning rate accordingly. If losses grow or become nonfinite, reduce the learning rate and restart with small parameters. If training stalls, inspect the wiring and gradients. Make sure every required coefficient is a Param, including the intercept. Use Test → Run inference for both Training set and Held-out test set. Their reported losses are MSE in mpg².

Standardize each raw input before applying learned coefficients. Keep the target in mpg. In Part C, square the already-standardized weight; do not fit a new standardizer to that squared feature. Refit a Standardize features block if you change its input or the training split. Reuse its fitted statistics for all validation and prediction. Equivalent Arithmetic scaling from earlier versions remains accepted by the grader.

## Compare your models

Compare each model’s training and validation MSE to see how adding features changes its predictions. No separate written answers or comprehension-question file is required.

## Grading

Parts A and B are worth 30 points each: 5 for a readable saved project with the correct dataset and split, 15 for the assigned model form and scaling with independently trainable coefficients, 5 for training MSE, and 5 for validation MSE. Part C is worth 40 points: the same 5 and 15 points for file/data and model form, plus 10 each for training and validation MSE. Performance points require the assigned model form. All thresholds below are inclusive.

**Full-credit performance thresholds**

| Model | Training MSE ≤ | Validation MSE ≤ |
| --- | --- | --- |
| One feature | 22 mpg² | 16 mpg² |
| Two features | 14 mpg² | 10 mpg² |
| Quadratic | 11 mpg² | 8 mpg² |

Parts are graded independently; you can submit partial work. Grading uses your saved parameters and actual graph operations, not a reported loss, screenshot, block label, or claimed number of training epochs.

Use at most 100 blocks, 200 wires, 16 trainable parameter values, and 64 values per tensor. Allowed operations are Dataset, Input, Param, Standardize features, Arithmetic, Matrix product, Concatenate, Target, MSE Loss, and Tensor transform’s Reshape/Transpose. Do not add activations, regularization, additional polynomial terms, or neural-network layers.

## Submission checklist

- Upload the three named JSON files together to the Gradescope assignment; do not upload a ZIP, Python export, or notebook.
- Open each saved model with File → Import to confirm it contains the correct trained model. Renaming a file does not change its model.
- Check Gradescope’s feedback. Correct incomplete work and resubmit before the deadline shown in Canvas.

Assignment version: mpg-01-v3. Data source: R. Quinlan, Auto MPG, UCI Machine Learning Repository, DOI 10.24432/C5859H; CC BY 4.0. The provided file selects two predictors, expands model year to four digits, and adds a deterministic split.
