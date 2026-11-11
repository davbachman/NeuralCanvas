<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 2: Gradient descent

Automate the slope and intercept adjustments from the previous lesson. Use the same model and compare three update sizes from exactly the same starting point. Submit **before.json, small.json, useful.json, excessive.json, trained.json**.

Start with a blank canvas and add a Dataset. In **Details → Browse datasets…**, choose **Regression → Simple regression examples → Line regression (1D)**, then **Use this dataset**. Keep the default split: 15 training and 5 validation examples. The app calls validation rows the held-out test set; they are validation data when used to choose a model.

## The update rule

Import your editable line or rebuild **prediction = b + w × x** with two scalar Params and an MSE Loss. Reset both Params to zero and save **before.json**. Do not import your fitted coefficients as the starting point.

For n training rows, with residual e = prediction − y:

```text
J = mean(e²)
dJ/dw = 2 mean(e x)
dJ/db = 2 mean(e)
w_new = w − learning_rate × dJ/dw
b_new = b − learning_rate × dJ/db
```

Both derivatives use the same old parameters. The gradient gives local sensitivity; the learning rate controls the size of the move. Lecture derives these two formulas directly; the general backpropagation algorithm comes later with neural networks.

## One full-batch update at three rates

Set **Dataset → Output mode → Numeric batch** and **Rows → Training**. In **Train → Training execution** select **Trace engine · SGD**. Use these settings for each one-update run:

| Train control | Value |
| --- | --- |
| Learning rate | As listed below |
| Examples per update | 15 |
| Epochs per run | 1 |
| Report loss every | 1 |

To enter a learning rate exactly, **double-click the displayed learning-rate number**, type the value, and press **Enter**. This is necessary for rate 1, which is above the slider’s range.

For each row below, import **before.json**, check w = b = 0, set the rate, and click **Run epochs** once. A full batch of 15 training rows makes this one gradient update. Save the result before doing further training.

| File | Learning rate | Points |
| --- | --- | --- |
| before.json | No update; both Params zero | 10 |
| small.json | 0.01 | 20 |
| useful.json | 0.1 | 30 |
| excessive.json | 1 | 20 |

Compare the resulting slopes, intercepts, and training MSE. Use the formulas to check the updates and the Data tab to inspect gradients. The large-rate model is deliberately a poor result; preserve it rather than training it until it improves. All three update artifacts are graded by their independently calculated parameters, not by a claim that a particular button was pressed.

## Repeat a useful update

Import before.json again. Set learning rate **0.1**, **Epochs per run → 50**, and **Report loss every → 1**, retaining 15 examples per update. Run training and save **trained.json** (20 points). Require training MSE ≤ **0.03** and validation MSE ≤ **0.04**.

Each file gives 40% credit for valid model construction and 60% for the expected parameter result. The one-step results must agree with a full-batch update from zero within 0.000001; the final model is graded on its fit. No cached gradient, epoch count, or reported score is trusted. Use the same block and size limits as the editable-line lesson. Assignment version: gradient-descent-02-v1.

## Submit and verify

Use **File → Save** to save each requested model JSON. Import it again to check its contents, then upload the named files together to Gradescope. Files are graded independently. No written explanation, screenshot, Python program, or ZIP is submitted. The grader recomputes outputs from saved graph operations and parameters on canonical data; it ignores labels, cached metrics, and claimed training history.
