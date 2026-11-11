<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 9: Follow gradients through a neural network

Gradient descent needs the derivative of the loss with respect to each parameter. Build a small network in [Neural Canvas](https://davbachman.github.io/NeuralCanvas/), then construct additional canvas branches that calculate a hidden weight's derivative. Extend the network so that one weight is used twice.

**Submit:** **chain.json** and **shared.json**. Each is worth 50 points. Submit saved models only.

## Use a familiar dataset

Select **Dataset → Details → Browse datasets…**. Choose **Regression → Simple regression examples → Line regression (1D)**, then **Use this dataset**. Keep the default data and split. Set **Output mode → One example**. Use x as input and y as the target.

This is a calculation investigation, not a fitting contest. Keep the specified parameter values in the files you submit. Use **Run forward** to inspect the branches. In **Train → Training execution**, choose **Trace engine · SGD**; use **Step** to follow forward and backward phases. Stop before the update phase, or restore the prescribed values before saving.

## Part A: One hidden unit — 50 points

Build the following prediction using Arithmetic blocks and one Activation block set to **Tanh**:

```text
z = w x + b
h = tanh(z)
prediction = v h + c
```

| Scalar Param | Value |
| --- | --- |
| w | 0.8 |
| b | −0.2 |
| v | 1.1 |
| c | 0.1 |

Connect prediction and the Dataset target to **Loss → Mean squared error**, with **Penalty → None**. Use four separate scalar Params. Leave them trainable and unconstrained. A single example has loss L = (prediction − y)².

Construct a branch that calculates dL/dw from the *current connected values*. Use Arithmetic blocks, branches from your existing Params and intermediate calculations, and the Dataset's x and y. The local derivatives you need are dL/dprediction = 2(prediction − y), d(tanh z)/dz = 1 − h², and the derivatives of multiplication and addition. Work out how they combine. Arithmetic supports powers such as **x1 ^ 2**.

Select the w Param after the backward phase and compare its gradient with the value of your derivative branch. Change the selected Dataset row and repeat. These inspections support the submitted construction; no screenshot or written answer is required. Save **chain.json**.

## Part B: A weight used along two paths — 50 points

Extend a copy of your model to two hidden units:

```text
h₁ = tanh(w x + b₁)
h₂ = tanh(w x + b₂)
prediction = v₁ h₁ + v₂ h₂ + c
```

Keep w = 0.8, b₁ = −0.2, v₁ = 1.1, and c = 0.1. Add b₂ = 0.4 and v₂ = −0.7. Use the **same w Param** in both hidden units: two Params that happen to have equal values are different models.

Construct three derivative outputs: the contribution to dL/dw through h₁, the contribution through h₂, and the complete dL/dw. Each contribution must use the error of the complete two-unit prediction. Determine how the contributions combine. Keep the three outputs on the canvas and save **shared.json**.

During the backward phase, inspect the two wires leaving w and the w Param itself. A wire shows its path's contribution; the Param shows the accumulated gradient. This pattern will return when a convolutional filter is reused at many image locations.

## Grading

| File | Requirements | Points |
| --- | --- | --- |
| chain.json | Correct forward network and fixed scalar values | 15 |
| chain.json | Connected chain-rule derivative construction | 35 |
| shared.json | Correct forward network and actual shared w | 15 |
| shared.json | First and second path contributions | 10 each |
| shared.json | Complete accumulated derivative | 15 |

The grader evaluates your operations on the canonical examples and with perturbed parameters and targets. Copying a displayed gradient into an Input, copying parameter values into a formula, or relying on cached gradients does not implement the derivative. Reference the existing Params and values through wires. Block names and positions do not affect grading.

Use at most 100 blocks and 200 wires. Allowed operations are Dataset, connected Input/Target aliases, scalar Params, Arithmetic, Add/Multiply, Tanh, and the single MSE Loss. Do not add other learned parameters or feed the target into predictions. The target may enter the derivative calculation. Reopen both saved files with **File → Import** before uploading them.

Version: backpropagation-09-v1. Dataset: the app's deterministic synthetic line examples.
