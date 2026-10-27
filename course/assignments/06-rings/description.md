<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 6: When a straight boundary is not enough

Build a linear classifier and a multilayer perceptron (MLP) in [Neural Canvas](https://davbachman.github.io/NeuralCanvas/). This exercise brings together classification and MLPs from earlier assignments, focusing on how to interpret their decision boundaries. Both receive the same two coordinates. Can adding a nonlinear hidden layer separate points that a straight decision boundary cannot?

## The data

Add a Dataset block.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Classification** → **Concentric rings**. Click **Use this dataset**. Keep the supplied data and split unchanged.

These synthetic points form two rings with **three classes: 0 = lower half of the outer ring, 1 = upper half of the outer ring, and 2 = the entire inner ring**. The library supplies **Multiclass classification** with `class` as the target. Keep the supplied split of **600 training and 300 held-out examples**: 200 training and 100 held-out points per class.

Use Numeric batch mode and Training examples. Concatenate the raw columns `x1` then `x2` on axis `1` to form X. Both coordinates already have comparable scales; use them directly without standardization. Do not add squared terms, a radius, or other engineered features: the hidden layer should learn the useful features.

## Part A: Linear classifier — 40 points

Build **logits = XW + b**, with **three outputs**, one score for each class. Determine the matrix and bias-vector shapes from the two inputs and three outputs. Initialize W with Xavier and b with Zeros.

Use **Cross entropy (logits)**, with logits connected directly to Loss and the Dataset's class label connected directly to the target input. Add a separate **Softmax** branch to inspect the three probabilities in its Data tab. This is the multiclass setup from the matrix-model and Iris classification assignment; do not select binary cross-entropy or put Softmax before Loss.

Train with the settings below and save as **linear.json**. Expect limited accuracy: straight decision boundaries cannot cleanly separate all three regions. A loss near 0.53 can be a correctly trained baseline, not a wiring mistake.

## Part B: MLP classifier — 60 points

Build a new model with **16 hidden units**:

**H = ReLU(XW₁ + b₁)**<br>**logits = HW₂ + b₂**

Determine the parameter shapes. Use separate Params for both layers, Xavier matrices, and zero bias vectors. Keep the same raw inputs, three output scores, cross-entropy Loss, and separate Softmax branch. Train and save as **mlp.json**.

## Train and compare

For both models, use **Tensor engine · minibatches** in Train. Change **Backend** from Auto to **Tensor CPU** and **Optimizer** from Adam to **SGD**. Set learning rate **0.1**, **32 examples per update**, and **100 epochs**. Enable **Reshuffle training examples each epoch** and report every **10 epochs**. Use no loss penalty. Change **Gradient norm limit** to **0**; leave **Early stopping patience** at **0**. Check these settings when reopening an older saved model, too.

Compare the decision regions in **Reporting**, then use **Test → Held-out test set → Run inference** to compare loss and accuracy. Inspect probabilities using the Softmax block's **Data** tab in the held-out inference context. Browse with the arrows beside **Examples** and match **Dataset row** numbers to the prediction report. How does the MLP's boundary differ from the linear model's? Why would adding another linear layer without ReLU fail to solve the problem? Discuss these questions in class; no written responses are required.

**Full-credit performance thresholds**

| Model | Training cross-entropy | Held-out cross-entropy | Held-out accuracy |
| --- | --- | --- | --- |
| Linear | ≤ 0.56 | ≤ 0.57 | Reported, not graded |
| MLP | ≤ 0.08 | ≤ 0.10 | ≥ 97% |

Different initializations can give different results. If the MLP falls short, check its architecture and training settings before adding epochs. The held-out split serves as validation while you tune the models.

## Submit

Upload **linear.json** and **mlp.json**, downloaded using File → Save. Keep the probability branches in both saved models. No screenshots or written responses are required.

Linear: 5 points for the saved model and original data, 20 for model form, 10 for training loss, and 5 for held-out loss. MLP: 5 for file/data, 25 for model form, and 10 each for training loss, held-out loss, and held-out accuracy. Performance credit requires the assigned architecture. The grader recomputes predictions from saved parameters.

## Optional extension: handwritten digits

In a separate copy, try transferring the same comparison to handwritten digits.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Images** → **Handwritten digits**. Click **Use this dataset**. Keep the supplied data and split unchanged.

Its pixels are already scaled to \[0,1\]. Reshape each image to `1, 64`, use ten output scores, and try 32 hidden units for the MLP. Compare accuracy and cross-entropy: a more expressive model need not improve accuracy when the linear baseline is already strong. This extension is not submitted or graded.

Assignment version: rings-06-v3. Synthetic data generated for this course with seed 20261007. This replaces the earlier digits and binary-rings versions of this assignment; submit models trained on the Concentric rings dataset.
