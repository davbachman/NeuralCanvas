<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 21: Test features through residual interventions

A feature can correlate with a token context without controlling the model's next prediction. Test that distinction by changing one feature direction inside your frozen transformer and measuring the resulting probabilities. The goal is a bounded, reproducible prediction effect, not a guarantee that a feature represents a clean concept or improves factual accuracy.

Start from your own **sparse.json** from the sparse-feature lesson. Preserve its 32-feature autoencoder, original transformer values, dataset and reconstruction objective in this file. Use copies for the interventions. No new training is needed.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Narrative generation** preset. Set **Training targets** to **Every input position**. Click **Use this dataset**. Keep the supplied data and split unchanged.

## Choose one feature and restore vocabulary predictions

Use the feature-context inspector to choose a candidate j. The index is local to your trained encoder: the grader does not require a reference feature number. Your chosen feature must satisfy the bounded activation and prediction requirements below; a feature that is always inactive cannot demonstrate an intervention.

In an intervention copy, choose **Train → Freeze existing parameters** so both transformer and autoencoder Params stay fixed. Reconnect the existing Loss prediction to the original vocabulary logits and its target to the Dataset's next-token target output. Choose **Cross entropy (logits)**, **Target source → Dataset target**, **Penalized activation → None**, **Valid-token mask → All valid positions** and **Parameter regularization → Penalty → None**. Keep the complete autoencoder branch and its learned values. This uses one Loss block at a time; do not create another Loss.

Use the same intervention location as before: the second block's final feed-forward residual r, immediately before the final output Layer norm. Keep **the original r feeding the autoencoder**. Only the path from r to that output Layer norm should receive r′. This prevents a feedback cycle and preserves the original residual's reconstruction error.

## Build three visible interventions

Let v<sub>j</sub> be row j of your saved decoder Wd, selected by a Slice connected directly to Wd. Let h<sub>j</sub> be the corresponding column of your original ReLU encoder output. Use ordinary Input, Slice, Arithmetic and Matrix product operations. Apply the intervention at **every input position**; use no position mask in these submitted files. The grader measures the final-position next-token distribution.

| File | Required construction | Purpose |
| --- | --- | --- |
| zero.json | r′ = r + 0 v<sub>j</sub> | A connected residual intervention with zero strength restores baseline predictions. |
| amplified.json | r′ = r + α v<sub>j</sub>, with 0 &lt; α ≤ 1 | Add a bounded amount of the selected direction. Start with α = 1. |
| ablated.json | r′ = r − h<sub>j</sub> v<sub>j</sub> | Remove the selected feature contribution while retaining the reconstruction error. |

Use the **same j** in all three intervention copies. Keep identical learned values across those copies and sparse.json. Use a fixed scalar strength: an Input or a numeric constant in Arithmetic. Do not add a trainable strength or extra Params. In the ablation file, derive h<sub>j</sub> from the actual encoder activations. Determine the appropriate Slice axes, bounds and broadcasting or matrix product shapes. Your feature-contribution construction from the previous lesson is useful here.

Do not replace r with r\_hat. Do not add a vocabulary bias, use copied decoder numbers or attach a disconnected steering branch: these constructions do not establish a residual intervention.

## Measure effects and controls

The grader independently evaluates the **765 canonical validation windows**. For your submitted feature, it ranks windows by h<sub>j</sub> at the final input position, resolving ties by canonical window order. The 32 highest-activation windows form the high-activation group; the 32 lowest form the control group. These sets depend on your learned feature, not on a fixed feature index.

For two next-token distributions p and q, **total variation = ½ Σ |p − q|** over all 25 vocabulary entries. It is zero for identical distributions and at most one. The grader checks zero-strength restoration, a measurable amplification effect, a stronger ablation effect on high-activation contexts, small changes on low-activation controls, and bounded collateral changes across all validation windows. It also evaluates the original model's canonical next-token cross entropy independently.

Investigate in the app: run **Test → Held-out test set → Run inference**, select h and open **Data**. Set **Examples to inspect** to 765 in the feature-context scan to check the largest activation over the validation windows. Use **Dataset row** to inspect the final row of h and of the separate vocabulary Softmax for the same example in zero.json, amplified.json and ablated.json. The Dataset block's Data view shows the corresponding token IDs and context. Re-run inference after changing a graph. For a typed prompt, **Dataset → Details → Inspect next-token probabilities** provides another quick comparison; that generation display excludes padding and shows only the top 12 tokens, while the grader uses the full 25-entry distribution.

Choose a feature whose activation does not exceed the stated bound over **any of the 32 positions in any validation window**. The bound prevents ablation from winning through an overwhelming residual change. Numerical acceptance requirements are listed in the rubric below. You may inspect several features and use a smaller α within the stated range; exhaustive tuning or training a new transformer is unnecessary.

Validation windows used to select a feature or strength are a validation set, not an untouched final test. A prediction shift demonstrates a causal effect of this intervention in this model. It does not show that the feature is uniquely responsible for a concept or that the resulting story is correct.

## Submit — four saved models, 100 points

**Numerical requirements, evaluated from saved operations**

| Check | Requirement |
| --- | --- |
| Preserved sparse autoencoder | Original generator cross entropy ≤ 0.60/0.65 training/validation; reconstruction MSE ≤ 0.20 on both; validation active features/token ≤ 12 and inactive features ≤ 16. |
| Selected feature activation for ablation | Maximum h<sub>j</sub> ≤ 4 over all positions in all validation windows. |
| Zero restoration | Maximum absolute probability difference ≤ 0.0000001. |
| Amplification | 0 &lt; α ≤ 1; mean final-position total variation ≥ 0.005. |
| Ablation high-activation contexts | Mean total variation over the highest 32 contexts ≥ 0.05. |
| Ablation low-activation controls | Mean total variation over the lowest 32 contexts ≤ 0.01. |
| Collateral changes, amplification and ablation | Mean total variation over all windows ≤ 0.10, and mean final-position cross-entropy increase ≤ 0.30 relative to the unchanged generator. |

| Component | Points |
| --- | --- |
| sparse.json: canonical data, valid frozen reference and autoencoder, useful reconstruction and sparse features | 20 |
| zero.json: valid connected construction and restored predictions | 20 |
| amplified.json: bounded decoder-direction construction and independently measured effect | 20 |
| ablated.json: encoder-derived feature removal, high-activation effect and controls | 30 |
| Identical transformer and autoencoder values, with the same selected feature across related files | 10 |

Valid components earn partial credit independently. The grader never executes submitted code and ignores cached predictions, losses, labels and training history. No written responses, screenshots or external quiz are required.

## Optional generation and clamping

In an intervention copy use **Dataset → Details → Continue the text** to compare the same prompt with zero, positive and ablated interventions. The edited graph is evaluated again for each generated token. Use **Most likely** for a deterministic illustration; free-running text is not the acceptance test.

Optional extension: clamp the feature to c using **r′ = r + (c − h<sub>j</sub>)v<sub>j</sub>**, or construct a token-position mask. Keep the original residual feeding the encoder. These variants are optional and are not additional submissions.
