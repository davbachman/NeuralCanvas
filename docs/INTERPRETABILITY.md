# Inspecting and intervening on internal representations

[Guide home](../README.md) · [Training](TRAINING.md) · [Blocks](BLOCKS.md) · [Saving and exporting](FILES-AND-EXPORT.md)

An internal representation can be a reconstruction target as well as an input to
the next calculation. This supports small sparse autoencoders built from ordinary
Param, Matrix product, Arithmetic and ReLU blocks. It does not guarantee that
individual features have a clean semantic meaning.

## Keep the reference model fixed

Save the trained model before extending it. In **Train**, choose **Freeze existing
parameters**, then add the new autoencoder parameters. Existing Params become
frozen and existing Dropout blocks use evaluation mode. Newly added Params are
trainable by default. To change one parameter, select it and use **Details →
Parameter training → Update setting → Trainable/Frozen**.

Frozen values are excluded from SGD, Adam, AdamW, decay and parameter
initialization. They remain part of the computation: gradients can pass through
an operation that uses frozen weights. Freezing a weight does not detach its
input. Frozen values cannot be edited or initialized in Details until unfrozen.

Dropout is a separate setting. **Dropout → Details → Always use evaluation mode**
passes activations unchanged during both training and evaluation, preserving
deterministic reconstruction targets. A frozen internal target must have only
frozen upstream parameters and deterministic upstream calculations.

## Construct and train a sparse autoencoder

For residual rows `r` of width `d`, construct `h = ReLU(r @ We + be)` and
`r_hat = h @ Wd + bd`. Choose the feature width `m` independently of the
vocabulary size. Internal token objectives require `[tokens, features]` tensors
that preserve token positions; flattened or resized token axes are unsupported.
The decoder has shape `[m, d]`: each row is one direction.

Select the decoder Param and choose **Norm constraint → Unit row norm · matrix
directions**. Every row is projected to Euclidean norm 1 on initialization and
after each update. A zero row becomes `[1, 0, …]`. This blocks the arbitrary
encoder-shrinking/decoder-growing rescaling that would defeat an activation L1
penalty. Use a suitable random initializer for useful distinct directions.

Use the existing single Loss block for the current training objective:

1. Connect `r_hat` to its prediction port and the original `r` to its target port.
2. In Details choose **Loss → Mean squared error**, then **Activation objective →
   Target source → Frozen internal activation**.
3. For internal reconstruction on a token dataset, padding is excluded
   automatically in both engines. Optionally set **Valid-token mask** to a binary
   mask calculation to select additional positions. A one includes a token and
   a zero excludes it; an explicit mask cannot reinclude padding.
4. Set **Penalized activation** to `h` and **Activation L1 strength λ** to the
   chosen nonnegative value. Keep this selection at λ = 0 for an unpenalized
   comparison with the same diagnostics.

The objective is `mean_valid_tokens,coordinates((r - r_hat)^2) +
λ * mean_valid_tokens(sum_features(abs(h)))`, plus any separately selected
parameter regularization. The L1 subgradient at exactly zero is zero. The target
side is detached in both engines, so reconstruction errors do not train the
reference branch. Explicit padding token IDs are excluded in both engines;
Tensor also excludes padding added to form minibatches. Use the same additional
position mask in both engines.

**Reporting** charts reconstruction/data loss and lists activation L1, total
objective, active features per token and inactive features separately. **Test →
Run inference** reports the same objective diagnostics on the selected split.
Active means exactly nonzero (tolerance 0); inactive means no nonzero activation
over the evaluated examples. A small positive ReLU output is active. Held-out
examples used to choose settings are a validation set.

## Inspect feature contexts

Run inference on the training set. Select the feature activation block and open
**Data → Inspect feature contexts**. Choose a zero-based **Feature index** and
**Examples to inspect**, then **Find high activations**. The scan uses the first
chosen number of examples in that inference split, excludes padding, and shows
up to 20 positive activations in descending order with their token contexts.
The selected token is bracketed. Increase the limit to inspect the whole split;
**Stop feature scan** cancels a running scan. Results describe only the inspected
examples. Run inference again after changing model values or wiring.

## Return to vocabulary predictions and intervene

Save the reconstruction project. In a copy, reconnect the existing Loss to the
transformer's vocabulary logits and original dataset targets, choose **Cross
entropy (logits)**, set **Target source → Dataset target**, **Penalized activation →
None** and **Valid-token mask → All valid positions**. Keep the autoencoder branch
complete and preserve its frozen trained parameters. Save a separate generation
or intervention project. There is one active Loss; this workflow does not add a
second objective or erase the learned autoencoder.

Use a Slice to select a decoder row, a fixed Input for bounded strength `alpha`,
and Arithmetic/Matrix product to construct `r + alpha * v`. Route that result to
the original downstream transformer calculation. For clamping, use
`r + (c - h_j) * v_j`; keep the original `r` feeding the feature encoder to avoid
a feedback cycle. A position mask can restrict the intervention to chosen token
positions. Do not replace `r` with the reconstruction: adding the intervention
preserves its reconstruction error.

Generation reevaluates the current graph as each token is appended. Use
deterministic next-token logits or probabilities on fixed prompts to test zero
strength, positive/negative strengths and control prompts. Free-running samples
can illustrate effects but do not establish reproducibility by themselves.

The mechanisms follow [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
and [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/).
Results on a small classroom transformer should not be interpreted as reproducing
large-model findings.
