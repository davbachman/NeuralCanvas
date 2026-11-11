# Revised course: 28 assignments

This is the current teaching order. The machine-readable catalog is [sequence.json](../sequence.json). Earlier dated reports retain the numbering and evidence from their original releases.

| Assignment | Topic | Learning purpose |
| --- | --- | --- |
| 1 | [Predictions, parameters, and loss](../assignments/01-predictions/description.md) | Manually fit a line, inspect residuals, and compare mean squared error. |
| 2 | [Gradient descent](../assignments/02-gradient-descent/description.md) | Derive regression gradients, follow updates, and investigate learning rate. |
| 3 | [Multiple features and feature scaling](../assignments/03-features/description.md) | Build regression models on real data with training-fitted scaling. |
| 4 | [Model flexibility and generalization](../assignments/04-generalization/description.md) | Compare linear and polynomial fits using training and validation data. |
| 5 | [Regularization](../assignments/05-regularization/description.md) | Investigate how L2 changes fitting and generalization. |
| 6 | [Binary classification](../assignments/06-classification/description.md) | Connect categorical targets, logits, sigmoid, cross-entropy, and thresholds. |
| 7 | [Matrix models and multiclass classification](../assignments/07-matrices/description.md) | Express models with matrices and classify multiple categories. |
| 8 | [Nonlinearity and neural networks](../assignments/08-mlp/description.md) | Construct nonlinear hidden layers for a regression task. |
| 9 | [Backpropagation](../assignments/09-backpropagation/description.md) | Trace and construct derivatives through a small neural network. |
| 10 | [Nonlinear classification](../assignments/10-rings/description.md) | Compare linear and MLP boundaries on concentric rings. |
| 11 | [Convolution and image classification](../assignments/11-convolution/description.md) | Build local receptive fields, shared filters, and pooling for digits. |
| 12 | [Word-count models](../assignments/12-word-counts/description.md) | Classify text and investigate information lost by discarding order. |
| 13 | [Learned embeddings](../assignments/13-embeddings/description.md) | Construct compact learned representations and preserve token order. |
| 14 | [Positions and separate questions](../assignments/14-positional-embeddings/description.md) | Represent whole stories, positions, questions, and padding masks. |
| 15 | [Cross-attention](../assignments/15-attention/description.md) | Let the question select useful story information. |
| 16 | [Next-token prediction](../assignments/16-next-word/description.md) | Build a bigram generator and perform autoregressive generation. |
| 17 | [Temperature and top-k sampling](../assignments/17-sampling/description.md) | Change output selection while keeping learned parameters fixed. |
| 18 | [Longer context through an MLP](../assignments/18-context/description.md) | Use ordered multiword context to overcome bigram limitations. |
| 19 | [Causal self-attention](../assignments/19-self-attention/description.md) | Share projections across positions with a causal mask. |
| 20 | [Residual connections and feed-forward layers](../assignments/20-residuals/description.md) | Preserve and transform contextual representations. |
| 21 | [Normalization and all-position training](../assignments/21-normalization/description.md) | Normalize internal features and learn from every target position. |
| 22 | [Stacked transformer blocks](../assignments/22-stacking/description.md) | Apply attention to representations contextualized by earlier blocks. |
| 23 | [Multiple attention heads](../assignments/23-multiple-heads/description.md) | Complete a transformer with distinct attention distributions. |
| 24 | [Instruction tuning](../assignments/24-instruction-tuning/description.md) | Adapt the pretrained generator to answer football-story questions. |
| 25 | [Learning from verifiable rewards](../assignments/25-rlvr/description.md) | Adapt the same transformer using rewards from a correctness checker. |
| 26 | [Learning from human preferences](../assignments/26-rlhf/description.md) | Fit a preference reward model and use its rewards to adapt the transformer. |
| 27 | [SAEs and interpretable features](../assignments/27-sparse-autoencoder/description.md) | Freeze the final policy, train an SAE, and investigate candidate features. |
| 28 | [Feature steering](../assignments/28-feature-interventions/description.md) | Use a compatible supplied SAE checkpoint to test controlled interventions. |

## Dependencies and assessment

Assignments 1–3 develop regression and optimization; assignments 4–5 separate the discovery of overfitting from regularization. Classification follows sigmoid, and explicit backpropagation follows the first nonlinear network. Convolution revisits shared parameter gradients.

Word counts lead into ordered embeddings, cross-attention, and a separate next-token generation sequence. Preserve the earlier cross-attention football evaluation for the instruction-tuning comparison. Sampling changes distributions without retraining.

The final sequence is instruction tuning → verifiable rewards → learned preference rewards → SAE feature discovery → steering. Reward lessons adapt the same transformer. Preserve the final reward-trained parameters before switching datasets for activation analysis. The guided steering checkpoint must contain a transformer and SAE trained against the same parameter state and activation location.

Every substantial required investigation produces an automatically graded artifact. Inspecting a gradient or a generated sample supports construction; cached metrics, claimed training history, screenshots, and prose are not evidence for credit. Use training-only preprocessing and identify held-out data used for tuning as validation.

Exact recipes, measured targets, and validation limits belong in each instructor guide. Browser and hosted Gradescope validation are distinct. Existing historical pilot results do not establish feasibility for a changed recipe.
