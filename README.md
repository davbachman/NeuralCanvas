# Neural Canvas

Neural Canvas is a visual workbench for learning and teaching neural networks. Build models from individual operations, train them in the browser, and inspect their values and gradients. Group calculations into neurons, layers, attention heads, and transformer blocks, then zoom between those levels.

**[Open Neural Canvas](https://davbachman.github.io/NeuralCanvas/)** · [Get started](#get-started) · [Datasets](#datasets) · [Course assignments](#course-assignments) · [Reference](#reference) · [Development](#development)

## About

Created by [David Bachman](https://pzacad.pitzer.edu/~dbachman/) with Codex. David also writes the AI substack [*Entropy Bonus*](https://profbachman.substack.com/).

## Get started

1. Open the app and start with a blank canvas, or import an [example model](public/models/README.md) through **File → Import**.
2. Add a Dataset block and choose a simple example from its dropdown or open **Browse datasets…**.
3. Add model operations from **Build**, connect their ports, and select blocks to edit their settings in **Details**.
4. Use **Train** to inspect a forward pass or train the model, then **Test** to evaluate it. Save the editable project as JSON or export a supported model to PyTorch.

For a guided first build, follow [Get started](docs/USAGE.md) or begin with [Assignment 1](https://davbachman.github.io/NeuralCanvas/course/01-predictions.html). Normal browser use requires no account or local installation.

## Reference

| Topic | Guide |
| --- | --- |
| Workspace | [Canvas navigation and grouping](docs/CANVAS.md) · [Block reference](docs/BLOCKS.md) |
| Model construction | [Build models from scratch](docs/BUILDING-MODELS.md) · [Example models and lessons](docs/EXAMPLES.md) |
| Training and evaluation | [Training, testing, and reports](docs/TRAINING.md) |
| Files and exports | [Save projects and export PyTorch models](docs/FILES-AND-EXPORT.md) |
| Local checkout | [Automatic updates on macOS](docs/AUTO-PULL.md) |

## Datasets

The dataset browser groups data by learning task. Open a family to choose a configuration and preview its inputs, targets, output shapes, and training/held-out counts. Click **Use this dataset** to add it to the selected Dataset block. The eight simple regression and classification examples are also available directly from the card's dropdown.

| Browser category | Available datasets |
| --- | --- |
| Regression | Auto MPG (full or small training sample), Concrete strength, Diabetes progression, California housing, and simple line, cubic, and plane regression examples. |
| Classification | Breast cancer measurements, Iris flowers, Wine cultivars, Palmer penguins, Concentric rings, and simple threshold, circle/center, parabola, XOR, and neuron-basics examples. |
| Text and sequences | Movie reviews (IMDb sentiment), Football stories, and repeating color-cycle and counting sequences. |
| Images | UCI handwritten digits: 8 × 8 images with ten digit classes. |
| Attention and probabilities | Query/message and three-token attention lookups, plus class-score and softmax experiments. |

### Football stories

Synthetic stories follow two people and a football through a sequence of events. The browser offers two tasks:

- **Narrative generation:** predict the next token, with training targets at the final position or at every input position.
- **Question answering:** use word counts, ordered tokens, separate story/question inputs, or a combined sequence that predicts the answer token. Choose football-location questions, mixed questions about locations and possession, or mixed training with the shared football-only evaluation set.

Each configuration supplies its vocabulary, input lengths, targets, and fixed split. The browser shows the supported combinations; the assignments specify which configuration to use.

### Custom data

Choose **Browse datasets…** at the bottom of the Dataset card’s dropdown, then **Custom data**. Import numeric CSV, text/review/question files, or prepared dataset JSON. Prepared JSON supplies its vocabulary, shapes, targets, and splits automatically. Selected data is embedded in saved model projects.

See the [dataset guide](docs/DATASETS.md) for controls and the [dataset sources](public/datasets/README.md) for attribution and exact preparations.

## Course assignments

[Browse the assignment descriptions on GitHub](course/assignments/README.md).

The 28-assignment course develops regression, optimization, classification, neural networks, transformers, reward learning, and sparse-feature interpretation. Students build editable models in Neural Canvas and submit saved JSON files for automatic grading. Preserve model checkpoints as directed for later assignments.

**Teaching the course:** [Instructor guides, Canvas handouts, Gradescope packages, source, and validation notes](course/README.md). Completed solutions remain private; the guided steering lesson supplies an explicitly authorized frozen checkpoint.

| Assignment | What students explore |
| --- | --- |
| [1. Predictions, parameters, and loss](https://davbachman.github.io/NeuralCanvas/course/01-predictions.html) | Manually fit a line, inspect residuals, and compare mean squared error. |
| [2. Gradient descent](https://davbachman.github.io/NeuralCanvas/course/02-gradient-descent.html) | Derive regression gradients, follow updates, and investigate learning rate. |
| [3. Multiple features and feature scaling](https://davbachman.github.io/NeuralCanvas/course/03-features.html) | Build regression models on real data with training-fitted scaling. |
| [4. Model flexibility and generalization](https://davbachman.github.io/NeuralCanvas/course/04-generalization.html) | Compare linear and polynomial fits using training and validation data. |
| [5. Regularization](https://davbachman.github.io/NeuralCanvas/course/05-regularization.html) | Investigate how L2 changes fitting and generalization. |
| [6. Binary classification](https://davbachman.github.io/NeuralCanvas/course/06-classification.html) | Connect categorical targets, logits, sigmoid, cross-entropy, and thresholds. |
| [7. Matrix models and multiclass classification](https://davbachman.github.io/NeuralCanvas/course/07-matrices.html) | Express models with matrices and classify multiple categories. |
| [8. Nonlinearity and neural networks](https://davbachman.github.io/NeuralCanvas/course/08-mlp.html) | Construct nonlinear hidden layers for a regression task. |
| [9. Backpropagation](https://davbachman.github.io/NeuralCanvas/course/09-backpropagation.html) | Trace and construct derivatives through a small neural network. |
| [10. Nonlinear classification](https://davbachman.github.io/NeuralCanvas/course/10-rings.html) | Compare linear and MLP boundaries on concentric rings. |
| [11. Convolution and image classification](https://davbachman.github.io/NeuralCanvas/course/11-convolution.html) | Build local receptive fields, shared filters, and pooling for digits. |
| [12. Word-count models](https://davbachman.github.io/NeuralCanvas/course/12-word-counts.html) | Classify text and investigate information lost by discarding order. |
| [13. Learned embeddings](https://davbachman.github.io/NeuralCanvas/course/13-embeddings.html) | Construct compact learned representations and preserve token order. |
| [14. Positions and separate questions](https://davbachman.github.io/NeuralCanvas/course/14-positional-embeddings.html) | Represent whole stories, positions, questions, and padding masks. |
| [15. Cross-attention](https://davbachman.github.io/NeuralCanvas/course/15-attention.html) | Let the question select useful story information. |
| [16. Next-token prediction](https://davbachman.github.io/NeuralCanvas/course/16-next-word.html) | Build a bigram generator and perform autoregressive generation. |
| [17. Temperature and top-k sampling](https://davbachman.github.io/NeuralCanvas/course/17-sampling.html) | Change output selection while keeping learned parameters fixed. |
| [18. Longer context through an MLP](https://davbachman.github.io/NeuralCanvas/course/18-context.html) | Use ordered multiword context to overcome bigram limitations. |
| [19. Causal self-attention](https://davbachman.github.io/NeuralCanvas/course/19-self-attention.html) | Share projections across positions with a causal mask. |
| [20. Residual connections and feed-forward layers](https://davbachman.github.io/NeuralCanvas/course/20-residuals.html) | Preserve and transform contextual representations. |
| [21. Normalization and all-position training](https://davbachman.github.io/NeuralCanvas/course/21-normalization.html) | Normalize internal features and learn from every target position. |
| [22. Stacked transformer blocks](https://davbachman.github.io/NeuralCanvas/course/22-stacking.html) | Apply attention to representations contextualized by earlier blocks. |
| [23. Multiple attention heads](https://davbachman.github.io/NeuralCanvas/course/23-multiple-heads.html) | Complete a transformer with distinct attention distributions. |
| [24. Instruction tuning](https://davbachman.github.io/NeuralCanvas/course/24-instruction-tuning.html) | Adapt the pretrained generator to answer football-story questions. |
| [25. Learning from verifiable rewards](https://davbachman.github.io/NeuralCanvas/course/25-rlvr.html) | Adapt the same transformer using rewards from a correctness checker. |
| [26. Learning from human preferences](https://davbachman.github.io/NeuralCanvas/course/26-rlhf.html) | Fit a preference reward model and use its rewards to adapt the transformer. |
| [27. SAEs and interpretable features](https://davbachman.github.io/NeuralCanvas/course/27-sparse-autoencoder.html) | Freeze the final policy, train an SAE, and investigate candidate features. |
| [28. Feature steering](https://davbachman.github.io/NeuralCanvas/course/28-feature-interventions.html) | Use a compatible supplied SAE checkpoint to test controlled interventions. |
