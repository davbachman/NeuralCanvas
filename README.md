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

For a guided first build, follow [Get started](docs/USAGE.md) or begin with [Assignment 1](https://davbachman.github.io/NeuralCanvas/course/01-mpg.html). Normal browser use requires no account or local installation.

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

These 19 assignments progress from regression to a transformer question-answering capstone. Students build editable models, select datasets from the browser, and save their work as JSON. Many assignments extend the preceding solution, so keep trained models for later comparisons and fine-tuning. Each link opens the student instructions.

**Teaching the course:** [Instructor guides, Canvas handouts, Gradescope autograders, sources, and curriculum notes](course/README.md) are included in this repository. Trained solutions and exams remain private.

### Regression and classification

| Assignment | What students explore |
| --- | --- |
| [1. Predicting fuel economy with models you build](https://davbachman.github.io/NeuralCanvas/course/01-mpg.html) | Build one-feature, two-feature, and quadratic regression models on Auto MPG, using training-fitted feature standardization. Compare how added features affect predictions. |
| [2. When a better fit makes worse predictions](https://davbachman.github.io/NeuralCanvas/course/02-regularization.html) | Use a small MPG training sample to investigate polynomial overfitting and L2 regularization. Compare training fit with validation performance. |
| [3. From scores to probabilities](https://davbachman.github.io/NeuralCanvas/course/03-classification.html) | Build one- and two-feature breast-cancer classifiers with logits, binary cross-entropy, and sigmoid probabilities. Explore how a decision threshold changes classifications. |
| [4. From weighted sums to matrix models](https://davbachman.github.io/NeuralCanvas/course/04-matrices.html) | Rewrite MPG regression as a matrix calculation, then classify three Iris species. Introduce concatenation, tensor parameters, matrix multiplication, softmax, and multiclass cross-entropy. |
| [5. When layers become a neural network](https://davbachman.github.io/NeuralCanvas/course/05-mlp.html) | Compare linear regression with a ReLU multilayer perceptron for predicting concrete strength. Investigate how a nonlinear hidden layer learns useful intermediate features. |
| [6. When a straight boundary is not enough](https://davbachman.github.io/NeuralCanvas/course/06-rings.html) | Compare linear and MLP classifiers on three-class concentric rings, examining how nonlinear features change decision boundaries. |

### Text representations and cross-attention

| Assignment | What students explore |
| --- | --- |
| [7. What can word counts tell us?](https://davbachman.github.io/NeuralCanvas/course/07-word-counts.html) | Build a guided IMDb sentiment classifier, then independently adapt word-count classification to football-story questions. |
| [8. Learn a compact representation of words](https://davbachman.github.io/NeuralCanvas/course/08-embeddings.html) | Preserve word order and compare one-hot inputs with learned word embeddings on football questions, using comparable parameter counts and training budgets. |
| [9. Positional embeddings and separate questions](https://davbachman.github.io/NeuralCanvas/course/09-positional-embeddings.html) | Add learned story positions and a separate question encoder to an embedding MLP. Establish a baseline on the common football-question benchmark. |
| [10. Let questions attend to story tokens](https://davbachman.github.io/NeuralCanvas/course/10-attention.html) | Replace the flattened story input with question-directed cross-attention. Compare accuracy and parameter count with the preceding MLP on the same football questions. |

### Language generation and transformers

| Assignment | What students explore |
| --- | --- |
| [11. Predict the next word](https://davbachman.github.io/NeuralCanvas/course/11-next-word.html) | Turn classification into generation with a football-narrative bigram model that predicts the next token from the preceding token. |
| [12. Control how the model chooses words](https://davbachman.github.io/NeuralCanvas/course/12-sampling.html) | Explore temperature and probabilistic sampling while preserving the trained bigram model's parameters. Apply these generation choices to later language models. |
| [13. Give the generator a longer memory](https://davbachman.github.io/NeuralCanvas/course/13-context.html) | Extend the bigram generator into a fixed-context MLP that uses an ordered sequence of embedded tokens to predict the next token. |
| [14. Share attention across word positions](https://davbachman.github.io/NeuralCanvas/course/14-self-attention.html) | Replace the context MLP with self-attention, learned positions, and a causal mask so each token can attend to its available context. |
| [15. Preserve and transform the representation](https://davbachman.github.io/NeuralCanvas/course/15-residuals.html) | Add an attention output projection, residual connections, and a shared feed-forward MLP to preserve and transform token representations. |
| [16. Normalize the block and learn at every position](https://davbachman.github.io/NeuralCanvas/course/16-normalization.html) | Add layer normalization and train on next-token targets at every position. Evaluate the trained model at the final position to compare it with earlier generators. |
| [17. Let attention build on attention](https://davbachman.github.io/NeuralCanvas/course/17-stacking.html) | Stack two independently parameterized transformer blocks so later attention can use already contextualized representations. Compare with the trained one-block model. |
| [18. Give each block two attention patterns](https://davbachman.github.io/NeuralCanvas/course/18-multiple-heads.html) | Give each transformer block two independently learned attention heads while keeping the total feature width fixed. Keep the trained generator for the capstone. |
| [19. Fine-tune your generator to answer questions](https://davbachman.github.io/NeuralCanvas/course/19-qa-finetuning.html) | Fine-tune the trained generator to predict answer tokens for football-story questions. Compare before and after fine-tuning, and with the earlier cross-attention model, on the common benchmark. |

