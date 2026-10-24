Created by David Bachman with Codex

# Neural Canvas

[Open Neural Canvas](https://davbachman.github.io/NeuralCanvas/)

Neural Canvas is a visual, editable machine-learning workbench for students. Start with a blank canvas, connect individual calculations into a model, and watch values move forward and gradients move backward. Group calculations into neurons, layers, attention heads, or larger modules, then zoom between those scales. You can train and test on built-in or imported data and export a working PyTorch version of a supported graph.

## Guide

| Page | What it covers |
| --- | --- |
| [Get started](docs/USAGE.md) | A first model, the sidebars, and the route from blank canvas to a training run |
| [Canvas and navigation](docs/CANVAS.md) | Adding, connecting, moving, grouping, zooming, and using the code outline |
| [Block reference](docs/BLOCKS.md) | Every block in the palette, its inputs, and common tensor shapes |
| [Build models from scratch](docs/BUILDING-MODELS.md) | Recipes for an MLP, attention, a transformer block, and a digit CNN |
| [Datasets](docs/DATASETS.md) | The dataset browser, teaching datasets, custom imports, feature and target wiring, and train/test splits |
| [Accelerated text training](docs/ACCELERATED-TEXT-TRAINING.md) | Padded batches, AdamW, WebGL, early stopping, and controlled IMDb results |
| [Text curriculum pilot](docs/TEXT-CURRICULUM.md) | Earlier IMDb and Alice experiments with embeddings, attention, and text generation |
| [Training, testing, and reports](docs/TRAINING.md) | Backpropagation, epoch runs, inference, predictions, accuracy, and troubleshooting |
| [Example models and lessons](docs/EXAMPLES.md) | Importable projects from a neuron through attention, a transformer, and a digit CNN |
| [Saving and exporting](docs/FILES-AND-EXPORT.md) | Project JSON, PyTorch files and notebooks, and dataset files |
| [Development](docs/DEVELOPMENT.md) | Local setup, checks, model generation, and dataset attribution |

The [importable model files](public/models/README.md) are editable projects made with the same blocks as the blank canvas.

## Datasets

Add a Dataset block and choose **Browse datasets…** from its dropdown or **Details**. The categorized browser includes the datasets used throughout the course, additional classroom datasets, and the original small examples. Open a family to choose its configuration, inspect example inputs and targets, and see output shapes and training/held-out counts before clicking **Use this dataset**.

| Browser category | Available datasets |
| --- | --- |
| Regression | Auto MPG (full or small training sample), Concrete strength, Diabetes progression, California housing, and simple line, cubic, and plane regression examples. |
| Classification | Breast cancer measurements, Iris flowers, Wine cultivars, Palmer penguins, Concentric rings, and simple threshold, circle/center, parabola, XOR, and neuron-basics examples. |
| Text and sequences | Movie reviews (IMDb sentiment), Football stories, and repeating color-cycle and counting sequences. |
| Images | UCI handwritten digits: 8 × 8 images with ten digit classes. |
| Attention and probabilities | Query/message and three-token attention lookups, plus class-score and softmax experiments. |

**Football stories** has configurations for narrative generation and question answering. Narratives can train on the final next token or on every input position. Question tasks offer word counts, ordered tokens, separate story/question inputs, or a combined input that predicts the answer token. Question coverage can be football location only, mixed questions about the football, people, and possession, or mixed training with a shared football-only evaluation set. The browser shows the supported combinations, vocabularies, and fixed splits.

The eight simple regression and classification examples are also direct shortcuts in the Dataset card's dropdown. For your own data, choose **Custom data** in the browser or the card's **Custom CSV…** or **Text / sentiment…** option. Custom imports accept numeric CSV, text/review/question files, and prepared dataset JSON; prepared JSON supplies its vocabulary, shapes, targets, and splits automatically. Selected data is embedded in saved model projects.

See the [dataset guide](docs/DATASETS.md) for controls and the [dataset sources](public/datasets/README.md) for attribution and exact preparations.

## Course assignments

These 19 assignments progress from regression to a transformer question-answering capstone. Students build editable models in Neural Canvas, use datasets from the browser, and save their models as JSON. Many assignments extend the preceding solution; keep the trained models for later comparisons and fine-tuning. Each link opens the student instructions.

| Assignment | What students explore |
| --- | --- |
| [1. Predicting fuel economy with models you build](https://davbachman.github.io/NeuralCanvas/course/01-mpg.html) | Build one-feature, two-feature, and quadratic regression models on Auto MPG, using training-fitted feature standardization. Compare how added features affect predictions. |
| [2. When a better fit makes worse predictions](https://davbachman.github.io/NeuralCanvas/course/02-regularization.html) | Use a small MPG training sample to investigate polynomial overfitting and L2 regularization. Compare training fit with validation performance. |
| [3. From scores to probabilities](https://davbachman.github.io/NeuralCanvas/course/03-classification.html) | Build one- and two-feature breast-cancer classifiers with logits, binary cross-entropy, and sigmoid probabilities. Explore how a decision threshold changes classifications. |
| [4. From weighted sums to matrix models](https://davbachman.github.io/NeuralCanvas/course/04-matrices.html) | Rewrite MPG regression as a matrix calculation, then classify three Iris species. Introduce concatenation, tensor parameters, matrix multiplication, softmax, and multiclass cross-entropy. |
| [5. When layers become a neural network](https://davbachman.github.io/NeuralCanvas/course/05-mlp.html) | Compare linear regression with a ReLU multilayer perceptron for predicting concrete strength. Investigate how a nonlinear hidden layer learns useful intermediate features. |
| [6. When a straight boundary is not enough](https://davbachman.github.io/NeuralCanvas/course/06-rings.html) | Compare linear and MLP classifiers on three-class concentric rings, examining how nonlinear features change decision boundaries. |
| [7. What can word counts tell us?](https://davbachman.github.io/NeuralCanvas/course/07-word-counts.html) | Build a guided IMDb sentiment classifier, then independently adapt word-count classification to football-story questions. |
| [8. Learn a compact representation of words](https://davbachman.github.io/NeuralCanvas/course/08-embeddings.html) | Preserve word order and compare one-hot inputs with learned word embeddings on football questions, using comparable parameter counts and training budgets. |
| [9. Positional embeddings and separate questions](https://davbachman.github.io/NeuralCanvas/course/09-positional-embeddings.html) | Add learned story positions and a separate question encoder to an embedding MLP. Establish a baseline on the common football-question benchmark. |
| [10. Let questions attend to story tokens](https://davbachman.github.io/NeuralCanvas/course/10-attention.html) | Replace the flattened story input with question-directed cross-attention. Compare accuracy and parameter count with the preceding MLP on the same football questions. |
| [11. Predict the next word](https://davbachman.github.io/NeuralCanvas/course/11-next-word.html) | Turn classification into generation with a football-narrative bigram model that predicts the next token from the preceding token. |
| [12. Control how the model chooses words](https://davbachman.github.io/NeuralCanvas/course/12-sampling.html) | Explore temperature and probabilistic sampling while preserving the trained bigram model's parameters. Apply these generation choices to later language models. |
| [13. Give the generator a longer memory](https://davbachman.github.io/NeuralCanvas/course/13-context.html) | Extend the bigram generator into a fixed-context MLP that uses an ordered sequence of embedded tokens to predict the next token. |
| [14. Share attention across word positions](https://davbachman.github.io/NeuralCanvas/course/14-self-attention.html) | Replace the context MLP with self-attention, learned positions, and a causal mask so each token can attend to its available context. |
| [15. Preserve and transform the representation](https://davbachman.github.io/NeuralCanvas/course/15-residuals.html) | Add an attention output projection, residual connections, and a shared feed-forward MLP to preserve and transform token representations. |
| [16. Normalize the block and learn at every position](https://davbachman.github.io/NeuralCanvas/course/16-normalization.html) | Add layer normalization and train on next-token targets at every position. Make an unchanged-weight final-position evaluation copy for a comparable measurement. |
| [17. Let attention build on attention](https://davbachman.github.io/NeuralCanvas/course/17-stacking.html) | Stack two independently parameterized transformer blocks so later attention can use already contextualized representations. Compare with the trained one-block model. |
| [18. Give each block two attention patterns](https://davbachman.github.io/NeuralCanvas/course/18-multiple-heads.html) | Give each transformer block two independently learned attention heads while keeping the total feature width fixed. Keep the trained generator for the capstone. |
| [19. Fine-tune your generator to answer questions](https://davbachman.github.io/NeuralCanvas/course/19-qa-finetuning.html) | Fine-tune the trained generator to predict answer tokens for football-story questions. Compare before and after fine-tuning, and with the earlier cross-attention model, on the common benchmark. |

Learn more about [David Bachman](https://pzacad.pitzer.edu/~dbachman/) and his AI podcast, [*Entropy Bonus*](https://profbachman.substack.com/).
