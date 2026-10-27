# Remaining semester: 20 class days, 19 assignments

Working plan revised for the approved whole-story sequence. Lectures prepare students for independent work
in Neural Canvas. Assignments 1–6 retain their issued materials. The approved
football progression now uses one corpus from generation through QA, eliminating
the separate pretraining homework. Assignment 18 supplies the trained checkpoint,
Assignment 12 teaches sampling on the bigram model, and Assignment 19 fine-tunes for questions.
The final class slot is available; this revision adds no replacement assignment.

| Day | Lecture or discussion | Corresponding assignment |
|---|---|---|
| 1 — Tuesday | Introduce the app; build linear regressors for its 1D and 2D toy datasets. | MPG: compare one input column, two input columns, then adding a squared feature. Keep the split fixed and compare training and validation error. |
| 2 | L2 for controlling overfitting; compare training and validation error as model flexibility and penalty strength change. | MPG: compare a linear baseline, an unregularized fifth-degree polynomial, and the same polynomial with L2 (Assignment 2). |
| 3 | Sigmoid, binary cross-entropy, and classification thresholds. | Breast-cancer diagnostic dataset: compare one- and two-feature logistic classifiers, then explore threshold tradeoffs ([Assignment 3](../assignments/03-classification/description.md)). |
| 4 | Express a familiar regression as Xw+b using concatenation and matrix Params; extend to logits, softmax, and multiclass cross-entropy. | Rebuild two-feature MPG regression with a matrix, then build an Iris multiclass classifier ([Assignment 4](../assignments/04-matrices/description.md)). |
| 5 | Compose linear maps; demonstrate that the result is still linear. Introduce nonlinear activations and an MLP. | Concrete compressive strength: compare a linear model with a regression MLP ([Assignment 5](../assignments/05-mlp/description.md)). |
| 6 | Consolidation exception: classification and MLP decision boundaries require no new lecture. This class is canceled; in a future offering, use the slot for a discussion of accuracy, fairness, and deployment decisions. | Three-class concentric rings (lower outer, upper outer, inner): compare linear and MLP classifiers; handwritten digits as an optional extension ([Assignment 6](../assignments/06-rings/description.md)). |
| 7 | Word-count vectors, vocabulary fitting and unknown words; transfer from sentiment to football questions. | Guided IMDb classifier, then independent football word-count classifier. Submit reviews.json and stories.json; review reversal is optional ([Assignment 7](../assignments/07-word-counts/description.md)). |
| 8 | Ordered one-hot inputs versus learned embeddings under a parameter/update budget. | On the same football examples, adapt an eight-unit one-hot MLP to four-dimensional embeddings and 32 hidden units. Ten epochs each, 3,190 versus 3,242 Params ([Assignment 8](../assignments/08-embeddings/description.md)). |
| 9 | Whole-story token positions, a separate question encoder and token padding masks. | Broaden football stories to person, football-location and possession questions; adapt Assignment 8’s embedding MLP into the 37,417-parameter story/question model and train five epochs. Submit baseline.json and an unchanged-weight football.json evaluation copy (5 points); its 240-question accuracy is the baseline for Assignment 10 and the final capstone ([Assignment 9](../assignments/09-positional-embeddings/description.md)). |
| 10 | Question-directed cross-attention and compact token selection; assess parameter cost and question-group tradeoffs. | Modify Assignment 9 into the 6,441-parameter attention model and train five epochs, then submit an unchanged-weight copy on the balanced football-location evaluation. Preserve it for Assignment 19 ([Assignment 10](../assignments/10-attention/description.md)). |
| 11 | Next-word vocabulary logits, cross-entropy and autoregressive generation. | Build a bigram generator on football narratives; feed a chosen token back to generate more text ([Assignment 11](../assignments/11-next-word/description.md)). |
| 12 | Temperature controls concentration; top-k restricts candidate tokens without changing learned weights. | Use Assignment 11’s bigram model to submit independently constructed temperature, top-k and combined branches plus an unchanged base model; optional generation stays in Neural Canvas ([Assignment 12](../assignments/12-sampling/description.md)). |
| 13 | A single word loses context; preserve ordered multiword context with an MLP. | Independently extend the bigram model to all 32 input tokens and a 64-unit hidden layer; compare loss and parameter cost ([Assignment 13](../assignments/13-context/description.md)). |
| 14 | Self-attention shares projections across positions; position embeddings and causal masking. | Build word-level causal attention on the same windows, using substantially fewer parameters than the flattened MLP ([Assignment 14](../assignments/14-self-attention/description.md)). |
| 15 | Residual paths preserve direct routes; shared nonlinear feed-forward transformations add capacity. | Add the output projection, two residual additions and token-wise MLP ([Assignment 15](../assignments/15-residuals/description.md)). |
| 16 | Layer normalization controls feature scales; all-position supervision uses more learning targets per window. | Train the normalized final-position model, train an all-position variant, then submit an unchanged-weight final-position evaluation copy. Compare metrics on the same prediction task ([Assignment 16](../assignments/16-normalization/description.md)). |
| 17 | Depth lets later attention use contextual information from earlier attention. | Build an independent second block and compare with Assignment 16’s all-position one-block model; extra compositional capacity has a cost ([Assignment 17](../assignments/17-stacking/description.md)). |
| 18 | Multiple heads provide distinct attention distributions at fixed total width and parameter count. | Replace each block’s head with two width-8 heads; retain both successive blocks ([Assignment 18](../assignments/18-multiple-heads/description.md)). |
| 19 | Adapt a pretrained generative transformer to answer tokens; compare on the same football task. | Reuse Assignment 18’s trained generator, prepare an unchanged-weight QA baseline and fine-tune. Submit earlier.json, pretrained.json, before.json and after.json ([Assignment 19](../assignments/19-qa-finetuning/description.md)). |
| 20 | Available class meeting after the assignment consolidation; no new topic assigned here. | No Assignment 20. |

## Teaching and assessment

Every substantial required investigation produces an automatically graded model.
Generation and interactive prompting are optional. No written explanations,
screenshots or separate quiz are required. New operations receive guidance;
familiar wiring and shape decisions become progressively independent.

Each lesson has a useful advantage or tradeoff to discuss; accuracy need not
improve at every step. The context MLP improves substantially over a bigram;
attention shares parameters; normalization improves the tested initializations;
depth and multiple heads add representational capacity; temperature controls
probability concentration without learning new weights. Architectural parallelism
is not a promise of a browser speedup.

Assignments 11–18 share the 25-token football corpus, 32-token contexts and
room-layout split. Assignment 18 trains ten all-position epochs from fresh
parameters. QA in Assignment 19 uses five additional answer-only epochs from
those values, keeping the epoch counter. Both use Adam 0.003 and batch 32.
WebGL can use a supported GPU; CPU is a fallback. Runtime and epochs are not
graded. Optional longer training requires checkpoints and may worsen accuracy.

The QA task uses exactly Assignment 10's 240 held-out football questions. Its
70% overall target replaces the former absolute/per-group 80% requirements.
Both event orders remain diagnostic. No penalty applies to a strong earlier
model and the comparison does not establish a pretraining advantage over scratch.

Current recipes, thresholds and validation are in
[NARRATIVE-REVISION.md](NARRATIVE-REVISION.md),
[STUDENT-CAPSTONE.md](STUDENT-CAPSTONE.md) and each assignment's Instructor/INSTRUCTOR.md.
Historical Alice, bAbI, two-fact and supplied-checkpoint reports retain their
original numbering and do not validate current datasets or laptop GPU runtime.
