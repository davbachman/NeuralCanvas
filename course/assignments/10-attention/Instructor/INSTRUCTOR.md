# Assignment 10 — Question-directed cross-attention over story tokens

Public teaching release: see [the course guide](../../../README.md) for standalone setup, packaging, and validation. Wiring, templates, grader sources, and these notes are public. Trained reference models, browser model exports, solution-based tests, and pilot/reproduction tools mentioned below remain in the private NeuralCanvas-Coursework checkout; their paths are historical development references, not required teaching files. Test your own saved models when configuring Gradescope.

Students modify Assignment 9's baseline graph, retaining the whole-story token
and position embeddings, Dataset word masks and question encoder. Replace only
the flattened story path with attention and resize the first classifier matrix.
Keep familiar shape/wiring work independent. See
[the revision](../../../docs/WHOLE-STORY-REVISION.md).

## Publish and handoff

Publish ../description.html. Students select both datasets in the browser; no dataset attachments are needed. Upload
Gradescope/gradescope-10-attention.zip. Require attention.json and football.json.
The latter becomes earlier.json in Assignment 19. Keep the trained main graph
as a construction reference for self-attention in Assignment 14.

## Data, architecture and recipe

Dataset bytes match Assignment 9: 3,840 training and 1,920 mixed held-out
questions; 25 tokens, nine answer labels. One whole story occupies 32 token
slots; the separately encoded question uses eight. The app supplies the four story/question inputs directly, in the same
shapes as Assignment 9; no input adapters or unused fact outputs remain.
The common football copy has exactly 240 held-out questions, with both event
orders equally represented, and matches Assignment 19's questions/answers/split.

Retain shared E (25 x 16), learned P (32 x 16), word masks and a 16-feature
question encoder. Q comes from the encoded masked question; K/V come from the
positioned story-token embeddings. Independent 16 x 16 projections, no biases.
Explicitly compute Q K-transpose / 4, exclude padded tokens with the story word
mask, Softmax, then multiply by V. Concatenate attended row first and encoded
question before Q second. Keep the 64-unit ReLU/nine-logit head and separate
output Softmax. Total: 6,441 parameters, 82.8% fewer than Assignment 9's 37,417.

Five continuous epochs, report every epoch, Tensor CPU, Adam .003, batch 32,
shuffle, clip norm 1, patience 0, no decay/penalty. Uniform tables, Xavier
matrices, zero biases. The changed architecture randomizes fresh; the evaluation
copy never randomizes or trains. Longer training is optional and can help or
hurt held-out accuracy. Auto/WebGL may be used; no hardware speed is graded.

## Grading and evidence

Point allocations are unchanged: main data/form/train/held-out/accuracy are
10/30/10/10/10; football data/form are 5/15; parameter identity earns ten.
New five-epoch limits are .80 training loss, .85 mixed held-out loss and 60%
mixed accuracy. Football performance is diagnostic, with no minimum or ceiling.
A student whose earlier model performs unusually well must not lose credit.

All three native epoch-five references pass: train losses .65458/.61333/.65354,
mixed accuracies 69.79/70.10/68.23%, football scores 140/145/165 out of 240
(58.33/60.42/68.75%). The checkpoints are the first five uninterrupted epochs of
ten-epoch native Tensor CPU runs, not optimizer restarts or selected best epochs.
Clean app: 42708105d80c2c47fd2b90a26f5add201a4c2ce5. Concurrent cloud checkpoint
times were 200–224 seconds, not isolated laptop/GPU measurements.

The MLP can outperform attention; compact selection is the demonstrated lesson.
The later transformer has contextual processing and 5,689 parameters. Its
previous five-QA-epoch results were 73.33–78.33%, with a separate UI run at 80%.
These complete course workflows also differ in supervision/pretraining budgets;
no isolated architecture/pretraining advantage is claimed.

The bounded evaluator requires canonical data, graph operations and shapes,
question input, word/score masks, parameter counts and output probabilities.
Equivalent affine score-mask expressions are normalized using bounded AST
parsing and exact rational coefficients. With the score mask present, omitting
explicit zeroing of padded story vectors is numerically equivalent and accepted.
No superseded sentence-encoder architecture is accepted.

## Reproduce and validate

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/football/build-later.py --days 10
RELEASE_ASSIGNMENTS=10-attention .venv/bin/python scripts/assignments/package-advanced.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/football/audit-release.py --days 10
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/10-attention/source -p test_grader.py -v
NEURALCANVAS_ROOT=<app> PLAYWRIGHT_ROOT=<playwright> APP_URL=<url> node scripts/assignments/football/browser-attention.cjs
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/football/validate-attention-ui.py
```

Independent metrics, retained checkpoint hashes and original native reports are
in reference/ and docs/football-release-validation.json. Current UI checks use
both mask formulas, both inference sets, unchanged-weight saves and reimports.
The actual saved football model is tested as capstone earlier.json. Extracted
ZIP checks are local; hosted Gradescope and the deployed app are not asserted.

## Direct story/question inputs and handout update (2026-10-22)

Assignments 9/10 and this capstone's earlier-model reference now use the app's
story-question representation. Six nontrainable input adapters are removed;
vocabulary, class IDs, original story/question text, document order/splits,
learned parameters, epoch counts, recipes and numerical targets are preserved.
The capstone's transformer QA input remains one 23-token sequence and an answer
next-token target. See ../../docs/DIRECT-STORY-INPUTS.md for conversion and checks.

The current handout uses file instructions in prose and a Prepared dataset
properties table for vocabulary, lengths, shapes, answer space and split sizes.
Independent Param-shape and model-wiring work is retained. Browser and native
checks use clean app 871a4a8a7dc733e6296e4c1d6c017abb18eef297; historical full-budget
training evidence retains its original app commit.

## Dataset browser workflow

Students open **Dataset → Details → Browse datasets…** and click **Use this dataset** after selecting the following choices. Datasets live in the app’s `public/datasets/` directory; no assignment-local copies or Canvas attachments are needed. Read [description.md](../description.md) on GitHub or use `../description.html` in Canvas. Model JSON submissions and grading targets are unchanged.

- **Text and sequences → Football stories**; preset: **Mixed-question classifier**.

- **Text and sequences → Football stories**; preset: **Mixed training · football benchmark**.


Browser/grader compatibility is checked with app 417be2e and the original canonical data. See [the dataset browser update](../../../docs/DATASET-BROWSER-ASSIGNMENTS.md).

## Loss-reporting update (2026-10-27)

The current student recipe reports training and held-out loss every epoch. The five-epoch default now produces six plotted points, including the starting point. This changes evaluation and plotting frequency, not parameter updates or grading. Early stopping is disabled. Extra reports require extra evaluation work; no runtime improvement is claimed. Retained pilot reports and the recipe metadata in `reference/data.json` record the original validation interval; they are historical evidence and are not grading criteria.
