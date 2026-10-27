# Assignment 9 — Positional embeddings and separate questions

Public teaching release: see [the course guide](../../../README.md) for standalone setup, packaging, and validation. Wiring, templates, grader sources, and these notes are public. Trained reference models, browser model exports, solution-based tests, and pilot/reproduction tools mentioned below remain in the private NeuralCanvas-Coursework checkout; their paths are historical development references, not required teaching files. Test your own saved models when configuring Gradescope.

Students adapt Assignment 8's embedding MLP to one whole story sequence and a
separate question. Learned token positions and padding masks replace the old
fact-row encoder/positions. Familiar classifier and question-encoder shapes
remain independent work. See [the revision](../../../docs/WHOLE-STORY-REVISION.md).

## Publish and handoff

Publish ../description.html. Students select both datasets in the browser; no dataset attachments are needed. Upload
Gradescope/gradescope-09-positional-embeddings.zip. Require baseline.json and
football.json. Keep instructor references private. Students retain baseline.json
for Assignment 10 and the football score for later comparisons. Same-named
football model files belong in their respective assignment folders.

The handout names the starting file, uses prose for dataset-browser choices and tables for dataset properties,
training and initialization settings, and retains the existing independent
question/classifier shape decisions. The optional custom-question activity is
preserved, using the Story and Question fields.

## Data and construction

Same original texts, answers, vocabulary, order and room-layout split: 3,840
training, 1,920 mixed held-out, 25 tokens and nine answers. football-test.json
has the same training set and 240 held-out ball-location questions, 120 in each
event order, exactly matching the capstone questions/answers/split.

The prepared data uses story-question with maxLength=32 and questionLength=8.
It supplies story IDs [32], question IDs [8], story position IDs [32], story
word mask [32,1], question word mask [8,1], and answer [1], in that output order. The story is document.text;
the separate question is document.question. Story positions are 0–31, including padded slots. No fact dimension, fact mask,
input reshapes or question slices are required.

Share E (25 x 16) between story and question lookups. Use a separate Embedding
lookup with learned P (32 x 16) and the Dataset story position IDs. Add its
position vectors to story word embeddings, then apply the story padding mask. Mask question
embeddings, flatten eight vectors, and encode to 16 features with a matrix,
bias and ReLU. Flatten all 32 story vectors, append the question, then use
528 x 64 + bias/ReLU and 64 x 9 + bias. Separate logits Softmax. Total: 37,417
parameters. Flattening already preserves order; explicit positions also prepare
token representations for the next assignment's attention.

## Recipe, grading and observed outcomes

Tensor CPU, Adam .003, batch 32, shuffle, five continuous epochs, report every
epoch, clip norm 1, patience 0, no decay or penalty. Uniform E/P, Xavier matrices,
zero biases. Optional Auto/WebGL and longer runs do not change grading.

Data/form/train points remain 10/60/25, with train loss <= .75. The fixed-weight
football evaluation earns five points, with no accuracy threshold. The grader
reconstructs operations and canonical data, verifies parameter identity across
the two files, and ignores cached metrics and claimed epochs. Missing or invalid
football.json loses only its five points.

Three seeds (137/211/307) at epoch five: train losses .58758/.58918/.63414;
mixed accuracies 72.66/73.23/71.41%; football scores 148/144/146 out of 240
(61.67/60.00/60.83%). These snapshots are the first five uninterrupted epochs of
ten-epoch native Tensor CPU pilots on clean app
42708105d80c2c47fd2b90a26f5add201a4c2ce5, with the same settings and no checkpoint
selection or optimizer restart. Concurrent cloud times at that checkpoint were
215–242 seconds; no isolated laptop/GPU timing claim is made.

Original pilot evidence remains in pilots/football/paragraph-qa. Current
references and reports are in reference/football-pilot; independent results are
in reference/model-metrics.json and docs/football-release-validation.json.
Superseded fact-row references remain in Git and original pilot directories.
Current loss targets do not use the previous 30-epoch recipe.

## Reproduce and validate

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/football/build-later.py --days 9
RELEASE_ASSIGNMENTS=09-positional-embeddings .venv/bin/python scripts/assignments/package-advanced.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/football/audit-release.py --days 9
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/09-positional-embeddings/source -p test_grader.py -v
NEURALCANVAS_ROOT=<app> PLAYWRIGHT_ROOT=<playwright> APP_URL=<url> node scripts/assignments/football/browser-common-benchmark.cjs
NEURALCANVAS_ROOT=<app> PLAYWRIGHT_ROOT=<playwright> APP_URL=<url> node scripts/assignments/football/browser-own-question.cjs
```

Tests cover canonical data, shape/semantic requirements, shared words, token
positions, padding, leakage, invalid/superseded architectures, longer training,
comparison identity and the extracted current ZIP. Actual browser checks cover
imports, both inference sets, both saves, custom questions, restore, and responsive
handout layout. They do not draw every wire or test hosted Gradescope/deployment.

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

## Position-lookup update (2026-10-27)

The story/question Dataset now outputs story position IDs at slot 2, followed
by story and question masks at slots 3 and 4 and the answer at slot 5. The
assigned model uses the position table through an Embedding lookup before
adding position vectors to word embeddings. This matches the later
self-attention construction. The capstone’s earlier.json uses the same updated
attention model. Graders require this wiring; existing saved-model compatibility
is not part of this update. Dataset files, learned parameter values/counts,
training budgets and numerical targets are preserved. Refresh the current
Gradescope ZIP and the HTML handout when publishing.

Current checks: [position-lookup validation](../../../docs/position-lookup-validation.json).
Trace predictions, Tensor CPU minibatch losses/gradients and one Adam update
match direct table addition with zero numerical difference on the checked models.
The actual browser-saved Assignment 9/10 models and the capstone handoff each
earn 100/100 under the rebuilt local ZIPs. This preserves the original training
evidence; no full-budget retraining was needed. Validation used app
`119696ccad8fbc557a5054cf05313d88477b3dac` with the position-output and handout changes present
(dirty checkout). These local checks do not upload to Canvas or Gradescope.
