# Assignment 7 — Word counts and football stories

Public teaching release: see [the course guide](../../../README.md) for standalone setup, packaging, and validation. Wiring, templates, grader sources, and these notes are public. Trained reference models, browser model exports, solution-based tests, and pilot/reproduction tools mentioned below remain in the private NeuralCanvas-Coursework checkout; their paths are historical development references, not required teaching files. Test your own saved models when configuring Gradescope.

Keep the guided IMDb sentiment model (60 points) and replace the independent
story classifier (40 points) with the football questions used in Assignment 8.
Students submit reviews.json and stories.json. No quiz or written grading.
The review-reversal activity remains explicitly optional because the saved
count-vector file does not establish that reversal was performed.

## Publish

Paste ../description.html into Canvas. Dataset attachments are unnecessary; students use the dataset browser.
Upload Gradescope/gradescope-counts-07.zip. The unpublished assignment is revised
in place. The IMDb data, model, recipe and rubric are unchanged. A Git push does
not publish to Canvas or Gradescope.

## Football data and lesson

There are 480 training and 240 held-out questions, five events each, two people
and six rooms. Questions ask where the football is. Room-layout groups remain
in one split, including both actor assignments and both drop orders. Dropping
before moving versus moving before dropping produces identical word counts but
different correct answers. Every held-out pair therefore defeats any deterministic
count-based classifier on at least one member. Do not disclose that result in
the independent task; use the saved predictions to motivate the next lecture.
The linear model's observed accuracy is 30.8%, below the 50% pairwise ceiling.
The held-out set was used during task selection and is validation data.

Import football-counts.json through the Dataset text importer. It embeds the
validated multiclass task, word/punctuation tokenizer, lowercase processing,
word counts, vocabulary and 23-token limit; other import-dialog controls are ignored. The fitted vocabulary has 16 entries. The model
has 16 × 6 weights and six biases (102 parameters), logits loss and a Softmax
branch. Train with Tensor CPU, Adam .02, batch 32, shuffle, 100 epochs in one run,
report every 10, clipping/stopping/minimum improvement 0, no decay or penalty.
Use Xavier and zero initialization. Train and held-out loss limits are 1.25 and
1.45; accuracy is informational. Never penalize a student for a better result.

## Evidence and grading

Three native app runs on clean commit 2b6b96d8d07ee52e0beca3c3432f9f4e5819af79
produced training loss 1.2069–1.2071, held-out loss 1.3527–1.3532 and 30.83%
accuracy. The independent NumPy grader agrees. Current evidence is in
reference/football-pilot and reference/browser; older backend-validation files
refer to the preceding story task. IMDb evidence is retained unchanged.
The retained browser training checks started with fresh editable graph fixtures,
imported the raw CSV, trained through the UI, inspected held-out probabilities
with original row numbers, and saved a submission. They do not reproduce every student wire or measure a
student laptop. Hosted Gradescope and the deployed student website are untested.

The grader reconstructs count features from canonical documents and fitted
vocabulary. It rejects data/split changes, target leakage, wrong architecture,
wrong loss, absent probability branches and malformed values. Cached scores and
epoch counts are ignored; required work is represented by the two model files.

## Reproduce

From the coursework root, with NEURALCANVAS_ROOT set to the clean app checkout:

```sh
COUNTS_ONLY=1 node scripts/assignments/football/pilot-early.mjs
.venv/bin/python scripts/assignments/football/install-early.py
node scripts/assignments/football/browser-early.cjs
.venv/bin/python scripts/assignments/package-word-counts.py
.venv/bin/python -m unittest discover -s assignments/07-word-counts/source -p test_grader.py -v
```

Browser reproduction also needs PLAYWRIGHT_ROOT and APP_URL. Pilots overwrite
private evidence. Packaging uses committed fixtures and needs no fresh training.

## Instruction organization update

The current Canvas handout uses short dataset import instructions and tables of dataset properties, training controls and initialization choices and makes the prior-model handoff explicit. Independent shape and wiring work, datasets, model forms, training budgets and grading rules are preserved. See [the whole-story and continuity revision](../../../docs/WHOLE-STORY-REVISION.md).

## Prepared dataset and control correspondence update (2026-10-22)

Both student datasets are prepared JSON, exported from reference/data.json with
only the display filename changed. No vocabulary refitting, new split, new
training, model change or numerical rubric change was made. Original CSV bytes
and their provenance hashes remain under reference/raw. Saved models with the
original CSV filename metadata remain valid; filenames do not change features.

The current browser check imports each JSON with deliberately incompatible
dialog controls, saves the model, checks exact embedded preprocessing, preserves
learned parameters, and earns full credit under the independent grader. It also
checks every settings table in Assignments 7–19 against rendered app labels,
option values and ordering. Evidence: ../../docs/settings-table-ui-validation.json,
clean app commit 871a4a8a7dc733e6296e4c1d6c017abb18eef297. This verifies the
new import workflow without repeating the retained full-budget training runs.

To check the app’s prepared files against committed canonical fixtures, run
`.venv/bin/python scripts/assignments/prepared_early.py` before packaging.
The football installer also performs this check. Do not run the older
prepare-text.py/browser-text.cjs workflows to rebuild this release; they
predate the current football sequence.

## Dataset browser workflow

Students open **Dataset → Details → Browse datasets…** and click **Use this dataset** after selecting the following choices. Datasets live in the app’s `public/datasets/` directory; no assignment-local copies or Canvas attachments are needed. Read [description.md](../description.md) on GitHub or use `../description.html` in Canvas. Model JSON submissions and grading targets are unchanged.

- **Text and sequences → Movie reviews**.

- **Text and sequences → Football stories**; preset: **Football location · word counts**.


Browser/grader compatibility is checked with app 417be2e and the original canonical data. See [the dataset browser update](../../../docs/DATASET-BROWSER-ASSIGNMENTS.md).
