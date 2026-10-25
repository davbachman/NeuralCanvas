# Assignment 8 — Compact embeddings within a parameter budget

Public teaching release: see [the course guide](../../README.md) for standalone setup, packaging, and validation. Wiring, templates, grader sources, and these notes are public. Trained reference models, browser model exports, solution-based tests, and pilot/reproduction tools mentioned below remain in the private NeuralCanvas-Coursework checkout; their paths are historical development references, not required teaching files. Test your own saved models when configuring Gradescope.

Use exactly Assignment 7's football documents and split, with the prepared
ordered-token representation in football-tokens.json. Students build an ordered
one-hot MLP, then independently adapt it to four-dimensional embeddings and a
wider hidden layer. Guidance explains how to add the new Embedding block; the
consequent reshape, layer dimensions and wiring remain student work.

## Publish

Paste Canvas/description.html into Canvas; no dataset attachment is needed. Upload
Gradescope/gradescope-embeddings-08.zip. Require onehot.json and embedding.json.
Only Canvas files are student-facing. This replaces the unpublished start/end
exercise in place; Git retains earlier releases. A push does not publish sites.

## Construction and recipe

The fixed 23-token input vocabulary has 17 entries, including unknown and pad.
One-hot: 23 × 17 → reshape 1 × 391 → 8-unit ReLU → 6 logits.
Embedding: table 17 × 4 → 23 × 4 → reshape 1 × 92 → 32-unit ReLU → 6 logits.
Both dense layers include biases; use Cross entropy (logits) and a separate
Softmax branch. Counts are 3,190 and 3,242 parameters (1.63% difference).

Both start fresh and train ten epochs in one run with Tensor CPU, Adam .02,
batch 32, shuffle, report every 5, no clipping, stopping, decay or loss penalty.
Uniform embedding, Xavier matrices, zero biases. Ten epochs is the comparison
budget; the grader evaluates model artifacts, not unverifiable training history.

| Model | Seed | Train loss | Held-out loss | Accuracy |
|---|---:|---:|---:|---:|
| One-hot | 137 | .6838 | .8216 | 65.83% |
| One-hot | 211 | .5770 | .7691 | 61.25% |
| One-hot | 307 | 1.3403 | 1.3822 | 25.83% |
| Embedding | 137 | .2596 | .9606 | 72.92% |
| Embedding | 211 | .2223 | .7322 | 74.58% |
| Embedding | 307 | .2207 | .7087 | 76.25% |

Embedding improves accuracy in all three matched initializations. It does not
necessarily improve held-out cross entropy: its mistakes may be more confident.
Do not promise perfection, equal runtime, or universal superiority. The first
screen with eight-dimensional embeddings showed no consistent accuracy advantage;
it is retained under pilots/football/course-sequence and was not selected.
The selected comparison matches parameter and update budgets, not elapsed time.

## Rubric and evidence

One-hot earns 40 points: data 5, form 20, train loss ≤1.50 for 10, held-out loss
≤1.55 for 5. Embedding earns 60: data 5, form 25, train loss ≤.35 for 10,
held-out loss ≤1.10 for 10, accuracy ≥70% for 10. Structural verification gates
performance credit. A stronger one-hot solution is never penalized.

Current native evidence, with independent metric agreement, is under
reference/football-pilot. Current UI exports are under reference/browser.
Training used clean app 2b6b96d8d07ee52e0beca3c3432f9f4e5819af79. The retained training UI checks imported
the original CSV, applied the reshape control, trained ten epochs, inspected
held-out outputs, and saved. They begin from fresh instructor graphs rather than drawing every wire.
Cloud timings are not laptop/GPU promises. Old start-end-pilot and
backend-validation folders document the preceding dataset, not this release.
Hosted Gradescope and the deployed student website are untested.

Tests independently reconstruct tokens and labels, simulate football state,
verify group-disjoint splits and matched opposite-answer pairs, and cover all
six pilots, UI exports, equivalent one-hot @ table lookup, invalid structures,
leakage, malformed submissions, and extracted upload ZIPs. No trained answer
model is distributed to students.

## Reproduce

```sh
EMBED_DIMS=4 EARLY_EPOCHS=10 node scripts/assignments/football/pilot-early.mjs
.venv/bin/python scripts/assignments/football/install-early.py
node scripts/assignments/football/browser-early.cjs
.venv/bin/python scripts/assignments/package-embeddings.py
.venv/bin/python -m unittest discover -s assignments/08-embeddings/source -p test_grader.py -v
```

Set NEURALCANVAS_ROOT; browser checks also require PLAYWRIGHT_ROOT and APP_URL.
Pilots overwrite evidence; packaging uses committed fixtures without retraining.

## Instruction organization update

The current Canvas handout uses short dataset import instructions and tables of dataset properties, training controls and initialization choices and makes the prior-model handoff explicit. Independent shape and wiring work, datasets, model forms, training budgets and grading rules are preserved. See [the whole-story and continuity revision](../../docs/WHOLE-STORY-REVISION.md).

## Prepared dataset and control correspondence update (2026-10-22)

Students replace Assignment 7's count dataset with football-tokens.json; the
JSON embeds the validated token representation, 17-entry vocabulary, 23-token
length and original split. They need not configure text import controls. The
same prepared file serves both Part A and Part B. Original CSV bytes and hashes
remain private under reference/raw; the canonical numerical data, model forms,
training budget and rubric are unchanged. Only the prepared file's display
filename differs from the original canonical textData metadata.

The current browser check imports and saves this file with deliberately wrong
dialog controls, confirms exact preprocessing and unchanged learned parameters,
and awards full credit with the independent grader. Both part models also pass
the prepared-data equivalence tests. It checks settings labels/options/order
throughout Assignments 7–19 on clean app commit
871a4a8a7dc733e6296e4c1d6c017abb18eef297. Evidence:
../../docs/settings-table-ui-validation.json. Full-budget training evidence is
retained, not rerun for this equivalent input packaging change.

Re-export committed prepared data with
`.venv/bin/python scripts/assignments/prepared_early.py` before packaging; the
football installer also performs this step. The older generic text preparation
and browser scripts predate this sequence and are not current release builders.

## Dataset browser workflow

Students open **Dataset → Details → Browse datasets…** and click **Use this dataset** after selecting the following choices. Dataset files retained beside the handout are canonical provenance/compatibility copies; they do not need to be attached in Canvas. Model JSON submissions and grading targets are unchanged.

- **Text and sequences → Football stories**; preset: **Football location · word counts**; Model inputs and prediction: **One ordered token sequence → answer class**.


Browser/grader compatibility is checked with app 417be2e and the original canonical data. See [the dataset browser update](../../docs/DATASET-BROWSER-ASSIGNMENTS.md).
