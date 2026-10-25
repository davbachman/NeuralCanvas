# Historical advanced-assignment validation

Current numbering: see [SAMPLING-SEQUENCE.md](SAMPLING-SEQUENCE.md). Sampling is
now Assignment 12; historical sections below retain their original numbering.

Current release (2026-10-21): Assignments 11–18 now use football narratives;
Assignment 18 supplies the pretrained checkpoint, and former Assignment 20 is
now Assignment 19 QA. The separate pretraining assignment was removed. See
[NARRATIVE-REVISION.md](NARRATIVE-REVISION.md) and
[STUDENT-CAPSTONE.md](STUDENT-CAPSTONE.md). The report below is historical;
its numbering, datasets and reproduction commands describe that earlier release.


Current football update (2026-10-20): [FOOTBALL-SEQUENCE-REVISION.md](FOOTBALL-SEQUENCE-REVISION.md) describes the implemented Assignments 7–10 and 20. Their earlier dataset/backend statements below are historical. The generation evidence for Assignments 11–19 is retained.

For the current synthetic-narrative Assignments 11–20, see [NARRATIVE-REVISION.md](NARRATIVE-REVISION.md). The measurements and earlier sequence below are historical evidence; they do not validate the replacement dataset.

This report preserves the sequence and evidence from before the stacking revision. Its release labels, archive paths and reproduction scripts describe that historical snapshot, available in Git history at `922605e`. Unpublished draft archives and obsolete scripts have since been removed; they are not alternate current assignments. For current numbering, releases and fresh two-block validation, see [Stacking revision](STACKING-REVISION.md). Historical measurements are not new two-block results.

Historical snapshot: **v2**, revised 2026-10-19. Read [the full trajectory and assessment review](ADVANCED-V2-REVIEW.md), [current suite results](advanced-v2-tests.json), and [current browser checks](advanced-v2-browser.json). The model recipes and three-seed evidence below are retained from v1; they are not new training runs. The revised handouts progressively reduce familiar-step guidance and make all required investigations produce graded artifacts.

The release extends the existing assignments through Day 20. Days 1–8 are unchanged. The extra days separate residual/feed-forward construction from normalization and add two-head attention. The final assignment reuses existing work and needs no new lecture. See the [course plan](REMAINING-SEMESTER-SCHEDULE.md).

## Data and exact models

Stories use the retained 1k-question bAbI Task 1 training source, split by whole story with three duplicate inputs removed: 798 training and 199 validation questions, vocabulary 23, six answers, ten facts and eight word slots. Assignment 9 retains source bytes and split rows. Assignment 10 uses the same ordered fact/question encoders, but replaces fixed flattening with question-to-fact attention. Assignment 11 changes only inference inputs.

Alice uses 551 retained paragraphs from Gutenberg, split before windows are constructed: 13,085 training windows and 2,680 validation windows. Character vocabulary 45, context 8, stride 8. Assignment 12 retains the original source. All Alice models use the same passages, vocabulary and windows. Assignments 17–19 add shifted supervision at every position; comparisons use the common final-position metric. Provenance includes exact source and distributed-file hashes.

The app recipes build ordinary editable palette blocks. No custom student operators or starter models are introduced. Uniform embedding initializers, Xavier matrices, zero bias vectors, unit layer-normalization scales, Tensor CPU, Adam 0.003, batch 32, shuffle, gradient clipping 1, no weight decay or loss penalty, and no early stopping are used throughout. Stories train 40 epochs; Alice models train 5. Nine trainable architectures each have completed runs at seeds 137, 211 and 307. The inference-only and synthesis assignments reuse those models. V2 additionally preserves a base file for the temperature comparison and requires the synthesis final-position evaluation copy as a fifth artifact.

Each `reference/app-pilot.json` records exact settings and source commits; `reference/pilots/` retains the trained seed files for the originating assignment. Models trained before the generation fix use app commit `bac2dc534cda31df7dc631debcb3c068bb4256ab`; later pilots and verification use `d85d4ff`. All app checkouts used for verification were clean and isolated. The shared app checkout was not changed.

## Findings and limits

- The structured MLP fits training stories but reaches only 34–44% held-out accuracy. Question-directed attention reaches 98–99% with the same split and fact/question encoders.
- The context MLP improves on the Alice bigram model. The MLP remains a strong baseline for this small dataset and short context.
- Residual/feed-forward additions improve over attention alone in these pilots. Layer normalization and multiple heads have their own structural lessons; neither is promised to improve this small model's loss in every run.
- All-position and final-position losses average different targets. The seed-137 all-position transformer reports 2.120314 on its training objective and 2.030311 on held-out final positions. The latter is the appropriate comparison with the final-position model.
- Generated text contains repetition and errors. Sampling changes diversity, not knowledge or reliability. Generation masks padding before top-k and renormalizes remaining probabilities; the raw inspection branches include all vocabulary entries.
- Three initialization seeds establish feasible thresholds, not universal convergence or browser timing guarantees. Instructor models and grading answers are private.

## Backend guidance update (2026-10-19)

The originating pilots above used Tensor CPU. The current handouts also allow Auto/WebGL after fresh browser training and independent grading; see [backend validation](BACKEND-VALIDATION.md) for per-model seed counts, full training budgets, package results and renderer limitations. Auto falls back to CPU when its WebGL compatibility probe fails. Existing CPU reference models and numerical targets are unchanged.

## Independent graders

Each upload ZIP contains only its runner, evaluator, canonical data/templates, and setup scripts. Each assignment's `source/` is self-contained. The shared implementation in `scripts/assignments/advanced/` is copied into release sources by `configure-releases.py`.

NumPy evaluation reconstructs inputs from canonical documents and executes saved graph parameters. It ignores cached predictions, metrics, epochs, and labels. Architecture checks include masks, causal direction, parameter independence, output probabilities, and exact objective; one-hot lookup and input aliases are accepted. Comparison fingerprints verify identical learned parameters while permitting a final-row Slice in the evaluation copy. Per-file subprocesses bound memory, runtime, and malformed input effects.

All twelve packaged reference submissions score 100/100. Each grader suite covers renaming/cache independence, data tampering, missing probability branches, target leakage, malformed graphs and unsafe expressions, one-hot equivalence, and the extracted ZIP. Comparison assignments test parameter fingerprints, including changed-weight rejection for Days 19 and 20. The intervention package also tests partial credit with a malformed third file and valid/invalid student-chosen location edits. Temperature branches accept independently verified equivalent scaling expressions. Results are in [advanced-tests.json](advanced-tests.json) and each assignment's `reference/test-log.txt`. App/independent reference metrics agree within 0.000002.

## Browser coverage and generation fix

Fresh Chromium sessions imported the issued prepared JSON datasets into editable reference graphs and exercised File → Save. Checks covered:

- Story baseline import and held-out inference (199 examples).
- Three applied story interventions, navigation through Test/Build/Train, forward inspection, and saved applied inputs. Recomputed predictions were garden, kitchen and bedroom, with unchanged learned parameters.
- Alice import, an additional CPU/Adam training epoch through the controls, held-out inference and generation.
- The all-position model's final-position evaluation copy, reimported with the final-target dataset and evaluated without parameter changes.
- Two-head generation at temperatures 0.5, 1 and 2 with top-k 45, preserving parameters and never emitting padding.

The saved browser files also pass their independent graders; see [advanced-browser-grading.json](advanced-browser-grading.json). Screenshots and saved files are under the relevant `reference/browser/` folders. These checks start from instructor graphs and do not simulate every student wire placement. They were local browser checks, not tests of the deployed website or hosted Gradescope.

The earlier generation reproduction is fixed: continuous generation and repeated one-token calls agree, including reserved-token prompts. Six focused app tests passed across `textGeneration.test.ts` and `TextGenerationControls.test.tsx`. Additional coursework verification checks greedy and seeded sampling across short/full contexts, reserved-token round trips, padding exclusion before top-k, and unchanged graph parameters; see Assignment 19's `reference/generation-verification.json`.

## Commands

From the coursework root, with Python 3.12/NumPy 1.26.4 and the app's Node dependencies installed:

```sh
.venv/bin/python scripts/assignments/package-advanced.py
.venv/bin/python -m unittest discover -s assignments/17-all-positions/source -p test_grader.py -v
```

Use the corresponding assignment directory to run another grader suite. No download or retraining is necessary for package verification. To repeat a model pilot into a new output folder:

```sh
NEURALCANVAS_ROOT=/path/to/NeuralCanvas PILOT_SEEDS=137,211,307 \
  node scripts/assignments/advanced/pilot.mjs \
  assignments/09-positional-embeddings/Canvas/stories.json /tmp/story-pilot sentence-mlp memory-1
NEURALCANVAS_ROOT=/path/to/NeuralCanvas PILOT_SEEDS=137,211,307 \
  node scripts/assignments/advanced/pilot.mjs \
  assignments/12-next-character/Canvas/alice.json /tmp/alice-pilot \
  bigram mlp attention residual transformer all-position two-heads
NEURALCANVAS_ROOT=/path/to/NeuralCanvas \
  node scripts/assignments/advanced/check-generation.mjs
```

`import-pilots.mjs` imports completed raw graph or saved-model files using a JSON manifest of `{kind, seed, path, appCommit}` entries, then independently reopens/evaluates them in the app. `prepare-models.mjs` derives the intervention, final-position and sampling copies from committed seed-137 pilots without changing parameters. `configure-releases.py` refreshes templates, source copies, datasets and grading tables; run it only intentionally, then rebuild packages and tests. These commands overwrite private evidence.

For browser checks, start the app and use an installed Playwright project and Chromium executable:

```sh
PLAYWRIGHT_ROOT=/path/to/playwright-project CHROMIUM_PATH=/path/to/chromium \
  APP_URL=http://127.0.0.1:5173/NeuralCanvas/ \
  node scripts/assignments/advanced/browser.cjs
PLAYWRIGHT_ROOT=/path/to/playwright-project CHROMIUM_PATH=/path/to/chromium \
  APP_URL=http://127.0.0.1:5173/NeuralCanvas/ \
  node scripts/assignments/advanced/browser-interventions.cjs
```

`prepare-stories.py` and `prepare-advanced-data.mjs` recreate base prepared datasets from retained raw sources. Do not regenerate issued datasets with changed upstream input or app preprocessing without a new version. Packaging records handout, dataset and upload ZIP hashes in each `reference/package-sha256.json`.
