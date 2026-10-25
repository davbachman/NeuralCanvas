# Whole-story questions and assignment continuity

Numbering note: this dated report retains its original assignment numbers. See
[SAMPLING-SEQUENCE.md](SAMPLING-SEQUENCE.md) for the current sequence after moving sampling to Assignment 12.

The instructor approved the whole-story replacements for Assignments 9 and 10,
with five-epoch starting budgets, and requested clearer instructions across
Assignments 7–19 while preserving the existing independent work.

## Current sequence and handoffs

| Assignment | Start from / change | Save for the next work |
| --- | --- | --- |
| 7 | Word-count sentiment classifier, then independent football classifier | stories.json model |
| 8 | Copy the football model, use ordered one-hot tokens and an MLP, then copy and adapt to learned embeddings | embedding.json |
| 9 | Copy the embedding MLP; import whole-story/separate-question data, add token positions and a question encoder | baseline.json; required fixed-weight football.json |
| 10 | Copy baseline; replace its story flatten with question-to-token attention and resize the first classifier matrix | attention.json; football.json for Assignment 19 |
| 11 | Deliberate new graph: a small next-token bigram generator | bigram.json |
| 12 | Copy bigram; expand to an ordered context MLP | context.json |
| 13 | Copy context; keep data/word embeddings/loss/probabilities, replace the context MLP with causal self-attention | self-attention.json |
| 14 | Copy self-attention; add projection, residual routes and a token-wise MLP | residual.json |
| 15 | Copy residual; add normalization, then all-position supervision and a fixed-weight evaluation copy | all_positions.json for 16; retain all three submissions |
| 16 | Copy all_positions, duplicate/reconnect an independent second block | stacked.json |
| 17 | Copy stacked; use two heads in both blocks at the same total width | multihead.json for both 18 and 19 |
| 18 | Preserve the trained model and add temperature branches | base.json and sampling.json; keep original multihead |
| 19 | Preserve the Assignment 17 checkpoint, import QA data, change the answer-logit selection and fine-tune | earlier / pretrained / before / after models |

Architecture changes retain editable graph structure and start with fresh
parameters. Evaluation copies preserve trained values. The final QA fine-tuning
also preserves the generator's initial trained values. These distinctions appear
in the handouts, together with tables for settings and clear file handoffs.
Independent matrix shapes, wiring and the existing constrained design tasks are
preserved. No written responses, starter solutions or additional required
investigations were introduced.

## Five-epoch validation and rubric

Both new models use Adam 0.003, batch 32, shuffle, clip 1, patience 0, no decay or
penalty, reporting every five epochs. The retained epoch-five snapshots are the
first five uninterrupted epochs of native ten-epoch pilots; there is no optimizer
restart or best-checkpoint selection. All three seeds use clean app
42708105d80c2c47fd2b90a26f5add201a4c2ce5 and Tensor CPU. Settings are preserved
in the saved five-epoch references. Cloud timings at that checkpoint were
200–242 seconds with concurrent runs; these are not isolated laptop/GPU claims.

| Model | Parameters | Train loss range | Mixed held-out loss | Mixed accuracy | Common football accuracy |
| --- | --- | --- | --- | --- | --- |
| 9: MLP | 37,417 | 0.588–0.634 | 0.661–0.689 | 71.41–73.23% | 60.00–61.67% |
| 10: cross-attention | 6,441 | 0.613–0.655 | 0.660–0.707 | 68.23–70.10% | 58.33–68.75% |

Assignment 9 retains its 10/60/25-point data/form/train rubric and 5-point
unchanged-weight football evaluation, with training loss at most 0.75.
Assignment 10 keeps its existing point allocations, using train/held-out loss
at most 0.80/0.85 and mixed accuracy at least 60%. Neither imposes a football
accuracy minimum or ceiling. Longer training is optional and receives credit
according to saved model quality. The old 30-epoch cutoffs do not apply.

The same canonical 3,840 training and 1,920 mixed held-out questions remain,
with an identical 240-question football comparison set. Original vocabulary,
texts, answers, order and room-layout split are preserved. In the current app,
one structured row carries the entire paragraph (`maxFacts=1`, `factWords=32`).
The graph reshapes it into a token sequence and uses word padding masks; fact
positions and the fact mask are unused. The first eight padded question slots
are retained, and every supplied question fits. The handout matches the app's
actual labels, including Fact 1 and the one-line whole-story experiment. The app
repository was not modified.

Five epochs make 600 QA updates. Ten narrative epochs in Assignment 17 make
480 updates on 1,534 windows; answer-only and all-position supervision also differ.
This is a practical course progression, not an isolated architecture or equal
computation experiment. The larger flattened MLP can still be strong, and no
later-model accuracy win is promised for every student.

Assignment 19 now accepts the new attention football.json as earlier.json.
Its narrative model, QA data, objectives, loss targets and 70% overall threshold
are preserved. The three prior transformer runs reached 73.33%, 77.08%, 78.33%,
with 5,689 parameters. Compared by seed with new Assignment 10, the gain is
9.58–16.67 percentage points. This does not establish a pretraining advantage over
training from scratch.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/football/build-later.py --days 9,10
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/football/build-student-capstone.py
RELEASE_ASSIGNMENTS=09-positional-embeddings,10-attention,19-qa-finetuning .venv/bin/python scripts/assignments/package-advanced.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/football/audit-release.py --days 9,10
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/football/validate-student-capstone.py
```

The builders replace current private references and generated sources; regenerate
UI checks and run the relevant per-assignment suites after rebuilding. Current
Canvas descriptions are edited directly. The narrative generator uses the shared
handout settings formatter so later regeneration retains the presentation and
handoffs. The original native evidence remains in pilots/football/paragraph-qa.
Local browser, NumPy and extracted ZIP validation are separate from hosted
Gradescope and the deployed app. A repository push does not publish to either
course site.

## Completed local checks

All 117 grader tests pass: Assignment 9 has 39, Assignment 10 has 41, and the
capstone has 37. They include the rebuilt extracted upload ZIPs, valid and
partial submissions, malformed files, masks, leakage, old architectures,
optional longer training, and parameter preservation. Current references earn
100/100. Actual browser-saved attention/football files and the capstone handoff
also earn 100/100 with the respective local ZIPs.

All three five-epoch seeds meet the new targets under independent NumPy
execution, with native loss/accuracy agreement. The original narrative and QA
checkpoints still pass. Browser checks cover mixed/football inference, Save as,
reimports, unchanged weights/epochs, and five custom story/question cases with
independent probabilities and restore/save checks. The 13 handouts pass phone
and desktop layout checks, including required settings tables and file handoffs;
representative settings screenshots were visually reviewed. The complete
single-version/package audit passes, including Assignment 17 → 19 ownership.

Datasets and grader configurations/sources for Assignments 7, 8 and 11–18 are
byte-for-byte unchanged. Assignments 1–6 are untouched. Original story/question
texts, labels, order, vocabulary and splits for 9/10 are retained. Changed
handouts and upload manifests are consistent. These checks are local; no Canvas
or Gradescope publishing was performed.

## Subsequent prepared-data and settings update (2026-10-22)

The later [prepared dataset update](PREPARED-DATASETS.md) changes the packaging
of Assignment 7/8 inputs to JSON without altering numerical inputs, vocabularies,
splits or grading thresholds. Their raw CSVs are now private provenance fixtures.
It corrects control labels/order in all 7–19 settings tables and separates
prepared-data properties from actual import controls. The earlier unchanged-data
statement above describes the whole-story revision before this packaging change.

## Subsequent direct-input update (2026-10-22)

The [direct story/question update](DIRECT-STORY-INPUTS.md) replaces the legacy
single-fact transport described above. The app supplies the intended story,
question and mask shapes directly. Data/parameter identity and the common task
are preserved, and dataset-properties tables replace filename-only import tables.
