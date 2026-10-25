# Question-answer fine-tuning capstone

Current release (2026-10-21): this two-fact pilot is historical. The course now
uses football narratives from Assignment 11 onward and reuses Assignment 17's
trained model in Assignment 19 QA. See [STUDENT-CAPSTONE.md](STUDENT-CAPSTONE.md)
and [NARRATIVE-REVISION.md](NARRATIVE-REVISION.md). Paths and assignment numbers
below refer to the earlier release retained in Git history.


Current football update (2026-10-20): [FOOTBALL-SEQUENCE-REVISION.md](FOOTBALL-SEQUENCE-REVISION.md) describes the implemented Assignments 7–10 and 20. Their earlier dataset/backend statements below are historical. The generation evidence for Assignments 11–19 is retained.

The released Assignment 20 adapts the two-block, two-head narrative transformer
to answer a question appended to a two-fact story. Assignments 11–19 now reserve
`where` in their shared 21-token vocabulary. Their story text, splits, windows,
other token IDs, training recipes and grading thresholds are unchanged. The
embedding and output dimensions, parameter-count hints, references and packages
were updated together. Assignments 1–10 are unchanged by this work.

The story generator, QA baseline and fine-tuned model each have **5,557
parameters**. No vocabulary expansion, additional Param or concatenation is
needed during the capstone. The original 32-position embedding table is retained;
the question task uses its first 15 positions. Only the final logits row receives
answer supervision. The learner changes the data, determines that Slice, and
continues training without randomizing. This introduces fine-tuning while keeping
the generative progression focused on generation.

## Task and split

Example input: `nora went to the kitchen . liam went to the bedroom . where is nora`
with the next-token target `kitchen`. There is no question mark. Each original
history provides two questions, one per person, in the original history split:
512 training and 256 held-out questions. The source has four names and six rooms.
The single answer target is outside the input window; the relevant room appears
in the preceding facts, as intended.

Ordered event histories do not cross splits, including narrative pretraining.
Some held-out histories describe a training state with reversed fact order.
The report therefore separately scores **114 questions with unordered
person/location states absent from training**. Do not describe all 256 questions
as entirely novel states or use these results as evidence of broad language
understanding. Held-out data was used during development and is validation.

## Exact native recipe and independent results

Clean app `2b6b96d8d07ee52e0beca3c3432f9f4e5819af79`, Chromium,
Tensor CPU, Adam 0.003, batch 32, shuffle each epoch, 150 epochs in one run,
report every 10, clip norm 1, patience 0, minimum improvement 0, no decay/penalty.
Each starting model was independently pretrained from fresh parameters on the
21-token narrative data for the assigned 10-epoch budget. Every original tensor
is preserved in its QA baseline. Fine-tuning updates the existing trainable
parameters, including the previously unused input embedding for `where`.

| Seed | Before accuracy | After accuracy | After held-out loss | Unseen-state accuracy | Both questions correct | Browser seconds |
|---|---:|---:|---:|---:|---:|---:|
| 137 | 0.0% | 98.4% | 0.0710 | 97.4% | 96.9% | 730 |
| 211 | 0.0% | 99.6% | 0.0039 | 99.1% | 99.2% | 409 |
| 307 | 0.0% | 100.0% | 0.0001 | 100.0% | 100.0% | 367 |

These full-budget browser-engine runs overlapped other validations on a cloud
machine limited to four CPU cores. Times are contended measurements, not laptop
guarantees or a CPU/GPU comparison. Actual UI callbacks are checked separately,
including dataset replacement, a worker training epoch, held-out inference,
paired prompt generation, File → Save, and responsive handouts. An isolated
timing run, if present, is recorded separately below; it does not replace the
three independent initialization checks.

The assignment uses conservative full-credit targets: answer accuracy at least
85%, training loss at most 0.4 and held-out loss at most 0.6. All three native
checkpoints pass. NumPy evaluation independently recomputes the logits/metrics
from saved operations and weights, with native agreement within 2e-5. It also
verifies canonical data, architecture, target isolation, both questions per
story, and the unchanged starting parameters. The grader does not trust cached
scores, names, epochs or claimed optimizer history.

## What this establishes, and alternatives

The lesson is that a next-token generator can be adapted to a conditional answer
task. It does **not** establish that narrative pretraining is necessary, faster
than learning QA from scratch, or better than the dedicated Assignment 10 model.
Assignment 10 has a different dataset and task; its accuracy is not a direct
comparison. The optional free prompt uses the trained two-fact format and known
names/rooms, not arbitrary questions.

Earlier private screening tested a punctuation token, no punctuation, and a
cloze-style question. Adding `?` was less reliable: one native 150-epoch seed
reached only 75%, versus 95.7% in another. Omitting it and extending the original
20-token checkpoint gave 97.7–100% across three native seeds. The released recipe
replaces that extension with a shared vocabulary from the start. It must not be
confused with the retained exploratory Concatenate graphs.

PyTorch screening compares fine-tuning and training from scratch under the same
QA update budget. Scratch models also learn this small task; results do not
justify a general pretraining advantage. PyTorch is a screening tool here, not
the student platform or the timing source. The released recipe is verified with
the native app and independent NumPy evaluation.

## Submission and release

Submit `generative.json` (30 points), `before.json` (20), and `after.json` (40).
Another 10 points require the generator and baseline to contain the same learned
model. The grader removes only the final-row Slice when comparing their semantic
parameter fingerprints; the reserved token's parameters are included. It cannot
prove a later training trajectory from a final JSON. No written answers, quizzes,
screenshots or required unsubmitted investigations are added. Direct interactive
generation is optional.

`assignments/20-qa-finetuning/Canvas/` contains only the handout and question data.
Its Gradescope folder contains one current ZIP; models, graders and evidence
remain private. Existing unpublished assignments are revised in place.
Local package tests are not hosted Gradescope validation. Neither Canvas nor
Gradescope nor the public app is modified by the Git push.

## Evidence and reproduction

- `pilots/qa-capstone/vocabulary-change.json`: only the reserved token was added.
- `pilots/qa-capstone/reserved-pretraining/`: three native full-budget runs for
  each of eight architectures, plus one-epoch Auto/WebGL compatibility checks.
- `pilots/qa-capstone/reserved-input-*` and `reserved-finetuned-*`: exact baseline
  files, final checkpoints, settings, logs, timings and app/browser provenance.
- `qa-capstone-validation.json` and Assignment 20's `reference/seed-validation.json`:
  independent metrics, paired-question/novel-state diagnostics and model hashes.
- Assignment 20's `reference/test-log.txt`, `packaged-test-results.json` and
  `browser-grading.json`: full-credit ZIP, malformed/partial submissions, leakage,
  baseline preservation, untrained-model rejection and actual UI exports.
- `pilots/qa-capstone/reserved-browser-ui/report.json`: actual UI workflow and
  browser errors/layout checks. Private `screen-*` and earlier `native-*` folders
  retain alternatives, including unsuccessful ones.

With the recorded app checkout and the local browser server available:

```sh
.venv/bin/python scripts/pilots/story-language/prepare-streams.py
.venv/bin/python scripts/pilots/story-language/run-streams.py --output pilots/qa-capstone/reserved-pretraining
.venv/bin/python scripts/pilots/story-language/run-streams.py --output pilots/qa-capstone/reserved-pretraining --seeds 137 --backend auto --epochs 1
.venv/bin/python scripts/pilots/story-language/run-capstone.py
.venv/bin/python scripts/pilots/story-language/time-capstone.py
.venv/bin/python scripts/assignments/advanced/configure-narratives.py
.venv/bin/python scripts/assignments/package-advanced.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/advanced/validate-narratives.py
.venv/bin/python scripts/assignments/qa-capstone/build.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/qa-capstone/validate.py
.venv/bin/python scripts/assignments/advanced/test-narratives.py
.venv/bin/python scripts/assignments/qa-capstone/test-release.py
node scripts/pilots/story-language/browser-capstone.cjs
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/qa-capstone/grade-ui.py
.venv/bin/python scripts/assignments/qa-capstone/write-report.py
```

Set `NEURALCANVAS_ROOT`, `PLAYWRIGHT_ROOT` and `APP_URL` for the local environment.
The committed screening controls need no regeneration for release checks. To
repeat them, use PyTorch and run `scripts/pilots/story-language/qa-capstone.py`
with `--out pilots/qa-capstone/screen-reserved --native-root pilots/qa-capstone/reserved-pretraining --style unpunctuated --epochs 150 --seeds 137,211,307`.
The earlier narrative browser checks, duplication check and final release audit
are listed in `NARRATIVE-REVISION.md`. Native runners skip completed reports;
use a new output directory for a genuine rerun instead of treating old evidence
as new training.

## Separate timing check and scratch controls

After the concurrent training jobs finished, the same seed-137 QA recipe completed in **6.14 minutes** (368.1 seconds), with **98.4%** held-out accuracy. Parameters identical to the earlier seed-137 run: **True**. This is an isolated cloud browser-engine check, not a student-laptop or full-UI timing guarantee.

The following are PyTorch screening controls at 150 epochs, using the same QA data, optimizer/rate/batch size and update budget. They are not native-app timings.

| Seed | From pretrained checkpoint | From scratch |
|---|---:|---:|
| 137 | 98.0% | 100.0% |
| 211 | 100.0% | 91.8% |
| 307 | 100.0% | 99.2% |

During local validation the development server stopped. A UI attempt failed before training and one pretraining attempt was interrupted by the development page reloading; neither is counted as a completed pilot. Retried runs and all UI checks passed. The engine-only harness now uses an isolated document to prevent a dev-server reconnect from reloading a training page. The app source was unchanged; actual UI callbacks are still tested separately.
